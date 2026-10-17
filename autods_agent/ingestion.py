"""
StreamingIngestAgent  (Phase 2 – Big Data Ingestion)
======================================================
Handles scalable loading of large datasets using three strategies:

1. **Pandas (small data)**  – direct ``pd.read_csv / read_parquet`` for files
   whose in-memory footprint is expected to be below *chunk_threshold_mb*.

2. **Dask (medium-large data)** – ``dask.dataframe.read_csv/read_parquet``
   with automatic partition sizing.  Materialises to pandas via ``.compute()``
   so downstream agents receive a standard ``pd.DataFrame``.  If Dask is not
   installed the agent falls back to pandas and logs a warning.

3. **Chunked pandas (fallback)** – iterative ``pd.read_csv(chunksize=…)``
   aggregation that keeps peak memory bounded when Dask is unavailable and
   the file is large.

Parquet support is provided via ``pyarrow`` (or ``fastparquet``); a graceful
warning is logged if neither is present.

Outputs
-------
The agent's ``run()`` method always returns a ``pd.DataFrame`` regardless of
which path was used, plus a metadata dict that the Orchestrator passes on to
the ``DataProfilerAgent``.

Usage (standalone)
------------------
>>> from autods_agent.ingestion import StreamingIngestAgent
>>> from autods_agent.utils import RunLog
>>> from autods_agent.llm_reasoner import RuleBasedReasoner
>>> agent = StreamingIngestAgent(RuleBasedReasoner(), RunLog())
>>> df, meta = agent.run("path/to/large.csv", target_col="label")
"""

from __future__ import annotations

import os
import warnings
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------

try:
    import dask.dataframe as dd
    _DASK_OK = True
except ImportError:
    _DASK_OK = False

try:
    import pyarrow  # noqa: F401
    _PARQUET_OK = True
except ImportError:
    try:
        import fastparquet  # noqa: F401
        _PARQUET_OK = True
    except ImportError:
        _PARQUET_OK = False

# Default threshold (MB) above which we switch to Dask / chunked reading
DEFAULT_CHUNK_THRESHOLD_MB = 100


class StreamingIngestAgent:
    """
    Scalable dataset ingestion agent.

    Parameters
    ----------
    reasoner : RuleBasedReasoner | AnthropicReasoner
        Shared reasoner instance (for ``decide()`` calls).
    log : RunLog
        Shared audit log.
    chunk_threshold_mb : float
        If the raw file is larger than this, Dask (or chunked-pandas) is used.
    dask_npartitions : int | None
        Target number of Dask partitions.  ``None`` → auto (1 per 64 MB).
    pandas_chunksize : int
        Row chunk size for the chunked-pandas fallback.
    sample_rows : int | None
        If set, the agent returns only the first *sample_rows* rows (useful
        for quick demos without loading the entire file).
    """

    def __init__(
        self,
        reasoner,
        log,
        chunk_threshold_mb: float = DEFAULT_CHUNK_THRESHOLD_MB,
        dask_npartitions: Optional[int] = None,
        pandas_chunksize: int = 50_000,
        sample_rows: Optional[int] = None,
    ):
        self.reasoner = reasoner
        self.log = log
        self.chunk_threshold_mb = chunk_threshold_mb
        self.dask_npartitions = dask_npartitions
        self.pandas_chunksize = pandas_chunksize
        self.sample_rows = sample_rows

    # ------------------------------------------------------------------
    def run(
        self,
        source: "str | pd.DataFrame",
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Load data from *source* and return ``(df, metadata)``.

        Parameters
        ----------
        source : str | pd.DataFrame
            File path (CSV or Parquet) **or** an already-loaded DataFrame.
        target_col : str, optional
            Target column name (used only for validation logging).

        Returns
        -------
        df : pd.DataFrame
            Loaded (and optionally sampled) dataset.
        metadata : dict
            Ingestion metadata: ``backend``, ``file_size_mb``, ``n_rows``,
            ``n_cols``, ``source_type``, ``parquet``.
        """
        # ---- pass-through for in-memory DataFrames -------------------------
        if isinstance(source, pd.DataFrame):
            df = source
            meta = {
                "backend": "in-memory",
                "file_size_mb": df.memory_usage(deep=True).sum() / 1e6,
                "n_rows": len(df),
                "n_cols": df.shape[1],
                "source_type": "dataframe",
                "parquet": False,
            }
            self.log.add(
                "Ingestion",
                f"source=in-memory, rows={len(df):,}, cols={df.shape[1]}",
                "DataFrame passed directly; no file I/O required.",
                extra=meta,
            )
            if self.sample_rows and len(df) > self.sample_rows:
                df = df.head(self.sample_rows)
            return df, meta

        # ---- file-based ingestion ------------------------------------------
        path = str(source)
        if not os.path.exists(path):
            raise FileNotFoundError(f"StreamingIngestAgent: file not found – {path!r}")

        file_size_mb = os.path.getsize(path) / 1e6
        is_parquet = path.lower().endswith((".parquet", ".pq", ".parq"))

        # Choose strategy
        if file_size_mb > self.chunk_threshold_mb:
            if _DASK_OK:
                strategy = "dask"
            else:
                strategy = "chunked-pandas"
                self.log.add(
                    "Ingestion",
                    "Dask not available – using chunked-pandas fallback",
                    f"pip install dask[dataframe] to enable Dask-backed ingestion "
                    f"for files > {self.chunk_threshold_mb} MB.",
                )
        else:
            strategy = "pandas"

        self.log.add(
            "Ingestion",
            f"strategy={strategy}, file={os.path.basename(path)}, "
            f"size={file_size_mb:.1f} MB, parquet={is_parquet}",
            self._strategy_reason(strategy, file_size_mb, is_parquet),
        )

        # ---- execute -------------------------------------------------------
        if strategy == "dask":
            df = self._load_dask(path, is_parquet, file_size_mb)
        elif strategy == "chunked-pandas":
            df = self._load_chunked(path, is_parquet)
        else:
            df = self._load_pandas(path, is_parquet)

        # ---- optional sampling --------------------------------------------
        if self.sample_rows and len(df) > self.sample_rows:
            df = df.head(self.sample_rows).copy()
            self.log.add(
                "Ingestion",
                f"sampled to {self.sample_rows:,} rows",
                "sample_rows limit applied to cap memory usage.",
            )

        # ---- validate target column ---------------------------------------
        if target_col and target_col not in df.columns:
            warnings.warn(
                f"StreamingIngestAgent: target_col={target_col!r} not found in loaded data. "
                f"Columns: {list(df.columns)}"
            )

        meta = {
            "backend": strategy,
            "file_size_mb": round(file_size_mb, 2),
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "source_type": "parquet" if is_parquet else "csv",
            "parquet": is_parquet,
        }
        self.log.add(
            "Ingestion",
            f"loaded {len(df):,} rows × {df.shape[1]} cols via {strategy}",
            f"Ingestion complete. Memory footprint: "
            f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB.",
            extra=meta,
        )
        return df, meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _strategy_reason(self, strategy: str, file_size_mb: float, is_parquet: bool) -> str:
        kind = "Parquet" if is_parquet else "CSV"
        if strategy == "dask":
            return (
                f"{kind} file ({file_size_mb:.1f} MB) exceeds threshold "
                f"({self.chunk_threshold_mb} MB). Using Dask distributed "
                f"ingestion for out-of-core parallelism."
            )
        if strategy == "chunked-pandas":
            return (
                f"{kind} file ({file_size_mb:.1f} MB) exceeds threshold "
                f"({self.chunk_threshold_mb} MB) but Dask is unavailable. "
                f"Falling back to chunked pandas (chunksize={self.pandas_chunksize:,})."
            )
        return (
            f"{kind} file ({file_size_mb:.1f} MB) is below the threshold "
            f"({self.chunk_threshold_mb} MB). Loading directly with pandas."
        )

    def _load_pandas(self, path: str, is_parquet: bool) -> pd.DataFrame:
        if is_parquet:
            if not _PARQUET_OK:
                raise ImportError(
                    "pyarrow or fastparquet is required to read Parquet files. "
                    "Install with: pip install pyarrow"
                )
            return pd.read_parquet(path)
        return pd.read_csv(path, low_memory=False)

    def _load_dask(self, path: str, is_parquet: bool, file_size_mb: float) -> pd.DataFrame:
        n_parts = self.dask_npartitions or max(2, int(file_size_mb / 64))
        try:
            if is_parquet:
                if not _PARQUET_OK:
                    raise ImportError("pyarrow or fastparquet required for Parquet.")
                ddf = dd.read_parquet(path)
            else:
                ddf = dd.read_csv(path, assume_missing=True, low_memory=False)
                # Repartition after read for balanced chunks
                ddf = ddf.repartition(npartitions=n_parts)

            self.log.add(
                "Ingestion",
                f"Dask: {ddf.npartitions} partitions created",
                "Dask lazy graph built; triggering .compute() to materialise to pandas.",
            )
            return ddf.compute()
        except Exception as exc:
            warnings.warn(f"Dask ingestion failed ({exc}); falling back to chunked pandas.")
            return self._load_chunked(path, is_parquet)

    def _load_chunked(self, path: str, is_parquet: bool) -> pd.DataFrame:
        """Chunked pandas read – memory-bounded fallback."""
        if is_parquet:
            # Parquet doesn't support chunked read simply; load fully
            if not _PARQUET_OK:
                raise ImportError("pyarrow or fastparquet required for Parquet.")
            return pd.read_parquet(path)

        chunks = []
        n_chunks = 0
        for chunk in pd.read_csv(path, chunksize=self.pandas_chunksize, low_memory=False):
            chunks.append(chunk)
            n_chunks += 1
        self.log.add(
            "Ingestion",
            f"chunked-pandas: {n_chunks} chunks × {self.pandas_chunksize:,} rows",
            "Chunked pandas concatenation complete.",
        )
        return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Convenience: CSV → Parquet converter
# ---------------------------------------------------------------------------

def csv_to_parquet(csv_path: str, parquet_path: Optional[str] = None,
                   compression: str = "snappy") -> str:
    """
    Convert a CSV file to Parquet format for faster future ingestion.

    Parameters
    ----------
    csv_path     : path to the input CSV
    parquet_path : output path (default: same directory, ``.parquet`` extension)
    compression  : Parquet compression codec (``snappy``, ``gzip``, ``none``)

    Returns
    -------
    str : absolute path to the written Parquet file.
    """
    if not _PARQUET_OK:
        raise ImportError(
            "pyarrow is required for Parquet conversion. "
            "Install with: pip install pyarrow"
        )
    if parquet_path is None:
        base = os.path.splitext(csv_path)[0]
        parquet_path = base + ".parquet"

    df = pd.read_csv(csv_path, low_memory=False)
    df.to_parquet(parquet_path, compression=compression, index=False)
    logger.info(f"CSV → Parquet: {csv_path!r} → {parquet_path!r} ({len(df):,} rows)")
    return parquet_path
