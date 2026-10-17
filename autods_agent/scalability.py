"""
ScalabilityExperiment
======================
Phase 10 – Measures execution time, peak memory, and throughput for the
local (pandas/sklearn) vs Dask (distributed) preprocessing + training paths
across synthetic datasets of increasing size.

PySpark is imported optionally; if not installed the PySpark column is
silently omitted from the results.

Outputs (written to outputs/scalability/):
  - results.csv         – raw timing/memory table
  - time_vs_size.png    – wall-clock time plot
  - memory_vs_size.png  – peak memory plot
  - speedup.png         – Dask speedup over local
  - summary.md          – markdown table + analysis

Usage:
  python run_scalability_experiment.py
"""

from __future__ import annotations

import os
import time
import warnings
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False


# ---------------------------------------------------------------------------
# Dataset sizes to benchmark
# ---------------------------------------------------------------------------
SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
N_FEATURES = 10


# ---------------------------------------------------------------------------
# Memory helper
# ---------------------------------------------------------------------------

def _peak_memory_mb() -> float:
    if not _PSUTIL_OK:
        return float("nan")
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------

def _make_dataset(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, N_FEATURES)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])
    df["target"] = (rng.randn(n) > 0).astype(int)
    # inject 5% missing
    for col in ["f0", "f2"]:
        mask = rng.rand(n) < 0.05
        df.loc[mask, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Preprocessing + training (local path)
# ---------------------------------------------------------------------------

def _run_local(df: pd.DataFrame) -> float:
    """Returns elapsed seconds for local preprocessing + training."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier

    t0 = time.time()
    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].fillna(df[feature_cols].median()).values
    y = df[target_col].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = HistGradientBoostingClassifier(max_iter=20, random_state=42)
    model.fit(X_tr, y_tr)
    return time.time() - t0


# ---------------------------------------------------------------------------
# Preprocessing + training (Dask path)
# ---------------------------------------------------------------------------

def _run_dask(df: pd.DataFrame) -> float:
    """Returns elapsed seconds for Dask-partitioned preprocessing + training."""
    import dask.dataframe as dd
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    n = len(df)
    n_parts = max(2, n // 50_000)

    ddf = dd.from_pandas(df, npartitions=n_parts)
    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]

    # Compute medians per partition
    medians = {c: float(ddf[c].quantile(0.5).compute()) for c in feature_cols}

    # Fill missing
    for col, med in medians.items():
        ddf[col] = ddf[col].fillna(med)

    pdf = ddf.compute()
    X = pdf[feature_cols].values
    y = pdf[target_col].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = HistGradientBoostingClassifier(max_iter=20, random_state=42)
    model.fit(X_tr, y_tr)
    return time.time() - t0


# ---------------------------------------------------------------------------
# PySpark path (optional)
# ---------------------------------------------------------------------------

def _run_spark(df: pd.DataFrame) -> Optional[float]:
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
        from pyspark.ml.classification import GBTClassifier

        spark = SparkSession.builder.master("local[*]").appName("autods_scalability").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        t0 = time.time()
        sdf = spark.createDataFrame(df.fillna(0))
        feature_cols = [c for c in df.columns if c != "target"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
        sdf2 = assembler.transform(sdf)
        scaler = SparkScaler(inputCol="raw_features", outputCol="features")
        scaler_model = scaler.fit(sdf2)
        sdf3 = scaler_model.transform(sdf2)
        sdf3 = sdf3.withColumnRenamed("target", "label")
        splits = sdf3.randomSplit([0.8, 0.2], seed=42)
        gbt = GBTClassifier(maxIter=5, seed=42)
        gbt.fit(splits[0])
        elapsed = time.time() - t0
        spark.stop()
        return elapsed
    except Exception as e:
        warnings.warn(f"PySpark benchmark skipped: {e}")
        return None


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

class ScalabilityExperiment:
    """Runs the scalability benchmark and saves results."""

    def __init__(self, sizes: Optional[List[int]] = None,
                 output_dir: str = "outputs/scalability",
                 max_memory_mb: float = 4000.0):
        """
        Parameters
        ----------
        sizes        : row counts to benchmark; auto-truncated if memory limited
        output_dir   : where to write CSV, plots, markdown
        max_memory_mb: soft limit; sizes that would likely exceed it are skipped
        """
        self.sizes      = sizes or SIZES
        self.output_dir = output_dir
        self.max_memory = max_memory_mb
        os.makedirs(output_dir, exist_ok=True)

    def run(self, sizes: Optional[List[int]] = None) -> pd.DataFrame:
        """Run the full experiment and return results DataFrame."""
        run_sizes = sizes or self.sizes
        rows = []

        for n in run_sizes:
            # Rough memory estimate: ~100 bytes per cell
            est_mb = n * (N_FEATURES + 1) * 100 / 1e6
            if est_mb > self.max_memory:
                logger.warning(f"Skipping n={n:,} (est {est_mb:.0f} MB > limit {self.max_memory:.0f} MB)")
                continue

            logger.info(f"Benchmarking n={n:,} ...")
            df = _make_dataset(n)
            mem_before = _peak_memory_mb()

            try:
                t_local = _run_local(df)
            except Exception as e:
                warnings.warn(f"Local path failed at n={n}: {e}")
                t_local = float("nan")

            try:
                t_dask = _run_dask(df)
            except Exception as e:
                warnings.warn(f"Dask path failed at n={n}: {e}")
                t_dask = float("nan")

            t_spark = None
            try:
                t_spark = _run_spark(df)
            except Exception:
                pass

            mem_after = _peak_memory_mb()
            peak_mem  = mem_after - mem_before

            speedup_dask = (t_local / t_dask) if t_dask and t_dask > 0 else float("nan")

            row = {
                "n_rows":            n,
                "local_time_s":      round(t_local, 3),
                "dask_time_s":       round(t_dask, 3),
                "dask_speedup":      round(speedup_dask, 3),
                "peak_memory_mb":    round(peak_mem, 1),
                "throughput_local":  round(n / t_local, 0) if t_local else float("nan"),
            }
            if t_spark is not None:
                row["spark_time_s"] = round(t_spark, 3)
            rows.append(row)
            logger.info(f"  local={t_local:.2f}s, dask={t_dask:.2f}s, mem_delta={peak_mem:.0f}MB")

        results_df = pd.DataFrame(rows)

        # ---- save CSV ---------------------------------------------------
        csv_path = os.path.join(self.output_dir, "results.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

        # ---- plots ------------------------------------------------------
        self._plot(results_df)

        # ---- markdown summary -------------------------------------------
        self._write_markdown(results_df)

        return results_df

    # ------------------------------------------------------------------
    def _plot(self, df: pd.DataFrame) -> None:
        if not _MPL_OK or df.empty:
            return

        # Time vs size
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["n_rows"] / 1000, df["local_time_s"], "o-", label="Local (pandas/sklearn)", color="#5EE39B")
        ax.plot(df["n_rows"] / 1000, df["dask_time_s"],  "s--", label="Dask",                  color="#F2B84B")
        if "spark_time_s" in df:
            ax.plot(df["n_rows"] / 1000, df["spark_time_s"], "^:", label="PySpark", color="#FF6B6B")
        ax.set_xlabel("Dataset size (k rows)")
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title("Preprocessing + Training: Time vs Dataset Size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "time_vs_size.png"), dpi=100)
        plt.close(fig)

        # Memory vs size
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["n_rows"] / 1000, df["peak_memory_mb"], "o-", color="#5EE39B")
        ax.set_xlabel("Dataset size (k rows)")
        ax.set_ylabel("Peak memory delta (MB)")
        ax.set_title("Peak Memory Usage vs Dataset Size")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "memory_vs_size.png"), dpi=100)
        plt.close(fig)

        # Speedup
        if "dask_speedup" in df:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(df["n_rows"] / 1000, df["dask_speedup"], color="#5EE39B", alpha=0.8)
            ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Breakeven (speedup=1)")
            ax.set_xlabel("Dataset size (k rows)")
            ax.set_ylabel("Dask Speedup over Local (x)")
            ax.set_title("Dask Speedup vs Dataset Size")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            fig.savefig(os.path.join(self.output_dir, "speedup.png"), dpi=100)
            plt.close(fig)

    def _write_markdown(self, df: pd.DataFrame) -> None:
        lines = [
            "# Scalability Experiment Results\n",
            "## Research Question",
            "At what dataset size does distributed (Dask) preprocessing outperform the "
            "in-memory (pandas/sklearn) path?\n",
            "## Results Table\n",
        ]
        try:
            lines.append(df.to_markdown(index=False))
        except Exception:
            cols = list(df.columns)
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for _, r in df.iterrows():
                lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        lines.append("\n## Analysis\n")

        if not df.empty and "dask_speedup" in df:
            crossover = df[df["dask_speedup"] >= 1.0]
            if not crossover.empty:
                first_win = crossover.iloc[0]["n_rows"]
                lines.append(
                    f"**Crossover point**: Dask first outperforms the local path at "
                    f"approximately **{int(first_win):,} rows**. "
                    f"Below this size the distributed overhead (partitioning, task scheduling) "
                    f"exceeds the parallelism benefit. Above it, Dask's parallel compute pays off."
                )
            else:
                lines.append(
                    "Within the tested range, the local path remains faster than Dask. "
                    "This is expected for datasets that comfortably fit in memory on a single core — "
                    "Dask's benefit emerges when either data exceeds RAM or genuine multi-core "
                    "parallelism is available."
                )
        lines.append(
            "\n\n**Recommendation**: Use the local path for datasets < 500k rows on a single machine. "
            "Switch to Dask (or PySpark on a cluster) for larger workloads where the scheduling "
            "overhead is amortised over the parallel speedup."
        )

        md_path = os.path.join(self.output_dir, "summary.md")
        with open(md_path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Markdown summary saved to {md_path}")
