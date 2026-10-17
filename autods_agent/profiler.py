"""
DataProfilerAgent  (v2)
========================
Extended in AutoDS-Agent 2.0 to provide a richer profile:
  - rows, columns, dtypes, missing values (count and %), duplicates
  - numeric vs categorical feature counts
  - target type (classification/regression, binary/multiclass)
  - dataset size in MB
  - high-cardinality columns (categorical cols with >50 unique values)
  - constant columns (1 unique value)
  - execution backend decision (local vs distributed)

This is the single most important decision for the Scalable ML / BDA half
of the project: the agent must reason about data volume vs available memory.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from .utils import human_bytes


class DataProfilerAgent:
    """First stage: profiles the raw dataset and makes routing decisions."""

    def __init__(self, reasoner, log, distributed_threshold_mb=500,
                 distributed_threshold_rows=1_000_000):
        self.reasoner = reasoner
        self.log = log
        self.distributed_threshold_mb   = distributed_threshold_mb
        self.distributed_threshold_rows = distributed_threshold_rows

    def run(self, df: pd.DataFrame, target_col: str) -> dict:
        size_mb   = df.memory_usage(deep=True).sum() / (1024 ** 2)
        n_rows, n_cols = df.shape
        target    = df[target_col]

        # ---- execution backend decision --------------------------------
        backend_decision, backend_reason = self.reasoner.decide(
            "execution_backend",
            {
                "size_mb":  size_mb,
                "n_rows":   n_rows,
                "distributed_threshold_mb":   self.distributed_threshold_mb,
                "distributed_threshold_rows": self.distributed_threshold_rows,
            },
        )
        self.log.add("Profiler", f"backend={backend_decision}", backend_reason,
                     extra={"size": human_bytes(size_mb * 1024 * 1024)})

        # ---- task type decision ----------------------------------------
        task_decision, task_reason = self.reasoner.decide(
            "task_type",
            {
                "target_n_unique": int(target.nunique()),
                "target_dtype":    str(target.dtype),
                "n_rows":          n_rows,
            },
        )
        self.log.add("Profiler", f"task_type={task_decision}", task_reason)

        # ---- extended stats --------------------------------------------
        feature_cols  = [c for c in df.columns if c != target_col]
        numeric_cols  = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols      = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
        high_card     = [c for c in cat_cols if df[c].nunique() > 50]
        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        n_dup         = int(df.duplicated().sum())
        missing       = df.isna().mean().to_dict()
        missing_count = df.isna().sum().to_dict()
        dtypes        = df.dtypes.astype(str).to_dict()

        n_unique_tgt  = int(target.nunique())
        if task_decision == "classification":
            target_subtype = "binary" if n_unique_tgt == 2 else "multiclass"
        else:
            target_subtype = "continuous"

        profile = {
            "n_rows":          n_rows,
            "n_cols":          n_cols,
            "size_mb":         size_mb,
            "backend":         backend_decision,
            "task_type":       task_decision,
            "target_subtype":  target_subtype,
            "missing_frac":    missing,
            "missing_count":   missing_count,
            "dtypes":          dtypes,
            "target_col":      target_col,
            "n_features":      n_cols - 1,
            "n_numeric":       len(numeric_cols),
            "n_categorical":   len(cat_cols),
            "numeric_cols":    numeric_cols,
            "cat_cols":        cat_cols,
            "high_cardinality_cols": high_card,
            "constant_cols":   constant_cols,
            "n_duplicates":    n_dup,
            "target_n_unique": n_unique_tgt,
        }
        return profile
