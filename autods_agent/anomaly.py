"""
AnomalyDetectionAgent
======================
Phase 3 – Detects anomalous rows using Isolation Forest and Local Outlier
Factor. By default, anomalies are only REPORTED (not removed); automatic
removal can be enabled via `remove_anomalies=True` in the Orchestrator config.

Results go into the RunLog and the final report.
"""

from __future__ import annotations

import warnings
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)

CONTAMINATION_DEFAULT = 0.05   # assumed fraction of anomalies


class AnomalyDetectionAgent:
    """
    Reports (and optionally removes) anomalous rows using two detectors:
      1. Isolation Forest  – efficient for high-dimensional tabular data
      2. Local Outlier Factor – density-based, catches local anomalies
    """

    def __init__(self, reasoner, log,
                 contamination: float = CONTAMINATION_DEFAULT,
                 remove_anomalies: bool = False):
        self.reasoner = reasoner
        self.log = log
        self.contamination = contamination
        self.remove_anomalies = remove_anomalies

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, profile: dict) -> Dict[str, Any]:
        """
        Parameters
        ----------
        df      : preprocessed numeric DataFrame (target column present)
        profile : dict from DataProfilerAgent (used for target_col)

        Returns
        -------
        dict with keys: if_anomaly_pct, lof_anomaly_pct,
                         top_anomalous_indices, anomaly_mask, cleaned_df
        """
        target_col = profile["target_col"]
        feature_cols = [c for c in df.columns if c != target_col]

        # Only run on numeric features; silently skip if nothing to work with
        num_cols = [c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            self.log.add("AnomalyDetection", "skipped",
                         "No numeric feature columns available for anomaly detection.")
            return self._empty_result(df)

        X = df[num_cols].fillna(df[num_cols].median()).values

        if_mask  = self._isolation_forest(X)
        lof_mask = self._lof(X)

        if_pct  = round(float(if_mask.sum())  / len(df) * 100, 2)
        lof_pct = round(float(lof_mask.sum()) / len(df) * 100, 2)

        # Union mask for "top anomalous" display
        union_mask = if_mask | lof_mask
        top_idx    = df.index[union_mask].tolist()[:20]

        self.log.add(
            "AnomalyDetection",
            f"IF={if_pct}% anomalous, LOF={lof_pct}% anomalous",
            f"Isolation Forest detected {if_pct}% anomalous rows; "
            f"Local Outlier Factor detected {lof_pct}% anomalous rows "
            f"(contamination assumption={self.contamination}). "
            f"{'Anomalies REMOVED from training data.' if self.remove_anomalies else 'Anomalies flagged only – not removed (default behaviour).'}"
        )

        cleaned_df = df
        if self.remove_anomalies:
            cleaned_df = df[~union_mask].reset_index(drop=True)
            self.log.add(
                "AnomalyDetection",
                f"removed {union_mask.sum()} rows",
                f"Removed rows flagged by at least one detector, "
                f"leaving {len(cleaned_df)} rows.",
            )

        return {
            "if_anomaly_pct":        if_pct,
            "lof_anomaly_pct":       lof_pct,
            "top_anomalous_indices": top_idx,
            "anomaly_mask":          union_mask.tolist(),
            "cleaned_df":            cleaned_df,
        }

    # ------------------------------------------------------------------
    def _isolation_forest(self, X: np.ndarray) -> np.ndarray:
        try:
            clf = IsolationForest(contamination=self.contamination,
                                  random_state=42, n_jobs=1)
            preds = clf.fit_predict(X)
            return preds == -1
        except Exception as e:
            warnings.warn(f"Isolation Forest failed: {e}")
            return np.zeros(len(X), dtype=bool)

    def _lof(self, X: np.ndarray) -> np.ndarray:
        try:
            clf = LocalOutlierFactor(contamination=self.contamination, n_jobs=1)
            preds = clf.fit_predict(X)
            return preds == -1
        except Exception as e:
            warnings.warn(f"LOF failed: {e}")
            return np.zeros(len(X), dtype=bool)

    @staticmethod
    def _empty_result(df: pd.DataFrame) -> dict:
        return {
            "if_anomaly_pct":        0.0,
            "lof_anomaly_pct":       0.0,
            "top_anomalous_indices": [],
            "anomaly_mask":          [False] * len(df),
            "cleaned_df":            df,
        }
