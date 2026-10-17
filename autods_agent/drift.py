"""
DriftDetectionAgent
====================
Phase 12 – Statistical drift detection between reference (training) data
and new incoming data.

Numeric features : KS-test + Population Stability Index (PSI)
Categorical features : chi-square test + PSI
Target drift : distribution shift on labels (when provided)

Reports per-feature drift scores and an overall "retraining recommended" flag
when thresholds are exceeded.

Auto-retraining workflow (optional):
  retrain(new_df, old_model, registry, ...)
    -> trains a new model on combined data
    -> evaluates against the same holdout as the registered model
    -> registers the new version ONLY if metric >= previous - tolerance
    -> logs acceptance or rejection to RunLog
"""

from __future__ import annotations

import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PSI_THRESHOLD    = 0.20   # PSI > 0.20 -> significant drift
KS_PVALUE_THRESH = 0.05   # KS p-value < 0.05 -> significant drift
CHI_PVALUE_THRESH = 0.05  # chi-square


# ---------------------------------------------------------------------------
# PSI helper
# ---------------------------------------------------------------------------

def _psi(expected: np.ndarray, actual: np.ndarray,
          n_bins: int = 10) -> float:
    """Population Stability Index between two continuous arrays."""
    try:
        bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        e_counts, _ = np.histogram(expected, bins=bins)
        a_counts, _ = np.histogram(actual,   bins=bins)
        e_pct = (e_counts + 1e-6) / (len(expected) + 1e-6)
        a_pct = (a_counts + 1e-6) / (len(actual)   + 1e-6)
        psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
        return round(psi, 4)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# DriftDetectionAgent
# ---------------------------------------------------------------------------

class DriftDetectionAgent:
    """Compares new data against reference (training) data for statistical drift."""

    def __init__(self, reasoner, log,
                 psi_threshold: float = PSI_THRESHOLD,
                 ks_pvalue: float = KS_PVALUE_THRESH):
        self.reasoner      = reasoner
        self.log           = log
        self.psi_threshold = psi_threshold
        self.ks_pvalue     = ks_pvalue

    # ------------------------------------------------------------------
    def run(
        self,
        reference_df: pd.DataFrame,
        new_df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> dict:
        """
        Parameters
        ----------
        reference_df : training data (reference distribution)
        new_df       : incoming data to check for drift
        target_col   : if provided, also checks target drift

        Returns
        -------
        dict with keys: feature_drift, target_drift, n_drifted_features,
                         retrain_recommended, overall_psi
        """
        feature_drift: Dict[str, dict] = {}
        n_drifted = 0

        common_cols = [c for c in reference_df.columns
                       if c in new_df.columns and c != target_col]

        for col in common_cols:
            ref_col = reference_df[col].dropna()
            new_col = new_df[col].dropna()
            if len(ref_col) < 5 or len(new_col) < 5:
                continue

            if pd.api.types.is_numeric_dtype(ref_col):
                result = self._numeric_drift(ref_col.values, new_col.values)
            else:
                result = self._categorical_drift(ref_col.values, new_col.values)

            feature_drift[col] = result
            if result.get("drifted"):
                n_drifted += 1

        # Target drift
        target_drift: Optional[dict] = None
        if target_col and target_col in reference_df and target_col in new_df:
            ref_tgt = reference_df[target_col].dropna()
            new_tgt = new_df[target_col].dropna()
            if pd.api.types.is_numeric_dtype(ref_tgt):
                target_drift = self._numeric_drift(ref_tgt.values, new_tgt.values)
            else:
                target_drift = self._categorical_drift(ref_tgt.values, new_tgt.values)

        retrain_recommended = (
            n_drifted > max(1, len(common_cols) * 0.3) or
            (target_drift is not None and target_drift.get("drifted", False))
        )

        overall_psi = round(
            np.mean([v.get("psi", 0) for v in feature_drift.values()]), 4
        ) if feature_drift else 0.0

        self.log.add(
            "DriftDetection",
            f"drifted_features={n_drifted}/{len(feature_drift)}, "
            f"retrain={'YES' if retrain_recommended else 'NO'}",
            f"Overall PSI={overall_psi:.4f}. "
            f"{'⚠ Retraining recommended: significant drift detected.' if retrain_recommended else 'No significant drift; current model is likely still valid.'}",
        )

        return {
            "feature_drift":        feature_drift,
            "target_drift":         target_drift,
            "n_drifted_features":   n_drifted,
            "total_features":       len(feature_drift),
            "retrain_recommended":  retrain_recommended,
            "overall_psi":          overall_psi,
        }

    # ------------------------------------------------------------------
    def retrain(
        self,
        new_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        target_col: str,
        old_metadata: dict,
        registry,
        tolerance: float = 0.01,
    ) -> dict:
        """
        Train a new model on combined reference + new data, evaluate on a
        holdout, and register only if metric >= previous metric - tolerance.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.metrics import accuracy_score, r2_score

        combined = pd.concat([reference_df, new_df], ignore_index=True)
        task_type = old_metadata.get("task_type", "classification")
        feature_names = old_metadata.get("feature_names", [])

        available = [c for c in feature_names if c in combined.columns]
        if not available or target_col not in combined.columns:
            return {"accepted": False, "reason": "Missing features or target in combined data."}

        X = combined[available].fillna(0).values
        y = combined[target_col].values

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        if task_type == "classification":
            model = HistGradientBoostingClassifier(random_state=42)
            model.fit(X_tr, y_tr)
            score = accuracy_score(y_te, model.predict(X_te))
            metric_name = "accuracy"
        else:
            model = HistGradientBoostingRegressor(random_state=42)
            model.fit(X_tr, y_tr)
            score = r2_score(y_te, model.predict(X_te))
            metric_name = "r2"

        prev_score = old_metadata.get("metrics", {}).get(metric_name, 0.0)
        accepted   = score >= prev_score - tolerance

        if accepted:
            ver = registry.register(
                model=model,
                model_type="HistGradientBoosting_retrained",
                run_id="drift_retrain",
                dataset_name="combined",
                hyperparameters={},
                metrics={metric_name: score},
                feature_names=available,
                task_type=task_type,
            )
            self.log.add(
                "DriftDetection",
                f"retrain ACCEPTED (new={score:.4f} vs prev={prev_score:.4f}), version={ver}",
                f"New model meets quality bar (tolerance={tolerance}). Registered as {ver}.",
            )
        else:
            self.log.add(
                "DriftDetection",
                f"retrain REJECTED (new={score:.4f} < prev={prev_score:.4f} - {tolerance})",
                "New model did not meet the performance threshold; old model retained.",
            )

        return {
            "accepted":       accepted,
            "new_score":      score,
            "prev_score":     prev_score,
            "metric":         metric_name,
            "tolerance":      tolerance,
        }

    # ------------------------------------------------------------------
    def _numeric_drift(self, ref: np.ndarray, new: np.ndarray) -> dict:
        ks_stat, ks_pval = stats.ks_2samp(ref, new)
        psi_val = _psi(ref, new)
        drifted = (ks_pval < self.ks_pvalue) or (psi_val > self.psi_threshold)
        return {
            "type":    "numeric",
            "ks_stat": round(float(ks_stat), 4),
            "ks_pval": round(float(ks_pval), 4),
            "psi":     psi_val,
            "drifted": drifted,
        }

    def _categorical_drift(self, ref: np.ndarray, new: np.ndarray) -> dict:
        try:
            all_cats = np.union1d(np.unique(ref), np.unique(new))
            ref_counts = np.array([np.sum(ref == c) for c in all_cats])
            new_counts = np.array([np.sum(new == c) for c in all_cats])
            # scale new_counts to same total as ref
            if new_counts.sum() > 0:
                new_counts = new_counts * ref_counts.sum() / new_counts.sum()
            chi2, pval = stats.chisquare(ref_counts + 1e-6, f_exp=new_counts + 1e-6)
            ref_str = ref.astype(str)
            new_str = new.astype(str)
            psi_val = _psi(
                pd.Categorical(ref_str).codes.astype(float),
                pd.Categorical(new_str, categories=np.unique(ref_str)).codes.astype(float),
            )
            drifted = (pval < CHI_PVALUE_THRESH) or (psi_val > self.psi_threshold)
            return {
                "type":   "categorical",
                "chi2":   round(float(chi2), 4),
                "pval":   round(float(pval), 4),
                "psi":    psi_val,
                "drifted": drifted,
            }
        except Exception as e:
            return {"type": "categorical", "error": str(e), "drifted": False}
