"""
WhatIfAgent
============
Phase 8 – What-If analysis after a model has been trained.

Provides two operations:
  1. predict(record)          – single-record prediction + probability
  2. compare(record, changes) – side-by-side comparison of original vs
                                modified record: prediction, probability,
                                and SHAP-based delta explanation.

Inputs are raw Python dicts using the *original* column names (before
feature engineering). The agent applies the same fitted transformations
that were used at training time (scaler / encoder stored in the Orchestrator).
"""

from __future__ import annotations

import warnings
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WhatIfAgent:
    """Single-record prediction and what-if comparison for a trained model."""

    def __init__(self, reasoner, log):
        self.reasoner = reasoner
        self.log = log

        # These are set by the Orchestrator after training
        self.model = None
        self.pipeline = None
        self.preprocessor_transform = None   # callable: pd.DataFrame -> np.ndarray
        self.feature_names: List[str] = []
        self.task_type: str = "classification"
        self.target_col: str = ""

    # ------------------------------------------------------------------
    def setup(self, model=None, preprocessor_transform=None, feature_names: List[str] | None = None,
               task_type: str = "classification", target_col: str = "", pipeline=None) -> None:
        """Called by the Orchestrator to wire in the trained pipeline or artifacts."""
        if pipeline is not None:
            self.pipeline = pipeline
            self.model = getattr(pipeline, "model", model)
        else:
            self.pipeline = None
            self.model = model

        self.preprocessor_transform = preprocessor_transform
        self.feature_names = feature_names or []
        self.task_type = task_type
        self.target_col = target_col

    # ------------------------------------------------------------------
    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict for a single record (dict of raw feature values).
        Returns prediction, probability/value, and top SHAP contributors.
        """
        if self.pipeline is None and self.model is None:
            return {"error": "WhatIfAgent not initialised (no trained model or pipeline)."}

        try:
            row_df = pd.DataFrame([record])
            if self.target_col in row_df.columns:
                row_df = row_df.drop(columns=[self.target_col])

            # Use full pipeline directly if available
            if self.pipeline is not None:
                pred = self.pipeline.predict(row_df)[0]
                result: Dict[str, Any] = {"prediction": _native(pred)}
                if self.task_type == "classification" and hasattr(self.pipeline, "predict_proba"):
                    try:
                        proba = self.pipeline.predict_proba(row_df)[0]
                        classes = getattr(self.pipeline, "classes_", getattr(self.model, "classes_", range(len(proba))))
                        result["probability"] = {
                            str(cls): round(float(p), 4)
                            for cls, p in zip(classes, proba)
                        }
                    except Exception:
                        pass
                X_trans = self.pipeline.transform_features(row_df)
                result["shap_contributors"] = self._shap_top(X_trans)
                self.log.add("WhatIf", f"predict -> {pred}",
                             f"Single-record prediction via FullInferencePipeline [{type(self.model).__name__}].")
                return result

            # Legacy path
            X = self._transform_record(record)
            pred = self.model.predict(X)[0]
            result: Dict[str, Any] = {"prediction": _native(pred)}

            if self.task_type == "classification":
                try:
                    proba = self.model.predict_proba(X)[0]
                    result["probability"] = {
                        str(cls): round(float(p), 4)
                        for cls, p in zip(self.model.classes_, proba)
                    }
                except Exception:
                    pass

            result["shap_contributors"] = self._shap_top(X)
            self.log.add("WhatIf", f"predict -> {pred}",
                         f"Single-record prediction via {type(self.model).__name__}.")
            return result
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    def compare(self, original: Dict[str, Any],
                 modified: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare original vs modified record side-by-side.
        Returns both predictions, probabilities, changed features, and
        SHAP-based explanation of what drove the change.
        """
        orig_result = self.predict(original)
        mod_result  = self.predict(modified)

        changed_features = [
            k for k in original
            if k != self.target_col and original.get(k) != modified.get(k)
        ]

        self.log.add(
            "WhatIf",
            f"compare: original={orig_result.get('prediction')}, "
            f"modified={mod_result.get('prediction')}",
            f"Changed features: {changed_features}. "
            f"SHAP contributors shown for modified record.",
        )

        return {
            "original_prediction": orig_result,
            "modified_prediction": mod_result,
            "changed_features":    changed_features,
        }

    # ------------------------------------------------------------------
    def _transform_record(self, record: Dict[str, Any]) -> np.ndarray:
        """Apply the same preprocessing that the training pipeline used."""
        if self.preprocessor_transform is not None:
            row_df = pd.DataFrame([record])
            return self.preprocessor_transform(row_df)
        # Fallback: just pick feature columns in order
        vals = [float(record.get(f, 0) or 0) for f in self.feature_names]
        return np.array(vals).reshape(1, -1)

    def _shap_top(self, X: np.ndarray,
                   n: int = 5) -> List[Dict[str, Any]]:
        """Return top-n SHAP contributors for a single transformed row."""
        try:
            import shap
            model_class = type(self.model).__name__
            tree_based  = any(k in model_class for k in
                              ["Forest", "GradientBoosting", "HistGradientBoosting"])
            if tree_based:
                exp = shap.TreeExplainer(self.model)
                sv  = exp.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[0]
                sv = np.array(sv).flatten()
            else:
                exp = shap.LinearExplainer(self.model, X)
                sv  = exp.shap_values(X).flatten()

            order = np.argsort(-np.abs(sv))[:n]
            return [
                {
                    "feature": self.feature_names[i] if i < len(self.feature_names) else f"f{i}",
                    "shap_value": round(float(sv[i]), 4),
                    "direction": "positive" if sv[i] > 0 else "negative",
                }
                for i in order
            ]
        except Exception as e:
            logger.warning(f"SHAP in WhatIf failed: {e}")
            return []


def _native(val):
    """Convert numpy scalar to Python native type for JSON serialisation."""
    if hasattr(val, "item"):
        return val.item()
    return val
