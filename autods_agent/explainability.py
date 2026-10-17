"""
ExplainabilityAgent  (v2)
==========================
Extended in AutoDS-Agent 2.0 to provide:
  - Global SHAP importance (summary/bar plot PNG)
  - Local explanation for a single record (top positive/negative contributors
    as a list + waterfall/bar plot PNG)
  - Full support for binary, multiclass (averaging across classes), and regression
  - Robust fallback to permutation importance when SHAP fails for a model type.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


class ExplainabilityAgent:
    """Global and local SHAP explanations with permutation importance fallback."""

    def __init__(self, reasoner, log, output_root: str = "outputs"):
        self.reasoner    = reasoner
        self.log         = log
        self.output_root = output_root

    # ------------------------------------------------------------------
    def run(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        task_type: str,
        run_id: str = "default",
        local_record: Optional[np.ndarray] = None,
    ) -> dict:
        out_dir = os.path.join(self.output_root, run_id, "explainability")
        os.makedirs(out_dir, exist_ok=True)

        model_class = type(model).__name__
        tree_based  = any(k in model_class for k in
                          ["Forest", "GradientBoosting", "HistGradientBoosting"])

        global_shap_vals: Optional[np.ndarray] = None
        method = ""
        plot_paths: Dict[str, str] = {}

        if tree_based:
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_raw = explainer.shap_values(X_test)

                # Normalize multiclass / 3D outputs to (n_samples, n_features)
                if isinstance(shap_raw, list):
                    # List of (n_samples, n_features) per class
                    shap_arr = np.mean([np.abs(s) for s in shap_raw], axis=0)
                elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
                    # (n_samples, n_features, n_classes)
                    shap_arr = np.mean(np.abs(shap_raw), axis=2)
                else:
                    shap_arr = np.abs(shap_raw)

                # Mean importance per feature -> shape: (n_features,)
                global_shap_vals = np.mean(shap_arr, axis=0).flatten()
                method = "SHAP TreeExplainer"

                p = self._plot_global_shap(global_shap_vals, feature_names, out_dir)
                if p:
                    plot_paths["global_shap"] = p
            except Exception as e:
                global_shap_vals = None
                method = f"permutation (SHAP fallback: {e})"

        if global_shap_vals is None or len(global_shap_vals) != len(feature_names):
            try:
                perm = permutation_importance(
                    model, X_test, y_test, n_repeats=5, random_state=42
                )
                global_shap_vals = perm.importances_mean.flatten()
                if not method:
                    method = "permutation importance"
            except Exception as e:
                # Ultimate fallback
                global_shap_vals = np.ones(len(feature_names)) / len(feature_names)
                method = f"uniform fallback ({e})"

        # Ensure correct length
        n_feat = len(feature_names)
        if len(global_shap_vals) > n_feat:
            global_shap_vals = global_shap_vals[:n_feat]
        elif len(global_shap_vals) < n_feat:
            padded = np.zeros(n_feat)
            padded[:len(global_shap_vals)] = global_shap_vals
            global_shap_vals = padded

        order = np.argsort(-global_shap_vals)
        ranked = [(feature_names[i], float(global_shap_vals[i])) for i in order if i < n_feat]

        # ---- local explanation ------------------------------------------
        local_explanation: List[dict] = []
        if local_record is None and len(X_test) > 0:
            local_record = X_test[0:1]

        if local_record is not None and tree_based:
            local_explanation, lp = self._local_shap(
                model, local_record, feature_names, out_dir
            )
            if lp:
                plot_paths["local_shap"] = lp

        self.log.add(
            "Explainability",
            f"method={method}",
            f"Chose {method} because model is {model_class} "
            f"({'tree-based, SHAP exact' if tree_based else 'non-tree, permutation importance'}).",
        )

        return {
            "method":            method,
            "ranked_importance": ranked,
            "global_shap_values": global_shap_vals.tolist(),
            "local_explanation": local_explanation,
            "plot_paths":        plot_paths,
        }

    # ------------------------------------------------------------------
    def _plot_global_shap(self, importance: np.ndarray, feature_names: List[str], out_dir: str) -> Optional[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            importance = np.asarray(importance).flatten()
            n_show = min(15, len(importance), len(feature_names))
            order = np.argsort(-importance)[:n_show]
            top_names = [feature_names[i][:25] for i in order]
            top_imp = [float(importance[i]) for i in order]

            fig, ax = plt.subplots(figsize=(8, max(4, n_show * 0.35)))
            ax.barh(top_names[::-1], top_imp[::-1], color="#5EE39B", alpha=0.85)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Global Feature Importance (SHAP)")
            ax.grid(True, alpha=0.25, axis="x")
            plt.tight_layout()
            p = os.path.join(out_dir, "global_shap.png")
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return p
        except Exception as e:
            warnings.warn(f"Global SHAP plot failed: {e}")
            return None

    # ------------------------------------------------------------------
    def _local_shap(
        self,
        model,
        record: np.ndarray,
        feature_names: List[str],
        out_dir: str
    ) -> Tuple[List[dict], Optional[str]]:
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            explainer = shap.TreeExplainer(model)
            shap_raw = explainer.shap_values(record)

            if isinstance(shap_raw, list):
                # Class 0 or first class values
                sv = np.array(shap_raw[0]).flatten()
            elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
                sv = shap_raw[0, :, 0].flatten()
            else:
                sv = np.array(shap_raw).flatten()

            n_feat = len(feature_names)
            if len(sv) > n_feat:
                sv = sv[:n_feat]
            elif len(sv) < n_feat:
                padded = np.zeros(n_feat)
                padded[:len(sv)] = sv
                sv = padded

            order = np.argsort(-np.abs(sv))[:10]
            result = [
                {
                    "feature":    feature_names[i],
                    "shap_value": round(float(sv[i]), 4),
                    "direction":  "positive" if sv[i] > 0 else "negative",
                }
                for i in order if i < n_feat
            ]

            top_f = [feature_names[i][:25] for i in order if i < n_feat]
            top_v = [float(sv[i]) for i in order if i < n_feat]
            colors = ["#5EE39B" if v >= 0 else "#FF6B6B" for v in top_v]

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(top_f[::-1], top_v[::-1], color=colors[::-1], alpha=0.85)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on prediction)")
            ax.set_title("Local Explanation for Sample Instance")
            ax.grid(True, alpha=0.25, axis="x")
            plt.tight_layout()
            p = os.path.join(out_dir, "local_shap.png")
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return result, p
        except Exception as e:
            warnings.warn(f"Local SHAP failed: {e}")
            return [], None
