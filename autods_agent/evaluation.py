"""
EvaluationAgent  (v2)
======================
Extended in AutoDS-Agent 2.0 to produce a richer metric suite and plots:

Classification:
  Accuracy, Precision, Recall, F1 (macro + weighted), ROC-AUC (macro OvR),
  confusion matrix PNG, ROC curve PNG, Precision-Recall curve PNG.

Regression:
  MAE, MSE, RMSE, R², actual-vs-predicted PNG, residual plot PNG.

Produces a model comparison table across all candidates and saves all plots
as PNGs in the run's output folder.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Any, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
import optuna

from .model_selection import _build_model

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# _FixedParamTrial – reconstruct a model from best_params without Optuna
# ---------------------------------------------------------------------------

class _FixedParamTrial:
    """Replay best_params as a mock Optuna trial."""
    def __init__(self, params: dict):
        self.params = params

    def suggest_float(self, name, *a, **k):
        return self.params[name]

    def suggest_int(self, name, *a, **k):
        return self.params[name]


# ---------------------------------------------------------------------------
# EvaluationAgent
# ---------------------------------------------------------------------------

class EvaluationAgent:
    """Fits, evaluates, and plots all tuned candidate models on a held-out split."""

    def __init__(self, reasoner, log, output_root: str = "outputs"):
        self.reasoner     = reasoner
        self.log          = log
        self.output_root  = output_root

    # ------------------------------------------------------------------
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        tuning_results: dict,
        run_id: str = "default",
        X_train_pre: Optional[np.ndarray] = None,
        X_test_pre:  Optional[np.ndarray] = None,
        y_train_pre: Optional[np.ndarray] = None,
        y_test_pre:  Optional[np.ndarray] = None,
    ) -> dict:
        """
        Parameters
        ----------
        X, y             : full feature matrix and labels
        task_type        : 'classification' or 'regression'
        tuning_results   : dict from HyperparameterTuningAgent
        run_id           : used to name the output sub-folder
        X_train_pre, ... : if provided (e.g. after imbalance resampling),
                            use these splits directly; otherwise split X, y
        """
        # ---- train/test split -----------------------------------------
        if X_train_pre is not None:
            X_train, X_test = X_train_pre, X_test_pre
            y_train, y_test = y_train_pre, y_test_pre
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if task_type == "classification" else None,
            )

        out_dir = os.path.join(self.output_root, run_id, "evaluation")
        os.makedirs(out_dir, exist_ok=True)

        scores: Dict[str, float] = {}
        fitted_models: Dict[str, Any] = {}
        metrics_detail: Dict[str, dict] = {}

        for model_name, res in tuning_results.items():
            try:
                trial = _FixedParamTrial(res["best_params"])
                model = _build_model(model_name, trial, task_type)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                if task_type == "classification":
                    m = self._classification_metrics(model, X_test, y_test, preds)
                    scores[model_name] = m.get("accuracy", 0.0)
                else:
                    m = self._regression_metrics(y_test, preds)
                    scores[model_name] = m.get("r2", float("-inf"))

                metrics_detail[model_name] = m
                fitted_models[model_name]  = model
                self.log.add("Evaluation", f"{model_name} test metrics", str(m))
            except Exception as e:
                warnings.warn(f"Evaluation of {model_name} failed: {e}")
                self.log.add("Evaluation", f"{model_name} FAILED", str(e))

        # Calculate improvement over dummy baseline
        dummy_score = scores.get("dummy", 0.0)
        for m_name in scores:
            imp = scores[m_name] - dummy_score
            metrics_detail[m_name]["improvement_over_dummy"] = round(imp, 4)
            if m_name in tuning_results:
                metrics_detail[m_name]["cv_mean"] = round(tuning_results[m_name].get("cv_mean", scores[m_name]), 4)
                metrics_detail[m_name]["cv_std"] = round(tuning_results[m_name].get("cv_std", 0.0), 4)

        # ---- final model choice (excluding dummy baseline) ------------
        candidate_scores = {k: v for k, v in scores.items() if k != "dummy"}
        if not candidate_scores:
            candidate_scores = scores

        final_decision, final_reasoning = self.reasoner.decide(
            "final_model_choice", {"scores": candidate_scores}
        )
        self.log.add("Evaluation", f"final_model={final_decision}", final_reasoning)

        final_model = fitted_models[final_decision]

        # ---- plots -------------------------------------------------------
        plot_paths: Dict[str, str] = {}
        self._make_plots(
            final_model, X_test, y_test,
            task_type, out_dir, plot_paths
        )

        return {
            "scores":           scores,
            "metrics_detail":   metrics_detail,
            "fitted_models":    fitted_models,
            "final_model_name": final_decision,
            "final_model":      final_model,
            "X_test":           X_test,
            "y_test":           y_test,
            "X_train":          X_train,
            "y_train":          y_train,
            "plot_paths":       plot_paths,
            "out_dir":          out_dir,
        }

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classification_metrics(model, X_test, y_test, preds) -> dict:
        classes    = np.unique(y_test)
        n_classes  = len(classes)
        acc        = float(accuracy_score(y_test, preds))
        prec       = float(precision_score(y_test, preds, average="macro", zero_division=0))
        rec        = float(recall_score(y_test, preds, average="macro", zero_division=0))
        f1_mac     = float(f1_score(y_test, preds, average="macro", zero_division=0))
        f1_wt      = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        try:
            if n_classes == 2:
                proba = model.predict_proba(X_test)[:, 1]
                pos_label = classes[1] if len(classes) > 1 else classes[0]
                y_bin = (y_test == pos_label).astype(int)
                auc   = float(roc_auc_score(y_bin, proba))
            else:
                proba = model.predict_proba(X_test)
                auc   = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")
        return {
            "accuracy":       round(acc,  4),
            "precision_macro": round(prec, 4),
            "recall_macro":   round(rec,  4),
            "f1_macro":       round(f1_mac, 4),
            "f1_weighted":    round(f1_wt, 4),
            "roc_auc":        round(auc,  4),
        }

    @staticmethod
    def _regression_metrics(y_test, preds) -> dict:
        mae  = float(mean_absolute_error(y_test, preds))
        mse  = float(mean_squared_error(y_test, preds))
        rmse = float(mse ** 0.5)
        r2   = float(r2_score(y_test, preds))
        return {
            "mae":  round(mae,  4),
            "mse":  round(mse,  4),
            "rmse": round(rmse, 4),
            "r2":   round(r2,   4),
        }

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _make_plots(self, model, X_test, y_test, task_type,
                     out_dir, plot_paths) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if task_type == "classification":
            self._plot_confusion_matrix(model, X_test, y_test, out_dir, plt, plot_paths)
            self._plot_roc_curve(model, X_test, y_test, out_dir, plt, plot_paths)
            self._plot_pr_curve(model, X_test, y_test, out_dir, plt, plot_paths)
        else:
            preds = model.predict(X_test)
            self._plot_actual_vs_predicted(y_test, preds, out_dir, plt, plot_paths)
            self._plot_residuals(y_test, preds, out_dir, plt, plot_paths)

    def _plot_confusion_matrix(self, model, X_test, y_test, out_dir, plt, out) -> None:
        try:
            from sklearn.metrics import ConfusionMatrixDisplay
            preds = model.predict(X_test)
            cm    = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(ax=ax, colorbar=False)
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            p = os.path.join(out_dir, "confusion_matrix.png")
            fig.savefig(p, dpi=100)
            plt.close(fig)
            out["confusion_matrix"] = p
        except Exception as e:
            warnings.warn(f"Confusion matrix plot failed: {e}")

    def _plot_roc_curve(self, model, X_test, y_test, out_dir, plt, out) -> None:
        try:
            from sklearn.metrics import roc_curve, auc
            classes   = np.unique(y_test)
            n_classes = len(classes)
            fig, ax   = plt.subplots(figsize=(6, 5))
            if n_classes == 2:
                proba = model.predict_proba(X_test)[:, 1]
                pos_label = classes[1] if len(classes) > 1 else classes[0]
                y_bin = (y_test == pos_label).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, proba)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}", color="#5EE39B")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "Multiclass ROC: see metrics table",
                        ha="center", va="center")
                ax.set_title("ROC Curve (Multiclass)")
            plt.tight_layout()
            p = os.path.join(out_dir, "roc_curve.png")
            fig.savefig(p, dpi=100)
            plt.close(fig)
            out["roc_curve"] = p
        except Exception as e:
            warnings.warn(f"ROC curve plot failed: {e}")

    def _plot_pr_curve(self, model, X_test, y_test, out_dir, plt, out) -> None:
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            classes   = np.unique(y_test)
            n_classes = len(classes)
            fig, ax   = plt.subplots(figsize=(6, 5))
            if n_classes == 2:
                proba = model.predict_proba(X_test)[:, 1]
                pos_label = classes[1] if len(classes) > 1 else classes[0]
                y_bin = (y_test == pos_label).astype(int)
                prec, rec, _ = precision_recall_curve(y_bin, proba)
                ap = average_precision_score(y_bin, proba)
                ax.step(rec, prec, where="post", color="#F2B84B", label=f"AP={ap:.4f}")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curve")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "Multiclass P-R: see metrics table",
                        ha="center", va="center")
                ax.set_title("Precision-Recall Curve (Multiclass)")
            plt.tight_layout()
            p = os.path.join(out_dir, "pr_curve.png")
            fig.savefig(p, dpi=100)
            plt.close(fig)
            out["pr_curve"] = p
        except Exception as e:
            warnings.warn(f"P-R curve plot failed: {e}")

    def _plot_actual_vs_predicted(self, y_test, preds, out_dir, plt, out) -> None:
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test, preds, alpha=0.4, s=10, color="#5EE39B")
            mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            plt.tight_layout()
            p = os.path.join(out_dir, "actual_vs_predicted.png")
            fig.savefig(p, dpi=100)
            plt.close(fig)
            out["actual_vs_predicted"] = p
        except Exception as e:
            warnings.warn(f"Actual vs Predicted plot failed: {e}")

    def _plot_residuals(self, y_test, preds, out_dir, plt, out) -> None:
        try:
            residuals = y_test - preds
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(preds, residuals, alpha=0.4, s=10, color="#F2B84B")
            ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residual")
            ax.set_title("Residual Plot")
            plt.tight_layout()
            p = os.path.join(out_dir, "residuals.png")
            fig.savefig(p, dpi=100)
            plt.close(fig)
            out["residuals"] = p
        except Exception as e:
            warnings.warn(f"Residual plot failed: {e}")
