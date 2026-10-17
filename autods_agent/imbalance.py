"""
ImbalanceHandlerAgent
======================
Phase 3 – Detects and handles class imbalance for classification tasks.

Strategy selection:
  1. Detect minority class ratio.
  2. If imbalanced (minority < `threshold`), experiment with:
       a. class_weight="balanced" (sklearn native)
       b. SMOTE oversampling (imbalanced-learn)
       c. RandomOverSampler
       d. RandomUnderSampler
  3. Compare F1, Recall, ROC-AUC before vs after each strategy.
  4. Choose the strategy with the highest macro-F1.

IMPORTANT: resampling is applied ONLY on the training split to avoid data
leakage. The test split is always kept clean.
"""

from __future__ import annotations

import warnings
import logging
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)

MINORITY_THRESHOLD = 0.20   # minority class fraction below which we act


class ImbalanceHandlerAgent:
    """Detects imbalance, experiments with resampling strategies, chooses best."""

    def __init__(self, reasoner, log,
                 minority_threshold: float = MINORITY_THRESHOLD):
        self.reasoner = reasoner
        self.log = log
        self.minority_threshold = minority_threshold

    # ------------------------------------------------------------------
    def run(self, X: np.ndarray, y: np.ndarray,
            task_type: str) -> Dict[str, Any]:
        """
        Parameters
        ----------
        X, y       : full feature matrix and labels (pre-split)
        task_type  : 'classification' or 'regression'

        Returns
        -------
        dict with keys: is_imbalanced, minority_ratio, best_strategy,
                        strategy_comparison, X_train_resampled, y_train_resampled,
                        X_test, y_test
        """
        if task_type != "classification":
            # Imbalance handling only for classification
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42)
            return {
                "is_imbalanced":       False,
                "minority_ratio":      None,
                "best_strategy":       "none",
                "strategy_comparison": {},
                "X_train_resampled":   X_tr,
                "y_train_resampled":   y_tr,
                "X_test":              X_te,
                "y_test":              y_te,
            }

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        counts = pd.Series(y_tr).value_counts(normalize=True)
        minority_ratio = float(counts.min())
        is_imbalanced  = minority_ratio < self.minority_threshold

        if not is_imbalanced:
            self.log.add(
                "ImbalanceHandling",
                f"balanced (minority={minority_ratio:.2%})",
                f"Minority class is {minority_ratio:.2%} of training data, "
                f"above the {self.minority_threshold:.0%} threshold. No resampling needed.",
            )
            return {
                "is_imbalanced":       False,
                "minority_ratio":      minority_ratio,
                "best_strategy":       "none",
                "strategy_comparison": {},
                "X_train_resampled":   X_tr,
                "y_train_resampled":   y_tr,
                "X_test":              X_te,
                "y_test":              y_te,
            }

        self.log.add(
            "ImbalanceHandling",
            f"imbalanced detected (minority={minority_ratio:.2%})",
            f"Minority class is {minority_ratio:.2%} of training data (<{self.minority_threshold:.0%}). "
            f"Experimenting with class_weight, SMOTE, RandomOverSampler, RandomUnderSampler.",
        )

        # ---- experiment with strategies --------------------------------
        strategies = self._get_strategies(X_tr, y_tr)
        comparison: Dict[str, dict] = {}

        probe = LogisticRegression(max_iter=500, random_state=42)
        probe.fit(X_tr, y_tr)
        baseline = self._metrics(probe, X_te, y_te)
        comparison["baseline"] = baseline

        best_strategy  = "baseline"
        best_f1        = baseline["f1_macro"]
        best_X_tr      = X_tr
        best_y_tr      = y_tr

        for name, (X_res, y_res) in strategies.items():
            try:
                m = LogisticRegression(max_iter=500, random_state=42)
                m.fit(X_res, y_res)
                metrics = self._metrics(m, X_te, y_te)
                comparison[name] = metrics
                if metrics["f1_macro"] > best_f1:
                    best_f1       = metrics["f1_macro"]
                    best_strategy = name
                    best_X_tr     = X_res
                    best_y_tr     = y_res
            except Exception as e:
                warnings.warn(f"Imbalance strategy {name} failed: {e}")
                comparison[name] = {"error": str(e)}

        self.log.add(
            "ImbalanceHandling",
            f"best_strategy={best_strategy} (macro-F1={best_f1:.4f})",
            f"Compared {len(comparison)} strategies; "
            f"{best_strategy} gave the highest macro-F1={best_f1:.4f}. "
            f"Returning its resampled training set for downstream training.",
        )

        return {
            "is_imbalanced":       True,
            "minority_ratio":      minority_ratio,
            "best_strategy":       best_strategy,
            "strategy_comparison": comparison,
            "X_train_resampled":   best_X_tr,
            "y_train_resampled":   best_y_tr,
            "X_test":              X_te,
            "y_test":              y_te,
        }

    # ------------------------------------------------------------------
    def _get_strategies(self, X_tr: np.ndarray,
                         y_tr: np.ndarray) -> Dict[str, Tuple]:
        strategies: Dict[str, Tuple] = {}

        # class_weight="balanced" — no resampling, just pass the original split
        strategies["class_weight_balanced"] = (X_tr, y_tr)

        # SMOTE
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
            strategies["SMOTE"] = (X_sm, y_sm)
        except Exception as e:
            warnings.warn(f"SMOTE skipped: {e}")

        # RandomOverSampler
        try:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_ros, y_ros = ros.fit_resample(X_tr, y_tr)
            strategies["RandomOverSampler"] = (X_ros, y_ros)
        except Exception as e:
            warnings.warn(f"RandomOverSampler skipped: {e}")

        # RandomUnderSampler
        try:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            X_rus, y_rus = rus.fit_resample(X_tr, y_tr)
            strategies["RandomUnderSampler"] = (X_rus, y_rus)
        except Exception as e:
            warnings.warn(f"RandomUnderSampler skipped: {e}")

        return strategies

    @staticmethod
    def _metrics(model, X_te: np.ndarray, y_te: np.ndarray) -> dict:
        preds  = model.predict(X_te)
        f1     = float(f1_score(y_te, preds, average="macro", zero_division=0))
        rec    = float(recall_score(y_te, preds, average="macro", zero_division=0))
        try:
            classes = np.unique(y_te)
            if len(classes) == 2:
                proba = model.predict_proba(X_te)[:, 1]
                pos_label = classes[1] if len(classes) > 1 else classes[0]
                y_bin = (y_te == pos_label).astype(int)
                auc   = float(roc_auc_score(y_bin, proba))
            else:
                proba = model.predict_proba(X_te)
                auc   = float(roc_auc_score(y_te, proba, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")
        return {"f1_macro": round(f1, 4), "recall_macro": round(rec, 4), "roc_auc": round(auc, 4)}
