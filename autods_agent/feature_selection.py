"""
FeatureSelectionAgent & FeatureSelectionTransformer (v3 - Leakage-Free)
=======================================================================
Refactored into a scikit-learn compatible transformer supporting fit() / transform():
  - Mutual Information (MI)
  - Univariate Statistical Filter (SelectKBest f_classif / f_regression)
  - Tree-based feature importance via RandomForest
  - Selected strictly on training split (no target leakage to test set)
"""

from __future__ import annotations

import warnings
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression,
    f_classif, f_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for feature selection.
    Fits selectors strictly on training data (X, y) to eliminate data leakage.
    """

    def __init__(self, reasoner=None, log=None, task_type: str = "classification", k: int | None = None):
        self.reasoner = reasoner
        self.log = log
        self.task_type = task_type
        self.k = k

        # Fitted parameters
        self.selected_indices_: list[int] = []
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []
        self.selection_method_: str = "none"

    def fit(self, X: np.ndarray | pd.DataFrame, y, feature_names: list[str] | None = None):
        if isinstance(X, pd.DataFrame):
            in_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            in_names = feature_names if feature_names else [f"f_{i}" for i in range(X_arr.shape[1])]

        n_features = X_arr.shape[1]
        y_arr = np.asarray(y)

        # Handle NaNs
        X_arr = np.nan_to_num(X_arr, nan=0.0)

        # Determine K
        k = self.k if self.k is not None else min(50, max(5, n_features // 2))

        if n_features <= k:
            self.selected_indices_ = list(range(n_features))
            self.selected_features_ = list(in_names)
            self.dropped_features_ = []
            self.selection_method_ = "skipped"
            if self.log:
                self.log.add(
                    "FeatureSelection",
                    f"skipped (n_features={n_features} <= k={k})",
                    "Feature count is small enough that selection would not reduce dimensionality."
                )
            return self

        # Choose method via reasoner or heuristic
        if self.reasoner:
            method_decision, method_reason = self.reasoner.decide(
                "feature_selection_method",
                {"n_features": n_features, "task_type": self.task_type, "k": k},
            )
        else:
            method_decision = "mutual_info"
            method_reason = "Default model-free non-linear dependency filter."

        if self.log:
            self.log.add("FeatureSelection", f"method={method_decision}, k={k}", method_reason)

        selected_indices = []
        try:
            if method_decision == "model_importance":
                if self.task_type == "classification":
                    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=1)
                else:
                    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=1)
                rf.fit(X_arr, y_arr)
                importances = rf.feature_importances_
                selected_indices = list(np.argsort(importances)[::-1][:k])

            elif method_decision == "select_k_best":
                score_func = f_classif if self.task_type == "classification" else f_regression
                selector = SelectKBest(score_func=score_func, k=k)
                selector.fit(X_arr, y_arr)
                selected_indices = list(selector.get_support(indices=True))

            else:  # mutual_info
                mi_func = mutual_info_classif if self.task_type == "classification" else mutual_info_regression
                mi_scores = mi_func(X_arr, y_arr, random_state=42)
                selected_indices = list(np.argsort(mi_scores)[::-1][:k])

        except Exception as e:
            warnings.warn(f"Feature selection method '{method_decision}' failed ({e}); falling back to top-{k} variance.")
            variances = np.var(X_arr, axis=0)
            selected_indices = list(np.argsort(variances)[::-1][:k])
            method_decision = "variance_fallback"

        selected_indices.sort()
        self.selected_indices_ = selected_indices
        self.selected_features_ = [in_names[i] for i in selected_indices]
        self.dropped_features_ = [in_names[i] for i in range(n_features) if i not in selected_indices]
        self.selection_method_ = method_decision

        if self.log:
            self.log.add(
                "FeatureSelection",
                f"retained {len(self.selected_features_)} / {n_features} features",
                f"Selected top {k} features via {method_decision}. Dropped {len(self.dropped_features_)} features."
            )
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        if not self.selected_indices_:
            return X_arr

        return X_arr[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return list(self.selected_features_)


class FeatureSelectionAgent:
    """Agent wrapping FeatureSelectionTransformer with audit logging."""

    def __init__(self, reasoner, log):
        self.reasoner = reasoner
        self.log = log
        self.transformer: FeatureSelectionTransformer | None = None
        self.last_result: dict = {}

    def fit(self, X: np.ndarray | pd.DataFrame, y, feature_names: list[str] | None = None, task_type: str = "classification"):
        self.transformer = FeatureSelectionTransformer(
            reasoner=self.reasoner, log=self.log, task_type=task_type
        )
        self.transformer.fit(X, y, feature_names=feature_names)
        self.last_result = {
            "selected_features": self.transformer.selected_features_,
            "dropped_features": self.transformer.dropped_features_,
            "method": self.transformer.selection_method_,
        }
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            return np.asarray(X)
        return self.transformer.transform(X)

    def get_feature_names_out(self) -> list[str]:
        if self.transformer is None:
            return []
        return self.transformer.get_feature_names_out()

    def run(self, df: pd.DataFrame, profile: dict) -> pd.DataFrame:
        target_col = profile.get("target_col")
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].values

        self.fit(X, y, feature_names=feature_cols, task_type=profile.get("task_type", "classification"))
        transformed = self.transform(X)
        selected_names = self.get_feature_names_out()

        out_df = pd.DataFrame(transformed, columns=selected_names, index=df.index)
        if target_col in df.columns:
            out_df[target_col] = df[target_col].values
        return out_df
