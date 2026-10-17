"""
FeatureEngineeringAgent & FeatureEngineeringTransformer (v3 - Leakage-Free)
===========================================================================
Refactored into a scikit-learn compatible transformer supporting fit() / transform():
  - Pairwise interaction features (a * b)
  - Polynomial square features (a ** 2)
  - Scale-invariant ratio features (a / (b + 1e-6))
  - Budget-capped expansion to prevent dimensional explosion
  - Audit logging of feature additions and rationales
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for interaction, polynomial, and ratio features.
    Learns feature combinations on training data only and computes them consistently.
    """

    def __init__(self, reasoner=None, log=None, max_interaction_features: int = 30):
        self.reasoner = reasoner
        self.log = log
        self.max_interaction_features = max_interaction_features

        # Fitted definitions
        self.input_feature_names_: list[str] = []
        self.interaction_pairs_: list[tuple[int, int, str]] = []
        self.poly_indices_: list[tuple[int, str]] = []
        self.ratio_pairs_: list[tuple[int, int, str]] = []
        self.feature_names_out_: list[str] = []

    def fit(self, X: np.ndarray | pd.DataFrame, y=None, feature_names: list[str] | None = None):
        if isinstance(X, pd.DataFrame):
            in_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            in_names = feature_names if feature_names else [f"f_{i}" for i in range(X_arr.shape[1])]

        self.input_feature_names_ = list(in_names)
        n_features = X_arr.shape[1]

        # Determine pairwise interactions
        n_possible = n_features * (n_features - 1) // 2
        if n_possible > self.max_interaction_features:
            k = max(2, int((2 * self.max_interaction_features) ** 0.5))
            rng = np.random.RandomState(42)
            chosen_indices = list(rng.choice(n_features, size=min(k, n_features), replace=False))
            reasoning = f"{n_features} features -> {n_possible} pairs exceeds {self.max_interaction_features} cap. Selected {len(chosen_indices)} features."
        else:
            chosen_indices = list(range(n_features))
            reasoning = f"{n_features} features -> {n_possible} pairs within budget."

        if self.log:
            self.log.add("FeatureEngineering", f"pairs from {len(chosen_indices)} features", reasoning)

        self.interaction_pairs_ = []
        added = 0
        for i, j in itertools.combinations(chosen_indices, 2):
            name = f"{in_names[i]}_x_{in_names[j]}"
            self.interaction_pairs_.append((i, j, name))
            added += 1
            if added >= self.max_interaction_features:
                break

        # Polynomial features (squares)
        poly_budget = min(10, max(0, self.max_interaction_features - added))
        self.poly_indices_ = []
        if poly_budget > 0:
            for idx in chosen_indices[:poly_budget]:
                name = f"{in_names[idx]}_sq"
                self.poly_indices_.append((idx, name))
            if self.poly_indices_ and self.log:
                self.log.add(
                    "FeatureEngineering",
                    f"added {len(self.poly_indices_)} polynomial (sq) features",
                    "Degree-2 squares of top signals to model non-linear dynamics."
                )

        # Ratio features
        remaining_budget = max(0, self.max_interaction_features - added - len(self.poly_indices_))
        ratio_budget = min(10, remaining_budget)
        self.ratio_pairs_ = []
        if ratio_budget > 0:
            for i, j in itertools.combinations(chosen_indices[:8], 2):
                if len(self.ratio_pairs_) >= ratio_budget:
                    break
                denom = X_arr[:, j]
                denom_nonzero = np.where(denom == 0, 1e-6, denom)
                ratios = X_arr[:, i] / denom_nonzero
                if not np.isinf(ratios).all() and not np.isnan(ratios).all():
                    name = f"{in_names[i]}_ratio_{in_names[j]}"
                    self.ratio_pairs_.append((i, j, name))

            if self.ratio_pairs_ and self.log:
                self.log.add(
                    "FeatureEngineering",
                    f"added {len(self.ratio_pairs_)} ratio features",
                    "A/B ratios to capture scale-invariant interactions."
                )

        # Compute output feature names
        out_names = list(self.input_feature_names_)
        out_names.extend([p[2] for p in self.interaction_pairs_])
        out_names.extend([p[1] for p in self.poly_indices_])
        out_names.extend([p[2] for p in self.ratio_pairs_])
        self.feature_names_out_ = out_names

        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        n_rows = X_arr.shape[0]
        n_extra = len(self.interaction_pairs_) + len(self.poly_indices_) + len(self.ratio_pairs_)
        if n_extra == 0:
            return X_arr

        extras = np.empty((n_rows, n_extra), dtype=float)
        col_idx = 0

        # Compute interactions
        for i, j, _ in self.interaction_pairs_:
            if i < X_arr.shape[1] and j < X_arr.shape[1]:
                extras[:, col_idx] = X_arr[:, i] * X_arr[:, j]
            else:
                extras[:, col_idx] = 0.0
            col_idx += 1

        # Compute squares
        for i, _ in self.poly_indices_:
            if i < X_arr.shape[1]:
                extras[:, col_idx] = X_arr[:, i] ** 2
            else:
                extras[:, col_idx] = 0.0
            col_idx += 1

        # Compute ratios
        for i, j, _ in self.ratio_pairs_:
            if i < X_arr.shape[1] and j < X_arr.shape[1]:
                denom = np.where(X_arr[:, j] == 0, 1e-6, X_arr[:, j])
                extras[:, col_idx] = np.nan_to_num(X_arr[:, i] / denom, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                extras[:, col_idx] = 0.0
            col_idx += 1

        return np.hstack([X_arr, extras])

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return list(self.feature_names_out_)


class FeatureEngineeringAgent:
    """Agent wrapping FeatureEngineeringTransformer with audit logging."""

    def __init__(self, reasoner, log, max_interaction_features: int = 30):
        self.reasoner = reasoner
        self.log = log
        self.max_interaction_features = max_interaction_features
        self.transformer = FeatureEngineeringTransformer(
            reasoner=reasoner, log=log, max_interaction_features=max_interaction_features
        )

    def fit(self, X: np.ndarray | pd.DataFrame, y=None, feature_names: list[str] | None = None):
        self.transformer.fit(X, y=y, feature_names=feature_names)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.transformer.transform(X)

    def get_feature_names_out(self) -> list[str]:
        return self.transformer.get_feature_names_out()

    def run(self, df: pd.DataFrame, profile: dict) -> pd.DataFrame:
        target_col = profile.get("target_col")
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].copy()

        self.fit(X, feature_names=feature_cols)
        transformed = self.transform(X)
        out_names = self.get_feature_names_out()

        out_df = pd.DataFrame(transformed, columns=out_names, index=df.index)
        if target_col in df.columns:
            out_df[target_col] = df[target_col].values
        return out_df
