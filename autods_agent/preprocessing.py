"""
PreprocessingAgent & TabularPreprocessor (v3 - Leakage-Free)
============================================================
Refactored into a scikit-learn compatible transformer supporting fit() / transform():
  - Target encoding for categorical classification labels
  - Duplicate detection & removal on training split
  - Drops ID-like columns (unique ratio > 0.95 for non-float columns)
  - Drops constant columns (nunique <= 1)
  - Warns on suspected target leakage (|corr| > 0.95 with target)
  - Drops columns exceeding 30% missingness threshold
  - Skew-aware imputation (mean for near-symmetric |skew| < 0.5, median otherwise)
  - Cardinality-aware encoding:
      * low cardinality (<= 10 unique): OneHotEncoder
      * high cardinality (> 10 unique): OrdinalEncoder
  - Standard scaling on numeric features
  - Full audit logging via RunLog and preprocessing_explanations
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for tabular data cleaning,
    imputation, cardinality-aware encoding, and scaling.
    Guarantees no data leakage: fits parameters strictly on training partition.
    """

    ONE_HOT_CARDINALITY_THRESHOLD = 10

    def __init__(self, reasoner=None, log=None, one_hot_threshold: int = ONE_HOT_CARDINALITY_THRESHOLD):
        self.reasoner = reasoner
        self.log = log
        self.one_hot_threshold = one_hot_threshold
        self.preprocessing_explanations: list[str] = []

        # Fitted parameters
        self.dropped_cols_: list[str] = []
        self.id_like_cols_: list[str] = []
        self.constant_cols_: list[str] = []
        self.leakage_warnings_: list[str] = []
        self.impute_values_: dict[str, any] = {}
        self.impute_strategies_: dict[str, str] = {}
        self.num_cols_: list[str] = []
        self.cat_cols_: list[str] = []
        self.one_hot_cols_: list[str] = []
        self.ordinal_cols_: list[str] = []
        self.one_hot_encoders_: dict[str, OneHotEncoder] = {}
        self.ordinal_encoder_: OrdinalEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_out_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        else:
            X = X.copy()

        self.preprocessing_explanations = []
        n_rows = len(X)
        cols_to_drop = []

        # 1. Detect and drop constant columns
        constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        self.constant_cols_ = constant_cols
        if constant_cols:
            cols_to_drop.extend(constant_cols)
            msg = f"Dropped {len(constant_cols)} constant column(s): {constant_cols}."
            self.preprocessing_explanations.append(msg)
            if self.log:
                self.log.add("Preprocessing[local]", "drop constant columns", msg)

        # 2. Detect and drop ID-like columns (unique ratio > 0.95 on non-float)
        id_like_cols = []
        for c in X.columns:
            if c in cols_to_drop:
                continue
            if not pd.api.types.is_float_dtype(X[c]):
                unique_ratio = X[c].nunique(dropna=True) / max(1, n_rows)
                if unique_ratio > 0.95 and n_rows >= 20:
                    id_like_cols.append(c)
        self.id_like_cols_ = id_like_cols
        if id_like_cols:
            cols_to_drop.extend(id_like_cols)
            msg = f"Dropped {len(id_like_cols)} ID-like column(s) (unique ratio > 95%): {id_like_cols}."
            self.preprocessing_explanations.append(msg)
            if self.log:
                self.log.add("Preprocessing[local]", "drop ID-like columns", msg)

        # 3. Target leakage check (if y is provided and numeric)
        if y is not None:
            try:
                y_arr = np.asarray(y)
                if pd.api.types.is_numeric_dtype(y_arr):
                    for c in X.columns:
                        if c in cols_to_drop:
                            continue
                        if pd.api.types.is_numeric_dtype(X[c]):
                            valid_mask = (~X[c].isna()) & (~np.isnan(y_arr))
                            if valid_mask.sum() > 5:
                                corr = float(np.corrcoef(X[c][valid_mask], y_arr[valid_mask])[0, 1])
                                if abs(corr) > 0.95:
                                    warn_msg = f"WARNING: Suspected target leakage! Column '{c}' has |correlation|={abs(corr):.4f} > 0.95 with target."
                                    self.leakage_warnings_.append(warn_msg)
                                    self.preprocessing_explanations.append(warn_msg)
                                    if self.log:
                                        self.log.add("Preprocessing[local]", f"target_leakage_warning: {c}", warn_msg)
            except Exception:
                pass

        # 4. Check missing values (> 30% drop threshold)
        for c in X.columns:
            if c in cols_to_drop:
                continue
            missing_frac = float(X[c].isna().mean())
            if missing_frac > 0.30:
                cols_to_drop.append(c)
                msg = f"Dropped column '{c}': {missing_frac:.1%} missing exceeds 30% threshold."
                self.preprocessing_explanations.append(msg)
                if self.log:
                    self.log.add("Preprocessing[local]", f"{c}: drop_column", msg)

        self.dropped_cols_ = list(dict.fromkeys(cols_to_drop))
        remaining_cols = [c for c in X.columns if c not in self.dropped_cols_]

        # 5. Fit skew-aware imputers on training data
        self.impute_values_ = {}
        self.impute_strategies_ = {}
        for c in remaining_cols:
            if X[c].isna().any():
                is_numeric = pd.api.types.is_numeric_dtype(X[c])
                if is_numeric:
                    skew = float(X[c].skew())
                    if abs(skew) < 0.5:
                        val = float(X[c].mean())
                        strategy = "mean"
                    else:
                        val = float(X[c].median())
                        strategy = "median"
                    self.impute_values_[c] = val
                    self.impute_strategies_[c] = strategy
                    self.preprocessing_explanations.append(f"Imputed '{c}' with {strategy} ({val:.4f}, skew={skew:.2f}).")
                else:
                    mode_val = X[c].mode(dropna=True)
                    val = str(mode_val.iloc[0]) if not mode_val.empty else "missing"
                    self.impute_values_[c] = val
                    self.impute_strategies_[c] = "mode"
                    self.preprocessing_explanations.append(f"Imputed '{c}' with mode ('{val}').")

        # Create temporary imputed view to fit encoders and scaler
        X_imputed = X[remaining_cols].copy()
        for c, val in self.impute_values_.items():
            X_imputed[c] = X_imputed[c].fillna(val)

        # 6. Fit categorical encoders
        num_cols = [c for c in remaining_cols if pd.api.types.is_numeric_dtype(X_imputed[c])]
        cat_cols = [c for c in remaining_cols if not pd.api.types.is_numeric_dtype(X_imputed[c])]
        self.num_cols_ = num_cols
        self.cat_cols_ = cat_cols

        one_hot_cols = [c for c in cat_cols if X_imputed[c].nunique() <= self.one_hot_threshold]
        ordinal_cols = [c for c in cat_cols if c not in one_hot_cols]
        self.one_hot_cols_ = one_hot_cols
        self.ordinal_cols_ = ordinal_cols

        self.one_hot_encoders_ = {}
        for c in one_hot_cols:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            enc.fit(X_imputed[[c]].astype(str))
            self.one_hot_encoders_[c] = enc
            self.preprocessing_explanations.append(f"One-hot encoded low-cardinality feature '{c}'.")

        if ordinal_cols:
            ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            ord_enc.fit(X_imputed[ordinal_cols].astype(str))
            self.ordinal_encoder_ = ord_enc
            self.preprocessing_explanations.append(f"Ordinal encoded {len(ordinal_cols)} high-cardinality features.")

        # 7. Fit StandardScaler on numeric features
        if num_cols:
            scaler = StandardScaler()
            scaler.fit(X_imputed[num_cols])
            self.scaler_ = scaler
            self.preprocessing_explanations.append(f"StandardScaler fit on {len(num_cols)} numeric features.")

        # Determine output feature names
        out_features = list(self.num_cols_)
        for c in self.one_hot_cols_:
            out_features.extend(list(self.one_hot_encoders_[c].get_feature_names_out([c])))
        out_features.extend(list(self.ordinal_cols_))
        self.feature_names_out_ = out_features

        if self.log:
            self.log.add(
                "Preprocessing[local]",
                "fit complete",
                f"Trained preprocessing on {n_rows} rows: dropped {len(self.dropped_cols_)} cols, "
                f"imputed {len(self.impute_values_)}, one-hot {len(self.one_hot_cols_)}, "
                f"ordinal {len(self.ordinal_cols_)}, scaled {len(self.num_cols_)} numeric features."
            )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        else:
            X = X.copy()

        # Drop ignored/dropped columns
        cols_to_drop = [c for c in self.dropped_cols_ if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)

        # Impute missing values
        for c, val in self.impute_values_.items():
            if c in X.columns:
                X[c] = X[c].fillna(val)
        # Any residual numeric NaN filled with 0
        for c in self.num_cols_:
            if c in X.columns and X[c].isna().any():
                X[c] = X[c].fillna(0.0)

        # Apply scaling to numeric features
        if self.scaler_ is not None and self.num_cols_:
            present_num = [c for c in self.num_cols_ if c in X.columns]
            if len(present_num) == len(self.num_cols_):
                scaled_num = self.scaler_.transform(X[self.num_cols_])
            else:
                scaled_df = pd.DataFrame(0.0, index=X.index, columns=self.num_cols_)
                for c in present_num:
                    scaled_df[c] = X[c]
                scaled_num = self.scaler_.transform(scaled_df)
        else:
            scaled_num = np.empty((len(X), 0))

        # Apply One-Hot Encoders
        ohe_parts = []
        for c in self.one_hot_cols_:
            if c in X.columns and c in self.one_hot_encoders_:
                enc = self.one_hot_encoders_[c]
                ohe_parts.append(enc.transform(X[[c]].astype(str)))
            elif c in self.one_hot_encoders_:
                enc = self.one_hot_encoders_[c]
                dummy_df = pd.DataFrame(["unknown"] * len(X), columns=[c])
                ohe_parts.append(enc.transform(dummy_df))

        # Apply Ordinal Encoders
        if self.ordinal_encoder_ is not None and self.ordinal_cols_:
            present_ord = [c for c in self.ordinal_cols_ if c in X.columns]
            if len(present_ord) == len(self.ordinal_cols_):
                ord_vals = self.ordinal_encoder_.transform(X[self.ordinal_cols_].astype(str))
            else:
                ord_df = pd.DataFrame("unknown", index=X.index, columns=self.ordinal_cols_)
                for c in present_ord:
                    ord_df[c] = X[c].astype(str)
                ord_vals = self.ordinal_encoder_.transform(ord_df)
        else:
            ord_vals = np.empty((len(X), 0))

        # Stack into single feature matrix
        arrays = [scaled_num]
        if ohe_parts:
            arrays.extend(ohe_parts)
        if ord_vals.size > 0:
            arrays.append(ord_vals)

        result = np.hstack(arrays)
        return result

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return list(self.feature_names_out_)


class PreprocessingAgent:
    """Agent wrapping TabularPreprocessor with audit logging and compatibility methods."""

    def __init__(self, reasoner, log):
        self.reasoner = reasoner
        self.log = log
        self.preprocessor = TabularPreprocessor(reasoner=reasoner, log=log)
        self.preprocessing_explanations: list[str] = []
        self._target_encoder: LabelEncoder | None = None
        self.target_classes: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.preprocessor.fit(X, y=y)
        self.preprocessing_explanations = list(self.preprocessor.preprocessing_explanations)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(X)

    def get_feature_names_out(self) -> list[str]:
        return self.preprocessor.get_feature_names_out()

    def run(self, df: pd.DataFrame, profile: dict) -> pd.DataFrame:
        """Backwards compatibility entrypoint: fits and transforms in one step."""
        target_col = profile.get("target_col")
        df_copy = df.copy()

        # Deduplication
        n_dup = int(df_copy.duplicated().sum())
        if n_dup > 0:
            df_copy = df_copy.drop_duplicates().reset_index(drop=True)
            msg = f"Removed {n_dup} duplicate rows."
            self.preprocessing_explanations.append(msg)
            if self.log:
                self.log.add("Preprocessing[local]", f"duplicates removed={n_dup}", msg)

        # Target encoding if classification
        if profile.get("task_type") == "classification" and target_col and target_col in df_copy.columns:
            if not pd.api.types.is_numeric_dtype(df_copy[target_col]) or df_copy[target_col].dtype == object or df_copy[target_col].dtype == bool:
                le = LabelEncoder()
                df_copy[target_col] = le.fit_transform(df_copy[target_col].astype(str))
                self._target_encoder = le
                self.target_classes = list(le.classes_)
                msg = f"Encoded target '{target_col}' ({self.target_classes}) into numeric labels [0..{len(self.target_classes)-1}]."
                self.preprocessing_explanations.append(msg)
                if self.log:
                    self.log.add("Preprocessing[local]", f"encoded target {target_col}", msg)

        if profile.get("backend") == "distributed":
            return self._run_distributed(df_copy, profile)

        X = df_copy.drop(columns=[target_col]) if target_col in df_copy.columns else df_copy
        y = df_copy[target_col] if target_col in df_copy.columns else None

        self.fit(X, y=y)
        transformed_arr = self.transform(X)
        transformed_df = pd.DataFrame(transformed_arr, columns=self.get_feature_names_out(), index=df_copy.index)

        if target_col in df_copy.columns:
            transformed_df[target_col] = df_copy[target_col].values

        return transformed_df

    def _run_distributed(self, df: pd.DataFrame, profile: dict) -> pd.DataFrame:
        import dask.dataframe as dd
        target_col = profile.get("target_col")
        df_copy = df.copy()

        n_partitions = max(4, profile.get("n_rows", 1000) // 200_000)
        ddf = dd.from_pandas(df_copy, npartitions=n_partitions)

        if self.log:
            self.log.add(
                "Preprocessing[distributed]",
                f"partitioned into {n_partitions} chunks",
                f"Split into {n_partitions} Dask partitions for out-of-core scaling."
            )

        num_cols = [c for c in df_copy.columns if c != target_col and pd.api.types.is_numeric_dtype(df_copy[c])]
        for col in num_cols:
            if df_copy[col].isna().any():
                med = float(ddf[col].quantile(0.5).compute())
                ddf[col] = ddf[col].fillna(med)
                self.preprocessing_explanations.append(f"Dask-median-imputed '{col}'.")

        pdf = ddf.compute()
        if num_cols:
            scaler = StandardScaler()
            pdf[num_cols] = scaler.fit_transform(pdf[num_cols])
            self.preprocessing_explanations.append(f"StandardScaler applied to {len(num_cols)} numeric columns.")

        if self.log:
            self.log.add(
                "Preprocessing[distributed]",
                "complete",
                f"Distributed preprocessing done ({n_partitions} partitions). Scaled {len(num_cols)} columns."
            )
        return pdf
