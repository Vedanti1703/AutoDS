"""
Orchestrator (v3 - Leakage-Free & Scalable Pipeline)
====================================================
AutoDS-Agent autonomous orchestrator:
  1. Profiling (DataProfilerAgent)
  2. Auto EDA (AutoEDAAgent)
  3. Train/Test Stratified Split FIRST (eliminates data leakage)
  4. Anomaly Detection on training partition
  5. Preprocessing fit on train only, transform test
  6. Feature Engineering fit on train only, transform test
  7. Feature Selection fit on train only, transform test
  8. Imbalance Handling on train only
  9. Model Selection (ModelSelectionAgent)
  10. Metric-Aware Hyperparameter Tuning (Optuna + Dummy baseline)
  11. Evaluation & Leaderboard (EvaluationAgent)
  12. Explainable AI with SHAP (ExplainabilityAgent)
  13. Assembly of FullInferencePipeline (end-to-end raw row inference)
  14. What-If Counterfactual Setup (WhatIfAgent using full pipeline)
  15. Model Registry Persistence (saving FullInferencePipeline)
  16. Structured CSV Results Export (CSVExportManager)
  17. Audit Reporting (ReportingAgent)
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .utils import RunLog
from .llm_reasoner import get_reasoner
from .profiler import DataProfilerAgent
from .eda import AutoEDAAgent
from .preprocessing import PreprocessingAgent, TabularPreprocessor
from .anomaly import AnomalyDetectionAgent
from .imbalance import ImbalanceHandlerAgent
from .feature_engineering import FeatureEngineeringAgent, FeatureEngineeringTransformer
from .feature_selection import FeatureSelectionAgent, FeatureSelectionTransformer
from .model_selection import ModelSelectionAgent, HyperparameterTuningAgent
from .evaluation import EvaluationAgent
from .explainability import ExplainabilityAgent
from .reporting import ReportingAgent
from .csv_export import CSVExportManager
from .whatif import WhatIfAgent
from .assistant import DataScienceAssistant, build_assistant_context
from .registry import ModelRegistry
from .drift import DriftDetectionAgent


class FullInferencePipeline(BaseEstimator):
    """
    Unified end-to-end inference pipeline:
    Raw Tabular Data -> Preprocessing -> Feature Engineering -> Feature Selection -> Final Estimator
    Predicts seamlessly from raw rows without external feature engineering scripts.
    """

    def __init__(
        self,
        preprocessor: TabularPreprocessor,
        feature_engineer: FeatureEngineeringTransformer,
        feature_selector: FeatureSelectionTransformer,
        model: Any,
        target_encoder: Optional[LabelEncoder] = None,
        target_col: str = "",
        feature_names: Optional[List[str]] = None,
    ):
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.feature_selector = feature_selector
        self.model = model
        self.target_encoder = target_encoder
        self.target_col = target_col
        self.feature_names = feature_names or []
        if hasattr(model, "classes_"):
            self.classes_ = model.classes_

    def transform_features(self, X: pd.DataFrame | dict | list | np.ndarray) -> np.ndarray:
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)

        if self.target_col and self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])

        # Step 1: Preprocess (impute, encode, scale)
        X_p = self.preprocessor.transform(df)

        # Step 2: Feature Engineering (interactions, squares, ratios)
        X_fe = self.feature_engineer.transform(X_p)

        # Step 3: Feature Selection (top-K features)
        X_fs = self.feature_selector.transform(X_fe)
        return X_fs

    def predict(self, X: pd.DataFrame | dict | list | np.ndarray) -> np.ndarray:
        X_trans = self.transform_features(X)
        preds = self.model.predict(X_trans)
        if self.target_encoder is not None:
            try:
                return self.target_encoder.inverse_transform(preds)
            except Exception:
                pass
        return preds

    def predict_proba(self, X: pd.DataFrame | dict | list | np.ndarray) -> np.ndarray:
        X_trans = self.transform_features(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_trans)
        raise AttributeError(f"Estimator {type(self.model).__name__} does not implement predict_proba.")


class Orchestrator:
    """The central autonomous data science agent orchestrator."""

    def __init__(
        self,
        reasoner=None,
        distributed_threshold_mb: float = 500,
        distributed_threshold_rows: int = 1_000_000,
        max_interaction_features: int = 30,
        remove_anomalies: bool = False,
        registry_dir: str = "model_registry",
        output_root: str = "outputs",
    ):
        self.log = RunLog()
        self.reasoner = reasoner or get_reasoner()
        self.output_root = output_root
        self.max_interaction_features = max_interaction_features

        # Agent initializations
        self.profiler = DataProfilerAgent(
            self.reasoner,
            self.log,
            distributed_threshold_mb,
            distributed_threshold_rows,
        )
        self.eda = AutoEDAAgent(self.reasoner, self.log, output_root=output_root)
        self.preprocessor = PreprocessingAgent(self.reasoner, self.log)
        self.anomaly_detector = AnomalyDetectionAgent(
            self.reasoner, self.log, remove_anomalies=remove_anomalies
        )
        self.imbalance_handler = ImbalanceHandlerAgent(self.reasoner, self.log)
        self.feature_engineer = FeatureEngineeringAgent(
            self.reasoner, self.log, max_interaction_features
        )
        self.feature_selector = FeatureSelectionAgent(self.reasoner, self.log)
        self.model_selector = ModelSelectionAgent(self.reasoner, self.log)
        self.tuner = HyperparameterTuningAgent(self.reasoner, self.log)
        self.evaluator = EvaluationAgent(self.reasoner, self.log, output_root=output_root)
        self.explainer = ExplainabilityAgent(self.reasoner, self.log, output_root=output_root)
        self.whatif = WhatIfAgent(self.reasoner, self.log)
        self.registry = ModelRegistry(registry_dir=registry_dir)
        self.drift_detector = DriftDetectionAgent(self.reasoner, self.log)
        self.csv_exporter = CSVExportManager(output_root=output_root, log=self.log)
        self.reporter = ReportingAgent(self.log)
        self.assistant = DataScienceAssistant(self.log)

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        dataset_name: str = "dataset",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full autonomous pipeline on the input dataset with strict leakage prevention.
        """
        run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"

        # 1. Profile Data
        profile = self.profiler.run(df, target_col)

        # 2. Auto EDA (visualizations & statistics on original data)
        eda_result = self.eda.run(df, profile, run_id=run_id)

        # Ensure no missing values in target column
        if df[target_col].isna().any():
            n_missing_tgt = int(df[target_col].isna().sum())
            df = df.dropna(subset=[target_col]).reset_index(drop=True)
            self.log.add(
                "Preprocessing[local]",
                f"dropped {n_missing_tgt} rows with missing target",
                f"Supervised learning requires ground truth target values; pruned {n_missing_tgt} rows missing '{target_col}'."
            )

        # Separate raw features and target
        X_raw = df.drop(columns=[target_col])
        y_raw = df[target_col]

        # Target encoding for categorical classification labels
        target_encoder = None
        target_classes = []
        if profile["task_type"] == "classification":
            if not pd.api.types.is_numeric_dtype(y_raw) or y_raw.dtype == object or y_raw.dtype == bool:
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y_raw.astype(str))
                target_classes = list(target_encoder.classes_)
                self.log.add(
                    "Preprocessing[local]",
                    f"encoded target {target_col}",
                    f"Mapped string categories {target_classes} to integer class labels."
                )
            else:
                y = y_raw.values
        else:
            y = y_raw.values

        # 3. SPLIT TRAIN/TEST FIRST (SEEDED, STRATIFIED)
        if profile["task_type"] == "classification":
            y_series = pd.Series(y)
            can_stratify = (y_series.value_counts().min() >= 2)
            strat = y if can_stratify else None
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_raw, y, test_size=0.2, random_state=42, stratify=strat
            )
        else:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_raw, y, test_size=0.2, random_state=42
            )

        self.log.add(
            "Orchestrator",
            "train_test_split (80/20)",
            f"Seeded split executed FIRST into {len(X_train_raw)} train and {len(X_test_raw)} test instances to strictly prevent data leakage."
        )

        # 4. Remove duplicates and detect anomalies strictly on training partition
        train_dups = X_train_raw.duplicated()
        if train_dups.sum() > 0:
            keep_mask = ~train_dups
            X_train_raw = X_train_raw[keep_mask].reset_index(drop=True)
            y_train = y_train[keep_mask.values]
            self.log.add(
                "Preprocessing[local]",
                f"train duplicates removed={train_dups.sum()}",
                f"Removed {train_dups.sum()} duplicate rows from the training split."
            )

        train_for_anomaly = X_train_raw.copy()
        train_for_anomaly[target_col] = y_train
        anomaly_result = self.anomaly_detector.run(train_for_anomaly, profile)
        if self.anomaly_detector.remove_anomalies and "cleaned_df" in anomaly_result:
            cleaned_train = anomaly_result["cleaned_df"]
            X_train_raw = cleaned_train.drop(columns=[target_col]).reset_index(drop=True)
            y_train = cleaned_train[target_col].values

        # 5. Fit Preprocessing strictly on train, transform both train and test
        self.preprocessor.fit(X_train_raw, y=y_train)
        X_train_clean = self.preprocessor.transform(X_train_raw)
        X_test_clean = self.preprocessor.transform(X_test_raw)
        clean_feature_names = self.preprocessor.get_feature_names_out()
        prep_explanations = list(self.preprocessor.preprocessing_explanations)

        # Also create clean_df for visualization & backward compatibility
        clean_df = pd.DataFrame(
            self.preprocessor.transform(X_raw),
            columns=clean_feature_names,
            index=df.index
        )
        clean_df[target_col] = y

        # 6. Fit Feature Engineering strictly on train, transform both
        self.feature_engineer.fit(X_train_clean, y=y_train, feature_names=clean_feature_names)
        X_train_fe = self.feature_engineer.transform(X_train_clean)
        X_test_fe = self.feature_engineer.transform(X_test_clean)
        fe_feature_names = self.feature_engineer.get_feature_names_out()

        # 7. Fit Feature Selection strictly on train, transform both
        self.feature_selector.fit(X_train_fe, y_train, feature_names=fe_feature_names, task_type=profile["task_type"])
        X_train_fs = self.feature_selector.transform(X_train_fe)
        X_test_fs = self.feature_selector.transform(X_test_fe)
        feature_names = self.feature_selector.get_feature_names_out()
        feature_selection_result = getattr(self.feature_selector, "last_result", {})

        # Full engineered feature dataframe for export and backward compatibility
        fs_df = pd.DataFrame(
            self.feature_selector.transform(self.feature_engineer.transform(self.preprocessor.transform(X_raw))),
            columns=feature_names,
            index=df.index
        )
        fs_df[target_col] = y

        # 8. Imbalance Handling on Training Split ONLY
        imbalance_result = None
        minority_ratio = None
        X_tr = X_train_fs
        y_tr = y_train
        X_te = X_test_fs
        y_te = y_test

        if profile["task_type"] == "classification":
            counts = pd.Series(y_train).value_counts(normalize=True)
            minority_ratio = float(counts.min()) if len(counts) > 0 else 0.5
            imbalance_result = self.imbalance_handler.run(X_train_fs, y_train, profile["task_type"])
            X_tr = imbalance_result.get("X_train_resampled", X_train_fs)
            y_tr = imbalance_result.get("y_train_resampled", y_train)

        # 9. Model Selection
        candidate_models = self.model_selector.run(profile)

        # 10. Metric-Aware Hyperparameter Tuning (with Dummy baseline)
        tuning_results = self.tuner.run(
            X_tr,
            y_tr,
            profile["task_type"],
            candidate_models,
            profile["n_rows"],
            minority_ratio=minority_ratio,
        )

        # 11. Model Evaluation & Comparison
        eval_results = self.evaluator.run(
            X=X_train_fs,
            y=y_train,
            task_type=profile["task_type"],
            tuning_results=tuning_results,
            run_id=run_id,
            X_train_pre=X_tr,
            X_test_pre=X_te,
            y_train_pre=y_tr,
            y_test_pre=y_te,
        )

        # 12. Explainability (SHAP & Permutation on test split)
        explain_results = self.explainer.run(
            eval_results["final_model"],
            eval_results["X_test"],
            eval_results["y_test"],
            feature_names,
            profile["task_type"],
            run_id=run_id,
        )

        # 13. Assemble Full Inference Pipeline (Packaging preprocessing + model)
        full_pipeline = FullInferencePipeline(
            preprocessor=self.preprocessor.preprocessor,
            feature_engineer=self.feature_engineer.transformer,
            feature_selector=self.feature_selector.transformer,
            model=eval_results["final_model"],
            target_encoder=target_encoder,
            target_col=target_col,
            feature_names=feature_names,
        )

        # 14. What-If Agent Setup (using full pipeline directly)
        self.whatif.setup(
            pipeline=full_pipeline,
            feature_names=feature_names,
            task_type=profile["task_type"],
            target_col=target_col,
        )

        # 15. Model Registry Registration (persisting FullInferencePipeline)
        final_name = eval_results["final_model_name"]
        best_params = tuning_results.get(final_name, {}).get("best_params", {})
        best_metrics = eval_results.get("metrics_detail", {}).get(final_name, {})
        reg_version = self.registry.register(
            model=full_pipeline,
            model_type=final_name,
            run_id=run_id,
            dataset_name=dataset_name,
            hyperparameters=best_params,
            metrics=best_metrics,
            feature_names=feature_names,
            task_type=profile["task_type"],
        )
        registry_info = {
            "version": reg_version,
            "model_type": final_name,
            "run_id": run_id,
            "dataset_name": dataset_name,
            "metrics": best_metrics,
            "model_path": os.path.join(self.registry.registry_dir, f"{run_id}_{final_name}_{reg_version}", "model.joblib"),
            "metadata_path": os.path.join(self.registry.registry_dir, f"{run_id}_{final_name}_{reg_version}", "metadata.json"),
        }

        # 16. Context-aware AI Assistant Setup
        assistant_ctx = build_assistant_context(
            profile=profile,
            eda_result=eda_result,
            preprocess_explanations=prep_explanations,
            anomaly_result=anomaly_result,
            imbalance_result=imbalance_result,
            tuning_results=tuning_results,
            eval_results=eval_results,
            explain_results=explain_results,
            feature_selection_result=feature_selection_result,
        )
        self.assistant.set_context(assistant_ctx)

        # 17. Export Cleaned & Engineered Datasets
        run_output_dir = os.path.join(self.output_root, run_id)
        os.makedirs(run_output_dir, exist_ok=True)
        clean_csv_path = os.path.join(run_output_dir, "cleaned_data.csv")
        clean_df.to_csv(clean_csv_path, index=False)

        engineered_csv_path = os.path.join(run_output_dir, "engineered_features.csv")
        fs_df.to_csv(engineered_csv_path, index=False)

        self.log.record(
            "ArtifactExport",
            "Exported Cleaned and Engineered Datasets",
            f"Saved cleaned dataset ({len(clean_df)} rows, {clean_df.shape[1]} columns) to {clean_csv_path} and feature matrix to {engineered_csv_path}.",
        )

        # 18. Structured CSV Results Export
        csv_paths = self.csv_exporter.export_all(
            run_id=run_id,
            profile=profile,
            eval_results=eval_results,
            tuning_results=tuning_results,
            explain_results=explain_results,
            preprocessing_explanations=prep_explanations,
            anomaly_result=anomaly_result,
            run_log=self.log,
            eda_result=eda_result,
        )
        csv_zip_path = self.csv_exporter.create_zip(run_id=run_id, csv_paths=csv_paths)

        # 19. Comprehensive Report Generation
        report_md = self.reporter.run(
            profile=profile,
            tuning_results=tuning_results,
            eval_results=eval_results,
            explain_results=explain_results,
            reasoner_name=self.reasoner.name,
            eda_result=eda_result,
            anomaly_result=anomaly_result,
            imbalance_result=imbalance_result,
            feature_selection_result=feature_selection_result,
            registry_info=registry_info,
            preprocessing_explanations=prep_explanations,
            csv_paths=csv_paths,
            csv_zip_path=csv_zip_path,
        )

        return {
            "profile": profile,
            "eda_result": eda_result,
            "preprocessing_explanations": prep_explanations,
            "cleaned_df": clean_df,
            "cleaned_csv_path": clean_csv_path,
            "engineered_df": fs_df,
            "engineered_csv_path": engineered_csv_path,
            "anomaly_result": anomaly_result,
            "imbalance_result": imbalance_result,
            "feature_selection_result": feature_selection_result,
            "tuning_results": tuning_results,
            "eval_results": eval_results,
            "explain_results": explain_results,
            "csv_paths": csv_paths,
            "csv_zip_path": csv_zip_path,
            "csv_dir": self.csv_exporter.get_csv_dir(run_id),
            "whatif": self.whatif,
            "assistant": self.assistant,
            "assistant_context": assistant_ctx,
            "registry_info": registry_info,
            "report_markdown": report_md,
            "run_log": self.log,
            "run_id": run_id,
            "feature_names": feature_names,
            "final_model": eval_results["final_model"],
            "full_pipeline": full_pipeline,
            "reference_df": df.copy(),
        }
