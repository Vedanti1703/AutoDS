"""
CSV Export Module
=================
Generates structured machine-readable CSV files containing the actual results
of the current pipeline execution:
  1. dataset_profile.csv      - Dataset scale, dimensions, and profiling metrics
  2. model_comparison.csv     - Candidate model metrics, tuning results, and winner
  3. predictions.csv          - Held-out test set actuals, predictions, and probabilities/residuals
  4. feature_importance.csv   - Ranked SHAP or permutation feature importances
  5. preprocessing_log.csv    - Structured record of all preprocessing decisions
  6. decision_log.csv         - Full audit trail of agent decisions and rationales
  7. anomalies.csv            - Outlier detection results (if anomaly detector was run)

Also manages ZIP packaging for downloading all generated CSV artifacts in a single archive.
"""

from __future__ import annotations

import os
import csv
import json
import time
import zipfile
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import RunLog, DecisionRecord

logger = logging.getLogger(__name__)


class CSVExportManager:
    """Manages the generation, validation, and packaging of run-specific CSV artifacts."""

    def __init__(self, output_root: str = "outputs", log: Optional[RunLog] = None):
        self.output_root = output_root
        self.log = log

    def get_csv_dir(self, run_id: str) -> str:
        """Return the absolute path to the run's CSV output directory."""
        return os.path.join(self.output_root, run_id, "csv")

    def export_all(
        self,
        run_id: str,
        profile: Optional[Dict[str, Any]] = None,
        eval_results: Optional[Dict[str, Any]] = None,
        tuning_results: Optional[Dict[str, Any]] = None,
        explain_results: Optional[Dict[str, Any]] = None,
        preprocessing_explanations: Optional[List[str]] = None,
        anomaly_result: Optional[Dict[str, Any]] = None,
        run_log: Optional[RunLog] = None,
        eda_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Execute export for all available artifacts in the current pipeline run.
        Returns a mapping of csv_key -> absolute_file_path for successfully generated files.
        """
        csv_dir = self.get_csv_dir(run_id)
        os.makedirs(csv_dir, exist_ok=True)

        log = run_log or self.log
        generated: Dict[str, str] = {}

        # 1. Dataset Profile CSV
        try:
            if profile:
                path = self.export_dataset_profile(run_id, profile, eda_result)
                if path and os.path.exists(path):
                    generated["dataset_profile"] = path
        except Exception as e:
            logger.warning(f"Failed to export dataset_profile.csv: {e}")
            if log:
                log.add("CSVExport", "dataset_profile.csv FAILED", str(e))

        # 2. Model Comparison CSV
        try:
            if eval_results:
                path = self.export_model_comparison(run_id, eval_results, tuning_results, profile)
                if path and os.path.exists(path):
                    generated["model_comparison"] = path
        except Exception as e:
            logger.warning(f"Failed to export model_comparison.csv: {e}")
            if log:
                log.add("CSVExport", "model_comparison.csv FAILED", str(e))

        # 3. Predictions CSV
        try:
            if eval_results:
                path = self.export_predictions(run_id, eval_results, profile)
                if path and os.path.exists(path):
                    generated["predictions"] = path
        except Exception as e:
            logger.warning(f"Failed to export predictions.csv: {e}")
            if log:
                log.add("CSVExport", "predictions.csv FAILED", str(e))

        # 4. Feature Importance CSV
        try:
            if explain_results:
                path = self.export_feature_importance(run_id, explain_results)
                if path and os.path.exists(path):
                    generated["feature_importance"] = path
        except Exception as e:
            logger.warning(f"Failed to export feature_importance.csv: {e}")
            if log:
                log.add("CSVExport", "feature_importance.csv FAILED", str(e))

        # 5. Preprocessing Log CSV
        try:
            if preprocessing_explanations is not None or log is not None:
                path = self.export_preprocessing_log(run_id, preprocessing_explanations, log)
                if path and os.path.exists(path):
                    generated["preprocessing_log"] = path
        except Exception as e:
            logger.warning(f"Failed to export preprocessing_log.csv: {e}")
            if log:
                log.add("CSVExport", "preprocessing_log.csv FAILED", str(e))

        # 6. Decision Log CSV
        try:
            if log:
                path = self.export_decision_log(run_id, log)
                if path and os.path.exists(path):
                    generated["decision_log"] = path
        except Exception as e:
            logger.warning(f"Failed to export decision_log.csv: {e}")
            if log:
                log.add("CSVExport", "decision_log.csv FAILED", str(e))

        # 7. Anomalies CSV
        try:
            if anomaly_result:
                path = self.export_anomalies(run_id, anomaly_result)
                if path and os.path.exists(path):
                    generated["anomalies"] = path
        except Exception as e:
            logger.warning(f"Failed to export anomalies.csv: {e}")
            if log:
                log.add("CSVExport", "anomalies.csv FAILED", str(e))

        if log:
            log.add(
                "CSVExport",
                f"exported {len(generated)} CSV files",
                f"Generated structured CSV artifacts in {csv_dir}: {', '.join(generated.keys())}",
            )

        return generated

    # ------------------------------------------------------------------
    # 1. Dataset Profile CSV
    # ------------------------------------------------------------------
    def export_dataset_profile(
        self,
        run_id: str,
        profile: Dict[str, Any],
        eda_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Exports dataset_profile.csv containing dataset scale, dimensions, and profiling metrics.
        """
        csv_path = os.path.join(self.get_csv_dir(run_id), "dataset_profile.csv")

        # Total missing values across all columns
        missing_count_dict = profile.get("missing_count", {})
        total_missing = sum(missing_count_dict.values()) if missing_count_dict else 0

        # High cardinality & constant columns
        high_card = profile.get("high_cardinality_cols", [])
        constant_cols = profile.get("constant_cols", [])

        # Format items cleanly
        high_card_str = "; ".join(high_card) if high_card else "None"
        constant_str = "; ".join(constant_cols) if constant_cols else "None"

        ext = eda_result.get("extended_profile", {}) if eda_result else {}
        n_num = profile.get("n_numeric", ext.get("n_numeric", 0))
        n_cat = profile.get("n_categorical", ext.get("n_categorical", 0))
        n_dup = profile.get("n_duplicates", ext.get("n_duplicates", 0))

        rows = [
            ("Metric", "Value"),
            ("Rows", str(profile.get("n_rows", ""))),
            ("Columns", str(profile.get("n_cols", ""))),
            ("Dataset_Size_MB", f"{profile.get('size_mb', 0.0):.4f}" if profile.get("size_mb") is not None else ""),
            ("Target_Column", str(profile.get("target_col", ""))),
            ("Problem_Type", str(profile.get("task_type", ""))),
            ("Target_Subtype", str(profile.get("target_subtype", ""))),
            ("Execution_Backend", str(profile.get("backend", ""))),
            ("Missing_Values", str(total_missing)),
            ("Duplicate_Rows", str(n_dup)),
            ("Numerical_Features", str(n_num)),
            ("Categorical_Features", str(n_cat)),
            ("High_Cardinality_Columns", high_card_str),
            ("Constant_Columns", constant_str),
        ]

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return csv_path

    # ------------------------------------------------------------------
    # 2. Model Comparison CSV
    # ------------------------------------------------------------------
    def export_model_comparison(
        self,
        run_id: str,
        eval_results: Dict[str, Any],
        tuning_results: Optional[Dict[str, Any]] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Exports model_comparison.csv including candidate models, held-out metrics,
        hyperparameter tuning details, and best model selection.
        """
        csv_path = os.path.join(self.get_csv_dir(run_id), "model_comparison.csv")
        task_type = (profile or {}).get("task_type", "classification")
        metrics_detail = eval_results.get("metrics_detail", {})
        final_model_name = eval_results.get("final_model_name", "")
        tuning = tuning_results or {}

        # Determine all candidate models
        all_models = list(metrics_detail.keys())
        for m in tuning.keys():
            if m not in all_models:
                all_models.append(m)

        if task_type == "classification":
            headers = [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC_AUC",
                "CV_Score",
                "Best_Model",
                "Best_Parameters",
                "Training_Time",
            ]
            rows: List[List[Any]] = [headers]
            for m_name in all_models:
                m_info = metrics_detail.get(m_name, {})
                t_info = tuning.get(m_name, {})
                best_params = json.dumps(t_info.get("best_params", {})) if t_info.get("best_params") else ""
                cv_score = f"{t_info.get('best_score'):.4f}" if t_info.get("best_score") is not None else ""
                is_best = "Yes" if m_name == final_model_name else "No"

                acc = f"{m_info.get('accuracy'):.4f}" if m_info.get("accuracy") is not None else ""
                prec = f"{m_info.get('precision_macro'):.4f}" if m_info.get("precision_macro") is not None else ""
                rec = f"{m_info.get('recall_macro'):.4f}" if m_info.get("recall_macro") is not None else ""
                f1 = f"{m_info.get('f1_macro', m_info.get('f1_weighted')):.4f}" if (m_info.get("f1_macro") is not None or m_info.get("f1_weighted") is not None) else ""
                auc_val = m_info.get("roc_auc")
                auc = f"{auc_val:.4f}" if (auc_val is not None and not (isinstance(auc_val, float) and np.isnan(auc_val))) else ""

                training_time = ""
                if "training_time" in m_info:
                    training_time = f"{m_info['training_time']:.2f}"
                elif "duration_sec" in t_info:
                    training_time = f"{t_info['duration_sec']:.2f}"

                rows.append([
                    m_name,
                    acc,
                    prec,
                    rec,
                    f1,
                    auc,
                    cv_score,
                    is_best,
                    best_params,
                    training_time,
                ])
        else:
            headers = [
                "Model",
                "MAE",
                "MSE",
                "RMSE",
                "R2",
                "CV_Score",
                "Best_Model",
                "Best_Parameters",
                "Training_Time",
            ]
            rows = [headers]
            for m_name in all_models:
                m_info = metrics_detail.get(m_name, {})
                t_info = tuning.get(m_name, {})
                best_params = json.dumps(t_info.get("best_params", {})) if t_info.get("best_params") else ""
                cv_score = f"{t_info.get('best_score'):.4f}" if t_info.get("best_score") is not None else ""
                is_best = "Yes" if m_name == final_model_name else "No"

                mae = f"{m_info.get('mae'):.4f}" if m_info.get("mae") is not None else ""
                mse = f"{m_info.get('mse'):.4f}" if m_info.get("mse") is not None else ""
                rmse = f"{m_info.get('rmse'):.4f}" if m_info.get("rmse") is not None else ""
                r2 = f"{m_info.get('r2'):.4f}" if m_info.get("r2") is not None else ""

                training_time = ""
                if "training_time" in m_info:
                    training_time = f"{m_info['training_time']:.2f}"
                elif "duration_sec" in t_info:
                    training_time = f"{t_info['duration_sec']:.2f}"

                rows.append([
                    m_name,
                    mae,
                    mse,
                    rmse,
                    r2,
                    cv_score,
                    is_best,
                    best_params,
                    training_time,
                ])

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return csv_path

    # ------------------------------------------------------------------
    # 3. Predictions CSV
    # ------------------------------------------------------------------
    def export_predictions(
        self,
        run_id: str,
        eval_results: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Exports predictions.csv containing test set actuals, predictions,
        and probabilities / residuals from the winning model.
        """
        final_model = eval_results.get("final_model")
        X_test = eval_results.get("X_test")
        y_test = eval_results.get("y_test")

        if final_model is None or X_test is None or y_test is None or len(X_test) == 0:
            logger.warning("No test set predictions available to export.")
            return None

        csv_path = os.path.join(self.get_csv_dir(run_id), "predictions.csv")
        task_type = (profile or {}).get("task_type", "classification")

        preds = final_model.predict(X_test)

        if task_type == "classification":
            headers = ["Row_Index", "Actual", "Predicted", "Probability"]
            rows: List[List[Any]] = [headers]

            probas: Optional[np.ndarray] = None
            if hasattr(final_model, "predict_proba"):
                try:
                    probas = final_model.predict_proba(X_test)
                except Exception:
                    probas = None

            for i in range(len(y_test)):
                prob_str = ""
                if probas is not None and len(probas) > i:
                    p_row = probas[i]
                    if len(p_row) == 2:
                        # Binary classification: positive class probability
                        prob_str = f"{float(p_row[1]):.4f}"
                    elif len(p_row) > 2:
                        # Multiclass: winning class probability or max
                        prob_str = f"{float(np.max(p_row)):.4f}"

                rows.append([
                    i,
                    y_test[i],
                    preds[i],
                    prob_str,
                ])
        else:
            headers = ["Row_Index", "Actual", "Predicted", "Residual"]
            rows = [headers]
            for i in range(len(y_test)):
                actual = float(y_test[i])
                pred = float(preds[i])
                residual = actual - pred
                rows.append([
                    i,
                    f"{actual:.4f}",
                    f"{pred:.4f}",
                    f"{residual:.4f}",
                ])

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return csv_path

    # ------------------------------------------------------------------
    # 4. Feature Importance CSV
    # ------------------------------------------------------------------
    def export_feature_importance(
        self,
        run_id: str,
        explain_results: Dict[str, Any],
    ) -> Optional[str]:
        """
        Exports feature_importance.csv using actual SHAP or permutation importance rankings.
        """
        ranked = explain_results.get("ranked_importance", [])
        if not ranked:
            logger.warning("No feature importance rankings available to export.")
            return None

        csv_path = os.path.join(self.get_csv_dir(run_id), "feature_importance.csv")
        headers = ["Rank", "Feature", "Importance"]
        rows: List[List[Any]] = [headers]

        for rank, item in enumerate(ranked, start=1):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                feat, score = item[0], item[1]
                rows.append([rank, str(feat), f"{float(score):.6f}"])

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return csv_path

    # ------------------------------------------------------------------
    # 5. Preprocessing Log CSV
    # ------------------------------------------------------------------
    def export_preprocessing_log(
        self,
        run_id: str,
        preprocessing_explanations: Optional[List[str]] = None,
        run_log: Optional[RunLog] = None,
    ) -> str:
        """
        Exports preprocessing_log.csv documenting every data cleaning, imputation,
        encoding, and scaling decision made during preprocessing.
        """
        csv_path = os.path.join(self.get_csv_dir(run_id), "preprocessing_log.csv")
        headers = ["Stage", "Technique", "Column", "Reason", "Status"]
        rows: List[List[Any]] = [headers]

        # First, check explanations list
        explanations = preprocessing_explanations or []

        # If explanations is empty, try to gather from RunLog
        if not explanations and run_log:
            for rec in run_log.records:
                if "Preprocessing" in rec.stage:
                    explanations.append(f"{rec.decision}: {rec.reasoning}")

        for exp in explanations:
            stage = "Preprocessing"
            technique = "Data Transformation"
            column = ""
            reason = exp
            status = "Applied"

            exp_lower = exp.lower()
            if "duplicate" in exp_lower:
                stage = "Missing Values & Cleaning"
                technique = "Duplicate Removal"
                column = "All Features"
                reason = exp
            elif "imputed" in exp_lower or "imputation" in exp_lower:
                stage = "Missing Values"
                if "median" in exp_lower:
                    technique = "Median Imputation"
                elif "mean" in exp_lower:
                    technique = "Mean Imputation"
                elif "mode" in exp_lower:
                    technique = "Mode Imputation"
                else:
                    technique = "Missing Value Imputation"

                # Extract column name if in single quotes
                if "'" in exp:
                    parts = exp.split("'")
                    if len(parts) >= 2:
                        column = parts[1]
                reason = exp
            elif "dropped column" in exp_lower or "dropped" in exp_lower:
                stage = "Missing Values"
                technique = "Drop Column"
                if "'" in exp:
                    parts = exp.split("'")
                    if len(parts) >= 2:
                        column = parts[1]
                reason = exp
            elif "one-hot" in exp_lower:
                stage = "Encoding"
                technique = "One Hot Encoding"
                if "'" in exp:
                    parts = exp.split("'")
                    if len(parts) >= 2:
                        column = parts[1]
                reason = "Low cardinality categorical feature"
            elif "ordinal" in exp_lower:
                stage = "Encoding"
                technique = "Ordinal Encoding"
                column = "High-cardinality nominals"
                reason = "High cardinality categorical feature (>10 unique)"
            elif "target" in exp_lower and "encoded" in exp_lower:
                stage = "Target Encoding"
                technique = "Label Encoding"
                if "'" in exp:
                    parts = exp.split("'")
                    if len(parts) >= 2:
                        column = parts[1]
                reason = "Categorical labels mapped to discrete integer indices"
            elif "standardscaler" in exp_lower or "scaling" in exp_lower:
                stage = "Scaling"
                technique = "Standard Scaling"
                column = "Numerical Features"
                reason = "Standardize features by removing mean and scaling to unit variance"

            rows.append([stage, technique, column, reason, status])

        if len(rows) == 1:
            # Fallback entry if no preprocessing was required
            rows.append(["Preprocessing", "None Required", "All", "Dataset is complete and clean", "Clean"])

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return csv_path

    # ------------------------------------------------------------------
    # 6. Decision Log CSV
    # ------------------------------------------------------------------
    def export_decision_log(
        self,
        run_id: str,
        run_log: RunLog,
    ) -> str:
        """
        Exports decision_log.csv containing the full audit trail of agent decisions,
        reasoning, and execution timing.
        """
        csv_path = os.path.join(self.get_csv_dir(run_id), "decision_log.csv")
        headers = ["Timestamp", "Stage", "Agent", "Decision", "Reasoning", "Status", "Duration_Sec"]
        rows: List[List[Any]] = [headers]

        agent_mapping = {
            "Profiler": "DataProfilerAgent",
            "AutoEDA": "AutoEDAAgent",
            "Preprocessing": "PreprocessingAgent",
            "AnomalyDetection": "AnomalyDetectionAgent",
            "ImbalanceHandling": "ImbalanceHandlerAgent",
            "FeatureEngineering": "FeatureEngineeringAgent",
            "FeatureSelection": "FeatureSelectionAgent",
            "ModelSelection": "ModelSelectionAgent",
            "Tuning": "HyperparameterTuningAgent",
            "Evaluation": "EvaluationAgent",
            "Explainability": "ExplainabilityAgent",
            "ArtifactExport": "CSVExportManager",
            "Reporting": "ReportingAgent",
            "WhatIf": "WhatIfAgent",
            "Assistant": "DataScienceAssistant",
            "ModelRegistry": "ModelRegistry",
            "DriftDetection": "DriftDetectionAgent",
            "CSVExport": "CSVExportManager",
        }

        for rec in run_log.records:
            t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(getattr(rec, "timestamp", time.time())))
            stage_base = rec.stage.split("[")[0].strip()
            agent_name = agent_mapping.get(stage_base, f"{stage_base}Agent")
            status = "Failed" if "FAILED" in rec.decision.upper() else "Success"
            dur = f"{getattr(rec, 'duration_sec', 0.0):.4f}"

            rows.append([
                t_str,
                rec.stage,
                agent_name,
                rec.decision,
                rec.reasoning,
                status,
                dur,
            ])

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return csv_path

    # ------------------------------------------------------------------
    # 7. Anomalies CSV
    # ------------------------------------------------------------------
    def export_anomalies(
        self,
        run_id: str,
        anomaly_result: Dict[str, Any],
    ) -> Optional[str]:
        """
        Exports anomalies.csv containing identified anomalous data points.
        Skips gracefully if no anomaly detection was performed.
        """
        mask = anomaly_result.get("anomaly_mask")
        top_indices = anomaly_result.get("top_anomalous_indices", [])

        if mask is None and not top_indices:
            logger.info("No anomaly mask available; skipping anomalies.csv.")
            return None

        csv_path = os.path.join(self.get_csv_dir(run_id), "anomalies.csv")
        headers = ["Row_Index", "Anomaly_Method", "Anomaly_Score", "Is_Anomaly"]
        rows: List[List[Any]] = [headers]

        if mask is not None and len(mask) > 0:
            for idx, is_anom in enumerate(mask):
                rows.append([
                    idx,
                    "Isolation Forest + LOF",
                    "",
                    "True" if is_anom else "False",
                ])
        elif top_indices:
            for idx in top_indices:
                rows.append([
                    idx,
                    "Isolation Forest + LOF",
                    "",
                    "True",
                ])

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return csv_path

    # ------------------------------------------------------------------
    # 8. Create ZIP Archive
    # ------------------------------------------------------------------
    def create_zip(
        self,
        run_id: str,
        csv_paths: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Packages all generated CSV files into a zip archive named autods_<run_id>_csv_results.zip.
        Only includes files that actually exist. Does not delete source CSV files.
        """
        csv_dir = self.get_csv_dir(run_id)
        zip_filename = f"autods_{run_id}_csv_results.zip"
        zip_path = os.path.join(self.output_root, run_id, zip_filename)

        # Gather target files
        files_to_zip: List[Tuple[str, str]] = []
        if os.path.exists(csv_dir):
            for fname in sorted(os.listdir(csv_dir)):
                if fname.endswith(".csv"):
                    full_p = os.path.join(csv_dir, fname)
                    if os.path.isfile(full_p):
                        files_to_zip.append((full_p, fname))

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for full_p, arcname in files_to_zip:
                zf.write(full_p, arcname=arcname)

        return zip_path
