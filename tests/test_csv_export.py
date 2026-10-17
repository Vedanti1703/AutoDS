"""
Tests for AutoDS-Agent 2.0 CSV Export Module & Download Endpoints
================================================================
Comprehensive test suite covering:
  1. CSV output directory creation
  2. Dataset profile CSV generation
  3. Model comparison CSV generation (classification & regression)
  4. Predictions CSV generation (probabilities & residuals)
  5. Feature importance CSV generation (SHAP & permutation)
  6. Preprocessing log CSV generation
  7. Decision log CSV generation
  8. Missing optional results handling & graceful skipping
  9. CSV headers validation
  10. Actual run data integrity (non-empty real values)
  11. ZIP packaging of generated CSV files
  12. Web UI CSV download endpoints
"""

import os
import csv
import json
import tempfile
import zipfile
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from autods_agent import (
    Orchestrator,
    CSVExportManager,
    RunLog,
    DecisionRecord,
    RuleBasedReasoner,
)
from webapp.app import app


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def classification_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # Inject a couple of NaNs for realistic preprocessing test
    df.loc[0, "mean radius"] = np.nan
    df.loc[1, "mean texture"] = np.nan
    return df.rename(columns={"target": "diagnosis"})


@pytest.fixture
def regression_data():
    data = load_diabetes(as_frame=True)
    df = data.frame.iloc[:80].copy()
    df.loc[0, "age"] = np.nan
    return df.rename(columns={"target": "progression"})


def test_csv_dir_creation(temp_output_dir):
    """1. Test that the CSV output directory is created."""
    exporter = CSVExportManager(output_root=temp_output_dir)
    csv_dir = exporter.get_csv_dir("test_run_001")
    assert not os.path.exists(csv_dir)

    exporter.export_all(run_id="test_run_001", profile={"n_rows": 10, "n_cols": 2, "target_col": "y", "task_type": "classification"})
    assert os.path.exists(csv_dir)
    assert os.path.isdir(csv_dir)


def test_dataset_profile_csv(temp_output_dir, classification_data):
    """2. Test dataset profile CSV is generated with headers and correct values."""
    orch = Orchestrator(output_root=temp_output_dir)
    result = orch.run(classification_data, target_col="diagnosis", run_id="test_prof_run")

    csv_path = os.path.join(temp_output_dir, "test_prof_run", "csv", "dataset_profile.csv")
    assert os.path.exists(csv_path)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) >= 10
        assert reader[0] == ["Metric", "Value"]

        metrics_dict = {row[0]: row[1] for row in reader[1:]}
        assert metrics_dict["Rows"] == str(len(classification_data))
        assert metrics_dict["Columns"] == str(classification_data.shape[1])
        assert metrics_dict["Target_Column"] == "diagnosis"
        assert metrics_dict["Problem_Type"] == "classification"
        assert metrics_dict["Execution_Backend"] == "local"


def test_model_comparison_csv_classification(temp_output_dir, classification_data):
    """3. Test model comparison CSV for classification."""
    orch = Orchestrator(output_root=temp_output_dir)
    result = orch.run(classification_data, target_col="diagnosis", run_id="test_cls_run")

    csv_path = os.path.join(temp_output_dir, "test_cls_run", "csv", "model_comparison.csv")
    assert os.path.exists(csv_path)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) >= 2  # Header + at least 1 candidate model
        header = reader[0]
        assert "Model" in header
        assert "Accuracy" in header
        assert "Precision" in header
        assert "Recall" in header
        assert "F1" in header
        assert "ROC_AUC" in header
        assert "Best_Model" in header

        # Verify best model flag
        best_flags = [row[header.index("Best_Model")] for row in reader[1:]]
        assert "Yes" in best_flags


def test_model_comparison_csv_regression(temp_output_dir, regression_data):
    """3b. Test model comparison CSV for regression (MAE, MSE, RMSE, R2)."""
    orch = Orchestrator(output_root=temp_output_dir)
    result = orch.run(regression_data, target_col="progression", run_id="test_reg_run")

    csv_path = os.path.join(temp_output_dir, "test_reg_run", "csv", "model_comparison.csv")
    assert os.path.exists(csv_path)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        header = reader[0]
        assert "Model" in header
        assert "MAE" in header
        assert "MSE" in header
        assert "RMSE" in header
        assert "R2" in header
        assert "Accuracy" not in header  # Should not force classification metrics


def test_predictions_csv(temp_output_dir, classification_data, regression_data):
    """4. Test predictions CSV generation for classification and regression."""
    orch = Orchestrator(output_root=temp_output_dir)

    # Classification predictions
    orch.run(classification_data, target_col="diagnosis", run_id="pred_cls_run")
    cls_csv = os.path.join(temp_output_dir, "pred_cls_run", "csv", "predictions.csv")
    assert os.path.exists(cls_csv)

    with open(cls_csv, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert reader[0] == ["Row_Index", "Actual", "Predicted", "Probability"]
        assert len(reader) > 1
        # Check first prediction row
        row1 = reader[1]
        assert row1[0] == "0"
        assert row1[1] in ["0", "1"]
        assert row1[2] in ["0", "1"]

    # Regression predictions
    orch.run(regression_data, target_col="progression", run_id="pred_reg_run")
    reg_csv = os.path.join(temp_output_dir, "pred_reg_run", "csv", "predictions.csv")
    assert os.path.exists(reg_csv)

    with open(reg_csv, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert reader[0] == ["Row_Index", "Actual", "Predicted", "Residual"]
        assert len(reader) > 1
        row1 = reader[1]
        assert float(row1[3]) == pytest.approx(float(row1[1]) - float(row1[2]), abs=1e-3)


def test_feature_importance_csv(temp_output_dir, classification_data):
    """5. Test feature importance CSV generation with ranked results."""
    orch = Orchestrator(output_root=temp_output_dir)
    orch.run(classification_data, target_col="diagnosis", run_id="feat_imp_run")

    csv_path = os.path.join(temp_output_dir, "feat_imp_run", "csv", "feature_importance.csv")
    assert os.path.exists(csv_path)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert reader[0] == ["Rank", "Feature", "Importance"]
        assert len(reader) > 1
        # Check rank ordering
        assert reader[1][0] == "1"
        assert float(reader[1][2]) >= float(reader[-1][2])


def test_preprocessing_log_csv(temp_output_dir, classification_data):
    """6. Test preprocessing log CSV with structured decisions."""
    orch = Orchestrator(output_root=temp_output_dir)
    orch.run(classification_data, target_col="diagnosis", run_id="prep_log_run")

    csv_path = os.path.join(temp_output_dir, "prep_log_run", "csv", "preprocessing_log.csv")
    assert os.path.exists(csv_path)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert reader[0] == ["Stage", "Technique", "Column", "Reason", "Status"]
        assert len(reader) > 1
        stages = [row[0] for row in reader[1:]]
        assert any("Missing" in s or "Scaling" in s or "Preprocessing" in s for s in stages)


def test_decision_log_csv(temp_output_dir, classification_data):
    """7. Test decision log CSV audit trail with timings and agent names."""
    orch = Orchestrator(output_root=temp_output_dir)
    orch.run(classification_data, target_col="diagnosis", run_id="dec_log_run")

    csv_path = os.path.join(temp_output_dir, "dec_log_run", "csv", "decision_log.csv")
    assert os.path.exists(csv_path)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert reader[0] == ["Timestamp", "Stage", "Agent", "Decision", "Reasoning", "Status", "Duration_Sec"]
        assert len(reader) > 5
        agents = [row[2] for row in reader[1:]]
        assert "DataProfilerAgent" in agents


def test_missing_optional_results_handling(temp_output_dir):
    """8. Test that missing optional results do not crash exporter and are skipped gracefully."""
    exporter = CSVExportManager(output_root=temp_output_dir)
    run_id = "test_sparse_run"

    # Only minimal profile provided, no models, no explainability, no anomalies
    generated = exporter.export_all(
        run_id=run_id,
        profile={"n_rows": 50, "n_cols": 3, "target_col": "target", "task_type": "classification"},
        eval_results=None,
        tuning_results=None,
        explain_results=None,
        anomaly_result=None,
    )

    assert "dataset_profile" in generated
    assert "predictions" not in generated
    assert "feature_importance" not in generated
    assert "anomalies" not in generated

    # Verify ZIP generation works even with sparse files
    zip_path = exporter.create_zip(run_id=run_id)
    assert os.path.exists(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "dataset_profile.csv" in namelist
        assert "predictions.csv" not in namelist


def test_zip_packaging(temp_output_dir, classification_data):
    """11. Test that ZIP archive contains all generated CSV files."""
    orch = Orchestrator(output_root=temp_output_dir)
    result = orch.run(classification_data, target_col="diagnosis", run_id="zip_test_run")

    zip_path = result.get("csv_zip_path")
    assert zip_path is not None
    assert os.path.exists(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        files = zf.namelist()
        assert "dataset_profile.csv" in files
        assert "model_comparison.csv" in files
        assert "predictions.csv" in files
        assert "feature_importance.csv" in files
        assert "preprocessing_log.csv" in files
        assert "decision_log.csv" in files


def test_flask_download_endpoints(temp_output_dir, classification_data):
    """12. Test Flask CSV download API endpoints."""
    run_id = "flask_test_run_123"
    try:
        orch = Orchestrator(output_root="outputs")
        result = orch.run(classification_data, target_col="diagnosis", run_id=run_id)

        client = app.test_client()

        # 1. Download dataset_profile.csv
        resp = client.get(f"/download/{run_id}/csv/dataset_profile.csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.content_type
        assert b"Metric,Value" in resp.data

        # 2. Download model_comparison.csv via /api/download route
        resp_model = client.get(f"/api/download/{run_id}/csv/model_comparison.csv")
        assert resp_model.status_code == 200
        assert b"Model,Accuracy" in resp_model.data

        # 3. Download all CSVs ZIP archive
        resp_zip = client.get(f"/download/{run_id}/csv/all")
        assert resp_zip.status_code == 200
        assert resp_zip.content_type == "application/zip"
        assert len(resp_zip.data) > 0

        # 4. Path traversal attempt should fail safely
        resp_traversal = client.get(f"/download/{run_id}/csv/../../app.py")
        assert resp_traversal.status_code in [400, 404]

        # 5. Non-existent file request
        resp_404 = client.get(f"/download/{run_id}/csv/non_existent.csv")
        assert resp_404.status_code == 400  # not in ALLOWED_CSV_FILENAMES
    finally:
        import shutil
        test_dir = os.path.join("outputs", run_id)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)
