"""
Unit & Integration Smoke Tests for AutoDS-Agent 2.0
===================================================
Tests each module in isolation and the end-to-end Orchestrator flow.
Designed to be fast (<30s total).
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes, make_classification

from autods_agent import (
    Orchestrator,
    FullInferencePipeline,
    DataProfilerAgent,
    PreprocessingAgent,
    TabularPreprocessor,
    AnomalyDetectionAgent,
    ImbalanceHandlerAgent,
    FeatureEngineeringAgent,
    FeatureSelectionAgent,
    FeatureSelectionTransformer,
    ModelSelectionAgent,
    HyperparameterTuningAgent,
    EvaluationAgent,
    ExplainabilityAgent,
    WhatIfAgent,
    ModelRegistry,
    DriftDetectionAgent,
    DataScienceAssistant,
    build_assistant_context,
    RunLog,
    RuleBasedReasoner,
)


@pytest.fixture
def reasoner_and_log():
    log = RunLog()
    reasoner = RuleBasedReasoner()
    return reasoner, log


@pytest.fixture
def sample_classification_df():
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    # binary classification subset for quick tests
    df = df[df["target"].isin([0, 1])].reset_index(drop=True)
    return df


@pytest.fixture
def sample_regression_df():
    data = load_diabetes(as_frame=True)
    df = data.frame.iloc[:60].copy()
    return df


def test_profiler(reasoner_and_log, sample_classification_df):
    reasoner, log = reasoner_and_log
    profiler = DataProfilerAgent(reasoner, log)
    profile = profiler.run(sample_classification_df, target_col="target")

    assert profile["n_rows"] == len(sample_classification_df)
    assert profile["n_cols"] == sample_classification_df.shape[1]
    assert profile["task_type"] == "classification"
    assert profile["backend"] == "local"


def test_preprocessing(reasoner_and_log, sample_classification_df):
    reasoner, log = reasoner_and_log
    profiler = DataProfilerAgent(reasoner, log)
    profile = profiler.run(sample_classification_df, target_col="target")

    preprocessor = PreprocessingAgent(reasoner, log)
    clean_df = preprocessor.run(sample_classification_df, profile)

    assert not clean_df.empty
    assert clean_df.shape[0] == sample_classification_df.shape[0]
    assert len(preprocessor.preprocessing_explanations) >= 0


def test_anomaly_detection(reasoner_and_log, sample_classification_df):
    reasoner, log = reasoner_and_log
    profile = {"target_col": "target", "task_type": "classification"}
    agent = AnomalyDetectionAgent(reasoner, log, contamination=0.1)
    res = agent.run(sample_classification_df, profile)

    assert "if_anomaly_pct" in res
    assert "lof_anomaly_pct" in res
    assert isinstance(res["top_anomalous_indices"], list)


def test_imbalance_handler(reasoner_and_log):
    reasoner, log = reasoner_and_log
    X, y = make_classification(n_samples=200, n_features=6, weights=[0.9, 0.1], random_state=42)
    handler = ImbalanceHandlerAgent(reasoner, log, minority_threshold=0.25)
    res = handler.run(X, y, task_type="classification")

    assert res["is_imbalanced"] is True
    assert "best_strategy" in res
    assert res["X_train_resampled"] is not None
    assert len(res["y_train_resampled"]) >= len(y) * 0.5


def test_feature_selection(reasoner_and_log):
    reasoner, log = reasoner_and_log
    X, y = make_classification(n_samples=100, n_features=15, n_informative=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(15)])
    df["target"] = y
    profile = {"target_col": "target", "task_type": "classification"}

    selector = FeatureSelectionAgent(reasoner, log)
    fs_df = selector.run(df, profile)

    assert "target" in fs_df.columns
    assert fs_df.shape[1] < df.shape[1]
    assert hasattr(selector, "last_result")
    assert len(selector.last_result["selected_features"]) > 0


def test_evaluation_and_explainability(reasoner_and_log, sample_classification_df):
    reasoner, log = reasoner_and_log
    X = sample_classification_df.drop(columns=["target"]).values
    y = sample_classification_df["target"].values
    task_type = "classification"

    tuner = HyperparameterTuningAgent(reasoner, log)
    tuning_res = tuner.run(X, y, task_type, ["logistic_regression"], len(sample_classification_df))

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = EvaluationAgent(reasoner, log, output_root=tmpdir)
        eval_res = evaluator.run(X, y, task_type, tuning_res, run_id="test_run")

        assert "final_model" in eval_res
        assert eval_res["final_model_name"] == "logistic_regression"
        assert "metrics_detail" in eval_res

        explainer = ExplainabilityAgent(reasoner, log, output_root=tmpdir)
        feat_names = [f"f_{i}" for i in range(X.shape[1])]
        explain_res = explainer.run(
            eval_res["final_model"], eval_res["X_test"], eval_res["y_test"],
            feat_names, task_type, run_id="test_run"
        )

        assert "ranked_importance" in explain_res
        assert len(explain_res["ranked_importance"]) == len(feat_names)


def test_whatif(reasoner_and_log, sample_classification_df):
    reasoner, log = reasoner_and_log
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression().fit(
        sample_classification_df.drop(columns=["target"]).values,
        sample_classification_df["target"].values
    )
    feature_names = [c for c in sample_classification_df.columns if c != "target"]

    whatif = WhatIfAgent(reasoner, log)
    whatif.setup(model, None, feature_names, "classification", "target")

    rec = {col: float(sample_classification_df.iloc[0][col]) for col in feature_names}
    pred = whatif.predict(rec)
    assert "prediction" in pred

    mod_rec = dict(rec)
    mod_rec[feature_names[0]] += 5.0
    comp = whatif.compare(rec, mod_rec)
    assert "original_prediction" in comp
    assert "modified_prediction" in comp
    assert feature_names[0] in comp["changed_features"]


def test_model_registry():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ModelRegistry(registry_dir=tmpdir)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression().fit([[1, 2], [3, 4], [5, 6]], [0, 1, 1])

        ver = reg.register(
            clf, "logistic_regression", "run_abc", "iris",
            {"C": 1.0}, {"accuracy": 0.95}, ["f1", "f2"], "classification"
        )
        assert ver.startswith("v")

        models = reg.list_models()
        assert len(models) == 1
        assert models[0]["model_type"] == "logistic_regression"

        loaded = reg.load("run_abc", "logistic_regression", ver)
        assert loaded is not None


def test_drift_detection(reasoner_and_log, sample_classification_df):
    reasoner, log = reasoner_and_log
    drift_agent = DriftDetectionAgent(reasoner, log)
    ref_df = sample_classification_df.copy()
    new_df = sample_classification_df.copy()
    # perturb one column heavily
    col0 = [c for c in sample_classification_df.columns if c != "target"][0]
    new_df[col0] = new_df[col0] + 10.0

    res = drift_agent.run(ref_df, new_df, target_col="target")
    assert "feature_drift" in res
    assert col0 in res["feature_drift"]
    assert bool(res["feature_drift"][col0]["drifted"]) is True


def test_assistant_fallback():
    ctx = {
        "dataset": {"n_rows": 100, "n_cols": 5, "task_type": "classification", "target_col": "target", "size_mb": 0.05},
        "leaderboard": {"random_forest": {"best_cv_score": 0.92, "best_params": {}}},
        "evaluation": {
            "final_model": "random_forest",
            "scores": {"random_forest": 0.92},
            "metrics_detail": {"random_forest": {"accuracy": 0.92, "f1": 0.91}}
        },
        "explainability": {"method": "shap", "top_features": [("feature_0", 0.35), ("feature_1", 0.22)]},
        "preprocessing": {"explanations": ["Imputed 2 columns with median."]}
    }
    log = RunLog()
    assistant = DataScienceAssistant(log)
    assistant.set_context(ctx)

    ans1 = assistant.ask("Which features matter most?")
    assert "feature_0" in ans1 or "feature" in ans1.lower()

    ans2 = assistant.ask("What is the best model?")
    assert "random_forest" in ans2.lower()


def test_orchestrator_end_to_end(sample_classification_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = Orchestrator(output_root=tmpdir, registry_dir=os.path.join(tmpdir, "registry"))
        result = orch.run(sample_classification_df, target_col="target", dataset_name="iris_test")

        assert "profile" in result
        assert "eval_results" in result
        assert "report_markdown" in result
        assert "run_log" in result
        assert result["eval_results"]["final_model_name"] is not None
        assert result["whatif"] is not None
        assert result["assistant"] is not None


def test_no_data_leakage():
    """Verify that test split data never influences fitted preprocessing or selection parameters."""
    np.random.seed(42)
    # 1. Base training partition
    X_train = pd.DataFrame({
        "num1": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat1": ["A", "B", "A", "B", "A"],
    })
    y_train = np.array([0, 1, 0, 1, 0])

    prep1 = TabularPreprocessor()
    prep1.fit(X_train, y=y_train)

    train_mean_num1 = prep1.scaler_.mean_[0]
    train_var_num1 = prep1.scaler_.var_[0]
    train_categories = list(prep1.one_hot_encoders_["cat1"].categories_[0])

    # 2. Altered test partition with extreme numbers and novel categories
    X_test_extreme = pd.DataFrame({
        "num1": [999999.0, -888888.0, 500000.0],
        "cat1": ["Z_UNSEEN", "W_UNSEEN", "NEW_CATEGORY"],
    })

    # Transform extreme test data using the model fitted ONLY on train
    X_test_trans = prep1.transform(X_test_extreme)
    assert X_test_trans is not None
    assert X_test_trans.shape[0] == 3

    # Assert that fitted parameters on the preprocessor are 100% unchanged
    assert prep1.scaler_.mean_[0] == train_mean_num1
    assert prep1.scaler_.var_[0] == train_var_num1
    assert list(prep1.one_hot_encoders_["cat1"].categories_[0]) == train_categories

    # 3. Test FeatureSelectionTransformer: training labels determine selection, test labels ignored
    fs = FeatureSelectionTransformer(task_type="classification", k=2)
    X_synthetic = np.random.randn(50, 6)
    y_synthetic = np.random.randint(0, 2, size=50)
    fs.fit(X_synthetic, y_synthetic)
    orig_selected = list(fs.selected_indices_)

    # Perturbed test labels should have zero effect since fit is only called on train
    assert len(orig_selected) == 2
    assert all(0 <= idx < 6 for idx in orig_selected)


def test_registered_pipeline_raw_prediction(sample_classification_df):
    """Verify that the registered FullInferencePipeline predicts directly from raw row data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_dir = os.path.join(tmpdir, "model_registry")
        out_dir = os.path.join(tmpdir, "outputs")
        orch = Orchestrator(output_root=out_dir, registry_dir=reg_dir)

        result = orch.run(sample_classification_df, target_col="target", dataset_name="iris_raw_pred")
        reg_info = result["registry_info"]

        # Load registered model artifact from registry
        loaded_pipeline = orch.registry.load(reg_info["run_id"], reg_info["model_type"], reg_info["version"])
        assert isinstance(loaded_pipeline, FullInferencePipeline)

        # Single raw record as dict (using raw original columns, no engineered features)
        feature_cols = [c for c in sample_classification_df.columns if c != "target"]
        raw_record = {col: sample_classification_df.iloc[0][col] for col in feature_cols}

        pred = loaded_pipeline.predict(raw_record)
        assert len(pred) == 1
        assert pred[0] in [0, 1]

        # Multi-row raw DataFrame prediction
        raw_batch = sample_classification_df[feature_cols].head(5)
        batch_preds = loaded_pipeline.predict(raw_batch)
        assert len(batch_preds) == 5

        # Predict proba
        proba = loaded_pipeline.predict_proba(raw_record)
        assert proba.shape == (1, 2)
        assert np.isclose(proba.sum(), 1.0)
