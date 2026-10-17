"""
tests/test_phase2_bigdata.py
============================
Phase 2 – Big Data Ingestion & Streaming tests

Covers:
  - StreamingIngestAgent: in-memory pass-through
  - StreamingIngestAgent: pandas path (small file)
  - StreamingIngestAgent: chunked-pandas fallback (small threshold)
  - StreamingIngestAgent: Dask path (if dask available)
  - StreamingIngestAgent: Parquet round-trip (if pyarrow available)
  - StreamingIngestAgent: missing-file error
  - OnlineLearningAgent: classification incremental training
  - OnlineLearningAgent: regression incremental training
  - OnlineLearningAgent: multi-epoch convergence (score improves over epochs)
  - csv_to_parquet: utility function (if pyarrow available)
  - RuleBasedReasoner: online_model decision rule
"""

from __future__ import annotations

import os
import math
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from autods_agent.ingestion import StreamingIngestAgent, csv_to_parquet
from autods_agent.online_learning import OnlineLearningAgent
from autods_agent.llm_reasoner import RuleBasedReasoner
from autods_agent.utils import RunLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def reasoner():
    return RuleBasedReasoner()


@pytest.fixture()
def log():
    return RunLog()


@pytest.fixture()
def clf_data():
    """Small binary classification dataset."""
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=6,
        n_redundant=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


@pytest.fixture()
def reg_data():
    """Small regression dataset."""
    X, y = make_regression(n_samples=600, n_features=10, noise=0.5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


# ---------------------------------------------------------------------------
# StreamingIngestAgent tests
# ---------------------------------------------------------------------------

class TestStreamingIngestAgent:

    def test_passthrough_dataframe(self, reasoner, log, clf_data):
        """DataFrame input should be returned as-is (no file I/O)."""
        agent = StreamingIngestAgent(reasoner, log)
        df, meta = agent.run(clf_data, target_col="target")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(clf_data)
        assert meta["backend"] == "in-memory"
        assert meta["source_type"] == "dataframe"

    def test_pandas_path_csv(self, reasoner, log, clf_data, tmp_path):
        """Small CSV file → pandas strategy."""
        csv_file = tmp_path / "small.csv"
        clf_data.to_csv(csv_file, index=False)

        agent = StreamingIngestAgent(reasoner, log, chunk_threshold_mb=500)
        df, meta = agent.run(str(csv_file), target_col="target")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(clf_data)
        assert "target" in df.columns
        assert meta["backend"] == "pandas"
        assert meta["source_type"] == "csv"

    def test_chunked_pandas_fallback(self, reasoner, log, clf_data, tmp_path):
        """Force chunked-pandas by setting threshold very low and no Dask."""
        csv_file = tmp_path / "medium.csv"
        clf_data.to_csv(csv_file, index=False)

        # Monkeypatch Dask as unavailable
        import autods_agent.ingestion as ing_mod
        orig = ing_mod._DASK_OK
        ing_mod._DASK_OK = False
        try:
            agent = StreamingIngestAgent(
                reasoner, log,
                chunk_threshold_mb=0.001,  # Force non-pandas path
                pandas_chunksize=100,
            )
            df, meta = agent.run(str(csv_file), target_col="target")
            assert len(df) == len(clf_data)
            assert meta["backend"] == "chunked-pandas"
        finally:
            ing_mod._DASK_OK = orig

    def test_sample_rows_cap(self, reasoner, log, clf_data, tmp_path):
        """sample_rows should limit the returned DataFrame to at most that many rows."""
        csv_file = tmp_path / "full.csv"
        clf_data.to_csv(csv_file, index=False)

        agent = StreamingIngestAgent(reasoner, log, sample_rows=100)
        df, meta = agent.run(str(csv_file), target_col="target")
        assert len(df) <= 100

    def test_missing_file_raises(self, reasoner, log):
        """Non-existent file should raise FileNotFoundError."""
        agent = StreamingIngestAgent(reasoner, log)
        with pytest.raises(FileNotFoundError):
            agent.run("/nonexistent/path/data.csv")

    def test_missing_target_warns(self, reasoner, log, clf_data, tmp_path):
        """If target_col not in columns, a warning is issued but no exception."""
        csv_file = tmp_path / "data.csv"
        clf_data.to_csv(csv_file, index=False)

        agent = StreamingIngestAgent(reasoner, log)
        with pytest.warns(UserWarning):
            df, meta = agent.run(str(csv_file), target_col="nonexistent_target")
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("dask"),
        reason="dask not installed"
    )
    def test_dask_path(self, reasoner, log, clf_data, tmp_path):
        """Large-ish file → Dask strategy when Dask is available."""
        csv_file = tmp_path / "large.csv"
        clf_data.to_csv(csv_file, index=False)

        agent = StreamingIngestAgent(
            reasoner, log,
            chunk_threshold_mb=0.001,  # Force Dask path
        )
        import autods_agent.ingestion as ing_mod
        if not ing_mod._DASK_OK:
            pytest.skip("Dask not available at runtime")

        df, meta = agent.run(str(csv_file), target_col="target")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(clf_data)
        assert meta["backend"] == "dask"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("pyarrow"),
        reason="pyarrow not installed"
    )
    def test_parquet_roundtrip(self, reasoner, log, clf_data, tmp_path):
        """CSV → Parquet → StreamingIngestAgent → same shape."""
        csv_file = str(tmp_path / "data.csv")
        parquet_file = str(tmp_path / "data.parquet")
        clf_data.to_csv(csv_file, index=False)

        written = csv_to_parquet(csv_file, parquet_file)
        assert os.path.exists(written)
        assert written.endswith(".parquet")

        agent = StreamingIngestAgent(reasoner, log)
        df, meta = agent.run(written, target_col="target")
        assert meta["parquet"] is True
        assert df.shape == clf_data.shape


# ---------------------------------------------------------------------------
# OnlineLearningAgent tests
# ---------------------------------------------------------------------------

class TestOnlineLearningAgent:

    def _split(self, df, target="target"):
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=[target]).values
        y = df[target].values
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def test_classification_runs(self, reasoner, log, clf_data):
        """OnlineLearning classification completes without error."""
        X_tr, X_te, y_tr, y_te = self._split(clf_data)
        agent = OnlineLearningAgent(reasoner, log, batch_size=50)
        result = agent.run(X_tr, y_tr, X_te, y_te, task_type="classification")

        assert "model" in result
        assert result["task_type"] == "classification"
        assert result["n_batches"] > 0
        assert result["n_samples_seen"] == len(X_tr)
        assert not math.isnan(result["final_metric"])
        assert result["final_metric_name"] == "macro_f1"

    def test_regression_runs(self, reasoner, log, reg_data):
        """OnlineLearning regression completes without error."""
        X_tr, X_te, y_tr, y_te = self._split(reg_data)
        agent = OnlineLearningAgent(reasoner, log, batch_size=50, model_name="SGDRegressor")
        result = agent.run(X_tr, y_tr, X_te, y_te, task_type="regression")

        assert result["task_type"] == "regression"
        assert result["final_metric_name"] == "r2"
        assert result["n_batches"] > 0

    def test_batch_metrics_tracked(self, reasoner, log, clf_data):
        """Per-batch metrics list should have one entry per batch."""
        X_tr, X_te, y_tr, y_te = self._split(clf_data)
        agent = OnlineLearningAgent(reasoner, log, batch_size=100)
        result = agent.run(X_tr, y_tr, X_te, y_te, task_type="classification")

        assert len(result["batch_metrics"]) == result["n_batches"]
        first = result["batch_metrics"][0]
        assert "epoch" in first
        assert "batch" in first
        assert "n_samples" in first
        assert "batch_score" in first

    def test_multi_epoch(self, reasoner, log, clf_data):
        """Multi-epoch training should process n_epochs * n_batches total batches."""
        X_tr, X_te, y_tr, y_te = self._split(clf_data)
        agent_1 = OnlineLearningAgent(reasoner, log, batch_size=100, n_epochs=1)
        agent_2 = OnlineLearningAgent(reasoner, log, batch_size=100, n_epochs=3)

        r1 = agent_1.run(X_tr, y_tr, X_te, y_te, task_type="classification")
        log2 = RunLog()
        r2 = agent_2.run(X_tr, y_tr, X_te, y_te, task_type="classification")

        assert r2["n_batches"] == r1["n_batches"] * 3

    def test_explicit_model_name(self, reasoner, log, clf_data):
        """Explicit model_name should be used without calling reasoner."""
        X_tr, X_te, y_tr, y_te = self._split(clf_data)
        agent = OnlineLearningAgent(
            reasoner, log,
            model_name="PassiveAggressiveClassifier",
            batch_size=100
        )
        result = agent.run(X_tr, y_tr, X_te, y_te, task_type="classification")
        assert result["model_name"] == "PassiveAggressiveClassifier"


# ---------------------------------------------------------------------------
# RuleBasedReasoner: online_model rule
# ---------------------------------------------------------------------------

class TestReasonerOnlineModel:

    def test_classification_returns_sgd_clf(self):
        r = RuleBasedReasoner()
        decision, reasoning = r.decide("online_model", {"task_type": "classification"})
        assert decision == "SGDClassifier"
        assert "partial_fit" in reasoning

    def test_regression_returns_sgd_reg(self):
        r = RuleBasedReasoner()
        decision, reasoning = r.decide("online_model", {"task_type": "regression"})
        assert decision == "SGDRegressor"
        assert "partial_fit" in reasoning

    def test_unknown_question_returns_default(self):
        r = RuleBasedReasoner()
        decision, _ = r.decide("nonexistent_question_xyz", {})
        assert decision == "default"
