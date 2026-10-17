"""
OnlineLearningAgent  (Phase 2 – Incremental / Streaming ML)
=============================================================
Trains models incrementally on data arriving in mini-batches using
scikit-learn's ``partial_fit`` API.

Supported models (all support ``partial_fit``):
  - SGDClassifier          (linear, Hinge / Log / modified_huber)
  - SGDRegressor           (linear, squared / Huber)
  - PassiveAggressiveClassifier
  - PassiveAggressiveRegressor
  - MultinomialNB / BernoulliNB  (classification only)
  - Perceptron               (classification only)

Design notes
------------
* The agent follows the existing pattern: takes ``reasoner`` and ``log``,
  records every decision via ``log.add(…)``.
* Data arrives as a ``pd.DataFrame``.  The agent splits it into mini-batches
  of ``batch_size`` rows, trains incrementally, and tracks per-batch metrics.
* For classification the agent needs ``classes`` on the first call to
  ``partial_fit``.  These are inferred from the first batch's target column.
* River / sklearn-stream libraries are NOT required; only core sklearn is used.

Outputs
-------
``run()`` returns a dict:
  {
    "model"          : fitted sklearn estimator,
    "model_name"     : str,
    "task_type"      : "classification" | "regression",
    "n_batches"      : int,
    "n_samples_seen" : int,
    "batch_metrics"  : List[dict],   # per-batch score history
    "final_metric"   : float,        # score on the full dataset
    "final_metric_name": str,
  }
"""

from __future__ import annotations

import warnings
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Available incremental model catalogue
# ---------------------------------------------------------------------------

_CLF_MODELS = {
    "SGDClassifier":                  lambda: _make_sgd_clf(),
    "PassiveAggressiveClassifier":    lambda: _make_pa_clf(),
    "Perceptron":                     lambda: _make_perceptron(),
    "BernoulliNB":                    lambda: _make_bnb(),
}

_REG_MODELS = {
    "SGDRegressor":                   lambda: _make_sgd_reg(),
    "PassiveAggressiveRegressor":     lambda: _make_pa_reg(),
}


def _make_sgd_clf():
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=42)

def _make_pa_clf():
    from sklearn.linear_model import PassiveAggressiveClassifier
    return PassiveAggressiveClassifier(max_iter=1, tol=None, random_state=42)

def _make_perceptron():
    from sklearn.linear_model import Perceptron
    return Perceptron(max_iter=1, tol=None, random_state=42)

def _make_bnb():
    from sklearn.naive_bayes import BernoulliNB
    return BernoulliNB()

def _make_sgd_reg():
    from sklearn.linear_model import SGDRegressor
    return SGDRegressor(loss="squared_error", max_iter=1, tol=None, random_state=42)

def _make_pa_reg():
    from sklearn.linear_model import PassiveAggressiveRegressor
    return PassiveAggressiveRegressor(max_iter=1, tol=None, random_state=42)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _score(model, X: np.ndarray, y: np.ndarray, task_type: str) -> float:
    try:
        if task_type == "classification":
            from sklearn.metrics import f1_score
            preds = model.predict(X)
            return float(f1_score(y, preds, average="macro", zero_division=0))
        else:
            from sklearn.metrics import r2_score
            return float(r2_score(y, model.predict(X)))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# OnlineLearningAgent
# ---------------------------------------------------------------------------

class OnlineLearningAgent:
    """
    Trains a model incrementally on mini-batches.

    Parameters
    ----------
    reasoner      : shared reasoner
    log           : shared RunLog
    model_name    : which model to use (or ``"auto"`` to pick based on task)
    batch_size    : rows per mini-batch (default 1 000)
    n_epochs      : how many passes over the data (default 1)
    scale_features: if True, apply ``StandardScaler`` online via ``partial_fit``
    """

    def __init__(
        self,
        reasoner,
        log,
        model_name: str = "auto",
        batch_size: int = 1_000,
        n_epochs: int = 1,
        scale_features: bool = True,
    ):
        self.reasoner = reasoner
        self.log = log
        self.model_name = model_name
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.scale_features = scale_features

    # ------------------------------------------------------------------
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = "classification",
        feature_names: Optional[List[str]] = None,
    ) -> dict:
        """
        Incrementally train on ``X_train / y_train`` and evaluate on ``X_test / y_test``.

        Returns
        -------
        dict  (see module docstring)
        """
        # ---- pick model --------------------------------------------------
        chosen_name = self._pick_model(task_type)
        catalogue = _CLF_MODELS if task_type == "classification" else _REG_MODELS
        if chosen_name not in catalogue:
            # fall back to default
            chosen_name = next(iter(catalogue))
        model = catalogue[chosen_name]()

        self.log.add(
            "OnlineLearning",
            f"model={chosen_name}, task={task_type}, batch_size={self.batch_size}, "
            f"n_epochs={self.n_epochs}",
            f"Selected {chosen_name} for incremental training "
            f"({len(X_train):,} training samples, batch_size={self.batch_size}).",
        )

        # ---- optional online scaler --------------------------------------
        scaler = None
        if self.scale_features:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

        classes = np.unique(y_train) if task_type == "classification" else None

        # ---- incremental training ----------------------------------------
        batch_metrics: List[dict] = []
        n_samples_seen = 0
        n_batches = 0
        n = len(X_train)

        for epoch in range(self.n_epochs):
            # shuffle each epoch
            idx = np.random.RandomState(epoch).permutation(n)
            X_ep = X_train[idx]
            y_ep = y_train[idx]

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                Xb = X_ep[start:end].astype(float)
                yb = y_ep[start:end]

                # handle NaN
                col_means = np.nanmean(Xb, axis=0)
                nan_mask = np.isnan(Xb)
                Xb[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

                if scaler is not None:
                    Xb = scaler.partial_fit(Xb).transform(Xb)

                try:
                    if task_type == "classification":
                        model.partial_fit(Xb, yb, classes=classes)
                    else:
                        model.partial_fit(Xb, yb)
                except Exception as e:
                    warnings.warn(f"OnlineLearning batch {n_batches} failed: {e}")
                    continue

                n_samples_seen += len(Xb)
                n_batches += 1

                # lightweight per-batch metric (on current batch)
                batch_score = _score(model, Xb, yb, task_type)
                batch_metrics.append({
                    "epoch": epoch,
                    "batch": n_batches,
                    "n_samples": n_samples_seen,
                    "batch_score": round(batch_score, 4),
                })

        # ---- final evaluation on test set --------------------------------
        X_test_f = np.array(X_test, dtype=float)
        Xte_means = np.nanmean(X_test_f, axis=0)
        nan_mask_te = np.isnan(X_test_f)
        X_test_f[nan_mask_te] = np.take(Xte_means, np.where(nan_mask_te)[1])
        if scaler is not None:
            X_test_f = scaler.transform(X_test_f)

        metric_name = "macro_f1" if task_type == "classification" else "r2"
        final_metric = _score(model, X_test_f, y_test, task_type)

        self.log.add(
            "OnlineLearning",
            f"training complete: {n_batches} batches, {n_samples_seen:,} samples, "
            f"test_{metric_name}={final_metric:.4f}",
            f"Incremental training finished in {n_batches} mini-batches. "
            f"Final test {metric_name}: {final_metric:.4f}.",
        )

        return {
            "model":             model,
            "model_name":        chosen_name,
            "task_type":         task_type,
            "n_batches":         n_batches,
            "n_epochs":          self.n_epochs,
            "n_samples_seen":    n_samples_seen,
            "batch_metrics":     batch_metrics,
            "final_metric":      round(final_metric, 4),
            "final_metric_name": metric_name,
            "scaler":            scaler,
        }

    # ------------------------------------------------------------------
    def _pick_model(self, task_type: str) -> str:
        if self.model_name != "auto":
            return self.model_name
        decision, reason = self.reasoner.decide(
            "online_model",
            {"task_type": task_type},
        )
        self.log.add("OnlineLearning", f"auto-selected model={decision}", reason)
        return decision
