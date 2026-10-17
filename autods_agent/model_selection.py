"""
ModelSelectionAgent + HyperparameterTuningAgent (v3 - Metric-Aware & Baseline Benchmark)
========================================================================================
- ModelSelectionAgent selects candidate model families based on task and scale.
- HyperparameterTuningAgent:
    * Implements metric-aware tuning (macro-F1 for imbalanced, ROC-AUC for binary,
      accuracy for multiclass, R2/RMSE for regression)
    * Always benchmarks a Dummy baseline (DummyClassifier / DummyRegressor)
    * Reports cross-validation mean +/- standard deviation (mu +/- sigma)
    * Computes each candidate model's improvement over the Dummy baseline
    * Logs scientific rationales for the tuning metric and budget
"""

from __future__ import annotations

import json
import numpy as np
import optuna
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


class _FixedParamTrial:
    """Replay best_params as a mock Optuna trial."""
    def __init__(self, params: dict):
        self.params = params

    def suggest_float(self, name, *a, **k):
        return self.params[name]

    def suggest_int(self, name, *a, **k):
        return self.params[name]


MODEL_REGISTRY = {
    "dummy": lambda trial, task: (
        DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor(strategy="mean")
    ),
    "logistic_regression": lambda trial: LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 1e2, log=True), max_iter=1000),
    "ridge": lambda trial: Ridge(
        alpha=trial.suggest_float("alpha", 1e-3, 1e2, log=True)),
    "random_forest": lambda trial, task: (
        RandomForestClassifier if task == "classification" else RandomForestRegressor)(
        n_estimators=trial.suggest_int("n_estimators", 15, 50),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        random_state=42, n_jobs=1),
    "gradient_boosting": lambda trial, task: (
        GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor)(
        n_estimators=trial.suggest_int("n_estimators", 30, 120),
        max_depth=trial.suggest_int("max_depth", 2, 5),
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        random_state=42),
    "hist_gradient_boosting": lambda trial, task: (
        HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor)(
        max_iter=trial.suggest_int("max_iter", 30, 150),
        max_depth=trial.suggest_int("max_depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        random_state=42),
}


def _build_model(name: str, trial, task: str):
    fn = MODEL_REGISTRY[name]
    try:
        return fn(trial, task)
    except TypeError:
        return fn(trial)


class ModelSelectionAgent:
    def __init__(self, reasoner, log):
        self.reasoner = reasoner
        self.log = log

    def run(self, profile: dict) -> list[str]:
        decision, reasoning = self.reasoner.decide(
            "candidate_models",
            {"task_type": profile["task_type"], "n_rows": profile["n_rows"],
             "n_features": profile.get("n_features", 10)},
        )
        models = json.loads(decision) if decision.startswith("[") else [decision]
        self.log.add("ModelSelection", f"candidates={models}", reasoning)
        return models


class HyperparameterTuningAgent:
    def __init__(self, reasoner, log):
        self.reasoner = reasoner
        self.log = log

    def run(
        self,
        X,
        y,
        task_type: str,
        candidate_models: list[str],
        n_rows: int,
        minority_ratio: float | None = None,
    ) -> dict:
        budget_decision, budget_reasoning = self.reasoner.decide(
            "tuning_budget", {"n_rows": n_rows, "n_models": len(candidate_models)}
        )
        n_trials = int(budget_decision)
        self.log.add("Tuning", f"n_trials={n_trials}", budget_reasoning)

        if n_rows > 100_000:
            cv_folds = 2
            cv_reasoning = f"{n_rows:,} rows is large enough that 2 folds give a stable CV estimate; using 2 to control compute."
        else:
            cv_folds = 3
            cv_reasoning = f"{n_rows:,} rows -> using standard 3-fold CV for a reliable estimate."
        self.log.add("Tuning", f"cv_folds={cv_folds}", cv_reasoning)

        # Metric-aware tuning selection
        if task_type == "classification":
            y_arr = np.asarray(y)
            n_classes = len(np.unique(y_arr))
            if minority_ratio is not None and minority_ratio < 0.20:
                scoring = "f1_macro"
                metric_reasoning = (
                    f"Minority class ratio is {minority_ratio:.1%} (<20% threshold) -> "
                    "tuning with macro-F1 to optimize balanced recall across classes."
                )
            elif n_classes == 2:
                scoring = "roc_auc"
                metric_reasoning = "Binary classification with balanced classes -> tuning with ROC-AUC."
            else:
                scoring = "accuracy"
                metric_reasoning = "Multiclass classification with balanced classes -> tuning with accuracy."
        else:
            scoring = "r2"
            metric_reasoning = "Regression task -> tuning with R² coefficient of determination."

        self.log.add("Tuning", f"tuning_metric={scoring}", metric_reasoning)

        # 1. Evaluate Dummy Baseline First
        dummy_model = DummyClassifier(strategy="prior") if task_type == "classification" else DummyRegressor(strategy="mean")
        try:
            dummy_cv_scores = cross_val_score(dummy_model, X, y, cv=cv_folds, scoring=scoring, n_jobs=1)
            dummy_cv_mean = float(dummy_cv_scores.mean())
            dummy_cv_std = float(dummy_cv_scores.std())
        except Exception:
            dummy_cv_mean = 0.0
            dummy_cv_std = 0.0

        results = {
            "dummy": {
                "best_score": dummy_cv_mean,
                "best_params": {},
                "n_trials": 1,
                "cv_mean": dummy_cv_mean,
                "cv_std": dummy_cv_std,
                "improvement_over_dummy": 0.0,
                "scoring_metric": scoring,
            }
        }
        self.log.add(
            "Tuning",
            f"dummy baseline: {scoring}={dummy_cv_mean:.4f} (+/- {dummy_cv_std:.4f})",
            "Zero-intelligence baseline against which all candidate models are compared."
        )

        # 2. Tune each candidate model
        for model_name in candidate_models:
            if model_name not in MODEL_REGISTRY or model_name == "dummy":
                continue

            def objective(trial):
                model = _build_model(model_name, trial, task_type)
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=1)
                return scores.mean()

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

            # Measure variance (mean +/- std) on best trial
            try:
                best_model = _build_model(model_name, _FixedParamTrial(study.best_params), task_type)
                best_cv_scores = cross_val_score(best_model, X, y, cv=cv_folds, scoring=scoring, n_jobs=1)
                cv_mean = float(best_cv_scores.mean())
                cv_std = float(best_cv_scores.std())
            except Exception:
                cv_mean = float(study.best_value)
                cv_std = 0.0

            improvement = cv_mean - dummy_cv_mean

            results[model_name] = {
                "best_score": study.best_value,
                "best_params": study.best_params,
                "n_trials": n_trials,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "improvement_over_dummy": improvement,
                "dummy_cv_mean": dummy_cv_mean,
                "scoring_metric": scoring,
            }
            self.log.add(
                "Tuning",
                f"{model_name}: {scoring}={cv_mean:.4f} (+/- {cv_std:.4f}), delta={improvement:+.4f}",
                f"Best of {n_trials} Optuna trials with params {study.best_params}. Improvement over dummy baseline: {improvement:+.4f}."
            )

        return results
