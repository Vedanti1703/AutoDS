# System Audit Report: AutoDS-Agent 2.0 Architecture & Correctness

**Date:** 2026-10-17  
**Auditor:** Senior ML / Big Data Engineer  
**Scope:** `autods_agent/`, `webapp/app.py`, `tests/`, `run_scalability_experiment.py`  
**Purpose:** Formal audit assessing data leakage, pipeline composition, tuning metrics, artifact persistence, and repository hygiene prior to the v3.0 Scalable ML upgrade.

---

## Executive Summary

A comprehensive architectural and code review of `AutoDS-Agent 2.0` was conducted. **All four suspected issues (a, b, c, d) are CONFIRMED.** 

While the system demonstrated high agent modularity, explainability logging, and rich frontend visualisations, critical methodological flaws existed in the data preparation flow, model packaging, and hyperparameter optimization. Specifically:
1. Preprocessing and feature selection leaked test partition statistics and labels prior to splitting.
2. The persistent model registry serialized bare scikit-learn estimators incapable of inference on raw tabular input.
3. Hyperparameter tuning blindly maximized accuracy even under severe class imbalance.
4. Test execution and manual runs directly polluted the project repository with temporary artifacts.

Detailed confirmations and evidence for each issue follow below.

---

## Audit Findings & Evidence

### Issue (a): Data Leakage Prior to Train/Test Split
*Status: **CONFIRMED***

#### Findings:
In `autods_agent/orchestrator.py` (lines 112–148), the pipeline execution sequence was:
1. `self.profiler.run(df, target_col)` (Full dataset)
2. `self.eda.run(df, profile)` (Full dataset)
3. `self.preprocessor.run(df, profile)` (Full dataset)
4. `self.anomaly_detector.run(clean_df, profile)` (Full dataset)
5. `self.feature_engineer.run(clean_df, profile)` (Full dataset)
6. `self.feature_selector.run(fe_df, profile)` (Full dataset)
7. `X = fs_df[feature_names].values; y = fs_df[target_col].values`
8. Train/test split was only invoked inside `self.imbalance_handler.run(X, y)` for classification tasks, or deferred until `self.evaluator.run()` for regression tasks.

#### Architectural Vulnerabilities:
- **Imputation Leakage:** In `autods_agent/preprocessing.py` (lines 140–154), column means and medians were calculated across the combined dataset. Test set values directly skewed the imputation fill values for training instances.
- **Scaling Leakage:** `StandardScaler.fit_transform(df[num_cols])` (`preprocessing.py`, line 207) computed global sample means $\mu$ and standard deviations $\sigma$ across all rows, transferring test distribution knowledge into training features.
- **Categorical Encoding Leakage:** `OneHotEncoder` and `OrdinalEncoder` observed all categories across the entire dataset, masking realistic out-of-vocabulary / unseen level scenarios at test time.
- **Feature Selection Target Leakage (Severe):** In `autods_agent/feature_selection.py` (lines 84–130), `mutual_info_classif`, `SelectKBest(f_classif)`, and `RandomForestClassifier.feature_importances_` were calculated on the **entire target vector $y$** including test samples. Features were selected precisely because they correlated with test labels, producing overly optimistic validation scores.

---

### Issue (b): Model Registry Saves Bare Estimator Without Preprocessing Pipeline
*Status: **CONFIRMED***

#### Findings:
In `autods_agent/orchestrator.py` (lines 235–244):
```python
reg_version = self.registry.register(
    model=eval_results["final_model"],
    model_type=final_name,
    run_id=run_id,
    ...
)
```
And in `autods_agent/registry.py` (lines 69–73):
```python
model_path = os.path.join(folder, "model.joblib")
joblib.dump(model, model_path)
```

#### Architectural Vulnerabilities:
- `eval_results["final_model"]` was a naked scikit-learn estimator (e.g., `LogisticRegression` or `RandomForestClassifier`) fitted on an in-memory NumPy array with engineered columns (e.g., 50 features including interaction products `colA_x_colB` and polynomial squares `colA_sq`).
- The saved `model.joblib` artifact could **not** predict directly on raw tabular rows (e.g., from a CSV or REST API endpoint).
- To circumvent this, `orchestrator.py` constructed an ad-hoc, brittle closure called `whatif_transform` (lines 183–222) using manual string parsing:
  ```python
  if "_x_" in col:
      p1, p2 = col.split("_x_")
  elif "_sq" in col:
      base = col.replace("_sq", "")
  elif "_ratio_" in col:
      parts = col.split("_ratio_")
  ```
  This ad-hoc string parsing could not be reconstructed when the model was loaded independently in production or external services.

---

### Issue (c): Tuning Optimizes Accuracy Even for Imbalanced Data
*Status: **CONFIRMED***

#### Findings:
In `autods_agent/model_selection.py` (lines 98–105):
```python
scoring = "accuracy" if task_type == "classification" else "r2"
results = {}

for model_name in candidate_models:
    def objective(trial):
        model = _build_model(model_name, trial, task_type)
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=1)
        return scores.mean()
```

#### Architectural Vulnerabilities:
- In classification problems with severe class imbalance (e.g. 95% majority, 5% minority), a trivial majority-class classifier achieves 95% accuracy while possessing 0% recall and 0% F1 on the critical minority class.
- Optuna's `TPESampler` optimized hyperparameters solely to maximize `scoring = "accuracy"`, systematically penalizing models that sacrificed minor accuracy to improve minority-class recall and macro-F1.
- In regression, `scoring` was hardcoded to `"r2"`, without tracking or optimizing Root Mean Squared Error (RMSE) or Mean Absolute Error (MAE).

---

### Issue (d): Artifact Pollution and Incomplete Ignore Specifications
*Status: **CONFIRMED***

#### Findings:
Inspection of `.gitignore`, `tests/`, and project directories revealed:
1. `.gitignore` only contained 5 lines:
   ```
   venv/
   __pycache__/
   *.pyc
   webapp/uploads/*.csv
   .env
   ```
   Missing critical rules: `.pytest_cache/`, `outputs/`, `model_registry/`, `webapp/uploads/` (non-csv files), and build artifacts.
2. Direct artifact emission into workspace roots during automated testing:
   - In `tests/test_csv_export.py` (line 270): `orch = Orchestrator(output_root="outputs")` wrote run artifacts directly into the repository root `outputs/`.
   - In `tests/test_pipeline_modules.py` (line 182): `reg = ModelRegistry()` wrote test model versions directly into `model_registry/` in the repository root.
   - Running test suites and demos left generated directories (`.pytest_cache`, `model_registry/`, `outputs/`) in the working directory.

---

## Remediation Plan (Phase 1)

1. **Strict Temporal Splitting (Section 1.1):**
   - Partition raw tabular data into `(train_df, test_df)` immediately after profiling and initial sanity checks, using stratified splitting for classification and seeded random splitting for regression.
   - Refactor preprocessing, feature engineering, and feature selection into reusable scikit-learn compatible `BaseEstimator` / `TransformerMixin` components (`fit()` and `transform()`).
   - Fit all transformations strictly on `train_df`, and apply `.transform()` on `test_df`.

2. **Unified End-to-End Pipeline & Registry Packaging (Section 1.2):**
   - Assemble the fitted preprocessing transformer and final trained estimator into a single `sklearn.pipeline.Pipeline`.
   - Register this unified pipeline in `ModelRegistry` so `pipeline.predict(raw_row_df)` and `pipeline.predict_proba(raw_row_df)` operate seamlessly on raw features without manual string parsing.

3. **Metric-Aware Hyperparameter Optimization (Section 1.3):**
   - Inspect class balance dynamically: if the minority class frequency $< 20\%$, tune with `scoring="f1_macro"` (or `balanced_accuracy`).
   - For balanced binary classification, tune with `scoring="roc_auc"`.
   - For balanced multiclass, tune with `scoring="accuracy"`.
   - For regression, support `scoring="neg_root_mean_squared_error"` and `scoring="r2"`.
   - Log explicit scientific rationales for the chosen optimization metric into `RunLog`.

4. **Dummy Baseline & Leaderboard Variance (Section 1.4):**
   - Evaluate `DummyClassifier(strategy="prior")` / `DummyRegressor(strategy="mean")` as baseline benchmarks on every run.
   - Measure each candidate model's delta improvement over the baseline.
   - Record cross-validation performance with both mean and standard deviation ($\mu \pm \sigma$).

5. **Data Quality & Leakage Filtering (Section 1.5):**
   - Automatically identify and prune ID-like columns (unique ratio $> 0.95$ on non-float data) and zero-variance constant columns.
   - Compute bivariate correlations with target labels and raise explicit warnings for suspicious features ($|\text{corr}| > 0.95$).

6. **Repository Hygiene & Test Isolation (Section 1.6 & 1.7):**
   - Update `.gitignore` to comprehensively exclude virtualenvs, caches, generated model stores, and outputs.
   - Refactor pytest suites to utilize `tmp_path` fixtures for all file operations.
   - Add targeted unit tests confirming zero data leakage and validating raw-record pipeline inference.
