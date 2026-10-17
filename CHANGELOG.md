# Changelog

All notable changes to the AutoDS-Agent project will be documented in this file.

## [Phase 2 - Big Data Ingestion & Streaming] - 2026-10-17

### Added
- **`StreamingIngestAgent`** (`autods_agent/ingestion.py`): Scalable dataset ingestion — pandas (small), Dask (large), chunked-pandas fallback (Dask unavailable). Supports CSV & Parquet. Graceful import errors logged.
- **`OnlineLearningAgent`** (`autods_agent/online_learning.py`): Incremental `partial_fit` training (SGDClassifier/Regressor, PassiveAggressive, Perceptron, BernoulliNB). Online scaler, per-batch metric tracking, multi-epoch, auto model selection.
- **`csv_to_parquet()`**: Utility to convert large CSVs to Parquet for faster future ingestion.
- **`RuleBasedReasoner._q_online_model`**: Routes to SGDClassifier (classification) or SGDRegressor (regression).
- **Tests** (`tests/test_phase2_bigdata.py`, 16 tests): ingestion paths, parquet roundtrip, online clf/reg, multi-epoch, metric tracking, reasoner rules.

### Changed
- `autods_agent/__init__.py`: exports `StreamingIngestAgent`, `csv_to_parquet`, `OnlineLearningAgent`; bumped `__version__` → `2.1.0`

## [Phase 1 - Correctness & Leakage Fixes] - 2026-10-17


### Added
- **Formal Audit**: Added [AUDIT.md](file:///c:/Users/Vedanti/Downloads/BDA-project--main/AUDIT.md) confirming issues (a) data leakage, (b) naked estimator registry, (c) tuning metric bias, and (d) repo pollution.
- **Scikit-Learn Transformers**:
  - `TabularPreprocessor`: Leakage-free fit/transform transformer with skew-aware imputation, one-hot/ordinal encoding, and standard scaling.
  - `FeatureEngineeringTransformer`: Leakage-free fit/transform transformer for pairwise interactions, degree-2 squares, and ratios.
  - `FeatureSelectionTransformer`: Leakage-free fit/transform transformer for Mutual Information, SelectKBest, and Random Forest feature selection.
- **FullInferencePipeline**: End-to-end pipeline combining preprocessing, feature engineering, feature selection, and the final estimator into a single serialized object that predicts directly on raw rows and dictionary records.
- **Metric-Aware Tuning**: Dynamic Optuna scoring selecting `f1_macro` for imbalanced classification (<20% minority), `roc_auc` for balanced binary, `accuracy` for balanced multiclass, and `r2` for regression.
- **Baseline Benchmarking**: `DummyClassifier` / `DummyRegressor` integrated into Optuna tuning and final model evaluation to measure each model's delta improvement over a zero-intelligence baseline, reporting CV mean $\pm$ std ($\mu \pm \sigma$).
- **Data Quality & Leakage Checks**: Automatic dropping of constant columns (`nunique <= 1`), ID-like columns (`unique_ratio > 0.95` on non-float), and warnings for suspected target leakage (`|corr| > 0.95`).
- **Tests Added**:
  - `test_no_data_leakage`: Proves that test partition perturbation and unseen categories have zero impact on fitted preprocessor/selection parameters.
  - `test_registered_pipeline_raw_prediction`: Proves that the registered model loaded from `ModelRegistry` predicts directly from raw input dictionaries and DataFrames.

### Changed
- **Orchestrator Execution Flow**: Partitioned raw data into train/test splits FIRST (seeded, stratified) prior to fitting any transformers. All preprocessing, anomaly detection, feature engineering, and feature selection are fit strictly on the training partition.
- **What-If Analysis**: Replaced manual string parsing with direct calls to `pipeline.predict()` and `pipeline.predict_proba()`.
- **Model Registry**: Now serializes `FullInferencePipeline` instead of bare estimators.
- **Repo Hygiene**: Updated `.gitignore` to exclude caches, virtual environments, outputs, and model registry artifacts; updated tests to execute within isolated temporary directories.
