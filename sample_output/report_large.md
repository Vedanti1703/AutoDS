# AutoDS-Agent 2.0: Autonomous Data Science Report

*Generated on 2026-09-29 14:45:29 | Reasoner Engine: `rule-based` | Pipeline Execution Time: 76.93s*

> **Research Problem Statement**: To develop an autonomous and explainable data-science platform that can automatically analyze an unseen dataset, perform preprocessing and feature engineering, select and optimize appropriate machine-learning models, explain predictions, support scalable data processing and assist users through an AI-based data-science assistant.

## 1. Dataset Profile & Data Scale
- **Total Rows**: 30,000
- **Total Columns**: 9
- **In-Memory Size**: 2.06 MB
- **Inferred Task Type**: `regression`
- **Target Column**: `target`
- **Execution Backend Chosen**: `distributed`
- **Numeric Features**: 8
- **Categorical Features**: 0
- **Duplicates Detected**: 0
- **Constant Columns**: 0

## 2. Exploratory Data Analysis (Auto EDA)
Generated diagnostic plots:
- **Correlation Heatmap**: `outputs\run_55fdaba1\eda\correlation_heatmap.png`
- **Distributions**: `outputs\run_55fdaba1\eda\distributions.png`
- **Missing Values**: `outputs\run_55fdaba1\eda\missing_values.png`

**Outlier Summary (IQR method)**:
| Feature | Outlier Count | Outlier % |
|---|---|---|
| `f0` | 198 | 0.7% |
| `f1` | 239 | 0.8% |
| `f2` | 214 | 0.7% |
| `f3` | 201 | 0.7% |
| `f4` | 201 | 0.7% |
| `f5` | 202 | 0.7% |
| `f6` | 176 | 0.6% |
| `f7` | 186 | 0.6% |

## 3. Preprocessing, Anomaly & Imbalance Handling
**Preprocessing Decisions Applied**:
- Dask-median-imputed 'f0'.
- Dask-median-imputed 'f3'.
- StandardScaler applied to 8 numeric columns.

**Anomaly Detection**:
- Isolation Forest flagged: **5.00%** of records
- Local Outlier Factor flagged: **5.00%** of records
- Top anomalous row indices: `[7, 28, 43, 83, 91]`

## 4. Feature Selection
- Method: `univariate`
- Features Selected (19): `['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f0_x_f2', 'f1_x_f4', 'f2_x_f3', 'f2_x_f4', 'f2_x_f7', 'f3_x_f4', 'f3_x_f5']`...
- Features Dropped (19): `['f0_x_f1', 'f0_x_f3', 'f0_x_f4', 'f0_x_f5', 'f0_x_f6', 'f0_x_f7', 'f1_x_f2', 'f1_x_f3', 'f1_x_f5', 'f1_x_f6']`...

## 5. Model Tuning Leaderboard (Optuna CV)
| Candidate Model | Best CV Score | Trials | Best Hyperparameters |
|---|---|---|---|
| `ridge` | 0.9739 | 6 | `{'alpha': 0.0060252157362038605}` |
| `hist_gradient_boosting` | 0.9610 | 6 | `{'max_iter': 75, 'max_depth': 10, 'learning_rate': 0.1205712628744377}` |
| `random_forest` | 0.8657 | 6 | `{'n_estimators': 28, 'max_depth': 8}` |

## 6. Held-out Test Evaluation
**Final Selected Model**: `ridge`

| Model | Metrics Summary |
|---|---|
| `ridge` **[Selected]** | mae=13.0695, mse=610.4137, rmse=24.7066, r2=0.9741 |
| `random_forest` | mae=43.3808, mse=3202.3735, rmse=56.5895, r2=0.8642 |
| `hist_gradient_boosting` | mae=19.7709, mse=917.1317, rmse=30.2842, r2=0.9611 |

**Evaluation Diagnostic Plots Saved**:
- Actual Vs Predicted: `outputs\run_55fdaba1\evaluation\actual_vs_predicted.png`
- Residuals: `outputs\run_55fdaba1\evaluation\residuals.png`

## 7. Model Explainability & Interpretability
- Interpretation Method: `permutation importance`

**Top 10 Most Influential Features**:
| Rank | Feature | Attribution Score |
|---|---|---|
| 1 | `f7` | 0.6412 |
| 2 | `f3` | 0.5826 |
| 3 | `f0` | 0.3007 |
| 4 | `f6` | 0.2071 |
| 5 | `f4` | 0.1558 |
| 6 | `f1` | 0.0294 |
| 7 | `f2` | 0.0123 |
| 8 | `f5` | 0.0047 |
| 9 | `f2_x_f4` | 0.0000 |
| 10 | `f3_x_f6` | 0.0000 |

## 8. Model Registry & Versioning
- **Version Tag**: `v6`
- **Model Path**: `model_registry\run_55fdaba1_ridge_v6\model.joblib`
- **Metadata Path**: `model_registry\run_55fdaba1_ridge_v6\metadata.json`

## 9. CSV Outputs
The following structured CSV files were generated:

- **`dataset_profile.csv`**: `outputs\run_55fdaba1\csv\dataset_profile.csv` — *Dataset dimensions, scaling, missingness, and profiling metrics*
- **`model_comparison.csv`**: `outputs\run_55fdaba1\csv\model_comparison.csv` — *Full candidate model leaderboard with tuning parameters and held-out scores*
- **`predictions.csv`**: `outputs\run_55fdaba1\csv\predictions.csv` — *Held-out test set actuals, model predictions, and class probabilities/residuals*
- **`feature_importance.csv`**: `outputs\run_55fdaba1\csv\feature_importance.csv` — *Ranked SHAP attribution scores and feature importance hierarchy*
- **`preprocessing_log.csv`**: `outputs\run_55fdaba1\csv\preprocessing_log.csv` — *Complete audit of data transformations, imputations, and encodings*
- **`decision_log.csv`**: `outputs\run_55fdaba1\csv\decision_log.csv` — *Granular agent reasoning log and execution duration for every stage*
- **`anomalies.csv`**: `outputs\run_55fdaba1\csv\anomalies.csv` — *Isolation Forest & Local Outlier Factor anomaly scoring flags*

- **Download All CSVs (ZIP Archive)**: `outputs\run_55fdaba1\autods_run_55fdaba1_csv_results.zip`

## 10. Autonomous Agent Audit Trail
Every decision made during the pipeline run is logged below with the agent's internal reasoning, timing, and operational parameters.

| Stage | Decision | Reasoning | Time (s) |
|---|---|---|---|
| Profiler | backend=distributed | Dataset is 2.1MB with 30,000 rows, exceeding thresholds (5MB / 25,000 rows). Routing to the Dask distributed execution path to keep memory use and wall-clock time bounded as data grows. | 0.00 |
| Profiler | task_type=regression | Target has dtype float64 and 30000 unique values relative to 30000 rows -> treating as continuous, so this is a regression task. | 0.00 |
| AutoEDA | extended profile: 8 numeric, 0 categorical, 0 constant cols | Computed detailed dataset statistics: duplicates=0, high-cardinality cols=[], target subtypes=continuous. | 0.00 |
| AutoEDA | generated 3 EDA plots | Plots saved to outputs\run_55fdaba1\eda: correlation_heatmap, distributions, missing_values. | 0.00 |
| Preprocessing[distributed] | partitioned into 4 chunks | Split the DataFrame into 4 Dask partitions so imputation/encoding/scaling execute in parallel across workers — scalability lever that lets the same logical pipeline handle data much larger than one machine's RAM. | 0.00 |
| Preprocessing[distributed] | complete | Distributed preprocessing done (4 partitions). Scaled 8 columns. | 0.00 |
| AnomalyDetection | IF=5.0% anomalous, LOF=5.0% anomalous | Isolation Forest detected 5.0% anomalous rows; Local Outlier Factor detected 5.0% anomalous rows (contamination assumption=0.05). Anomalies flagged only – not removed (default behaviour). | 0.00 |
| FeatureEngineering | pairs from 8 cols | 8 features -> 28 pairwise interactions, within budget; generating all of them. | 0.00 |
| FeatureEngineering | added 2 polynomial (sq) features | Squares of top 2 numeric columns to capture nonlinear relationships; low cost on top of existing features. | 0.00 |
| FeatureEngineering | total added: 30 engineered features | Created 28 interaction, 2 polynomial, 0 ratio features as low-cost nonlinear signals. | 0.00 |
| FeatureSelection | method=univariate, k=19 | 38 features, regression task — univariate F-test is fast and appropriate for continuous targets. | 0.00 |
| FeatureSelection | kept 19/38 features, dropped 19 | Top features by univariate: ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f0_x_f2', 'f1_x_f4'].... Dropped: ['f0_x_f1', 'f0_x_f3', 'f0_x_f4', 'f0_x_f5', 'f0_x_f6', 'f0_x_f7', 'f1_x_f2', 'f1_x_f3', 'f1_x_f5', 'f1_x_f6'].... | 0.00 |
| ModelSelection | candidates=['ridge', 'random_forest', 'hist_gradient_boosting'] | 30,000 rows / 8 features is large enough that plain gradient boosting would be slow; substituting the histogram-based variant for scalability. | 0.00 |
| Tuning | n_trials=6 | Allocating 6 Optuna trials per model (3 models) given 30,000 rows -- balances search quality against the quadratic-ish cost of CV on larger data. | 0.00 |
| Tuning | cv_folds=3 | 30,000 rows -- using standard 3-fold CV for a reliable estimate. | 0.00 |
| Tuning | ridge: best r2=0.9739 | Best of 6 Optuna trials using params {'alpha': 0.0060252157362038605}. | 0.00 |
| Tuning | random_forest: best r2=0.8657 | Best of 6 Optuna trials using params {'n_estimators': 28, 'max_depth': 8}. | 0.00 |
| Tuning | hist_gradient_boosting: best r2=0.9610 | Best of 6 Optuna trials using params {'max_iter': 75, 'max_depth': 10, 'learning_rate': 0.1205712628744377}. | 0.00 |
| Evaluation | ridge test metrics | {'mae': 13.0695, 'mse': 610.4137, 'rmse': 24.7066, 'r2': 0.9741} | 0.00 |
| Evaluation | random_forest test metrics | {'mae': 43.3808, 'mse': 3202.3735, 'rmse': 56.5895, 'r2': 0.8642} | 0.00 |
| Evaluation | hist_gradient_boosting test metrics | {'mae': 19.7709, 'mse': 917.1317, 'rmse': 30.2842, 'r2': 0.9611} | 0.00 |
| Evaluation | final_model=ridge | Selected ridge with score 0.9741, the highest among 3 tuned candidates (runner-up hist_gradient_boosting scored 0.9611). | 0.00 |
| Explainability | method=permutation importance | Chose permutation importance because model is Ridge (non-tree, permutation importance). | 0.00 |
| ArtifactExport | Exported Cleaned and Engineered Datasets | Saved cleaned dataset (30000 rows, 9 columns) to outputs\run_55fdaba1\cleaned_data.csv and feature matrix to outputs\run_55fdaba1\engineered_features.csv. | 0.00 |
| CSVExport | exported 7 CSV files | Generated structured CSV artifacts in outputs\run_55fdaba1\csv: dataset_profile, model_comparison, predictions, feature_importance, preprocessing_log, decision_log, anomalies | 0.00 |
