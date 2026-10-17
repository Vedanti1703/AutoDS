# AutoDS-Agent 2.0: Autonomous Data Science Report

*Generated on 2026-09-29 14:50:06 | Reasoner Engine: `rule-based` | Pipeline Execution Time: 8.74s*

> **Research Problem Statement**: To develop an autonomous and explainable data-science platform that can automatically analyze an unseen dataset, perform preprocessing and feature engineering, select and optimize appropriate machine-learning models, explain predictions, support scalable data processing and assist users through an AI-based data-science assistant.

## 1. Dataset Profile & Data Scale
- **Total Rows**: 442
- **Total Columns**: 11
- **In-Memory Size**: 0.04 MB
- **Inferred Task Type**: `regression`
- **Target Column**: `target`
- **Execution Backend Chosen**: `local`
- **Numeric Features**: 10
- **Categorical Features**: 0
- **Duplicates Detected**: 0
- **Constant Columns**: 0

## 2. Exploratory Data Analysis (Auto EDA)
Generated diagnostic plots:
- **Correlation Heatmap**: `sample_output\regression\outputs\run_d02181ea\eda\correlation_heatmap.png`
- **Distributions**: `sample_output\regression\outputs\run_d02181ea\eda\distributions.png`

**Outlier Summary (IQR method)**:
| Feature | Outlier Count | Outlier % |
|---|---|---|
| `bmi` | 3 | 0.7% |
| `s1` | 8 | 1.8% |
| `s2` | 7 | 1.6% |
| `s3` | 7 | 1.6% |
| `s4` | 2 | 0.5% |
| `s5` | 4 | 0.9% |
| `s6` | 9 | 2.0% |

## 3. Preprocessing, Anomaly & Imbalance Handling
**Preprocessing Decisions Applied**:
- StandardScaler applied to 10 numeric columns.

**Anomaly Detection**:
- Isolation Forest flagged: **5.20%** of records
- Local Outlier Factor flagged: **5.20%** of records
- Top anomalous row indices: `[10, 11, 15, 23, 26]`

## 4. Feature Selection
- Method: `univariate`
- Features Selected (20): `['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 's5_x_s2', 's5_x_bmi', 's5_x_s6', 's2_x_age', 's2_x_bmi', 's4_x_s6']`...
- Features Dropped (20): `['sex', 's5_x_sex', 's5_x_age', 's5_x_s4', 'sex_x_s2', 'sex_x_age', 'sex_x_s4', 'sex_x_bmi', 'sex_x_s6', 's2_x_s4']`...

## 5. Model Tuning Leaderboard (Optuna CV)
| Candidate Model | Best CV Score | Trials | Best Hyperparameters |
|---|---|---|---|
| `ridge` | 0.4731 | 10 | `{'alpha': 56.69849511478853}` |
| `random_forest` | 0.4453 | 10 | `{'n_estimators': 21, 'max_depth': 4}` |
| `gradient_boosting` | 0.4138 | 10 | `{'n_estimators': 46, 'max_depth': 3, 'learning_rate': 0.05958389350068958}` |

## 6. Held-out Test Evaluation
**Final Selected Model**: `ridge`

| Model | Metrics Summary |
|---|---|
| `ridge` **[Selected]** | mae=44.4570, mse=2955.8442, rmse=54.3677, r2=0.4421 |
| `random_forest` | mae=44.4757, mse=2986.4543, rmse=54.6485, r2=0.4363 |
| `gradient_boosting` | mae=45.2155, mse=3008.8285, rmse=54.8528, r2=0.4321 |

**Evaluation Diagnostic Plots Saved**:
- Actual Vs Predicted: `sample_output\regression\outputs\run_d02181ea\evaluation\actual_vs_predicted.png`
- Residuals: `sample_output\regression\outputs\run_d02181ea\evaluation\residuals.png`

## 7. Model Explainability & Interpretability
- Interpretation Method: `permutation importance`

**Top 10 Most Influential Features**:
| Rank | Feature | Attribution Score |
|---|---|---|
| 1 | `s5` | 0.1481 |
| 2 | `bmi` | 0.1215 |
| 3 | `bmi_sq` | 0.0262 |
| 4 | `bp` | 0.0160 |
| 5 | `s3` | 0.0151 |
| 6 | `s6_sq` | 0.0089 |
| 7 | `bmi_x_s6` | 0.0083 |
| 8 | `s2` | 0.0076 |
| 9 | `s5_x_s6` | 0.0074 |
| 10 | `s4_x_s6` | 0.0056 |

## 8. Model Registry & Versioning
- **Version Tag**: `v7`
- **Model Path**: `model_registry\run_d02181ea_ridge_v7\model.joblib`
- **Metadata Path**: `model_registry\run_d02181ea_ridge_v7\metadata.json`

## 9. CSV Outputs
The following structured CSV files were generated:

- **`dataset_profile.csv`**: `sample_output\regression\outputs\run_d02181ea\csv\dataset_profile.csv` - *Dataset dimensions, scaling, missingness, and profiling metrics*
- **`model_comparison.csv`**: `sample_output\regression\outputs\run_d02181ea\csv\model_comparison.csv` - *Full candidate model leaderboard with tuning parameters and held-out scores*
- **`predictions.csv`**: `sample_output\regression\outputs\run_d02181ea\csv\predictions.csv` - *Held-out test set actuals, model predictions, and class probabilities/residuals*
- **`feature_importance.csv`**: `sample_output\regression\outputs\run_d02181ea\csv\feature_importance.csv` - *Ranked SHAP attribution scores and feature importance hierarchy*
- **`preprocessing_log.csv`**: `sample_output\regression\outputs\run_d02181ea\csv\preprocessing_log.csv` - *Complete audit of data transformations, imputations, and encodings*
- **`decision_log.csv`**: `sample_output\regression\outputs\run_d02181ea\csv\decision_log.csv` - *Granular agent reasoning log and execution duration for every stage*
- **`anomalies.csv`**: `sample_output\regression\outputs\run_d02181ea\csv\anomalies.csv` - *Isolation Forest & Local Outlier Factor anomaly scoring flags*

- **Download All CSVs (ZIP Archive)**: `sample_output\regression\outputs\run_d02181ea\autods_run_d02181ea_csv_results.zip`

## 10. Autonomous Agent Audit Trail
Every decision made during the pipeline run is logged below with the agent's internal reasoning, timing, and operational parameters.

| Stage | Decision | Reasoning | Time (s) |
|---|---|---|---|
| Profiler | backend=local | Dataset is 0.0MB with 442 rows, within in-memory thresholds. Routing to the pandas/sklearn local path -- distributed overhead would not pay off here. | 0.00 |
| Profiler | task_type=regression | Target has dtype float64 and 214 unique values relative to 442 rows -> treating as continuous, so this is a regression task. | 0.00 |
| AutoEDA | extended profile: 10 numeric, 0 categorical, 0 constant cols | Computed detailed dataset statistics: duplicates=0, high-cardinality cols=[], target subtypes=continuous. | 0.00 |
| AutoEDA | generated 2 EDA plots | Plots saved to sample_output\regression\outputs\run_d02181ea\eda: correlation_heatmap, distributions. | 0.00 |
| Preprocessing[local] | complete | Dropped 0 sparse columns, one-hot 0, ordinal 0 categoricals, scaled 10 numeric columns. | 0.00 |
| AnomalyDetection | IF=5.2% anomalous, LOF=5.2% anomalous | Isolation Forest detected 5.2% anomalous rows; Local Outlier Factor detected 5.2% anomalous rows (contamination assumption=0.05). Anomalies flagged only – not removed (default behaviour). | 0.00 |
| FeatureEngineering | pairs from 7 cols | 10 features -> 45 possible pairwise interactions, exceeds the 30 cap. Sampling 7 columns to pair instead. | 0.00 |
| FeatureEngineering | added 7 polynomial (sq) features | Squares of top 7 numeric columns to capture nonlinear relationships; low cost on top of existing features. | 0.00 |
| FeatureEngineering | added 2 ratio features | A/B ratio features encode scale-invariant relationships between numeric columns; useful when relative magnitude matters more than absolute. | 0.00 |
| FeatureEngineering | total added: 30 engineered features | Created 21 interaction, 7 polynomial, 2 ratio features as low-cost nonlinear signals. | 0.00 |
| FeatureSelection | method=univariate, k=20 | 40 features, regression task — univariate F-test is fast and appropriate for continuous targets. | 0.00 |
| FeatureSelection | kept 20/40 features, dropped 20 | Top features by univariate: ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 's5_x_s2'].... Dropped: ['sex', 's5_x_sex', 's5_x_age', 's5_x_s4', 'sex_x_s2', 'sex_x_age', 'sex_x_s4', 'sex_x_bmi', 'sex_x_s6', 's2_x_s4'].... | 0.00 |
| ModelSelection | candidates=['ridge', 'random_forest', 'gradient_boosting'] | 442 rows / 10 features is small enough to try the full candidate set of regression models. | 0.00 |
| Tuning | n_trials=10 | Allocating 10 Optuna trials per model (3 models) given 442 rows -- balances search quality against the quadratic-ish cost of CV on larger data. | 0.00 |
| Tuning | cv_folds=3 | 442 rows -- using standard 3-fold CV for a reliable estimate. | 0.00 |
| Tuning | ridge: best r2=0.4731 | Best of 10 Optuna trials using params {'alpha': 56.69849511478853}. | 0.00 |
| Tuning | random_forest: best r2=0.4453 | Best of 10 Optuna trials using params {'n_estimators': 21, 'max_depth': 4}. | 0.00 |
| Tuning | gradient_boosting: best r2=0.4138 | Best of 10 Optuna trials using params {'n_estimators': 46, 'max_depth': 3, 'learning_rate': 0.05958389350068958}. | 0.00 |
| Evaluation | ridge test metrics | {'mae': 44.457, 'mse': 2955.8442, 'rmse': 54.3677, 'r2': 0.4421} | 0.00 |
| Evaluation | random_forest test metrics | {'mae': 44.4757, 'mse': 2986.4543, 'rmse': 54.6485, 'r2': 0.4363} | 0.00 |
| Evaluation | gradient_boosting test metrics | {'mae': 45.2155, 'mse': 3008.8285, 'rmse': 54.8528, 'r2': 0.4321} | 0.00 |
| Evaluation | final_model=ridge | Selected ridge with score 0.4421, the highest among 3 tuned candidates (runner-up random_forest scored 0.4363). | 0.00 |
| Explainability | method=permutation importance | Chose permutation importance because model is Ridge (non-tree, permutation importance). | 0.00 |
| ArtifactExport | Exported Cleaned and Engineered Datasets | Saved cleaned dataset (442 rows, 11 columns) to sample_output\regression\outputs\run_d02181ea\cleaned_data.csv and feature matrix to sample_output\regression\outputs\run_d02181ea\engineered_features.csv. | 0.00 |
| CSVExport | exported 7 CSV files | Generated structured CSV artifacts in sample_output\regression\outputs\run_d02181ea\csv: dataset_profile, model_comparison, predictions, feature_importance, preprocessing_log, decision_log, anomalies | 0.00 |
