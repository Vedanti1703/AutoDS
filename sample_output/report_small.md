# AutoDS-Agent 2.0: Autonomous Data Science Report

*Generated on 2026-10-17 21:39:45 | Reasoner Engine: `rule-based` | Pipeline Execution Time: 23.36s*

> **Research Problem Statement**: To develop an autonomous and explainable data-science platform that can automatically analyze an unseen dataset, perform preprocessing and feature engineering, select and optimize appropriate machine-learning models, explain predictions, support scalable data processing and assist users through an AI-based data-science assistant.

## 1. Dataset Profile & Data Scale
- **Total Rows**: 569
- **Total Columns**: 31
- **In-Memory Size**: 0.13 MB
- **Inferred Task Type**: `classification`
- **Target Column**: `diagnosis`
- **Execution Backend Chosen**: `local`
- **Numeric Features**: 30
- **Categorical Features**: 0
- **Duplicates Detected**: 0
- **Constant Columns**: 0

## 2. Exploratory Data Analysis (Auto EDA)
Generated diagnostic plots:
- **Correlation Heatmap**: `outputs\run_0114e7a4\eda\correlation_heatmap.png`
- **Distributions**: `outputs\run_0114e7a4\eda\distributions.png`
- **Boxplots**: `outputs\run_0114e7a4\eda\boxplots.png`
- **Class Distribution**: `outputs\run_0114e7a4\eda\class_distribution.png`
- **Missing Values**: `outputs\run_0114e7a4\eda\missing_values.png`

**Outlier Summary (IQR method)**:
| Feature | Outlier Count | Outlier % |
|---|---|---|
| `mean radius` | 11 | 2.1% |
| `mean texture` | 7 | 1.3% |
| `mean perimeter` | 13 | 2.3% |
| `mean area` | 25 | 4.4% |
| `mean smoothness` | 6 | 1.1% |
| `mean compactness` | 16 | 2.8% |
| `mean concavity` | 18 | 3.2% |
| `mean concave points` | 10 | 1.8% |

## 3. Preprocessing, Anomaly & Imbalance Handling
**Preprocessing Decisions Applied**:
- Imputed 'mean radius' with median (13.3800, skew=0.85).
- Imputed 'mean texture' with median (18.7000, skew=0.72).
- Imputed 'worst symmetry' with median (0.2829, skew=1.44).
- StandardScaler fit on 30 numeric features.

**Anomaly Detection**:
- Isolation Forest flagged: **5.05%** of records
- Local Outlier Factor flagged: **5.05%** of records
- Top anomalous row indices: `[192, 0, 521, 213, 352]`

**Class Imbalance Analysis**:
- Imbalance Detected: **No** (Minority class ratio: 37.4%)

## 4. Feature Selection
- Method: `mutual_info`
- Features Selected (30): `['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'radius error', 'perimeter error', 'area error', 'concavity error', 'concave points error', 'worst radius', 'worst texture']`...
- Features Dropped (30): `['mean symmetry', 'mean fractal dimension', 'texture error', 'smoothness error', 'compactness error', 'symmetry error', 'fractal dimension error', 'worst fractal dimension', 'worst concave points_x_compactness error', 'worst concave points_x_concave points error']`...

## 5. Model Tuning Leaderboard (Optuna CV)
| Candidate Model | Best CV Score | Trials | Best Hyperparameters |
|---|---|---|---|
| `logistic_regression` | 0.9926 | 10 | `{'C': 0.9846738873614566}` |
| `gradient_boosting` | 0.9909 | 10 | `{'n_estimators': 94, 'max_depth': 2, 'learning_rate': 0.2708160864249968}` |
| `random_forest` | 0.9904 | 10 | `{'n_estimators': 20, 'max_depth': 3}` |
| `dummy` | 0.5000 | 1 | `{}` |

## 6. Held-out Test Evaluation
**Final Selected Model**: `logistic_regression`

| Model | Metrics Summary |
|---|---|
| `dummy` | accuracy=0.6316, precision_macro=0.3158, recall_macro=0.5000, f1_macro=0.3871, f1_weighted=0.4890, roc_auc=0.5000, improvement_over_dummy=0.0000, cv_mean=0.5000, cv_std=0.0000 |
| `logistic_regression` **[Selected]** | accuracy=0.9649, precision_macro=0.9623, recall_macro=0.9623, f1_macro=0.9623, f1_weighted=0.9649, roc_auc=0.9957, improvement_over_dummy=0.3333, cv_mean=0.9926, cv_std=0.0078 |
| `random_forest` | accuracy=0.9474, precision_macro=0.9435, recall_macro=0.9435, f1_macro=0.9435, f1_weighted=0.9474, roc_auc=0.9883, improvement_over_dummy=0.3158, cv_mean=0.9904, cv_std=0.0042 |
| `gradient_boosting` | accuracy=0.9561, precision_macro=0.9551, recall_macro=0.9504, f1_macro=0.9526, f1_weighted=0.9560, roc_auc=0.9940, improvement_over_dummy=0.3245, cv_mean=0.9909, cv_std=0.0034 |

**Evaluation Diagnostic Plots Saved**:
- Confusion Matrix: `outputs\run_0114e7a4\evaluation\confusion_matrix.png`
- Roc Curve: `outputs\run_0114e7a4\evaluation\roc_curve.png`
- Pr Curve: `outputs\run_0114e7a4\evaluation\pr_curve.png`

## 7. Model Explainability & Interpretability
- Interpretation Method: `permutation importance`

**Top 10 Most Influential Features**:
| Rank | Feature | Attribution Score |
|---|---|---|
| 1 | `worst concave points` | 0.0246 |
| 2 | `worst texture` | 0.0228 |
| 3 | `worst smoothness` | 0.0175 |
| 4 | `worst area` | 0.0175 |
| 5 | `worst concavity` | 0.0123 |
| 6 | `radius error` | 0.0105 |
| 7 | `worst radius` | 0.0105 |
| 8 | `area error` | 0.0088 |
| 9 | `worst perimeter` | 0.0088 |
| 10 | `mean area` | 0.0070 |

## 8. Model Registry & Versioning
- **Version Tag**: `v17`
- **Model Path**: `model_registry\run_0114e7a4_logistic_regression_v17\model.joblib`
- **Metadata Path**: `model_registry\run_0114e7a4_logistic_regression_v17\metadata.json`

## 9. CSV Outputs
The following structured CSV files were generated:

- **`dataset_profile.csv`**: `outputs\run_0114e7a4\csv\dataset_profile.csv` - *Dataset dimensions, scaling, missingness, and profiling metrics*
- **`model_comparison.csv`**: `outputs\run_0114e7a4\csv\model_comparison.csv` - *Full candidate model leaderboard with tuning parameters and held-out scores*
- **`predictions.csv`**: `outputs\run_0114e7a4\csv\predictions.csv` - *Held-out test set actuals, model predictions, and class probabilities/residuals*
- **`feature_importance.csv`**: `outputs\run_0114e7a4\csv\feature_importance.csv` - *Ranked SHAP attribution scores and feature importance hierarchy*
- **`preprocessing_log.csv`**: `outputs\run_0114e7a4\csv\preprocessing_log.csv` - *Complete audit of data transformations, imputations, and encodings*
- **`decision_log.csv`**: `outputs\run_0114e7a4\csv\decision_log.csv` - *Granular agent reasoning log and execution duration for every stage*
- **`anomalies.csv`**: `outputs\run_0114e7a4\csv\anomalies.csv` - *Isolation Forest & Local Outlier Factor anomaly scoring flags*

- **Download All CSVs (ZIP Archive)**: `outputs\run_0114e7a4\autods_run_0114e7a4_csv_results.zip`

## 10. Autonomous Agent Audit Trail
Every decision made during the pipeline run is logged below with the agent's internal reasoning, timing, and operational parameters.

| Stage | Decision | Reasoning | Time (s) |
|---|---|---|---|
| Profiler | backend=local | Dataset is 0.1MB with 569 rows, within in-memory thresholds. Routing to the pandas/sklearn local path -- distributed overhead would not pay off here. | 0.00 |
| Profiler | task_type=classification | Target has dtype int64 with only 2 unique values -> treating as categorical, so this is a classification task. | 0.00 |
| AutoEDA | extended profile: 30 numeric, 0 categorical, 0 constant cols | Computed detailed dataset statistics: duplicates=0, high-cardinality cols=[], target subtypes=binary. | 0.00 |
| AutoEDA | generated 5 EDA plots | Plots saved to outputs\run_0114e7a4\eda: correlation_heatmap, distributions, boxplots, class_distribution, missing_values. | 0.00 |
| Orchestrator | train_test_split (80/20) | Seeded split executed FIRST into 455 train and 114 test instances to strictly prevent data leakage. | 0.00 |
| AnomalyDetection | IF=5.05% anomalous, LOF=5.05% anomalous | Isolation Forest detected 5.05% anomalous rows; Local Outlier Factor detected 5.05% anomalous rows (contamination assumption=0.05). Anomalies flagged only – not removed (default behaviour). | 0.00 |
| Preprocessing[local] | fit complete | Trained preprocessing on 455 rows: dropped 0 cols, imputed 3, one-hot 0, ordinal 0, scaled 30 numeric features. | 0.00 |
| FeatureEngineering | pairs from 7 features | 30 features -> 435 pairs exceeds 30 cap. Selected 7 features. | 0.00 |
| FeatureEngineering | added 7 polynomial (sq) features | Degree-2 squares of top signals to model non-linear dynamics. | 0.00 |
| FeatureEngineering | added 2 ratio features | A/B ratios to capture scale-invariant interactions. | 0.00 |
| FeatureSelection | method=mutual_info, k=30 | 60 features, classification task — Mutual Information captures nonlinear feature-target dependencies, robust to outliers. | 0.00 |
| FeatureSelection | retained 30 / 60 features | Selected top 30 features via mutual_info. Dropped 30 features. | 0.00 |
| ImbalanceHandling | balanced (minority=37.36%) | Minority class is 37.36% of training data, above the 20% threshold. No resampling needed. | 0.00 |
| ModelSelection | candidates=['logistic_regression', 'random_forest', 'gradient_boosting'] | 569 rows / 30 features is small enough to try the full candidate set of classification models. | 0.00 |
| Tuning | n_trials=10 | Allocating 10 Optuna trials per model (3 models) given 569 rows -- balances search quality against the quadratic-ish cost of CV on larger data. | 0.00 |
| Tuning | cv_folds=3 | 569 rows -> using standard 3-fold CV for a reliable estimate. | 0.00 |
| Tuning | tuning_metric=roc_auc | Binary classification with balanced classes -> tuning with ROC-AUC. | 0.00 |
| Tuning | dummy baseline: roc_auc=0.5000 (+/- 0.0000) | Zero-intelligence baseline against which all candidate models are compared. | 0.00 |
| Tuning | logistic_regression: roc_auc=0.9926 (+/- 0.0078), delta=+0.4926 | Best of 10 Optuna trials with params {'C': 0.9846738873614566}. Improvement over dummy baseline: +0.4926. | 0.00 |
| Tuning | random_forest: roc_auc=0.9904 (+/- 0.0042), delta=+0.4904 | Best of 10 Optuna trials with params {'n_estimators': 20, 'max_depth': 3}. Improvement over dummy baseline: +0.4904. | 0.00 |
| Tuning | gradient_boosting: roc_auc=0.9909 (+/- 0.0034), delta=+0.4909 | Best of 10 Optuna trials with params {'n_estimators': 94, 'max_depth': 2, 'learning_rate': 0.2708160864249968}. Improvement over dummy baseline: +0.4909. | 0.00 |
| Evaluation | dummy test metrics | {'accuracy': 0.6316, 'precision_macro': 0.3158, 'recall_macro': 0.5, 'f1_macro': 0.3871, 'f1_weighted': 0.489, 'roc_auc': 0.5} | 0.00 |
| Evaluation | logistic_regression test metrics | {'accuracy': 0.9649, 'precision_macro': 0.9623, 'recall_macro': 0.9623, 'f1_macro': 0.9623, 'f1_weighted': 0.9649, 'roc_auc': 0.9957} | 0.00 |
| Evaluation | random_forest test metrics | {'accuracy': 0.9474, 'precision_macro': 0.9435, 'recall_macro': 0.9435, 'f1_macro': 0.9435, 'f1_weighted': 0.9474, 'roc_auc': 0.9883} | 0.00 |
| Evaluation | gradient_boosting test metrics | {'accuracy': 0.9561, 'precision_macro': 0.9551, 'recall_macro': 0.9504, 'f1_macro': 0.9526, 'f1_weighted': 0.956, 'roc_auc': 0.994} | 0.00 |
| Evaluation | final_model=logistic_regression | Selected logistic_regression with score 0.9649, the highest among 3 tuned candidates (runner-up gradient_boosting scored 0.9561). | 0.00 |
| Explainability | method=permutation importance | Chose permutation importance because model is LogisticRegression (non-tree, permutation importance). | 0.00 |
| ArtifactExport | Exported Cleaned and Engineered Datasets | Saved cleaned dataset (569 rows, 31 columns) to outputs\run_0114e7a4\cleaned_data.csv and feature matrix to outputs\run_0114e7a4\engineered_features.csv. | 0.00 |
| CSVExport | exported 7 CSV files | Generated structured CSV artifacts in outputs\run_0114e7a4\csv: dataset_profile, model_comparison, predictions, feature_importance, preprocessing_log, decision_log, anomalies | 0.00 |
