# AutoDS-Agent 2.0: Autonomous Data Science Report

*Generated on 2026-09-29 14:49:44 | Reasoner Engine: `rule-based` | Pipeline Execution Time: 11.45s*

> **Research Problem Statement**: To develop an autonomous and explainable data-science platform that can automatically analyze an unseen dataset, perform preprocessing and feature engineering, select and optimize appropriate machine-learning models, explain predictions, support scalable data processing and assist users through an AI-based data-science assistant.

## 1. Dataset Profile & Data Scale
- **Total Rows**: 569
- **Total Columns**: 31
- **In-Memory Size**: 0.13 MB
- **Inferred Task Type**: `classification`
- **Target Column**: `target`
- **Execution Backend Chosen**: `local`
- **Numeric Features**: 30
- **Categorical Features**: 0
- **Duplicates Detected**: 0
- **Constant Columns**: 0

## 2. Exploratory Data Analysis (Auto EDA)
Generated diagnostic plots:
- **Correlation Heatmap**: `sample_output\binary_classification\outputs\run_688ffb97\eda\correlation_heatmap.png`
- **Distributions**: `sample_output\binary_classification\outputs\run_688ffb97\eda\distributions.png`
- **Boxplots**: `sample_output\binary_classification\outputs\run_688ffb97\eda\boxplots.png`
- **Class Distribution**: `sample_output\binary_classification\outputs\run_688ffb97\eda\class_distribution.png`

**Outlier Summary (IQR method)**:
| Feature | Outlier Count | Outlier % |
|---|---|---|
| `mean radius` | 14 | 2.5% |
| `mean texture` | 7 | 1.2% |
| `mean perimeter` | 13 | 2.3% |
| `mean area` | 25 | 4.4% |
| `mean smoothness` | 6 | 1.1% |
| `mean compactness` | 16 | 2.8% |
| `mean concavity` | 18 | 3.2% |
| `mean concave points` | 10 | 1.8% |

## 3. Preprocessing, Anomaly & Imbalance Handling
**Preprocessing Decisions Applied**:
- StandardScaler applied to 30 numeric columns.

**Anomaly Detection**:
- Isolation Forest flagged: **5.10%** of records
- Local Outlier Factor flagged: **5.10%** of records
- Top anomalous row indices: `[0, 3, 9, 12, 38]`

**Class Imbalance Analysis**:
- Imbalance Detected: **No** (Minority class ratio: 37.4%)

## 4. Feature Selection
- Method: `mutual_info`
- Features Selected (30): `['worst perimeter', 'worst area', 'worst radius', 'mean concave points', 'worst concave points', 'mean perimeter', 'mean concavity', 'mean radius', 'mean area', 'area error', 'worst concavity', 'perimeter error', 'radius error', 'worst compactness', 'mean compactness']`...
- Features Dropped (30): `['mean symmetry', 'mean fractal dimension', 'texture error', 'smoothness error', 'symmetry error', 'fractal dimension error', 'worst fractal dimension', 'worst concave points_x_compactness error', 'worst concave points_x_concave points error', 'worst concave points_x_mean symmetry']`...

## 5. Model Tuning Leaderboard (Optuna CV)
| Candidate Model | Best CV Score | Trials | Best Hyperparameters |
|---|---|---|---|
| `logistic_regression` | 0.9781 | 10 | `{'C': 0.9846738873614566}` |
| `random_forest` | 0.9582 | 10 | `{'n_estimators': 36, 'max_depth': 7}` |
| `gradient_boosting` | 0.9561 | 10 | `{'n_estimators': 94, 'max_depth': 2, 'learning_rate': 0.2708160864249968}` |

## 6. Held-out Test Evaluation
**Final Selected Model**: `logistic_regression`

| Model | Metrics Summary |
|---|---|
| `logistic_regression` **[Selected]** | accuracy=0.9737, precision_macro=0.9697, recall_macro=0.9742, f1_macro=0.9719, f1_weighted=0.9737, roc_auc=0.9954 |
| `random_forest` | accuracy=0.9561, precision_macro=0.9551, recall_macro=0.9504, f1_macro=0.9526, f1_weighted=0.9560, roc_auc=0.9927 |
| `gradient_boosting` | accuracy=0.9649, precision_macro=0.9672, recall_macro=0.9573, f1_macro=0.9619, f1_weighted=0.9647, roc_auc=0.9960 |

**Evaluation Diagnostic Plots Saved**:
- Confusion Matrix: `sample_output\binary_classification\outputs\run_688ffb97\evaluation\confusion_matrix.png`
- Roc Curve: `sample_output\binary_classification\outputs\run_688ffb97\evaluation\roc_curve.png`
- Pr Curve: `sample_output\binary_classification\outputs\run_688ffb97\evaluation\pr_curve.png`

## 7. Model Explainability & Interpretability
- Interpretation Method: `permutation importance`

**Top 10 Most Influential Features**:
| Rank | Feature | Attribution Score |
|---|---|---|
| 1 | `radius error` | 0.0263 |
| 2 | `worst smoothness` | 0.0211 |
| 3 | `worst texture` | 0.0211 |
| 4 | `worst concavity` | 0.0193 |
| 5 | `area error` | 0.0175 |
| 6 | `worst area` | 0.0158 |
| 7 | `worst radius` | 0.0158 |
| 8 | `worst concave points` | 0.0158 |
| 9 | `worst perimeter` | 0.0140 |
| 10 | `mean concave points` | 0.0123 |

## 8. Model Registry & Versioning
- **Version Tag**: `v27`
- **Model Path**: `model_registry\run_688ffb97_logistic_regression_v27\model.joblib`
- **Metadata Path**: `model_registry\run_688ffb97_logistic_regression_v27\metadata.json`

## 9. CSV Outputs
The following structured CSV files were generated:

- **`dataset_profile.csv`**: `sample_output\binary_classification\outputs\run_688ffb97\csv\dataset_profile.csv` - *Dataset dimensions, scaling, missingness, and profiling metrics*
- **`model_comparison.csv`**: `sample_output\binary_classification\outputs\run_688ffb97\csv\model_comparison.csv` - *Full candidate model leaderboard with tuning parameters and held-out scores*
- **`predictions.csv`**: `sample_output\binary_classification\outputs\run_688ffb97\csv\predictions.csv` - *Held-out test set actuals, model predictions, and class probabilities/residuals*
- **`feature_importance.csv`**: `sample_output\binary_classification\outputs\run_688ffb97\csv\feature_importance.csv` - *Ranked SHAP attribution scores and feature importance hierarchy*
- **`preprocessing_log.csv`**: `sample_output\binary_classification\outputs\run_688ffb97\csv\preprocessing_log.csv` - *Complete audit of data transformations, imputations, and encodings*
- **`decision_log.csv`**: `sample_output\binary_classification\outputs\run_688ffb97\csv\decision_log.csv` - *Granular agent reasoning log and execution duration for every stage*
- **`anomalies.csv`**: `sample_output\binary_classification\outputs\run_688ffb97\csv\anomalies.csv` - *Isolation Forest & Local Outlier Factor anomaly scoring flags*

- **Download All CSVs (ZIP Archive)**: `sample_output\binary_classification\outputs\run_688ffb97\autods_run_688ffb97_csv_results.zip`

## 10. Autonomous Agent Audit Trail
Every decision made during the pipeline run is logged below with the agent's internal reasoning, timing, and operational parameters.

| Stage | Decision | Reasoning | Time (s) |
|---|---|---|---|
| Profiler | backend=local | Dataset is 0.1MB with 569 rows, within in-memory thresholds. Routing to the pandas/sklearn local path -- distributed overhead would not pay off here. | 0.00 |
| Profiler | task_type=classification | Target has dtype int32 with only 2 unique values -> treating as categorical, so this is a classification task. | 0.00 |
| AutoEDA | extended profile: 30 numeric, 0 categorical, 0 constant cols | Computed detailed dataset statistics: duplicates=0, high-cardinality cols=[], target subtypes=binary. | 0.00 |
| AutoEDA | generated 4 EDA plots | Plots saved to sample_output\binary_classification\outputs\run_688ffb97\eda: correlation_heatmap, distributions, boxplots, class_distribution. | 0.00 |
| Preprocessing[local] | complete | Dropped 0 sparse columns, one-hot 0, ordinal 0 categoricals, scaled 30 numeric columns. | 0.00 |
| AnomalyDetection | IF=5.1% anomalous, LOF=5.1% anomalous | Isolation Forest detected 5.1% anomalous rows; Local Outlier Factor detected 5.1% anomalous rows (contamination assumption=0.05). Anomalies flagged only – not removed (default behaviour). | 0.00 |
| FeatureEngineering | pairs from 7 cols | 30 features -> 435 possible pairwise interactions, exceeds the 30 cap. Sampling 7 columns to pair instead. | 0.00 |
| FeatureEngineering | added 7 polynomial (sq) features | Squares of top 7 numeric columns to capture nonlinear relationships; low cost on top of existing features. | 0.00 |
| FeatureEngineering | added 2 ratio features | A/B ratio features encode scale-invariant relationships between numeric columns; useful when relative magnitude matters more than absolute. | 0.00 |
| FeatureEngineering | total added: 30 engineered features | Created 21 interaction, 7 polynomial, 2 ratio features as low-cost nonlinear signals. | 0.00 |
| FeatureSelection | method=mutual_info, k=30 | 60 features, classification task — Mutual Information captures nonlinear feature-target dependencies, robust to outliers. | 0.00 |
| FeatureSelection | kept 30/60 features, dropped 30 | Top features by mutual_info: ['worst perimeter', 'worst area', 'worst radius', 'mean concave points', 'worst concave points', 'mean perimeter', 'mean concavity', 'mean radius', 'mean area', 'area error'].... Dropped: ['mean symmetry', 'mean fractal dimension', 'texture error', 'smoothness error', 'symmetry error', 'fractal dimension error', 'worst fractal dimension', 'worst concave points_x_compactness error', 'worst concave points_x_concave points error', 'worst concave points_x_mean symmetry'].... | 0.00 |
| ImbalanceHandling | balanced (minority=37.36%) | Minority class is 37.36% of training data, above the 20% threshold. No resampling needed. | 0.00 |
| ModelSelection | candidates=['logistic_regression', 'random_forest', 'gradient_boosting'] | 569 rows / 30 features is small enough to try the full candidate set of classification models. | 0.00 |
| Tuning | n_trials=10 | Allocating 10 Optuna trials per model (3 models) given 569 rows -- balances search quality against the quadratic-ish cost of CV on larger data. | 0.00 |
| Tuning | cv_folds=3 | 569 rows -- using standard 3-fold CV for a reliable estimate. | 0.00 |
| Tuning | logistic_regression: best accuracy=0.9781 | Best of 10 Optuna trials using params {'C': 0.9846738873614566}. | 0.00 |
| Tuning | random_forest: best accuracy=0.9582 | Best of 10 Optuna trials using params {'n_estimators': 36, 'max_depth': 7}. | 0.00 |
| Tuning | gradient_boosting: best accuracy=0.9561 | Best of 10 Optuna trials using params {'n_estimators': 94, 'max_depth': 2, 'learning_rate': 0.2708160864249968}. | 0.00 |
| Evaluation | logistic_regression test metrics | {'accuracy': 0.9737, 'precision_macro': 0.9697, 'recall_macro': 0.9742, 'f1_macro': 0.9719, 'f1_weighted': 0.9737, 'roc_auc': 0.9954} | 0.00 |
| Evaluation | random_forest test metrics | {'accuracy': 0.9561, 'precision_macro': 0.9551, 'recall_macro': 0.9504, 'f1_macro': 0.9526, 'f1_weighted': 0.956, 'roc_auc': 0.9927} | 0.00 |
| Evaluation | gradient_boosting test metrics | {'accuracy': 0.9649, 'precision_macro': 0.9672, 'recall_macro': 0.9573, 'f1_macro': 0.9619, 'f1_weighted': 0.9647, 'roc_auc': 0.996} | 0.00 |
| Evaluation | final_model=logistic_regression | Selected logistic_regression with score 0.9737, the highest among 3 tuned candidates (runner-up gradient_boosting scored 0.9649). | 0.00 |
| Explainability | method=permutation importance | Chose permutation importance because model is LogisticRegression (non-tree, permutation importance). | 0.00 |
| ArtifactExport | Exported Cleaned and Engineered Datasets | Saved cleaned dataset (569 rows, 31 columns) to sample_output\binary_classification\outputs\run_688ffb97\cleaned_data.csv and feature matrix to sample_output\binary_classification\outputs\run_688ffb97\engineered_features.csv. | 0.00 |
| CSVExport | exported 7 CSV files | Generated structured CSV artifacts in sample_output\binary_classification\outputs\run_688ffb97\csv: dataset_profile, model_comparison, predictions, feature_importance, preprocessing_log, decision_log, anomalies | 0.00 |
