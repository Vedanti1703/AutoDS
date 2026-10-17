# AutoDS-Agent 2.0: Autonomous Data Science Report

*Generated on 2026-09-29 14:50:37 | Reasoner Engine: `rule-based` | Pipeline Execution Time: 31.19s*

> **Research Problem Statement**: To develop an autonomous and explainable data-science platform that can automatically analyze an unseen dataset, perform preprocessing and feature engineering, select and optimize appropriate machine-learning models, explain predictions, support scalable data processing and assist users through an AI-based data-science assistant.

## 1. Dataset Profile & Data Scale
- **Total Rows**: 1,000
- **Total Columns**: 13
- **In-Memory Size**: 0.10 MB
- **Inferred Task Type**: `classification`
- **Target Column**: `label`
- **Execution Backend Chosen**: `local`
- **Numeric Features**: 12
- **Categorical Features**: 0
- **Duplicates Detected**: 0
- **Constant Columns**: 0

## 2. Exploratory Data Analysis (Auto EDA)
Generated diagnostic plots:
- **Correlation Heatmap**: `sample_output\imbalanced_classification\outputs\run_562617e0\eda\correlation_heatmap.png`
- **Distributions**: `sample_output\imbalanced_classification\outputs\run_562617e0\eda\distributions.png`
- **Boxplots**: `sample_output\imbalanced_classification\outputs\run_562617e0\eda\boxplots.png`
- **Class Distribution**: `sample_output\imbalanced_classification\outputs\run_562617e0\eda\class_distribution.png`

**Outlier Summary (IQR method)**:
| Feature | Outlier Count | Outlier % |
|---|---|---|
| `feat_0` | 6 | 0.6% |
| `feat_1` | 20 | 2.0% |
| `feat_2` | 5 | 0.5% |
| `feat_3` | 13 | 1.3% |
| `feat_4` | 3 | 0.3% |
| `feat_5` | 6 | 0.6% |
| `feat_6` | 5 | 0.5% |
| `feat_7` | 6 | 0.6% |

## 3. Preprocessing, Anomaly & Imbalance Handling
**Preprocessing Decisions Applied**:
- StandardScaler applied to 12 numeric columns.

**Anomaly Detection**:
- Isolation Forest flagged: **5.00%** of records
- Local Outlier Factor flagged: **5.00%** of records
- Top anomalous row indices: `[56, 58, 68, 93, 112]`

**Class Imbalance Analysis**:
- Imbalance Detected: **Yes** (Minority class ratio: 5.4%)
- Selected Resampling Strategy: **`baseline`**

| Strategy | Macro F1 | Recall | ROC-AUC |
|---|---|---|---|
| `baseline` | 0.0000 | 0.0000 | 0.9024 |
| `class_weight_balanced` | 0.0000 | 0.0000 | 0.9024 |
| `SMOTE` | 0.0000 | 0.0000 | 0.8980 |
| `RandomOverSampler` | 0.0000 | 0.0000 | 0.8740 |
| `RandomUnderSampler` | 0.0000 | 0.0000 | 0.8793 |

## 4. Feature Selection
- Method: `mutual_info`
- Features Selected (21): `['feat_1', 'feat_1_sq', 'feat_8_sq', 'feat_2_x_feat_1', 'feat_0_x_feat_5', 'feat_10_x_feat_2', 'feat_5_sq', 'feat_8', 'feat_10_div_feat_9', 'feat_4', 'feat_0_x_feat_8', 'feat_8_x_feat_5', 'feat_11', 'feat_10_x_feat_9', 'feat_9_x_feat_0']`...
- Features Dropped (21): `['feat_0', 'feat_2', 'feat_3', 'feat_6', 'feat_7', 'feat_9', 'feat_10_x_feat_0', 'feat_10_x_feat_8', 'feat_10_x_feat_5', 'feat_10_x_feat_1']`...

## 5. Model Tuning Leaderboard (Optuna CV)
| Candidate Model | Best CV Score | Trials | Best Hyperparameters |
|---|---|---|---|
| `logistic_regression` | 0.9488 | 10 | `{'C': 0.0745934328572655}` |
| `random_forest` | 0.9475 | 10 | `{'n_estimators': 20, 'max_depth': 3}` |
| `gradient_boosting` | 0.9475 | 10 | `{'n_estimators': 105, 'max_depth': 2, 'learning_rate': 0.01855998084649059}` |

## 6. Held-out Test Evaluation
**Final Selected Model**: `logistic_regression`

| Model | Metrics Summary |
|---|---|
| `logistic_regression` **[Selected]** | accuracy=0.9500, precision_macro=0.9749, recall_macro=0.5455, f1_macro=0.5704, f1_weighted=0.9298, roc_auc=0.8975 |
| `random_forest` | accuracy=0.9500, precision_macro=0.9749, recall_macro=0.5455, f1_macro=0.5704, f1_weighted=0.9298, roc_auc=0.7174 |
| `gradient_boosting` | accuracy=0.9500, precision_macro=0.9749, recall_macro=0.5455, f1_macro=0.5704, f1_weighted=0.9298, roc_auc=0.7727 |

**Evaluation Diagnostic Plots Saved**:
- Confusion Matrix: `sample_output\imbalanced_classification\outputs\run_562617e0\evaluation\confusion_matrix.png`
- Roc Curve: `sample_output\imbalanced_classification\outputs\run_562617e0\evaluation\roc_curve.png`
- Pr Curve: `sample_output\imbalanced_classification\outputs\run_562617e0\evaluation\pr_curve.png`

## 7. Model Explainability & Interpretability
- Interpretation Method: `permutation importance`

**Top 10 Most Influential Features**:
| Rank | Feature | Attribution Score |
|---|---|---|
| 1 | `feat_1_sq` | 0.0050 |
| 2 | `feat_1` | 0.0040 |
| 3 | `feat_11` | 0.0030 |
| 4 | `feat_9_sq` | 0.0020 |
| 5 | `feat_10` | 0.0010 |
| 6 | `feat_5_sq` | 0.0010 |
| 7 | `feat_10_x_feat_9` | 0.0010 |
| 8 | `feat_4` | 0.0010 |
| 9 | `feat_5` | 0.0000 |
| 10 | `feat_2_sq` | 0.0000 |

## 8. Model Registry & Versioning
- **Version Tag**: `v30`
- **Model Path**: `model_registry\run_562617e0_logistic_regression_v30\model.joblib`
- **Metadata Path**: `model_registry\run_562617e0_logistic_regression_v30\metadata.json`

## 9. CSV Outputs
The following structured CSV files were generated:

- **`dataset_profile.csv`**: `sample_output\imbalanced_classification\outputs\run_562617e0\csv\dataset_profile.csv` - *Dataset dimensions, scaling, missingness, and profiling metrics*
- **`model_comparison.csv`**: `sample_output\imbalanced_classification\outputs\run_562617e0\csv\model_comparison.csv` - *Full candidate model leaderboard with tuning parameters and held-out scores*
- **`predictions.csv`**: `sample_output\imbalanced_classification\outputs\run_562617e0\csv\predictions.csv` - *Held-out test set actuals, model predictions, and class probabilities/residuals*
- **`feature_importance.csv`**: `sample_output\imbalanced_classification\outputs\run_562617e0\csv\feature_importance.csv` - *Ranked SHAP attribution scores and feature importance hierarchy*
- **`preprocessing_log.csv`**: `sample_output\imbalanced_classification\outputs\run_562617e0\csv\preprocessing_log.csv` - *Complete audit of data transformations, imputations, and encodings*
- **`decision_log.csv`**: `sample_output\imbalanced_classification\outputs\run_562617e0\csv\decision_log.csv` - *Granular agent reasoning log and execution duration for every stage*
- **`anomalies.csv`**: `sample_output\imbalanced_classification\outputs\run_562617e0\csv\anomalies.csv` - *Isolation Forest & Local Outlier Factor anomaly scoring flags*

- **Download All CSVs (ZIP Archive)**: `sample_output\imbalanced_classification\outputs\run_562617e0\autods_run_562617e0_csv_results.zip`

## 10. Autonomous Agent Audit Trail
Every decision made during the pipeline run is logged below with the agent's internal reasoning, timing, and operational parameters.

| Stage | Decision | Reasoning | Time (s) |
|---|---|---|---|
| Profiler | backend=local | Dataset is 0.1MB with 1,000 rows, within in-memory thresholds. Routing to the pandas/sklearn local path -- distributed overhead would not pay off here. | 0.00 |
| Profiler | task_type=classification | Target has dtype int32 with only 2 unique values -> treating as categorical, so this is a classification task. | 0.00 |
| AutoEDA | extended profile: 12 numeric, 0 categorical, 0 constant cols | Computed detailed dataset statistics: duplicates=0, high-cardinality cols=[], target subtypes=binary. | 0.00 |
| AutoEDA | generated 4 EDA plots | Plots saved to sample_output\imbalanced_classification\outputs\run_562617e0\eda: correlation_heatmap, distributions, boxplots, class_distribution. | 0.00 |
| Preprocessing[local] | complete | Dropped 0 sparse columns, one-hot 0, ordinal 0 categoricals, scaled 12 numeric columns. | 0.00 |
| AnomalyDetection | IF=5.0% anomalous, LOF=5.0% anomalous | Isolation Forest detected 5.0% anomalous rows; Local Outlier Factor detected 5.0% anomalous rows (contamination assumption=0.05). Anomalies flagged only – not removed (default behaviour). | 0.00 |
| FeatureEngineering | pairs from 7 cols | 12 features -> 66 possible pairwise interactions, exceeds the 30 cap. Sampling 7 columns to pair instead. | 0.00 |
| FeatureEngineering | added 7 polynomial (sq) features | Squares of top 7 numeric columns to capture nonlinear relationships; low cost on top of existing features. | 0.00 |
| FeatureEngineering | added 2 ratio features | A/B ratio features encode scale-invariant relationships between numeric columns; useful when relative magnitude matters more than absolute. | 0.00 |
| FeatureEngineering | total added: 30 engineered features | Created 21 interaction, 7 polynomial, 2 ratio features as low-cost nonlinear signals. | 0.00 |
| FeatureSelection | method=mutual_info, k=21 | 42 features, classification task — Mutual Information captures nonlinear feature-target dependencies, robust to outliers. | 0.00 |
| FeatureSelection | kept 21/42 features, dropped 21 | Top features by mutual_info: ['feat_1', 'feat_1_sq', 'feat_8_sq', 'feat_2_x_feat_1', 'feat_0_x_feat_5', 'feat_10_x_feat_2', 'feat_5_sq', 'feat_8', 'feat_10_div_feat_9', 'feat_4'].... Dropped: ['feat_0', 'feat_2', 'feat_3', 'feat_6', 'feat_7', 'feat_9', 'feat_10_x_feat_0', 'feat_10_x_feat_8', 'feat_10_x_feat_5', 'feat_10_x_feat_1'].... | 0.00 |
| ImbalanceHandling | imbalanced detected (minority=5.38%) | Minority class is 5.38% of training data (<20%). Experimenting with class_weight, SMOTE, RandomOverSampler, RandomUnderSampler. | 0.00 |
| ImbalanceHandling | best_strategy=baseline (macro-F1=0.6299) | Compared 5 strategies; baseline gave the highest macro-F1=0.6299. Returning its resampled training set for downstream training. | 0.00 |
| ModelSelection | candidates=['logistic_regression', 'random_forest', 'gradient_boosting'] | 1,000 rows / 12 features is small enough to try the full candidate set of classification models. | 0.00 |
| Tuning | n_trials=10 | Allocating 10 Optuna trials per model (3 models) given 1,000 rows -- balances search quality against the quadratic-ish cost of CV on larger data. | 0.00 |
| Tuning | cv_folds=3 | 1,000 rows -- using standard 3-fold CV for a reliable estimate. | 0.00 |
| Tuning | logistic_regression: best accuracy=0.9488 | Best of 10 Optuna trials using params {'C': 0.0745934328572655}. | 0.00 |
| Tuning | random_forest: best accuracy=0.9475 | Best of 10 Optuna trials using params {'n_estimators': 20, 'max_depth': 3}. | 0.00 |
| Tuning | gradient_boosting: best accuracy=0.9475 | Best of 10 Optuna trials using params {'n_estimators': 105, 'max_depth': 2, 'learning_rate': 0.01855998084649059}. | 0.00 |
| Evaluation | logistic_regression test metrics | {'accuracy': 0.95, 'precision_macro': 0.9749, 'recall_macro': 0.5455, 'f1_macro': 0.5704, 'f1_weighted': 0.9298, 'roc_auc': 0.8975} | 0.00 |
| Evaluation | random_forest test metrics | {'accuracy': 0.95, 'precision_macro': 0.9749, 'recall_macro': 0.5455, 'f1_macro': 0.5704, 'f1_weighted': 0.9298, 'roc_auc': 0.7174} | 0.00 |
| Evaluation | gradient_boosting test metrics | {'accuracy': 0.95, 'precision_macro': 0.9749, 'recall_macro': 0.5455, 'f1_macro': 0.5704, 'f1_weighted': 0.9298, 'roc_auc': 0.7727} | 0.00 |
| Evaluation | final_model=logistic_regression | Selected logistic_regression with score 0.9500, the highest among 3 tuned candidates (runner-up random_forest scored 0.9500). | 0.00 |
| Explainability | method=permutation importance | Chose permutation importance because model is LogisticRegression (non-tree, permutation importance). | 0.00 |
| ArtifactExport | Exported Cleaned and Engineered Datasets | Saved cleaned dataset (1000 rows, 13 columns) to sample_output\imbalanced_classification\outputs\run_562617e0\cleaned_data.csv and feature matrix to sample_output\imbalanced_classification\outputs\run_562617e0\engineered_features.csv. | 0.00 |
| CSVExport | exported 7 CSV files | Generated structured CSV artifacts in sample_output\imbalanced_classification\outputs\run_562617e0\csv: dataset_profile, model_comparison, predictions, feature_importance, preprocessing_log, decision_log, anomalies | 0.00 |
