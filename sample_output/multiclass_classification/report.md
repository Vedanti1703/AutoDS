# AutoDS-Agent 2.0: Autonomous Data Science Report

*Generated on 2026-09-29 14:49:57 | Reasoner Engine: `rule-based` | Pipeline Execution Time: 12.54s*

> **Research Problem Statement**: To develop an autonomous and explainable data-science platform that can automatically analyze an unseen dataset, perform preprocessing and feature engineering, select and optimize appropriate machine-learning models, explain predictions, support scalable data processing and assist users through an AI-based data-science assistant.

## 1. Dataset Profile & Data Scale
- **Total Rows**: 178
- **Total Columns**: 14
- **In-Memory Size**: 0.02 MB
- **Inferred Task Type**: `classification`
- **Target Column**: `target`
- **Execution Backend Chosen**: `local`
- **Numeric Features**: 13
- **Categorical Features**: 0
- **Duplicates Detected**: 0
- **Constant Columns**: 0

## 2. Exploratory Data Analysis (Auto EDA)
Generated diagnostic plots:
- **Correlation Heatmap**: `sample_output\multiclass_classification\outputs\run_a5301837\eda\correlation_heatmap.png`
- **Distributions**: `sample_output\multiclass_classification\outputs\run_a5301837\eda\distributions.png`
- **Boxplots**: `sample_output\multiclass_classification\outputs\run_a5301837\eda\boxplots.png`
- **Class Distribution**: `sample_output\multiclass_classification\outputs\run_a5301837\eda\class_distribution.png`

**Outlier Summary (IQR method)**:
| Feature | Outlier Count | Outlier % |
|---|---|---|
| `malic_acid` | 3 | 1.7% |
| `ash` | 3 | 1.7% |
| `alcalinity_of_ash` | 4 | 2.2% |
| `magnesium` | 4 | 2.2% |
| `proanthocyanins` | 2 | 1.1% |
| `color_intensity` | 4 | 2.2% |
| `hue` | 1 | 0.6% |

## 3. Preprocessing, Anomaly & Imbalance Handling
**Preprocessing Decisions Applied**:
- StandardScaler applied to 13 numeric columns.

**Anomaly Detection**:
- Isolation Forest flagged: **5.06%** of records
- Local Outlier Factor flagged: **5.06%** of records
- Top anomalous row indices: `[59, 69, 73, 78, 95]`

**Class Imbalance Analysis**:
- Imbalance Detected: **No** (Minority class ratio: 26.8%)

## 4. Feature Selection
- Method: `mutual_info`
- Features Selected (21): `['flavanoids', 'proline', 'color_intensity', 'od280/od315_of_diluted_wines', 'alcohol', 'hue', 'total_phenols', 'color_intensity_x_malic_acid', 'od280/od315_of_diluted_wines_div_alcohol', 'proanthocyanins', 'malic_acid', 'color_intensity_sq', 'od280/od315_of_diluted_wines_x_color_intensity', 'alcalinity_of_ash', 'magnesium']`...
- Features Dropped (22): `['ash', 'nonflavanoid_phenols', 'od280/od315_of_diluted_wines_x_proanthocyanins', 'od280/od315_of_diluted_wines_x_ash', 'color_intensity_x_alcohol', 'color_intensity_x_proanthocyanins', 'color_intensity_x_ash', 'alcohol_x_proanthocyanins', 'alcohol_x_total_phenols', 'alcohol_x_ash']`...

## 5. Model Tuning Leaderboard (Optuna CV)
| Candidate Model | Best CV Score | Trials | Best Hyperparameters |
|---|---|---|---|
| `logistic_regression` | 0.9718 | 10 | `{'C': 0.9846738873614566}` |
| `random_forest` | 0.9577 | 10 | `{'n_estimators': 28, 'max_depth': 8}` |
| `gradient_boosting` | 0.9437 | 10 | `{'n_estimators': 94, 'max_depth': 2, 'learning_rate': 0.2708160864249968}` |

## 6. Held-out Test Evaluation
**Final Selected Model**: `random_forest`

| Model | Metrics Summary |
|---|---|
| `logistic_regression` | accuracy=0.9722, precision_macro=0.9778, recall_macro=0.9667, f1_macro=0.9710, f1_weighted=0.9720, roc_auc=0.9989 |
| `random_forest` **[Selected]** | accuracy=1.0000, precision_macro=1.0000, recall_macro=1.0000, f1_macro=1.0000, f1_weighted=1.0000, roc_auc=1.0000 |
| `gradient_boosting` | accuracy=1.0000, precision_macro=1.0000, recall_macro=1.0000, f1_macro=1.0000, f1_weighted=1.0000, roc_auc=1.0000 |

**Evaluation Diagnostic Plots Saved**:
- Confusion Matrix: `sample_output\multiclass_classification\outputs\run_a5301837\evaluation\confusion_matrix.png`
- Roc Curve: `sample_output\multiclass_classification\outputs\run_a5301837\evaluation\roc_curve.png`
- Pr Curve: `sample_output\multiclass_classification\outputs\run_a5301837\evaluation\pr_curve.png`

## 7. Model Explainability & Interpretability
- Interpretation Method: `SHAP TreeExplainer`

**Top 10 Most Influential Features**:
| Rank | Feature | Attribution Score |
|---|---|---|
| 1 | `proline` | 0.0791 |
| 2 | `color_intensity` | 0.0747 |
| 3 | `od280/od315_of_diluted_wines` | 0.0711 |
| 4 | `alcohol` | 0.0602 |
| 5 | `flavanoids` | 0.0436 |
| 6 | `hue` | 0.0383 |
| 7 | `color_intensity_x_malic_acid` | 0.0195 |
| 8 | `total_phenols` | 0.0157 |
| 9 | `magnesium` | 0.0138 |
| 10 | `alcalinity_of_ash` | 0.0120 |

**Explainability Visualizations**:
- Global Shap: `sample_output\multiclass_classification\outputs\run_a5301837\explainability\global_shap.png`
- Local Shap: `sample_output\multiclass_classification\outputs\run_a5301837\explainability\local_shap.png`

## 8. Model Registry & Versioning
- **Version Tag**: `v8`
- **Model Path**: `model_registry\run_a5301837_random_forest_v8\model.joblib`
- **Metadata Path**: `model_registry\run_a5301837_random_forest_v8\metadata.json`

## 9. CSV Outputs
The following structured CSV files were generated:

- **`dataset_profile.csv`**: `sample_output\multiclass_classification\outputs\run_a5301837\csv\dataset_profile.csv` - *Dataset dimensions, scaling, missingness, and profiling metrics*
- **`model_comparison.csv`**: `sample_output\multiclass_classification\outputs\run_a5301837\csv\model_comparison.csv` - *Full candidate model leaderboard with tuning parameters and held-out scores*
- **`predictions.csv`**: `sample_output\multiclass_classification\outputs\run_a5301837\csv\predictions.csv` - *Held-out test set actuals, model predictions, and class probabilities/residuals*
- **`feature_importance.csv`**: `sample_output\multiclass_classification\outputs\run_a5301837\csv\feature_importance.csv` - *Ranked SHAP attribution scores and feature importance hierarchy*
- **`preprocessing_log.csv`**: `sample_output\multiclass_classification\outputs\run_a5301837\csv\preprocessing_log.csv` - *Complete audit of data transformations, imputations, and encodings*
- **`decision_log.csv`**: `sample_output\multiclass_classification\outputs\run_a5301837\csv\decision_log.csv` - *Granular agent reasoning log and execution duration for every stage*
- **`anomalies.csv`**: `sample_output\multiclass_classification\outputs\run_a5301837\csv\anomalies.csv` - *Isolation Forest & Local Outlier Factor anomaly scoring flags*

- **Download All CSVs (ZIP Archive)**: `sample_output\multiclass_classification\outputs\run_a5301837\autods_run_a5301837_csv_results.zip`

## 10. Autonomous Agent Audit Trail
Every decision made during the pipeline run is logged below with the agent's internal reasoning, timing, and operational parameters.

| Stage | Decision | Reasoning | Time (s) |
|---|---|---|---|
| Profiler | backend=local | Dataset is 0.0MB with 178 rows, within in-memory thresholds. Routing to the pandas/sklearn local path -- distributed overhead would not pay off here. | 0.00 |
| Profiler | task_type=classification | Target has dtype int32 with only 3 unique values -> treating as categorical, so this is a classification task. | 0.00 |
| AutoEDA | extended profile: 13 numeric, 0 categorical, 0 constant cols | Computed detailed dataset statistics: duplicates=0, high-cardinality cols=[], target subtypes=multiclass. | 0.00 |
| AutoEDA | generated 4 EDA plots | Plots saved to sample_output\multiclass_classification\outputs\run_a5301837\eda: correlation_heatmap, distributions, boxplots, class_distribution. | 0.00 |
| Preprocessing[local] | complete | Dropped 0 sparse columns, one-hot 0, ordinal 0 categoricals, scaled 13 numeric columns. | 0.00 |
| AnomalyDetection | IF=5.06% anomalous, LOF=5.06% anomalous | Isolation Forest detected 5.06% anomalous rows; Local Outlier Factor detected 5.06% anomalous rows (contamination assumption=0.05). Anomalies flagged only – not removed (default behaviour). | 0.00 |
| FeatureEngineering | pairs from 7 cols | 13 features -> 78 possible pairwise interactions, exceeds the 30 cap. Sampling 7 columns to pair instead. | 0.00 |
| FeatureEngineering | added 7 polynomial (sq) features | Squares of top 7 numeric columns to capture nonlinear relationships; low cost on top of existing features. | 0.00 |
| FeatureEngineering | added 2 ratio features | A/B ratio features encode scale-invariant relationships between numeric columns; useful when relative magnitude matters more than absolute. | 0.00 |
| FeatureEngineering | total added: 30 engineered features | Created 21 interaction, 7 polynomial, 2 ratio features as low-cost nonlinear signals. | 0.00 |
| FeatureSelection | method=mutual_info, k=21 | 43 features, classification task — Mutual Information captures nonlinear feature-target dependencies, robust to outliers. | 0.00 |
| FeatureSelection | kept 21/43 features, dropped 22 | Top features by mutual_info: ['flavanoids', 'proline', 'color_intensity', 'od280/od315_of_diluted_wines', 'alcohol', 'hue', 'total_phenols', 'color_intensity_x_malic_acid', 'od280/od315_of_diluted_wines_div_alcohol', 'proanthocyanins'].... Dropped: ['ash', 'nonflavanoid_phenols', 'od280/od315_of_diluted_wines_x_proanthocyanins', 'od280/od315_of_diluted_wines_x_ash', 'color_intensity_x_alcohol', 'color_intensity_x_proanthocyanins', 'color_intensity_x_ash', 'alcohol_x_proanthocyanins', 'alcohol_x_total_phenols', 'alcohol_x_ash'].... | 0.00 |
| ImbalanceHandling | balanced (minority=26.76%) | Minority class is 26.76% of training data, above the 20% threshold. No resampling needed. | 0.00 |
| ModelSelection | candidates=['logistic_regression', 'random_forest', 'gradient_boosting'] | 178 rows / 13 features is small enough to try the full candidate set of classification models. | 0.00 |
| Tuning | n_trials=10 | Allocating 10 Optuna trials per model (3 models) given 178 rows -- balances search quality against the quadratic-ish cost of CV on larger data. | 0.00 |
| Tuning | cv_folds=3 | 178 rows -- using standard 3-fold CV for a reliable estimate. | 0.00 |
| Tuning | logistic_regression: best accuracy=0.9718 | Best of 10 Optuna trials using params {'C': 0.9846738873614566}. | 0.00 |
| Tuning | random_forest: best accuracy=0.9577 | Best of 10 Optuna trials using params {'n_estimators': 28, 'max_depth': 8}. | 0.00 |
| Tuning | gradient_boosting: best accuracy=0.9437 | Best of 10 Optuna trials using params {'n_estimators': 94, 'max_depth': 2, 'learning_rate': 0.2708160864249968}. | 0.00 |
| Evaluation | logistic_regression test metrics | {'accuracy': 0.9722, 'precision_macro': 0.9778, 'recall_macro': 0.9667, 'f1_macro': 0.971, 'f1_weighted': 0.972, 'roc_auc': 0.9989} | 0.00 |
| Evaluation | random_forest test metrics | {'accuracy': 1.0, 'precision_macro': 1.0, 'recall_macro': 1.0, 'f1_macro': 1.0, 'f1_weighted': 1.0, 'roc_auc': 1.0} | 0.00 |
| Evaluation | gradient_boosting test metrics | {'accuracy': 1.0, 'precision_macro': 1.0, 'recall_macro': 1.0, 'f1_macro': 1.0, 'f1_weighted': 1.0, 'roc_auc': 1.0} | 0.00 |
| Evaluation | final_model=random_forest | Selected random_forest with score 1.0000, the highest among 3 tuned candidates (runner-up gradient_boosting scored 1.0000). | 0.00 |
| Explainability | method=SHAP TreeExplainer | Chose SHAP TreeExplainer because model is RandomForestClassifier (tree-based, SHAP exact). | 0.00 |
| ArtifactExport | Exported Cleaned and Engineered Datasets | Saved cleaned dataset (178 rows, 14 columns) to sample_output\multiclass_classification\outputs\run_a5301837\cleaned_data.csv and feature matrix to sample_output\multiclass_classification\outputs\run_a5301837\engineered_features.csv. | 0.00 |
| CSVExport | exported 7 CSV files | Generated structured CSV artifacts in sample_output\multiclass_classification\outputs\run_a5301837\csv: dataset_profile, model_comparison, predictions, feature_importance, preprocessing_log, decision_log, anomalies | 0.00 |
