# AutoDS-Agent 2.0: An Explainable and Scalable Autonomous Data Science Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest%20passing-brightgreen.svg)]()
[![AutoML](https://img.shields.io/badge/AutoML-Optuna%20%2B%20Dask-brightgreen.svg)]()

AutoDS-Agent 2.0 is an autonomous, explainable, and scale-aware multi-agent data science platform that automates the end-to-end machine learning lifecycle from raw tabular data ingestion to continuous production monitoring.

---

## 🎯 Research Problem Statement

> *"To develop an autonomous and explainable data-science platform that can automatically analyze an unseen dataset, perform preprocessing and feature engineering, select and optimize appropriate machine-learning models, explain predictions, support scalable data processing and assist users through an AI-based data-science assistant."*

---

## 🏗️ System Architecture

```
                                    +-----------------------+
                                    | Raw Tabular Data CSV  |
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | DataProfilerAgent     | -> Local vs Distributed Execution Routing
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | AutoEDAAgent          | -> Diagnostic Plots (Correlation, Distributions, Outliers)
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | PreprocessingAgent    | -> Imputation (Mean/Median/Mode), Cardinality-Aware Encoding
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | AnomalyDetectionAgent | -> Isolation Forest & Local Outlier Factor (LOF)
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | ImbalanceHandlerAgent | -> SMOTE / RandomOverSampler / Class Weighting
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | FeatureEngineering    | -> Interaction Pairs, Degree-2 Polynomials, Ratios
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | FeatureSelectionAgent | -> Mutual Information (MI), SelectKBest, Tree Importance
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | ModelSelectionAgent   | -> Multi-family Candidate Search (Trees, Boosting, Linear)
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | HyperparameterTuning  | -> Scale-Adaptive Optuna Bayesian Optimization (TPE)
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | EvaluationAgent       | -> Held-out Metrics, Confusion Matrix, ROC/PR Curves
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | ExplainabilityAgent   | -> Global & Local SHAP Waterfall + Permutation Fallback
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | WhatIfAgent           | -> Real-time Counterfactual Feature Tweak Analysis
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | ModelRegistry         | -> Persistent Joblib Storage + JSON Metadata Sidecar
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | DriftDetectionAgent   | -> KS-Test, Chi-Square, PSI & Auto-Retraining Gating
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | DataScienceAssistant  | -> Context-Aware Conversational Assistant
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | ReportingAgent        | -> Complete Audit Trail Markdown & HTML Report
                                    +-----------------------+
```

---

## 🚀 Key Features by Phase

### 1. Automated Profiling & Exploratory Data Analysis (EDA)
- **Scale-Aware Profiling**: Ingests arbitrary CSVs, infers task type (classification vs. regression, binary vs. multiclass), computes in-memory size and cardinality.
- **Diagnostic Visualizations**: Generates and exports publication-ready PNGs for correlation heatmaps, feature distribution histograms, box plots by class, class distribution charts, and IQR-based outlier summaries.

### 2. Intelligent Preprocessing & Data Quality
- **Skew-Aware Imputation**: Applies mean imputation for near-symmetric numerical features ($|\text{skew}| < 1.0$), median for skewed distributions, and mode for categorical features. Drops columns exceeding 30% missingness.
- **Cardinality-Aware Encoding**: One-hot encodes low-cardinality nominal variables ($\le 10$ unique levels) and ordinal encodes high-cardinality features.
- **Duplicate Handling**: Automatically detects, logs, and prunes redundant records.
- **Explainable Decisions**: Records every cleaning action with explicit scientific justification in the `RunLog`.

### 3. Anomaly Detection & Imbalance Handling
- **Multi-Detector Anomaly Scoring**: Executes Isolation Forest and Local Outlier Factor (LOF) concurrently. Identifies outlier percentages and flags top anomaly indices (configurable removal).
- **Class Imbalance Strategy Search**: Detects minority class ratios $< 20\%$ and benchmarks `class_weight="balanced"`, `SMOTE`, `RandomOverSampler`, and `RandomUnderSampler` on training folds to prevent data leakage. Automatically adopts the highest macro-F1 strategy.

### 4. Feature Engineering & Feature Selection
- **Non-Linear Transformations**: Computes budget-capped pairwise interaction features, degree-2 polynomial squares of correlated signals, and ratio features.
- **Statistical Feature Selection**: Uses Mutual Information (MI), univariate $F$-tests (`SelectKBest`), and Random Forest feature importance to select the top-$K$ most informative features.

### 5. Multi-Family AutoML & Optuna Bayesian Optimization
- **Diverse Candidate Families**:
  - *Classification*: Logistic Regression, Random Forest, Gradient Boosting, Histogram Gradient Boosting.
  - *Regression*: Ridge, Random Forest, Gradient Boosting, Histogram Gradient Boosting.
- **Scale-Adaptive Budgeting**: Automatically scales Optuna trials (5 to 10) and cross-validation folds (2 to 3) based on row volume and memory constraints.

### 6. Comprehensive Model Evaluation & Diagnostic Plots
- **Classification Suite**: Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1 (Macro/Weighted), ROC-AUC (Multi-class OvR), Confusion Matrix, ROC curves, and Precision-Recall curves.
- **Regression Suite**: MAE, MSE, RMSE, $R^2$, actual vs. predicted scatter plots, and residual distribution plots.

### 7. Explainable AI (XAI) with SHAP
- **Global Explanations**: Summary bar plots showing mean absolute SHAP values across all features.
- **Local Explanations**: Waterfall attributions for individual records detailing positive and negative feature contributions.
- **Robust Fallback**: Permutation feature importance fallback if SHAP tree extraction encounters non-standard estimators.

### 8. Interactive What-If Counterfactual Analysis
- **Dynamic Prediction Form**: Evaluates single-record predictions and class probabilities on the fly.
- **Side-by-Side Comparison**: Tweak feature inputs to evaluate counterfactual differences and identify driving features.

### 9. Context-Aware AI Data Scientist Assistant
- **Fact-Grounded Q&A**: Queries an embedded run-context object containing real metrics, leaderboard results, SHAP scores, and anomaly counts.
- **Dual Reasoner Backend**:
  - *RuleBasedAnswerer* (Default): Deterministic, offline, and template-driven. Never hallucinates metrics.
  - *LLMAnswerer* (Optional with `USE_LLM=1`): Interfaces with Claude via Anthropic API for conversational explanations with automated rule-based fallback.

### 10. Big Data & Scalability Benchmarking
- **Volume-Aware Routing**: Routes workloads $< 500\text{MB}$ / $1\text{M}$ rows to in-memory pandas/scikit-learn and larger workloads to distributed Dask DataFrames.
- **Benchmark Suite (`run_scalability_experiment.py`)**: Evaluates execution time, peak memory usage (via `psutil`), and throughput across synthetic datasets (10K to 1M rows), exporting comparative scaling curves and crossover analysis.

### 11. Persistent Model Registry & Versioning
- **Local Artifact Store**: Saves trained models with `joblib` and accompanying `metadata.json` sidecars (hyperparameters, metrics, features, timestamps).
- **Version Management**: Automated version incrementing (`v1`, `v2`, ...), metadata retrieval, and cross-version metric comparisons.

### 12. Continuous Data Drift Detection & Auto-Retraining
- **Statistical Drift Tests**: Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI) for numeric features; Chi-Square test for categorical features.
- **Target Shift Monitoring**: Detects distribution shifts in target labels.
- **Performance-Gated Retraining**: Automatically trains a candidate model on combined reference and drifted data, registering the new version only if held-out performance meets tolerance thresholds.

### 13. Full-Featured Web Application
- Sleek dark-mode tabbed interface covering **Overview**, **Auto EDA**, **Preprocessing & Quality**, **Models & Tuning**, **Evaluation & SHAP**, **What-If Analysis**, **AI Assistant**, **Model Registry**, and **Drift Monitoring**.
- Non-blocking background worker with live polling and decision log streaming.

---

## 📊 CSV Export & Downstream Analytics

### Why CSV Export Was Added
While Markdown and HTML reports offer readable executive summaries, real-world data science workflows require **structured, machine-readable data artifacts** for downstream analytics, external reporting, model auditing, and integration with business intelligence (BI) tooling.

The **CSV Export Module** (`autods_agent/csv_export.py`) automatically extracts the real results of every pipeline execution and writes them into standardized CSV files immediately following evaluation and explainability.

### Generated CSV Artifacts
Every successful pipeline run produces a dedicated `outputs/<run_id>/csv/` directory containing:

| File | Contents & Metrics | Typical Downstream Use Case |
|---|---|---|
| **`dataset_profile.csv`** | Dimensions (rows/cols), memory footprint, task type, target subtype, missing values, duplicate counts, cardinality | Data cataloging, ETL profiling, metadata tracking |
| **`model_comparison.csv`** | Full candidate model leaderboard with tuning hyperparams, CV scores, held-out metrics (Accuracy, Precision, Recall, F1, ROC-AUC for classification; MAE, MSE, RMSE, R² for regression), training time, and winner flag | Model governance, algorithm benchmarking, executive scorecards |
| **`predictions.csv`** | Test-set instance actuals, model predictions, class probabilities (classification) or residuals (regression) | Error analysis, calibration curves, threshold tuning, audit trails |
| **`feature_importance.csv`** | Ranked feature attributions from SHAP (or permutation importance fallback) | Feature selection review, stakeholder explainability, regulatory compliance |
| **`preprocessing_log.csv`** | Complete record of imputations, one-hot/ordinal encodings, duplicate removals, and standard scalings applied | Reproducibility audits, data pipeline documentation |
| **`decision_log.csv`** | Complete audit trail of every agent decision, stage name, reasoner rationale, status, and duration in seconds | Autonomous system auditing, pipeline explainability |
| **`anomalies.csv`** | Row-level outlier flags generated by Isolation Forest and Local Outlier Factor (LOF) | Data quality monitoring, fraudulent record triage |

### Directory & Archive Structure
```text
outputs/
└── <run_id>/
    ├── report.md
    ├── autods_<run_id>_csv_results.zip    <-- Complete bundled archive
    ├── cleaned_data.csv
    ├── engineered_features.csv
    ├── eda/
    ├── evaluation/
    ├── explainability/
    └── csv/                               <-- Structured CSV exports
        ├── dataset_profile.csv
        ├── model_comparison.csv
        ├── predictions.csv
        ├── feature_importance.csv
        ├── preprocessing_log.csv
        ├── decision_log.csv
        └── anomalies.csv
```

### How to Download
- **Web UI**: In the Results view, use the **Download Results** panel on the Overview tab to download individual CSVs or click **Download All CSVs** to retrieve the complete `.zip` archive.
- **REST Endpoints**:
  - `GET /download/<run_id>/csv/<filename>` (e.g., `/download/run_123/csv/model_comparison.csv`)
  - `GET /download/<run_id>/csv/all` (downloads `autods_<run_id>_csv_results.zip`)
- **Python / CLI**: Directly access files under `outputs/<run_id>/csv/` or use `CSVExportManager`.

### Downstream Integration & Analytics
Exported CSV results can be loaded directly into:
- **Pandas / Python**: `pd.read_csv("outputs/<run_id>/csv/model_comparison.csv")` for custom plotting and automated pipelines.
- **Microsoft Excel**: Pivot tables and conditional formatting on candidate model metrics and predictions.
- **Power BI & Tableau**: Build interactive dashboards tracking feature importance over time, prediction residual distributions, and preprocessing statistics.

### 🌐 Big Data Analytics (BDA) Connection
CSV export supports downstream data analytics by producing structured machine-readable results ready for further exploration in Pandas, Excel, Power BI, and Tableau.

> **Note**: CSV export is a downstream data formatting and export capability. The core Big Data Analytics (BDA) components of this project remain:
> - **Large-volume data processing** & out-of-core computation
> - **Distributed Dask DataFrames** and partition-based parallel execution
> - **Scale-aware routing** (automatic local vs. distributed execution paths)
> - **Scalability & memory benchmarking** (`run_scalability_experiment.py`)
> - **Performance & throughput scaling analysis** across increasing dataset volumes (10K to 1M rows)

---

## 📂 Project Structure

```
BDA-project--main/
├── autods_agent/                 # Core Autonomous Data Science Package
│   ├── __init__.py               # Package exports and version metadata
│   ├── orchestrator.py           # Central 18-stage agent orchestrator
│   ├── csv_export.py             # Structured CSV export manager & ZIP packager
│   ├── profiler.py               # Dataset profiling & scale routing agent
│   ├── eda.py                    # Automated exploratory data analysis agent
│   ├── preprocessing.py          # Data cleaning, imputation & encoding agent
│   ├── anomaly.py                # Isolation Forest & LOF anomaly detector
│   ├── imbalance.py              # Class imbalance detection & SMOTE handler
│   ├── feature_engineering.py    # Non-linear & polynomial feature agent
│   ├── feature_selection.py      # Mutual Information & K-Best selector
│   ├── model_selection.py        # Model families & Optuna tuning agent
│   ├── evaluation.py             # Metric computation & diagnostic plot agent
│   ├── explainability.py         # SHAP global/local XAI agent
│   ├── whatif.py                 # Counterfactual prediction & comparison agent
│   ├── assistant.py              # Context-aware AI data scientist assistant
│   ├── registry.py               # Model registry & artifact versioning
│   ├── drift.py                  # Statistical drift & auto-retraining agent
│   ├── scalability.py            # Local vs Dask scalability benchmarking
│   ├── reporting.py              # Markdown & HTML report generator
│   ├── llm_reasoner.py           # Rule-based & Anthropic LLM reasoners
│   └── utils.py                  # Audit RunLog & timing decorators
├── webapp/                       # Interactive Flask Web Platform
│   ├── app.py                    # REST API backend & background thread runner
│   ├── templates/
│   │   └── index.html            # Tabbed responsive frontend
│   └── static/
│       ├── css/style.css         # Dark console design system
│       ├── js/app.js             # Reactive tab & What-If logic
│       └── sample_data/          # Bundled demo datasets
├── tests/                        # Fast Pytest Test Suite
│   ├── test_pipeline_modules.py  # Smoke & integration tests
│   └── test_csv_export.py        # CSV export & download endpoint tests
├── outputs/                      # Generated Diagnostic Plots & Reports
│   └── scalability/              # Scaling benchmark curves & tables
├── sample_output/                # Reference Demonstration Reports
│   ├── binary_classification/    # Breast Cancer demo artifacts
│   ├── multiclass_classification/# Wine demo artifacts
│   ├── regression/               # Diabetes demo artifacts
│   └── imbalanced_classification/# Synthetic 95:5 demo artifacts
├── model_registry/               # Persistent Model Artifacts & Sidecars
├── run_demo_small.py             # Small dataset in-memory demo
├── run_demo_large.py             # Large dataset distributed Dask demo
├── run_multidataset_demo.py      # Multi-dataset 4-task demonstration suite
├── run_scalability_experiment.py # Scalability & Big Data benchmark runner
├── requirements.txt              # Production dependency specifications
└── README.md                     # Comprehensive platform documentation
```

---

## ⚡ Installation & Quick Start

### 1. Prerequisites
Python 3.10+ is recommended.

```bash
# Clone repository and enter directory
cd BDA-project--main

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Demos

#### Small Dataset Demo (In-Memory Path)
```bash
python run_demo_small.py
```

#### Large Dataset Demo (Distributed Dask Path)
```bash
python run_demo_large.py
```

#### Multi-Dataset Demonstration Suite (Binary, Multiclass, Regression, Imbalance)
```bash
python run_multidataset_demo.py
```

### 3. Run Scalability Benchmarks
```bash
python run_scalability_experiment.py
```
*Outputs are written to `outputs/scalability/` with `results.csv`, `time_vs_size.png`, `memory_vs_size.png`, `speedup.png`, and `summary.md`.*

### 4. Run Pytest Test Suite
```bash
python -m pytest -v
```

### 5. Launch the Web Application
```bash
python webapp/app.py
```
Open your browser and navigate to **`http://localhost:5050`**.
- Click **"Load Breast Cancer Demo Dataset"** or drag and drop any CSV.
- Click **"Launch Autonomous Agent"** to watch live streaming decisions.
- Explore all 9 interactive tabs (**Auto EDA**, **SHAP Explainability**, **What-If Analysis**, **AI Assistant**, **Model Registry**, **Drift Monitoring**).

---

## ⚙️ Assumptions & Design Decisions

1. **Deterministic Default Operation**: By default, `AutoDS-Agent 2.0` operates 100% offline using `RuleBasedReasoner` to guarantee reproducible grading and execution in network-restricted environments.
2. **Leakage-Free Resampling**: Anomaly detection and class imbalance resampling (`SMOTE`, `RandomOverSampler`) are computed strictly on training partitions to prevent optimistic test bias.
3. **Adaptive Optuna Budgeting**: Optuna trial counts and CV folds automatically adjust based on dataset row volume to balance search quality with compute time.
4. **Graceful Explainability Fallback**: For tree-based models, SHAP `TreeExplainer` is used; for non-tree models (or upon SHAP incompatibility), permutation feature importance is automatically utilized.
5. **Memory-Aware Benchmarking**: `run_scalability_experiment.py` inspects host system memory (`psutil`) and automatically restricts benchmark dataset sizes in low-memory environments.

---

## 🔮 Limitations & Future Work

- **Deep Learning Architectures**: Future releases can integrate PyTorch tabular transformers (FT-Transformer, TabNet).
- **Cluster Orchestration**: Extend Dask and PySpark support from local single-node multi-processing to remote Kubernetes / Slurm clusters.
- **Automated Feature Synthesis**: Add LLM-guided domain-specific feature synthesis for text and datetime columns.

---

## 📄 License
This project is licensed under the MIT License.
