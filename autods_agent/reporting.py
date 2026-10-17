"""
ReportingAgent (v2)
====================
Assembles everything the AutoDS-Agent 2.0 pipeline produced into a complete,
human-readable Markdown and HTML report:
  1. Executive Summary & Metadata
  2. Dataset Profile & Auto EDA Insights
  3. Preprocessing, Anomaly Detection & Imbalance Handling
  4. Feature Engineering & Feature Selection
  5. Model Selection & Optuna Tuning Leaderboard
  6. Held-out Test Evaluation (Metrics & Plots)
  7. Explainable AI (SHAP Global & Local Attributions)
  8. Model Registry Entry
  9. Full Autonomous Agent Audit Trail (Every decision + reasoning)
"""

from __future__ import annotations

import os
import time
from typing import Dict, Any, List, Optional


class ReportingAgent:
    """Generates structured Markdown and HTML reports from pipeline artifacts."""

    def __init__(self, log):
        self.log = log

    def run(
        self,
        profile: dict,
        tuning_results: dict,
        eval_results: dict,
        explain_results: dict,
        reasoner_name: str,
        eda_result: Optional[dict] = None,
        anomaly_result: Optional[dict] = None,
        imbalance_result: Optional[dict] = None,
        feature_selection_result: Optional[dict] = None,
        registry_info: Optional[dict] = None,
        preprocessing_explanations: Optional[List[str]] = None,
        csv_paths: Optional[Dict[str, str]] = None,
        csv_zip_path: Optional[str] = None,
    ) -> str:
        lines: List[str] = []

        # Title & Meta
        lines.append("# AutoDS-Agent 2.0: Autonomous Data Science Report\n")
        lines.append(
            f"*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Reasoner Engine: `{reasoner_name}` | "
            f"Pipeline Execution Time: {self.log.total_elapsed():.2f}s*\n"
        )
        lines.append(
            "> **Research Problem Statement**: To develop an autonomous and explainable data-science "
            "platform that can automatically analyze an unseen dataset, perform preprocessing and feature "
            "engineering, select and optimize appropriate machine-learning models, explain predictions, "
            "support scalable data processing and assist users through an AI-based data-science assistant.\n"
        )

        # 1. Dataset Profile
        lines.append("## 1. Dataset Profile & Data Scale")
        lines.append(f"- **Total Rows**: {profile.get('n_rows', 0):,}")
        lines.append(f"- **Total Columns**: {profile.get('n_cols', 0)}")
        lines.append(f"- **In-Memory Size**: {profile.get('size_mb', 0.0):.2f} MB")
        lines.append(f"- **Inferred Task Type**: `{profile.get('task_type')}`")
        lines.append(f"- **Target Column**: `{profile.get('target_col')}`")
        lines.append(f"- **Execution Backend Chosen**: `{profile.get('backend')}`")

        if eda_result and "extended_profile" in eda_result:
            ext = eda_result["extended_profile"]
            lines.append(f"- **Numeric Features**: {ext.get('n_numeric', 0)}")
            lines.append(f"- **Categorical Features**: {ext.get('n_categorical', 0)}")
            lines.append(f"- **Duplicates Detected**: {ext.get('n_duplicates', 0)}")
            lines.append(f"- **Constant Columns**: {ext.get('n_constant', 0)}")
            if ext.get("high_cardinality_cols"):
                lines.append(f"- **High-Cardinality Columns**: {', '.join(ext['high_cardinality_cols'])}")
        lines.append("")

        # 2. Auto EDA Insights & Visualizations
        if eda_result:
            lines.append("## 2. Exploratory Data Analysis (Auto EDA)")
            plots = eda_result.get("plot_paths", {})
            if plots:
                lines.append("Generated diagnostic plots:")
                for name, path in plots.items():
                    rel_name = name.replace("_", " ").title()
                    lines.append(f"- **{rel_name}**: `{path}`")
            outliers = eda_result.get("outlier_summary", {})
            if outliers:
                lines.append("\n**Outlier Summary (IQR method)**:")
                lines.append("| Feature | Outlier Count | Outlier % |")
                lines.append("|---|---|---|")
                for col, info in list(outliers.items())[:8]:
                    lines.append(f"| `{col}` | {info.get('n_outliers', 0)} | {info.get('pct', 0):.1f}% |")
            lines.append("")

        # 3. Data Quality, Preprocessing & Imbalance
        lines.append("## 3. Preprocessing, Anomaly & Imbalance Handling")
        if preprocessing_explanations:
            lines.append("**Preprocessing Decisions Applied**:")
            for exp in preprocessing_explanations:
                lines.append(f"- {exp}")
            lines.append("")

        if anomaly_result:
            lines.append("**Anomaly Detection**:")
            lines.append(f"- Isolation Forest flagged: **{anomaly_result.get('if_anomaly_pct', 0):.2f}%** of records")
            lines.append(f"- Local Outlier Factor flagged: **{anomaly_result.get('lof_anomaly_pct', 0):.2f}%** of records")
            top_anom = anomaly_result.get("top_anomalous_indices", [])
            if top_anom:
                lines.append(f"- Top anomalous row indices: `{top_anom[:5]}`")
            lines.append("")

        if imbalance_result and profile.get("task_type") == "classification":
            lines.append("**Class Imbalance Analysis**:")
            is_imb = imbalance_result.get("is_imbalanced", False)
            ratio = imbalance_result.get("minority_ratio", 0.5)
            lines.append(f"- Imbalance Detected: **{'Yes' if is_imb else 'No'}** (Minority class ratio: {ratio:.1%})")
            if is_imb:
                lines.append(f"- Selected Resampling Strategy: **`{imbalance_result.get('best_strategy')}`**")
                strat_comp = imbalance_result.get("strategy_comparison", {})
                if strat_comp:
                    lines.append("\n| Strategy | Macro F1 | Recall | ROC-AUC |")
                    lines.append("|---|---|---|---|")
                    for s_name, s_metrics in strat_comp.items():
                        lines.append(
                            f"| `{s_name}` | {s_metrics.get('macro_f1', 0):.4f} | "
                            f"{s_metrics.get('recall', 0):.4f} | {s_metrics.get('roc_auc', 0):.4f} |"
                        )
            lines.append("")

        # 4. Feature Selection
        if feature_selection_result:
            lines.append("## 4. Feature Selection")
            kept = feature_selection_result.get("selected_features", [])
            dropped = feature_selection_result.get("dropped_features", [])
            lines.append(f"- Method: `{feature_selection_result.get('method')}`")
            lines.append(f"- Features Selected ({len(kept)}): `{kept[:15]}`" + ("..." if len(kept) > 15 else ""))
            if dropped:
                lines.append(f"- Features Dropped ({len(dropped)}): `{dropped[:10]}`" + ("..." if len(dropped) > 10 else ""))
            lines.append("")

        # 5. Model Leaderboard
        lines.append("## 5. Model Tuning Leaderboard (Optuna CV)")
        lines.append("| Candidate Model | Best CV Score | Trials | Best Hyperparameters |")
        lines.append("|---|---|---|---|")
        for name, res in sorted(tuning_results.items(), key=lambda kv: -kv[1].get("best_score", 0)):
            lines.append(
                f"| `{name}` | {res.get('best_score', 0):.4f} | "
                f"{res.get('n_trials', 0)} | `{res.get('best_params', {})}` |"
            )
        lines.append("")

        # 6. Held-out Test Evaluation
        lines.append("## 6. Held-out Test Evaluation")
        lines.append(f"**Final Selected Model**: `{eval_results.get('final_model_name')}`\n")
        lines.append("| Model | Metrics Summary |")
        lines.append("|---|---|")
        for name, metrics in eval_results.get("metrics_detail", {}).items():
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            marker = " **[Selected]**" if name == eval_results.get("final_model_name") else ""
            lines.append(f"| `{name}`{marker} | {metrics_str} |")
        lines.append("")

        eval_plots = eval_results.get("plot_paths", {})
        if eval_plots:
            lines.append("**Evaluation Diagnostic Plots Saved**:")
            for p_name, p_path in eval_plots.items():
                lines.append(f"- {p_name.replace('_', ' ').title()}: `{p_path}`")
            lines.append("")

        # 7. Explainability
        lines.append("## 7. Model Explainability & Interpretability")
        lines.append(f"- Interpretation Method: `{explain_results.get('method', 'shap')}`")
        ranked = explain_results.get("ranked_importance", [])
        if ranked:
            lines.append("\n**Top 10 Most Influential Features**:")
            lines.append("| Rank | Feature | Attribution Score |")
            lines.append("|---|---|---|")
            for i, (feat, score) in enumerate(ranked[:10], 1):
                lines.append(f"| {i} | `{feat}` | {score:.4f} |")
        exp_plots = explain_results.get("plot_paths", {})
        if exp_plots:
            lines.append("\n**Explainability Visualizations**:")
            for ep_name, ep_path in exp_plots.items():
                lines.append(f"- {ep_name.replace('_', ' ').title()}: `{ep_path}`")
        lines.append("")

        # 8. Model Registry Info
        if registry_info:
            lines.append("## 8. Model Registry & Versioning")
            lines.append(f"- **Version Tag**: `{registry_info.get('version', 'v1')}`")
            lines.append(f"- **Model Path**: `{registry_info.get('model_path', '')}`")
            lines.append(f"- **Metadata Path**: `{registry_info.get('metadata_path', '')}`")
            lines.append("")

        # 9. CSV Outputs
        lines.append("## 9. CSV Outputs")
        lines.append("The following structured CSV files were generated:\n")
        all_expected_csvs = [
            ("dataset_profile.csv", "dataset_profile", "Dataset dimensions, scaling, missingness, and profiling metrics"),
            ("model_comparison.csv", "model_comparison", "Full candidate model leaderboard with tuning parameters and held-out scores"),
            ("predictions.csv", "predictions", "Held-out test set actuals, model predictions, and class probabilities/residuals"),
            ("feature_importance.csv", "feature_importance", "Ranked SHAP attribution scores and feature importance hierarchy"),
            ("preprocessing_log.csv", "preprocessing_log", "Complete audit of data transformations, imputations, and encodings"),
            ("decision_log.csv", "decision_log", "Granular agent reasoning log and execution duration for every stage"),
            ("anomalies.csv", "anomalies", "Isolation Forest & Local Outlier Factor anomaly scoring flags"),
        ]

        for fname, key, desc in all_expected_csvs:
            if csv_paths and key in csv_paths:
                lines.append(f"- **`{fname}`**: `{csv_paths[key]}` - *{desc}*")
            else:
                lines.append(f"- **`{fname}`**: *Not generated (result unavailable for this run configuration)*")

        if csv_zip_path:
            lines.append(f"\n- **Download All CSVs (ZIP Archive)**: `{csv_zip_path}`")
        lines.append("")

        # 10. Audit Trail
        lines.append("## 10. Autonomous Agent Audit Trail")
        lines.append(
            "Every decision made during the pipeline run is logged below with the "
            "agent's internal reasoning, timing, and operational parameters.\n"
        )
        lines.append(self.log.as_markdown())
        lines.append("")

        return "\n".join(lines)
