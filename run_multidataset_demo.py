"""
Multi-Dataset Demonstration
============================
Phase 14: Demonstrates AutoDS-Agent 2.0 on 4 diverse problem types:
  1. Binary Classification: Breast Cancer dataset
  2. Multiclass Classification: Wine dataset
  3. Regression: Diabetes dataset
  4. Class Imbalance: Synthetic high-imbalance dataset (95:5 class ratio)

Outputs are saved to sample_output/ for reproducible reference.
"""

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, make_classification

from autods_agent import Orchestrator

SAMPLE_DIR = "sample_output"
os.makedirs(SAMPLE_DIR, exist_ok=True)


def run_binary_demo():
    print("\n" + "=" * 60)
    print("DEMO 1: Binary Classification (Breast Cancer)")
    print("=" * 60)
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    target_col = "target"

    out_folder = os.path.join(SAMPLE_DIR, "binary_classification")
    os.makedirs(out_folder, exist_ok=True)

    orch = Orchestrator(output_root=os.path.join(out_folder, "outputs"))
    res = orch.run(df, target_col=target_col, dataset_name="breast_cancer")

    report_path = os.path.join(out_folder, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(res["report_markdown"])

    print(f"-> Selected model: {res['eval_results']['final_model_name']}")
    print(f"-> Metrics: {res['eval_results']['metrics_detail'][res['eval_results']['final_model_name']]}")
    print(f"-> Report saved to: {report_path}")
    return res


def run_multiclass_demo():
    print("\n" + "=" * 60)
    print("DEMO 2: Multiclass Classification (Wine)")
    print("=" * 60)
    data = load_wine(as_frame=True)
    df = data.frame.copy()
    target_col = "target"

    out_folder = os.path.join(SAMPLE_DIR, "multiclass_classification")
    os.makedirs(out_folder, exist_ok=True)

    orch = Orchestrator(output_root=os.path.join(out_folder, "outputs"))
    res = orch.run(df, target_col=target_col, dataset_name="wine")

    report_path = os.path.join(out_folder, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(res["report_markdown"])

    print(f"-> Selected model: {res['eval_results']['final_model_name']}")
    print(f"-> Metrics: {res['eval_results']['metrics_detail'][res['eval_results']['final_model_name']]}")
    print(f"-> Report saved to: {report_path}")
    return res


def run_regression_demo():
    print("\n" + "=" * 60)
    print("DEMO 3: Regression (Diabetes)")
    print("=" * 60)
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    target_col = "target"

    out_folder = os.path.join(SAMPLE_DIR, "regression")
    os.makedirs(out_folder, exist_ok=True)

    orch = Orchestrator(output_root=os.path.join(out_folder, "outputs"))
    res = orch.run(df, target_col=target_col, dataset_name="diabetes")

    report_path = os.path.join(out_folder, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(res["report_markdown"])

    print(f"-> Selected model: {res['eval_results']['final_model_name']}")
    print(f"-> Metrics: {res['eval_results']['metrics_detail'][res['eval_results']['final_model_name']]}")
    print(f"-> Report saved to: {report_path}")
    return res


def run_imbalanced_demo():
    print("\n" + "=" * 60)
    print("DEMO 4: Imbalanced Classification (Synthetic 95:5)")
    print("=" * 60)
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_informative=8,
        weights=[0.95, 0.05],
        random_state=42,
    )
    cols = [f"feat_{i}" for i in range(12)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    target_col = "label"

    out_folder = os.path.join(SAMPLE_DIR, "imbalanced_classification")
    os.makedirs(out_folder, exist_ok=True)

    orch = Orchestrator(output_root=os.path.join(out_folder, "outputs"))
    res = orch.run(df, target_col=target_col, dataset_name="imbalanced_synthetic")

    report_path = os.path.join(out_folder, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(res["report_markdown"])

    print(f"-> Imbalance detected: {res['imbalance_result']['is_imbalanced']}")
    print(f"-> Best resampling strategy: {res['imbalance_result']['best_strategy']}")
    print(f"-> Selected model: {res['eval_results']['final_model_name']}")
    print(f"-> Metrics: {res['eval_results']['metrics_detail'][res['eval_results']['final_model_name']]}")
    print(f"-> Report saved to: {report_path}")
    return res


def main():
    print("AutoDS-Agent 2.0: Multi-Dataset Autonomous Demonstration Suite")
    run_binary_demo()
    run_multiclass_demo()
    run_regression_demo()
    run_imbalanced_demo()
    print("\n" + "=" * 60)
    print("All 4 demonstrations completed successfully!")
    print(f"Reports and artifacts saved under: {os.path.abspath(SAMPLE_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
