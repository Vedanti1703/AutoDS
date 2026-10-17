"""
Demo 1: small, in-memory dataset -> agent should route to the LOCAL
(pandas/sklearn) execution path.

Dataset: sklearn's breast cancer dataset (classification), with a few
missing values injected artificially so imputation logic actually fires.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from autods_agent import Orchestrator

np.random.seed(0)

data = load_breast_cancer(as_frame=True)
df = data.frame.rename(columns={"target": "diagnosis"})

# inject missingness so ImputationStrategy decisions actually trigger
for col in ["mean radius", "mean texture", "worst symmetry"]:
    mask = np.random.rand(len(df)) < 0.08
    df.loc[mask, col] = np.nan

print(f"Dataset shape: {df.shape}")

orch = Orchestrator()
result = orch.run(df, target_col="diagnosis")

with open("sample_output/report_small.md", "w") as f: 
    f.write(result["report_markdown"])

print("\n=== DONE: small-dataset demo ===")
print(f"Backend chosen: {result['profile']['backend']}")
print(f"Task type: {result['profile']['task_type']}")
print(f"Final model: {result['eval_results']['final_model_name']}")
print(f"Scores: {result['eval_results']['scores']}")
