"""
Demo 2: large-ish synthetic dataset -> agent should route to the DISTRIBUTED
(Dask) execution path.

Dataset: sklearn's make_regression, with injected missingness.

NOTE ON SCALE: this repo's sandbox has a single CPU core and a hard
per-command time limit, so the numbers below (30k rows, a 25k-row
distributed threshold) are picked to *demonstrate* the routing logic and
the distributed preprocessing path quickly. In `autods_agent.Orchestrator`
the real thresholds default to 500MB / 1,000,000 rows -- tuned for an
actual multi-core machine or cluster, not a demo sandbox. To reproduce this
on genuinely large data, just construct `Orchestrator()` with the defaults
(or your own thresholds) and give it a bigger dataframe; nothing else in
the pipeline changes.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from autods_agent import Orchestrator

np.random.seed(0)

X, y = make_regression(n_samples=30_000, n_features=8, noise=10.0, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = y

# inject missingness
for col in ["f0", "f3"]:
    mask = np.random.rand(len(df)) < 0.05
    df.loc[mask, col] = np.nan

print(f"Dataset shape: {df.shape}")
print(f"Approx size in memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Lowered thresholds so this sandbox-sized dataset still demonstrates the
# distributed path (see NOTE above for production-scale defaults).
orch = Orchestrator(distributed_threshold_mb=5, distributed_threshold_rows=25_000)
result = orch.run(df, target_col="target")

with open("sample_output/report_large.md", "w") as f:
    f.write(result["report_markdown"])

print("\n=== DONE: large-dataset demo ===")
print(f"Backend chosen: {result['profile']['backend']}")
print(f"Task type: {result['profile']['task_type']}")
print(f"Final model: {result['eval_results']['final_model_name']}")
print(f"Scores: {result['eval_results']['scores']}")
