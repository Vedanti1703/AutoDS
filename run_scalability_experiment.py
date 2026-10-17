"""
Run Scalability Experiment
==========================
Benchmarks local (pandas/sklearn) vs distributed (Dask) preprocessing and training
across synthetic datasets of increasing scale.

Outputs:
  - outputs/scalability/results.csv
  - outputs/scalability/time_vs_size.png
  - outputs/scalability/memory_vs_size.png
  - outputs/scalability/speedup.png
  - outputs/scalability/summary.md
"""

import sys
import psutil
from autods_agent.scalability import ScalabilityExperiment


def main():
    print("=" * 70)
    print("AutoDS-Agent 2.0: Scalability & Big Data Benchmark")
    print("=" * 70)

    # Check available system memory to choose appropriate dataset scale
    avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"Available system memory: {avail_gb:.2f} GB")

    if avail_gb < 4.0:
        sizes = [10_000, 25_000, 50_000, 100_000]
        print(f"Memory constrained environment (<4GB): running on sizes {sizes}")
    else:
        sizes = [10_000, 50_000, 100_000, 250_000, 500_000]
        print(f"Running benchmark on sizes {sizes}")

    exp = ScalabilityExperiment(output_dir="outputs/scalability")
    df_results = exp.run(sizes=sizes)

    print("\nBenchmark Completed Successfully!")
    print(df_results)
    print("\nResults and diagnostic figures saved in outputs/scalability/")


if __name__ == "__main__":
    main()
