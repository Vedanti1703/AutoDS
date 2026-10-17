"""
AutoEDAAgent
=============
Phase 1 – Extended profiling + automatic EDA plot generation.

Generates and saves:
  - Correlation heatmap (numeric features)
  - Numeric feature distributions (histograms)
  - Box plots (numeric vs target for classification)
  - Class distribution bar chart (classification)
  - Missing-value bar chart
  - Outlier summary (IQR method)

All plots saved to  outputs/<run_id>/eda/  and paths returned in the
result dict so the report and web UI can reference them.
"""

from __future__ import annotations

import os
import warnings
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper – safe import of matplotlib / seaborn (non-fatal if unavailable)
# ---------------------------------------------------------------------------

def _imports_ok() -> bool:
    try:
        import matplotlib  # noqa: F401
        import seaborn     # noqa: F401
        return True
    except ImportError:
        return False


def _setup_mpl():
    """Use non-interactive backend so it works in headless / server mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid", palette="muted")
    return plt, sns


# ---------------------------------------------------------------------------
# AutoEDAAgent
# ---------------------------------------------------------------------------

class AutoEDAAgent:
    """Generates a full set of EDA plots and an extended dataset profile."""

    def __init__(self, reasoner, log, output_root: str = "outputs"):
        self.reasoner = reasoner
        self.log = log
        self.output_root = output_root

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, profile: dict, run_id: str) -> dict:
        """
        Parameters
        ----------
        df       : raw (un-preprocessed) DataFrame
        profile  : dict produced by DataProfilerAgent
        run_id   : unique string used to name the output folder

        Returns
        -------
        eda_result dict with keys:
            extended_profile  - richer stats dict
            plot_paths        - {plot_name: absolute_path}
            outlier_summary   - {col: {n_outliers, pct}}
        """
        target_col = profile["target_col"]
        task_type  = profile["task_type"]

        # ---- extended profile -----------------------------------------
        extended = self._extended_profile(df, target_col, profile)
        self.log.add(
            "AutoEDA",
            f"extended profile: {extended['n_numeric']} numeric, "
            f"{extended['n_categorical']} categorical, "
            f"{extended['n_constant']} constant cols",
            f"Computed detailed dataset statistics: duplicates={extended['n_duplicates']}, "
            f"high-cardinality cols={extended['high_cardinality_cols']}, "
            f"target subtypes={extended['target_subtype']}.",
        )

        # ---- output dir -----------------------------------------------
        out_dir = os.path.join(self.output_root, run_id, "eda")
        os.makedirs(out_dir, exist_ok=True)

        plot_paths: Dict[str, str] = {}
        outlier_summary: Dict[str, dict] = {}

        if not _imports_ok():
            warnings.warn("matplotlib/seaborn not available - EDA plots skipped.")
            self.log.add("AutoEDA", "plots skipped", "matplotlib/seaborn not installed.")
            return {
                "extended_profile": extended,
                "plot_paths": plot_paths,
                "outlier_summary": outlier_summary,
            }

        try:
            plt, sns = _setup_mpl()

            numeric_cols = [c for c in df.columns
                            if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
            cat_cols     = [c for c in df.columns
                            if c != target_col and not pd.api.types.is_numeric_dtype(df[c])]

            # 1. Correlation heatmap
            if len(numeric_cols) >= 2:
                p = self._plot_correlation(df, numeric_cols, target_col,
                                           task_type, out_dir, plt, sns)
                if p:
                    plot_paths["correlation_heatmap"] = p

            # 2. Numeric distributions
            if numeric_cols:
                p = self._plot_distributions(df, numeric_cols, out_dir, plt)
                if p:
                    plot_paths["distributions"] = p

            # 3. Box plots (classification only; use first 6 numeric cols)
            if task_type == "classification" and numeric_cols:
                p = self._plot_boxplots(df, numeric_cols[:6], target_col, out_dir, plt)
                if p:
                    plot_paths["boxplots"] = p

            # 4. Class distribution
            if task_type == "classification":
                p = self._plot_class_distribution(df, target_col, out_dir, plt)
                if p:
                    plot_paths["class_distribution"] = p

            # 5. Missing-value chart
            p = self._plot_missing(df, out_dir, plt)
            if p:
                plot_paths["missing_values"] = p

            # 6. Outlier summary (IQR)
            outlier_summary = self._compute_outliers(df, numeric_cols)

            self.log.add(
                "AutoEDA",
                f"generated {len(plot_paths)} EDA plots",
                f"Plots saved to {out_dir}: {', '.join(plot_paths.keys())}.",
            )

        except Exception as exc:
            warnings.warn(f"EDA plot generation failed: {exc}")
            self.log.add("AutoEDA", "plot error", str(exc))

        return {
            "extended_profile": extended,
            "plot_paths": plot_paths,
            "outlier_summary": outlier_summary,
            "out_dir": out_dir,
        }

    # ------------------------------------------------------------------
    # Extended profile helpers
    # ------------------------------------------------------------------

    def _extended_profile(self, df: pd.DataFrame, target_col: str,
                           base_profile: dict) -> dict:
        """Compute richer dataset statistics."""
        numeric_cols = [c for c in df.columns
                        if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
        cat_cols     = [c for c in df.columns
                        if c != target_col and not pd.api.types.is_numeric_dtype(df[c])]

        # High-cardinality: categorical cols with >50 unique values
        high_card = [c for c in cat_cols if df[c].nunique() > 50]

        # Constant columns (only 1 unique value)
        constant  = [c for c in df.columns if df[c].nunique() <= 1]

        # Duplicates
        n_dup = int(df.duplicated().sum())

        # Missing per column
        missing_counts = df.isna().sum().to_dict()
        missing_pct    = (df.isna().mean() * 100).round(2).to_dict()

        # Target subtype
        tgt = df[target_col]
        n_unique_tgt = int(tgt.nunique())
        if base_profile["task_type"] == "classification":
            target_subtype = "binary" if n_unique_tgt == 2 else "multiclass"
        else:
            target_subtype = "continuous"

        return {
            "n_rows":              base_profile["n_rows"],
            "n_cols":              base_profile["n_cols"],
            "size_mb":             round(base_profile["size_mb"], 3),
            "n_numeric":           len(numeric_cols),
            "n_categorical":       len(cat_cols),
            "n_constant":          len(constant),
            "constant_cols":       constant,
            "high_cardinality_cols": high_card,
            "n_duplicates":        n_dup,
            "missing_counts":      missing_counts,
            "missing_pct":         missing_pct,
            "target_subtype":      target_subtype,
            "target_n_unique":     n_unique_tgt,
        }

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _plot_correlation(self, df, numeric_cols, target_col, task_type,
                           out_dir, plt, sns) -> Optional[str]:
        try:
            cols = numeric_cols[:20]   # cap at 20 for readability
            if task_type == "classification":
                tgt_enc = pd.Categorical(df[target_col]).codes
                corr_df = df[cols].copy()
                corr_df["_target"] = tgt_enc
            else:
                corr_df = df[cols + [target_col]]
            corr = corr_df.corr()

            fig, ax = plt.subplots(figsize=(min(14, len(corr.columns)), min(12, len(corr.columns))))
            sns.heatmap(corr, annot=len(cols) <= 12, fmt=".2f", cmap="RdYlGn",
                        center=0, ax=ax, linewidths=0.3, cbar_kws={"shrink": 0.8})
            ax.set_title("Feature Correlation Heatmap", fontsize=13, pad=12)
            plt.tight_layout()
            path = os.path.join(out_dir, "correlation_heatmap.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception as e:
            logger.warning(f"Correlation heatmap failed: {e}")
            return None

    def _plot_distributions(self, df, numeric_cols, out_dir, plt) -> Optional[str]:
        try:
            cols = numeric_cols[:12]
            n = len(cols)
            ncols_grid = min(4, n)
            nrows_grid = (n + ncols_grid - 1) // ncols_grid
            fig, axes = plt.subplots(nrows_grid, ncols_grid,
                                     figsize=(5 * ncols_grid, 4 * nrows_grid))
            axes = np.array(axes).flatten()
            for i, col in enumerate(cols):
                axes[i].hist(df[col].dropna(), bins=30, color="#5EE39B", alpha=0.75, edgecolor="none")
                axes[i].set_title(col[:30], fontsize=10)
                axes[i].set_xlabel("")
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            fig.suptitle("Numeric Feature Distributions", fontsize=13, y=1.01)
            plt.tight_layout()
            path = os.path.join(out_dir, "distributions.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception as e:
            logger.warning(f"Distributions plot failed: {e}")
            return None

    def _plot_boxplots(self, df, numeric_cols, target_col, out_dir, plt) -> Optional[str]:
        try:
            cols = numeric_cols[:6]
            n = len(cols)
            ncols_grid = min(3, n)
            nrows_grid = (n + ncols_grid - 1) // ncols_grid
            fig, axes = plt.subplots(nrows_grid, ncols_grid,
                                     figsize=(5 * ncols_grid, 4 * nrows_grid))
            axes = np.array(axes).flatten()
            for i, col in enumerate(cols):
                data_by_class = [df.loc[df[target_col] == cls, col].dropna().values
                                 for cls in sorted(df[target_col].unique())]
                try:
                    axes[i].boxplot(data_by_class,
                                    tick_labels=[str(c) for c in sorted(df[target_col].unique())],
                                    patch_artist=True)
                except TypeError:
                    axes[i].boxplot(data_by_class,
                                    labels=[str(c) for c in sorted(df[target_col].unique())],
                                    patch_artist=True)
                axes[i].set_title(col[:25], fontsize=10)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            fig.suptitle("Box Plots by Target Class", fontsize=13)
            plt.tight_layout()
            path = os.path.join(out_dir, "boxplots.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception as e:
            logger.warning(f"Box plots failed: {e}")
            return None

    def _plot_class_distribution(self, df, target_col, out_dir, plt) -> Optional[str]:
        try:
            counts = df[target_col].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(max(6, len(counts)), 4))
            bars = ax.bar([str(c) for c in counts.index], counts.values, color="#5EE39B", alpha=0.85)
            ax.bar_label(bars, padding=3, fontsize=10)
            ax.set_title("Target Class Distribution", fontsize=13)
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            plt.tight_layout()
            path = os.path.join(out_dir, "class_distribution.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception as e:
            logger.warning(f"Class distribution plot failed: {e}")
            return None

    def _plot_missing(self, df, out_dir, plt) -> Optional[str]:
        try:
            missing = df.isna().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            if missing.empty:
                return None
            fig, ax = plt.subplots(figsize=(max(6, len(missing)), 4))
            pct = (missing / len(df) * 100).values
            bars = ax.barh([str(c) for c in missing.index], pct, color="#F2B84B", alpha=0.85)
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
            ax.set_title("Missing Value Percentage by Column", fontsize=13)
            ax.set_xlabel("% Missing")
            ax.axvline(30, color="red", linestyle="--", linewidth=1, alpha=0.6, label="30% drop threshold")
            ax.legend(fontsize=9)
            plt.tight_layout()
            path = os.path.join(out_dir, "missing_values.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception as e:
            logger.warning(f"Missing values plot failed: {e}")
            return None

    def _compute_outliers(self, df: pd.DataFrame,
                           numeric_cols: List[str]) -> Dict[str, dict]:
        """IQR-based outlier detection per numeric column."""
        result = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((series < lower) | (series > upper)).sum())
            if n_out > 0:
                result[col] = {
                    "n_outliers": n_out,
                    "pct": round(n_out / len(series) * 100, 2),
                    "lower_fence": round(float(lower), 4),
                    "upper_fence": round(float(upper), 4),
                }
        return result
