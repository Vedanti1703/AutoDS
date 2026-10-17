"""
ModelRegistry
==============
Phase 11 – Persistent local model registry.

Stores each trained model with joblib + a JSON metadata sidecar:
  - dataset name, model type, hyperparameters, metrics, version
    (auto-incrementing per model type), training date, feature list, run_id.

Storage layout:
  model_registry/
    <run_id>_<model_type>_v<version>/
      model.joblib
      metadata.json

Public API:
  register(...)       -> version string
  list_models()       -> list of metadata dicts
  load(run_id, ...)   -> sklearn model
  compare_versions(model_type) -> comparison table
  get_best(metric)    -> metadata of best registered model
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional

import joblib

logger = logging.getLogger(__name__)

REGISTRY_DIR = "model_registry"


class ModelRegistry:
    """Local file-based model registry with JSON metadata."""

    def __init__(self, registry_dir: str = REGISTRY_DIR):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def register(
        self,
        model: Any,
        model_type: str,
        run_id: str,
        dataset_name: str,
        hyperparameters: dict,
        metrics: dict,
        feature_names: List[str],
        task_type: str = "classification",
    ) -> str:
        """
        Save model + metadata. Returns the version string (e.g. 'v3').
        """
        version = self._next_version(model_type)
        folder  = os.path.join(self.registry_dir,
                               f"{run_id}_{model_type}_v{version}")
        os.makedirs(folder, exist_ok=True)

        # Save model
        model_path = os.path.join(folder, "model.joblib")
        try:
            joblib.dump(model, model_path)
        except Exception as e:
            warnings.warn(f"Could not save model with joblib: {e}")

        # Save metadata
        metadata = {
            "run_id":          run_id,
            "model_type":      model_type,
            "version":         f"v{version}",
            "dataset_name":    dataset_name,
            "task_type":       task_type,
            "hyperparameters": hyperparameters,
            "metrics":         metrics,
            "feature_names":   feature_names,
            "training_date":   time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path":      model_path,
            "folder":          folder,
        }
        meta_path = os.path.join(folder, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Registered {model_type} v{version} in {folder}")
        return f"v{version}"

    # ------------------------------------------------------------------
    def list_models(self) -> List[dict]:
        """Return a list of all registered model metadata dicts."""
        results = []
        for entry in sorted(os.listdir(self.registry_dir)):
            meta_path = os.path.join(self.registry_dir, entry, "metadata.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path) as f:
                        results.append(json.load(f))
                except Exception:
                    pass
        return results

    # ------------------------------------------------------------------
    def load(self, folder_or_run_id: str, model_type: Optional[str] = None, version: Optional[str] = None) -> Any:
        """Load a model by folder path or by (run_id, model_type, version)."""
        if model_type is not None and version is not None:
            folder = os.path.join(self.registry_dir, f"{folder_or_run_id}_{model_type}_{version}")
        else:
            folder = folder_or_run_id
        model_path = os.path.join(folder, "model.joblib")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No model.joblib found in {folder}")
        return joblib.load(model_path)

    # ------------------------------------------------------------------
    def compare_versions(self, model_type: str) -> List[dict]:
        """Compare all registered versions of a given model type."""
        return [m for m in self.list_models() if m.get("model_type") == model_type]

    # ------------------------------------------------------------------
    def get_best(self, metric: str = "accuracy",
                  task_type: Optional[str] = None) -> Optional[dict]:
        """Return metadata of the registered model with the highest `metric`."""
        candidates = self.list_models()
        if task_type:
            candidates = [c for c in candidates if c.get("task_type") == task_type]
        if not candidates:
            return None
        def _score(m):
            return m.get("metrics", {}).get(metric, float("-inf"))
        return max(candidates, key=_score)

    # ------------------------------------------------------------------
    def _next_version(self, model_type: str) -> int:
        """Auto-increment version number for a given model_type."""
        existing = self.list_models()
        versions = []
        for m in existing:
            if m.get("model_type") == model_type:
                v = m.get("version", "v0")
                try:
                    versions.append(int(re.sub(r"[^0-9]", "", v)))
                except ValueError:
                    pass
        return max(versions, default=0) + 1
