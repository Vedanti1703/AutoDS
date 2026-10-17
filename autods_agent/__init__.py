"""
AutoDS-Agent 2.0
================
An Explainable and Scalable Autonomous Data Science Platform.

Combines:
  - Advanced AutoML, hyperparameter tuning (Optuna), and multi-family model search
  - Automated EDA and diagnostic data visualization
  - Robust preprocessing, skew-aware imputation, and cardinality-aware encoding
  - Anomaly detection (Isolation Forest + LOF) and class imbalance handling (SMOTE, resamplers)
  - Budget-capped feature engineering and statistical feature selection
  - Model evaluation with comprehensive diagnostic plots
  - Explainable AI (SHAP global summary, local waterfall, and permutation fallback)
  - Interactive What-If counterfactual analysis
  - Context-aware AI Data Scientist Assistant
  - Local Model Registry with artifact versioning
  - Continuous data drift monitoring and automated retraining
  - Scale-aware routing (pandas/sklearn vs Dask/PySpark) and scalability benchmarking
"""

from .orchestrator import Orchestrator, FullInferencePipeline
from .profiler import DataProfilerAgent
from .eda import AutoEDAAgent
from .preprocessing import PreprocessingAgent, TabularPreprocessor
from .anomaly import AnomalyDetectionAgent
from .imbalance import ImbalanceHandlerAgent
from .feature_engineering import FeatureEngineeringAgent, FeatureEngineeringTransformer
from .feature_selection import FeatureSelectionAgent, FeatureSelectionTransformer
from .model_selection import ModelSelectionAgent, HyperparameterTuningAgent
from .evaluation import EvaluationAgent
from .explainability import ExplainabilityAgent
from .whatif import WhatIfAgent
from .assistant import DataScienceAssistant, build_assistant_context
from .registry import ModelRegistry
from .drift import DriftDetectionAgent
from .scalability import ScalabilityExperiment
from .ingestion import StreamingIngestAgent, csv_to_parquet
from .online_learning import OnlineLearningAgent
from .reporting import ReportingAgent
from .csv_export import CSVExportManager
from .utils import RunLog, DecisionRecord
from .llm_reasoner import get_reasoner, RuleBasedReasoner, AnthropicReasoner

__all__ = [
    "Orchestrator",
    "FullInferencePipeline",
    "DataProfilerAgent",
    "AutoEDAAgent",
    "PreprocessingAgent",
    "TabularPreprocessor",
    "AnomalyDetectionAgent",
    "ImbalanceHandlerAgent",
    "FeatureEngineeringAgent",
    "FeatureEngineeringTransformer",
    "FeatureSelectionAgent",
    "FeatureSelectionTransformer",
    "ModelSelectionAgent",
    "HyperparameterTuningAgent",
    "EvaluationAgent",
    "ExplainabilityAgent",
    "WhatIfAgent",
    "DataScienceAssistant",
    "build_assistant_context",
    "ModelRegistry",
    "DriftDetectionAgent",
    "ScalabilityExperiment",
    "StreamingIngestAgent",
    "csv_to_parquet",
    "OnlineLearningAgent",
    "ReportingAgent",
    "CSVExportManager",
    "RunLog",
    "DecisionRecord",
    "get_reasoner",
    "RuleBasedReasoner",
    "AnthropicReasoner",
]

__version__ = "2.1.0"
