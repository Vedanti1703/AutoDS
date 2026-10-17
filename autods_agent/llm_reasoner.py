"""
LLMReasoner
===========
The "brain" the Orchestrator consults at each decision point. Two backends:

1. RuleBasedReasoner (default, always available, fully offline & deterministic
   -- important for reproducible grading and for running on machines with no
   internet access). Encodes the same heuristics a data scientist would use.

2. AnthropicReasoner (optional): if ANTHROPIC_API_KEY is set in the
   environment and the `anthropic` package is installed, decisions are
   delegated to a live Claude model, with the rule-based reasoner as a
   fallback if the API call fails for any reason. This is what lets you
   actually study "does an LLM-driven agent make reliable ML-lifecycle
   decisions" -- run the whole pipeline with USE_LLM=1 and compare its
   decisions/logs against the rule-based run.

Both backends implement the same interface: `decide(question, context) -> (decision, reasoning)`
so the rest of the codebase never needs to know which one is active.
"""

import os
import json
from typing import Tuple, Any


class BaseReasoner:
    name = "base"

    def decide(self, question: str, context: dict) -> Tuple[str, str]:
        raise NotImplementedError


class RuleBasedReasoner(BaseReasoner):
    """Deterministic, transparent heuristics -- no external calls."""
    name = "rule-based"

    def decide(self, question: str, context: dict) -> Tuple[str, str]:
        handler = getattr(self, f"_q_{question}", None)
        if handler is None:
            return "default", f"No rule defined for '{question}'; using default behavior."
        return handler(context)

    # ---- individual decision rules -------------------------------------

    def _q_execution_backend(self, ctx: dict) -> Tuple[str, str]:
        size_mb = ctx["size_mb"]
        n_rows = ctx["n_rows"]
        threshold_mb = ctx.get("distributed_threshold_mb", 500)
        threshold_rows = ctx.get("distributed_threshold_rows", 1_000_000)
        if size_mb >= threshold_mb or n_rows >= threshold_rows:
            return ("distributed",
                    f"Dataset is {size_mb:.1f}MB with {n_rows:,} rows, "
                    f"exceeding thresholds ({threshold_mb}MB / {threshold_rows:,} rows). "
                    f"Routing to the Dask distributed execution path to keep "
                    f"memory use and wall-clock time bounded as data grows.")
        return ("local",
                f"Dataset is {size_mb:.1f}MB with {n_rows:,} rows, within "
                f"in-memory thresholds. Routing to the pandas/sklearn local "
                f"path -- distributed overhead would not pay off here.")

    def _q_task_type(self, ctx: dict) -> Tuple[str, str]:
        n_unique = ctx["target_n_unique"]
        dtype = ctx["target_dtype"]
        n_rows = ctx["n_rows"]
        if "float" in dtype and n_unique > max(20, 0.05 * n_rows):
            return ("regression",
                    f"Target has dtype {dtype} and {n_unique} unique values "
                    f"relative to {n_rows} rows -> treating as continuous, so "
                    f"this is a regression task.")
        return ("classification",
                f"Target has dtype {dtype} with only {n_unique} unique values "
                f"-> treating as categorical, so this is a classification task.")

    def _q_imputation_strategy(self, ctx: dict) -> Tuple[str, str]:
        missing_frac = ctx["missing_frac"]
        if missing_frac > 0.3:
            return ("drop_column",
                    f"Column is {missing_frac:.0%} missing -- too sparse to "
                    f"impute reliably, dropping it.")
        if ctx["is_numeric"]:
            return ("median",
                    f"Numeric column with {missing_frac:.0%} missing -- median "
                    f"imputation is robust to outliers and skew.")
        return ("mode",
                f"Categorical column with {missing_frac:.0%} missing -- filling "
                f"with the most frequent category.")

    def _q_candidate_models(self, ctx: dict) -> Tuple[str, str]:
        task = ctx["task_type"]
        n_rows = ctx["n_rows"]
        n_features = ctx["n_features"]
        models = []
        if task == "classification":
            models = ["logistic_regression", "random_forest", "gradient_boosting"]
        else:
            models = ["ridge", "random_forest", "gradient_boosting"]
        if n_rows > 20_000 or n_features > 200:
            # drop the slowest option under time/scale pressure
            models = [m for m in models if m != "gradient_boosting"] + ["hist_gradient_boosting"]
            reasoning = (f"{n_rows:,} rows / {n_features} features is large enough that "
                         f"plain gradient boosting would be slow; substituting the "
                         f"histogram-based variant for scalability.")
        else:
            reasoning = (f"{n_rows:,} rows / {n_features} features is small enough to "
                         f"try the full candidate set of {task} models.")
        return (json.dumps(models), reasoning)

    def _q_tuning_budget(self, ctx: dict) -> Tuple[str, str]:
        n_rows = ctx["n_rows"]
        n_models = ctx["n_models"]
        if n_rows > 100_000:
            trials = 5
        elif n_rows > 10_000:
            trials = 6
        else:
            trials = 10
        return (str(trials),
                f"Allocating {trials} Optuna trials per model ({n_models} models) "
                f"given {n_rows:,} rows -- balances search quality against the "
                f"quadratic-ish cost of CV on larger data.")

    def _q_feature_selection_method(self, ctx: dict) -> Tuple[str, str]:
        n_features = ctx["n_features"]
        task_type  = ctx["task_type"]
        if n_features > 100:
            return ("mutual_info",
                    f"{n_features} features is large — Mutual Information is model-free "
                    f"and scales well; chosen over model-based importance to avoid "
                    f"a full tree fit just for selection.")
        if task_type == "classification":
            return ("mutual_info",
                    f"{n_features} features, classification task — Mutual Information "
                    f"captures nonlinear feature-target dependencies, robust to outliers.")
        return ("univariate",
                f"{n_features} features, regression task — univariate F-test is fast "
                f"and appropriate for continuous targets.")

    def _q_online_model(self, ctx: dict) -> Tuple[str, str]:
        task = ctx.get("task_type", "classification")
        if task == "classification":
            return (
                "SGDClassifier",
                "SGDClassifier with log_loss supports partial_fit, probability "
                "estimates, and is the most versatile incremental classifier for "
                "streaming tabular data.",
            )
        return (
            "SGDRegressor",
            "SGDRegressor with squared_error supports partial_fit and provides "
            "online gradient descent for continuous target regression.",
        )

    def _q_final_model_choice(self, ctx: dict) -> Tuple[str, str]:
        scores: dict = ctx["scores"]
        best = max(scores, key=scores.get)
        sorted_scores = sorted(scores.items(), key=lambda kv: -kv[1])
        gap = ""
        if len(sorted_scores) > 1:
            gap = f" (runner-up {sorted_scores[1][0]} scored {sorted_scores[1][1]:.4f})"
        return (best,
                f"Selected {best} with score {scores[best]:.4f}, the highest "
                f"among {len(scores)} tuned candidates{gap}.")


class AnthropicReasoner(BaseReasoner):
    """Delegates decisions to a live Claude model. Falls back to
    RuleBasedReasoner transparently if the API is unavailable or errors."""
    name = "anthropic-llm"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self._fallback = RuleBasedReasoner()
        self._client = None
        try:
            import anthropic  # noqa: F401
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
        except Exception:
            self._client = None

    def decide(self, question: str, context: dict) -> Tuple[str, str]:
        if self._client is None:
            decision, reasoning = self._fallback.decide(question, context)
            return decision, f"[fallback: no ANTHROPIC_API_KEY] {reasoning}"

        prompt = (
            "You are the reasoning module of an autonomous data-science agent.\n"
            f"Decision point: '{question}'\n"
            f"Context (JSON): {json.dumps(context, default=str)}\n\n"
            "Respond ONLY with strict JSON: "
            '{"decision": "<short machine-readable value>", "reasoning": "<1-3 sentence justification>"}'
        )
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            ).strip()
            text = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            return parsed["decision"], parsed["reasoning"]
        except Exception as e:
            decision, reasoning = self._fallback.decide(question, context)
            return decision, f"[fallback after LLM error: {e}] {reasoning}"


def get_reasoner() -> BaseReasoner:
    """Factory: use the Anthropic reasoner if USE_LLM=1 and a key is present,
    otherwise the deterministic rule-based reasoner."""
    if os.environ.get("USE_LLM") == "1":
        return AnthropicReasoner()
    return RuleBasedReasoner()
