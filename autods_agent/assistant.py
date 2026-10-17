"""
DataScienceAssistant
=====================
Phase 9 – Context-aware AI Data Scientist Assistant.

NOT a generic chatbot. Builds a context object from the actual run results
(profile, preprocessing decisions, model leaderboard, metrics, SHAP, etc.)
and answers data-science questions about THAT specific run.

Two backends (same interface as the Reasoner pattern):
  - RuleBasedAnswerer (default): template-driven, composes answers from
    real numbers in the context. Never invents numbers.
  - LLM Answerer (USE_LLM=1): sends context + question to Claude with a
    strict instruction to only use provided context, falls back to
    rule-based on error.
"""

from __future__ import annotations

import os
import json
import warnings
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_assistant_context(
    profile: dict,
    eda_result: Optional[dict],
    preprocess_explanations: List[str],
    anomaly_result: Optional[dict],
    imbalance_result: Optional[dict],
    tuning_results: dict,
    eval_results: dict,
    explain_results: dict,
    feature_selection_result: Optional[dict] = None,
) -> dict:
    """
    Assembles a single context dict from all pipeline artifacts.
    This is what the assistant queries — always real numbers, never made-up.
    """
    ctx: Dict[str, Any] = {}

    # Dataset profile
    ctx["dataset"] = {
        "n_rows":    profile.get("n_rows"),
        "n_cols":    profile.get("n_cols"),
        "size_mb":   profile.get("size_mb"),
        "task_type": profile.get("task_type"),
        "backend":   profile.get("backend"),
        "target_col": profile.get("target_col"),
    }
    if eda_result:
        ctx["dataset"]["target_subtype"] = eda_result.get("extended_profile", {}).get("target_subtype")
        ctx["dataset"]["n_duplicates"]   = eda_result.get("extended_profile", {}).get("n_duplicates")
        ctx["dataset"]["outlier_summary"] = eda_result.get("outlier_summary", {})

    # Preprocessing
    ctx["preprocessing"] = {
        "explanations": preprocess_explanations,
    }

    # Anomaly detection
    if anomaly_result:
        ctx["anomaly"] = {
            "if_pct":  anomaly_result.get("if_anomaly_pct"),
            "lof_pct": anomaly_result.get("lof_anomaly_pct"),
        }

    # Imbalance
    if imbalance_result:
        ctx["imbalance"] = {
            "is_imbalanced":    imbalance_result.get("is_imbalanced"),
            "minority_ratio":   imbalance_result.get("minority_ratio"),
            "best_strategy":    imbalance_result.get("best_strategy"),
            "strategy_comparison": imbalance_result.get("strategy_comparison", {}),
        }

    # Model leaderboard
    ctx["leaderboard"] = {
        name: {"best_cv_score": res.get("best_score"),
               "best_params": res.get("best_params")}
        for name, res in tuning_results.items()
    }

    # Evaluation
    ctx["evaluation"] = {
        "scores":           eval_results.get("scores", {}),
        "metrics_detail":   eval_results.get("metrics_detail", {}),
        "final_model":      eval_results.get("final_model_name"),
    }

    # SHAP / explainability
    if explain_results:
        ctx["explainability"] = {
            "method": explain_results.get("method"),
            "top_features": explain_results.get("ranked_importance", [])[:10],
        }

    # Feature selection
    if feature_selection_result:
        ctx["feature_selection"] = feature_selection_result

    return ctx


# ---------------------------------------------------------------------------
# Rule-based answerer
# ---------------------------------------------------------------------------

class RuleBasedAnswerer:
    """Template-driven answerer that composes answers from real context numbers."""

    def answer(self, question: str, context: dict) -> str:
        q_lower = question.lower()

        # Model performance questions
        if any(w in q_lower for w in ["best model", "which model", "final model", "winner"]):
            return self._answer_best_model(context)

        if any(w in q_lower for w in ["feature", "important", "shap", "top feature"]):
            return self._answer_features(context)

        if any(w in q_lower for w in ["recall", "low recall", "precision"]):
            return self._answer_recall(context)

        if any(w in q_lower for w in ["imbalance", "imbalanced", "minority", "class ratio"]):
            return self._answer_imbalance(context)

        if any(w in q_lower for w in ["anomaly", "outlier", "anomalous"]):
            return self._answer_anomaly(context)

        if any(w in q_lower for w in ["preprocess", "impute", "encode", "scale", "missing"]):
            return self._answer_preprocessing(context)

        if any(w in q_lower for w in ["overfit", "generaliz", "generalis"]):
            return self._answer_overfitting(context)

        if any(w in q_lower for w in ["dataset", "data", "rows", "columns", "size"]):
            return self._answer_dataset(context)

        return (
            "I can answer questions about: model selection & performance, feature importance, "
            "preprocessing decisions, class imbalance, anomaly detection, and dataset statistics. "
            "Try asking: 'Why did the final model win?', 'Which features matter most?', "
            "'Why is recall low?', or 'What preprocessing was applied?'"
        )

    # ------------------------------------------------------------------
    def _answer_best_model(self, ctx: dict) -> str:
        ev = ctx.get("evaluation", {})
        final = ev.get("final_model", "N/A")
        scores = ev.get("scores", {})
        lb = ctx.get("leaderboard", {})
        if not scores:
            return "No evaluation data available yet."
        sorted_sc = sorted(scores.items(), key=lambda x: -x[1])
        top_str = "; ".join(f"{n}: {s:.4f}" for n, s in sorted_sc)
        metrics = ev.get("metrics_detail", {}).get(final, {})
        m_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        params = lb.get(final, {}).get("best_params", {})
        return (
            f"The final model selected by the agent is **{final}**.\n\n"
            f"**Leaderboard scores:** {top_str}\n\n"
            f"**{final} test metrics:** {m_str}\n\n"
            f"**Best hyperparameters:** {json.dumps(params)}\n\n"
            f"It was chosen because it achieved the highest held-out test score "
            f"({scores.get(final, 0):.4f}) among all tuned candidates."
        )

    def _answer_features(self, ctx: dict) -> str:
        exp = ctx.get("explainability", {})
        method = exp.get("method", "unknown")
        top = exp.get("top_features", [])
        if not top:
            return "Feature importance data is not available."
        feat_str = "\n".join(
            f"{i+1}. **{f}**: {v:.4f}" for i, (f, v) in enumerate(top[:10])
        )
        return (
            f"Feature importance was computed using **{method}**.\n\n"
            f"**Top features:**\n{feat_str}\n\n"
            f"Higher values mean the feature had a larger average impact on model predictions."
        )

    def _answer_recall(self, ctx: dict) -> str:
        ev = ctx.get("evaluation", {})
        final = ev.get("final_model", "")
        metrics = ev.get("metrics_detail", {}).get(final, {})
        recall = metrics.get("recall_macro", metrics.get("recall", None))
        imb = ctx.get("imbalance", {})
        parts = []
        if recall is not None:
            parts.append(f"The recall for {final} is **{recall:.4f}**.")
        if imb.get("is_imbalanced"):
            mr = imb.get("minority_ratio", 0)
            parts.append(
                f"The dataset is imbalanced (minority class ratio: {mr:.2%}), "
                f"which typically suppresses recall on the minority class. "
                f"Best imbalance strategy applied: **{imb.get('best_strategy')}**."
            )
        else:
            parts.append("The dataset is not significantly imbalanced.")
        parts.append(
            "To improve recall: consider adjusting the classification threshold, "
            "using SMOTE, or weighting minority class more heavily."
        )
        return "\n\n".join(parts)

    def _answer_imbalance(self, ctx: dict) -> str:
        imb = ctx.get("imbalance", {})
        if not imb:
            return "Imbalance analysis was not run (may not be a classification task)."
        if not imb.get("is_imbalanced"):
            mr = imb.get("minority_ratio", 0)
            return f"The dataset is balanced — minority class ratio is {mr:.2%}, above the 20% threshold."
        mr = imb.get("minority_ratio", 0)
        best = imb.get("best_strategy", "none")
        comp = imb.get("strategy_comparison", {})
        comp_str = "; ".join(
            f"{k}: F1={v.get('f1_macro', 'N/A'):.4f}" if isinstance(v, dict) and "f1_macro" in v else f"{k}: error"
            for k, v in comp.items()
        )
        return (
            f"The dataset is **imbalanced** — minority class is {mr:.2%} of training data.\n\n"
            f"Strategies tested: {comp_str}\n\n"
            f"Best strategy: **{best}** (highest macro-F1)."
        )

    def _answer_anomaly(self, ctx: dict) -> str:
        an = ctx.get("anomaly", {})
        if not an:
            return "Anomaly detection was not run."
        return (
            f"Anomaly detection results:\n"
            f"- **Isolation Forest**: {an.get('if_pct', 0):.2f}% rows flagged as anomalous\n"
            f"- **Local Outlier Factor**: {an.get('lof_pct', 0):.2f}% rows flagged as anomalous\n\n"
            f"By default, anomalies are flagged only and not removed from training."
        )

    def _answer_preprocessing(self, ctx: dict) -> str:
        exps = ctx.get("preprocessing", {}).get("explanations", [])
        if not exps:
            return "Preprocessing explanation log is empty."
        return "**Preprocessing decisions made by the agent:**\n\n" + "\n".join(f"- {e}" for e in exps)

    def _answer_overfitting(self, ctx: dict) -> str:
        ev = ctx.get("evaluation", {})
        lb = ctx.get("leaderboard", {})
        final = ev.get("final_model", "")
        cv_score = lb.get(final, {}).get("best_cv_score")
        test_score = ev.get("scores", {}).get(final)
        if cv_score and test_score:
            gap = abs(cv_score - test_score)
            verdict = "appears well-generalised" if gap < 0.05 else "may have mild overfitting"
            return (
                f"For {final}: CV score = {cv_score:.4f}, test score = {test_score:.4f}, "
                f"gap = {gap:.4f}. The model **{verdict}**."
            )
        return "CV and test scores not available for overfitting comparison."

    def _answer_dataset(self, ctx: dict) -> str:
        ds = ctx.get("dataset", {})
        return (
            f"**Dataset summary:**\n"
            f"- Rows: {ds.get('n_rows', 'N/A'):,}\n"
            f"- Columns: {ds.get('n_cols', 'N/A')}\n"
            f"- Size: {ds.get('size_mb', 'N/A'):.2f} MB\n"
            f"- Task type: {ds.get('task_type', 'N/A')}\n"
            f"- Target: `{ds.get('target_col', 'N/A')}`\n"
            f"- Execution backend: {ds.get('backend', 'N/A')}\n"
            f"- Duplicates: {ds.get('n_duplicates', 'N/A')}\n"
        )


# ---------------------------------------------------------------------------
# LLM answerer
# ---------------------------------------------------------------------------

class LLMAnswerer:
    """Delegates to Claude with full context; falls back to RuleBasedAnswerer."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self._client = None
        self._fallback = RuleBasedAnswerer()
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
        except Exception:
            pass

    def answer(self, question: str, context: dict) -> str:
        if self._client is None:
            return self._fallback.answer(question, context)
        try:
            ctx_str = json.dumps(context, default=str)[:6000]  # keep under token limit
            prompt = (
                "You are an AI data scientist assistant embedded inside AutoDS-Agent 2.0.\n"
                "Answer the user's question using ONLY the data provided in the run context below.\n"
                "Never invent numbers not present in the context. Be concise (<=200 words).\n\n"
                f"Run context (JSON):\n{ctx_str}\n\n"
                f"User question: {question}"
            )
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
        except Exception as e:
            logger.warning(f"LLM answer failed, falling back: {e}")
            return self._fallback.answer(question, context)


# ---------------------------------------------------------------------------
# DataScienceAssistant (public API)
# ---------------------------------------------------------------------------

class DataScienceAssistant:
    """Context-aware assistant; delegates to LLM or rule-based answerer."""

    def __init__(self, log):
        self.log = log
        self.context: dict = {}
        if os.environ.get("USE_LLM") == "1":
            self._answerer = LLMAnswerer()
        else:
            self._answerer = RuleBasedAnswerer()

    def set_context(self, context: dict) -> None:
        """Load run-specific context."""
        self.context = context

    def ask(self, question: str) -> str:
        """Answer a user question about the run."""
        if not self.context:
            return "No run context available. Please run the pipeline first."
        answer = self._answerer.answer(question, self.context)
        self.log.add(
            "Assistant",
            f"answered: {question[:60]}",
            f"Replied via {type(self._answerer).__name__}.",
        )
        return answer
