"""Conversation progression metric using LLM-as-judge (whole conversation)."""

from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore

_CONVERSATION_PROGRESSION_DIMENSION_KEYS = (
    "unnecessary_tool_calls",
    "information_loss",
    "redundant_statements",
    "question_quality",
)


@register_metric
class ConversationProgressionJudgeMetric(ConversationTextJudgeMetric):
    """LLM-based conversation progression metric (whole conversation).

    Evaluates whether the assistant consistently moved the conversation
    forward and made progress toward resolving the user's issue.

    Rating scale: 3 (excellent), 2 (ok), 1 (poor)
    Normalized: 3→1.0, 2→0.5, 1→0.0
    """

    name = "conversation_progression"
    description = "LLM judge evaluation of whether the assistant moved the conversation forward productively"
    category = "experience"
    rating_scale = (1, 3)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {"conversation_trace": transcript_text}

    def build_metric_score(
        self,
        rating: int,
        normalized: float,
        response: dict,
        prompt: str,
        context: MetricContext,
        raw_response: str | None = None,
    ) -> MetricScore:
        """Build MetricScore with analysis details and per-dimension issue-flag sub-metrics."""
        dimensions = response.get("dimensions", {}) or {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=dimensions,
            entry_keys=_CONVERSATION_PROGRESSION_DIMENSION_KEYS,
            flag_field="flagged",
            detail_fields=("rating", "evidence"),
        )

        analysis = {
            "dimensions": dimensions,
            "flags_count": response.get("flags_count", ""),
        }
        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "explanation": analysis,
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
            sub_metrics=sub_metrics or None,
        )
