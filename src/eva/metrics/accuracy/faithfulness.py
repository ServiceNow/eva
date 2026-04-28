"""Faithfulness metric using LLM-as-judge (whole conversation)."""

import json
from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.pipeline_prompts import get_user_turns_disclaimer
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore

_FAITHFULNESS_DIMENSION_KEYS = (
    "fabricating_tool_parameters",
    "misrepresenting_tool_result",
    "violating_policies",
    "failing_to_disambiguate",
    "hallucination",
)

# --- Pipeline-specific prompt text for faithfulness evaluation ---

_CASCADE_DISAMBIGUATION_CONTEXT = (
    "Since the assistant is working from a speech-to-text transcript, it should account for potential "
    "transcription errors, and clarify any ambiguity in the user's intent, especially when they lead to "
    "write/irreversible operations. It's not needed to clarify if the tools called are simple lookups, "
    "but if the lookups fail, the assistant is expected to clarify the user's intent."
)

_S2S_DISAMBIGUATION_CONTEXT = (
    "Since the assistant processes raw audio directly (speech-to-speech), it should account for potential "
    "audio perception errors — mishearing letters, numbers, names, or codes is common with spoken input. "
    "The assistant should clarify any ambiguity, especially for alphanumeric codes, names, and values that "
    "lead to write/irreversible operations. It's not needed to clarify if the tools called are simple "
    "lookups, but if the lookups fail, the assistant is expected to clarify the user's intent. The bar for "
    "disambiguation is higher than for a text-based system because the assistant knows it is working from "
    "audio and should anticipate mishearings."
)


@register_metric
class FaithfulnessJudgeMetric(ConversationTextJudgeMetric):
    """LLM-based faithfulness metric (whole conversation).

    Evaluates whether the assistant remains faithful to information, policies,
    and instructions (no hallucinations, grounded tool calls, policy adherence,
    proper disambiguation).

    Rating scale: 1 (faithful), 0 (violations)
    Normalized: 1→1.0, 0→0.0
    """

    name = "faithfulness"
    description = (
        "LLM judge evaluation of whether the assistant remains faithful to information, policies, and instructions"
    )
    category = "accuracy"
    default_model = "us.anthropic.claude-opus-4-6-v1"
    rating_scale = (1, 3)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        user_turns_disclaimer = get_user_turns_disclaimer(context.is_audio_native)
        if context.is_audio_native:
            disambiguation_context = _S2S_DISAMBIGUATION_CONTEXT
        else:
            disambiguation_context = _CASCADE_DISAMBIGUATION_CONTEXT

        return {
            "agent_instructions": context.agent_instructions,
            "agent_role": context.agent_role,
            "available_tools": json.dumps(context.agent_tools, indent=4),
            "conversation_trace": transcript_text,
            "current_date_time": context.current_date_time,
            "user_turns_disclaimer": user_turns_disclaimer,
            "disambiguation_context": disambiguation_context,
        }

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
        dimensions = response.get("dimensions", {}) if isinstance(response, dict) else {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=dimensions,
            entry_keys=_FAITHFULNESS_DIMENSION_KEYS,
            flag_field="flagged",
            detail_fields=("rating", "evidence"),
        )

        analysis = {"dimensions": dimensions}
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
