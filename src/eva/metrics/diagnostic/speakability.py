"""Speakability metric using LLM-as-judge (conversation-level).

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

from typing import Any

from eva.metrics.base import MetricContext, PerTurnConversationJudgeMetric
from eva.metrics.registry import register_metric
from eva.models.config import PipelineType


@register_metric
class SpeakabilityJudgeMetric(PerTurnConversationJudgeMetric):
    """LLM-based speakability metric (conversation-level).

    Evaluates all assistant turns at once for voice-friendliness:
    - Proper sentence breaks for natural speech
    - No awkward acronyms/abbreviations when spoken
    - Numbers formatted appropriately for TTS
    - Punctuation that guides natural pauses
    - No visual-only elements (tables, links, formatting)

    Rating scale: 0 (voice-unfriendly), 1 (voice-friendly)
    Normalized: Same as raw score (already 0-1)

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "speakability"
    description = "Debug metric: LLM judge evaluation of text voice-friendliness per turn"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE, PipelineType.AUDIO_LLM})
    rating_scale = (0, 1)

    def get_expected_turn_ids(self, context: MetricContext) -> list[int]:
        """Return turn IDs for non-empty intended assistant turns."""
        return [
            turn_id
            for turn_id in sorted(context.intended_assistant_turns.keys())
            if context.intended_assistant_turns[turn_id]
        ]

    def format_transcript(self, context: MetricContext) -> str:
        """Format intended assistant turns for the judge prompt."""
        lines = []
        for turn_id in sorted(context.intended_assistant_turns.keys()):
            content = context.intended_assistant_turns[turn_id]
            if content:
                lines.append(f"[Turn {turn_id}]\n{content}")
        return "\n\n".join(lines)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {"assistant_turns_formatted": transcript_text}
