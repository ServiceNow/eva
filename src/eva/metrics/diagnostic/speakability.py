"""Speakability metric using LLM-as-judge (conversation-level).

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

import re
from typing import Any

from eva.metrics.base import MetricContext, PerTurnConversationJudgeMetric
from eva.metrics.registry import register_metric
from eva.models.config import PipelineType
from eva.utils.log_processing import AnnotationLabel


def _strip_annotation_labels(text: str) -> str:
    """Remove EVA-inserted annotation labels, keeping the agent's own line breaks.

    Unlike ``strip_labels``, newlines are preserved so that markdown lists and
    tables in the agent's text remain visible to the judge.
    """
    for label in AnnotationLabel:
        text = text.replace(label, "")
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


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
    version = "v0.3"
    description = "Debug metric: LLM judge evaluation of text voice-friendliness per turn"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE, PipelineType.AUDIO_LLM})
    rating_scale = (0, 1)

    def get_expected_turn_ids(self, context: MetricContext) -> list[int]:
        """Return turn IDs for intended assistant turns that are non-empty once labels are stripped."""
        return [
            turn_id
            for turn_id in sorted(context.intended_assistant_turns.keys())
            if _strip_annotation_labels(context.intended_assistant_turns[turn_id] or "")
        ]

    def format_transcript(self, context: MetricContext) -> str:
        """Format intended assistant turns for the judge prompt, without EVA annotation labels."""
        lines = []
        for turn_id in sorted(context.intended_assistant_turns.keys()):
            content = _strip_annotation_labels(context.intended_assistant_turns[turn_id] or "")
            if content:
                lines.append(f"[Turn {turn_id}]\n{content}")
        return "\n\n".join(lines)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {"assistant_turns_formatted": transcript_text}
