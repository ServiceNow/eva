"""Conciseness metric using LLM-as-judge (conversation-level)."""

from typing import Any

from eva.metrics.base import MetricContext, PerTurnConversationJudgeMetric
from eva.metrics.registry import register_metric


@register_metric
class ConcisenessJudgeMetric(PerTurnConversationJudgeMetric):
    """LLM-based conciseness metric (conversation-level).

    Evaluates all assistant turns at once using the full conversation transcript
    (user + assistant) for context, then aggregates the scores using mean (default)
    or other aggregation methods.

    Rating scale: 3 (highly concise), 2 (adequate), 1 (not concise)
    Normalized: 3→1.0, 2→0.5, 1→0.0
    """

    name = "conciseness"
    description = "LLM judge evaluation of assistant response conciseness"
    category = "experience"
    rating_scale = (1, 3)

    def get_transcript_trace(self, context: MetricContext) -> list[dict]:
        """Use message-native traces for continuous-stream Telnyx conversations."""
        if context.is_continuous_assistant_stream and context.message_trace:
            return context.message_trace
        return super().get_transcript_trace(context)

    def get_expected_turn_ids(self, context: MetricContext) -> list[int]:
        """Return unique turn IDs from conversation trace, preserving order."""
        trace = self.get_transcript_trace(context)
        return list(dict.fromkeys(e.get("turn_id") for e in trace if e.get("turn_id") is not None))

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {"conversation_turns": transcript_text}

    def process_turn_item(self, item: dict, turn_id: int, rating: int | None, context: MetricContext) -> dict[str, Any]:
        """Extract and validate failure_modes from the judge response item."""
        if rating is None:
            return {"failure_modes": []}

        failure_modes = item.get("failure_modes", [])
        if isinstance(failure_modes, str):
            failure_modes = [failure_modes] if failure_modes else []
        elif isinstance(failure_modes, list):
            failure_modes = [str(fm) for fm in failure_modes if fm]
        else:
            failure_modes = []

        if rating == self.rating_scale[1] and failure_modes:
            self.logger.warning(
                f"[{context.record_id}] Turn {turn_id}: rating={rating} but failure_modes={failure_modes}; clearing"
            )
            failure_modes = []

        return {"failure_modes": failure_modes}
