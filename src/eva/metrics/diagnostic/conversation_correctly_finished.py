"""Conversation-correctly-finished diagnostic metric."""

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.diagnostic.conv_finish_classifier import (
    build_conv_finish_sub_metrics,
    build_final_turn_flag_sub_metrics,
    classify_conv_finish_failure,
    extract_conv_finish_signals,
)
from eva.metrics.processor import is_agent_timeout_on_user_turn, last_audio_speaker
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


@register_metric
class ConversationCorrectlyFinishedMetric(CodeMetric):
    """0.0 when the agent timed out on the user's final turn; 1.0 otherwise.

    On failures (score 0.0) it also emits per-cause diagnostic sub-metrics
    (``conversation_correctly_finished.<cause>_rate``) splitting *why* the agent went silent.
    Causes are only classified when the conversation failed — see
    ``docs/metrics/conv_finish_submetrics.md``.
    """

    name = "conversation_correctly_finished"
    version = "v0.2"
    description = "Diagnostic metric: 0.0 when agent failed to respond to the user's final turn"
    category = "diagnostic"
    exclude_from_pass_at_k = True

    async def compute(self, context: MetricContext) -> MetricScore:
        try:
            reason = context.conversation_ended_reason
            speaker = last_audio_speaker(
                context.audio_timestamps_user_turns,
                context.audio_timestamps_assistant_turns,
            )
            missed_turn = is_agent_timeout_on_user_turn(
                reason,
                context.audio_timestamps_user_turns,
                context.audio_timestamps_assistant_turns,
            )
            score = 0.0 if missed_turn else 1.0

            # Classify the cause only when the conversation actually failed.
            sub_metrics = None
            if missed_turn:
                human_reason = "conversation ended with inactivity_timeout and user was the last speaker"
                signals = extract_conv_finish_signals(context)
                classification = classify_conv_finish_failure(signals)
                sub_metrics = build_conv_finish_sub_metrics(classification, self.name)
                # Orthogonal input-characteristic flags (short / acknowledgement / spelled final turn).
                sub_metrics.update(build_final_turn_flag_sub_metrics(signals.user_final_words, self.name))
            elif reason == "inactivity_timeout":
                human_reason = f"inactivity_timeout but last speaker was {speaker!r}"
            else:
                human_reason = f"conversation ended with reason={reason!r}"

            return MetricScore(
                name=self.name,
                score=score,
                normalized_score=score,
                details={
                    "conversation_ended_reason": reason,
                    "last_audio_speaker": speaker,
                    "reason": human_reason,
                },
                sub_metrics=sub_metrics,
            )

        except Exception as e:
            return self._handle_error(e, context)
