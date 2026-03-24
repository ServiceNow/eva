"""Response speed metric measuring latency between user and assistant.

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.
"""

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


@register_metric
class ResponseSpeedMetric(CodeMetric):
    """Response speed metric.

    Measures the elapsed time between the end of the user's utterance
    and the beginning of the assistant's response.

    Reports raw latency values in seconds — no normalization applied.

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "response_speed"
    description = "Debug metric: latency between user utterance end and assistant response start"
    category = "diagnostic"
    exclude_from_pass_at_k = True

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute response speed from Pipecat's UserBotLatencyObserver measurements."""
        try:
            # Check if we have response speed latencies from UserBotLatencyObserver
            if not context.response_speed_latencies:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=None,
                    error="No response latencies available (UserBotLatencyObserver data missing)",
                )

            # Use latencies measured by Pipecat's UserBotLatencyObserver
            # These measure the time from user stopped speaking to assistant started speaking
            speeds = []
            per_turn_speeds = []

            for response_speed in context.response_speed_latencies:
                # Filter out invalid values (negative or extremely large)
                if 0 < response_speed < 1000:  # Sanity check: under 1000 seconds
                    speeds.append(response_speed)
                    per_turn_speeds.append(round(response_speed, 3))
                else:
                    self.logger.warning(
                        f"[{context.record_id}] Unusual response speed detected and dropped: {response_speed} seconds"
                    )

            if not speeds:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=None,
                    error="No valid response speeds computed",
                )

            mean_speed = sum(speeds) / len(speeds)
            max_speed = max(speeds)

            return MetricScore(
                name=self.name,
                score=round(mean_speed, 3),  # Mean response speed in seconds
                normalized_score=None,  # Raw latency in seconds; not normalizable to [0,1]
                details={
                    "mean_speed_seconds": round(mean_speed, 3),
                    "max_speed_seconds": round(max_speed, 3),
                    "num_turns": len(speeds),
                    "per_turn_speeds": per_turn_speeds,
                },
            )

        except Exception as e:
            return self._handle_error(e, context)
