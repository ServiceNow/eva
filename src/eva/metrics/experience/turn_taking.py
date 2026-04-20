"""Turn-taking metric using ElevenLabs latency data only (no LLM judge)."""

from typing import Any

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import aggregate_per_turn_scores
from eva.models.results import MetricScore

LABEL_EARLY = "Early / Interrupting"
LABEL_ON_TIME = "On-Time"
LABEL_LATE = "Late"

LABEL_TO_RATING: dict[str, int] = {
    LABEL_EARLY: -1,
    LABEL_ON_TIME: 0,
    LABEL_LATE: 1,
}


@register_metric
class TurnTakingMetric(CodeMetric):
    """Turn-taking metric computed from ElevenLabs per-turn latencies.

    Latency thresholds (user_end → assistant_start):
      latency < 200 ms             → Early / Interrupting  (-1)
      200 ms <= latency < 4000 ms  → On-Time               ( 0)
      latency >= 4000 ms           → Late                  (+1)

    Normalized: 1 - abs(aggregated rating). Perfect timing (all On-Time) = 1.0.
    """

    name = "turn_taking"
    description = "Turn-taking evaluation based on per-turn latency"
    category = "experience"
    rating_scale = (-1, 1)
    aggregation = "abs_mean"

    @staticmethod
    def _get_turn_ids_with_turn_taking(context: MetricContext) -> list[int]:
        """Return sorted turn IDs that have both user and assistant audio timestamps (excludes greeting)."""
        return sorted(
            context.audio_timestamps_user_turns.keys() & context.audio_timestamps_assistant_turns.keys() - {0}
        )

    def _compute_per_turn_latency_and_timing_labels(
        self,
        context: MetricContext,
        turn_keys: list[int],
    ) -> tuple[dict[int, float], dict[int, str]]:
        """Return {turn_id: latency} and {turn_id: timing_label} for each turn key."""
        latencies: dict[int, float] = {}
        labels: dict[int, str] = {}
        for turn_id in turn_keys:
            latency_s = context.latency_assistant_turns[turn_id]
            latencies[turn_id] = latency_s
            latency_ms = latency_s * 1000
            if latency_ms < 200:
                labels[turn_id] = LABEL_EARLY
            elif latency_ms < 4000:
                labels[turn_id] = LABEL_ON_TIME
            else:
                labels[turn_id] = LABEL_LATE
        return latencies, labels

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute turn-taking score from ElevenLabs per-turn latencies."""
        try:
            turn_keys = self._get_turn_ids_with_turn_taking(context)

            per_turn_latency, per_turn_timing_labels = self._compute_per_turn_latency_and_timing_labels(
                context, turn_keys
            )

            numeric_ratings: dict[int, int] = {tid: LABEL_TO_RATING[lbl] for tid, lbl in per_turn_timing_labels.items()}

            details: dict[str, Any] = {
                "per_turn_timing_labels": per_turn_timing_labels,
                "per_turn_latency": per_turn_latency,
                "num_turns": max(
                    max(context.audio_timestamps_user_turns, default=0),
                    max(context.audio_timestamps_assistant_turns, default=0),
                ),
                "num_evaluated": len(numeric_ratings),
            }

            if not numeric_ratings:
                self.logger.info(
                    f"[{context.record_id}] No turns with both user and assistant audio timestamps; skipping metric."
                )
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=None,
                    details=details,
                    error="No turns with both user and assistant audio timestamps",
                )

            aggregated_score = aggregate_per_turn_scores(list(numeric_ratings.values()), self.aggregation)
            if aggregated_score is None:
                self.logger.warning(
                    f"[{context.record_id}] Score aggregation returned None (method={self.aggregation})."
                )
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=None,
                    error="Aggregation failed",
                    details=details,
                )

            normalized_score = 1.0 - min(1.0, abs(aggregated_score))
            avg_rating = sum(numeric_ratings.values()) / len(numeric_ratings)

            details["aggregation"] = self.aggregation

            return MetricScore(
                name=self.name,
                score=round(avg_rating, 3),
                normalized_score=round(normalized_score, 3),
                details=details,
            )

        except Exception as e:
            return self._handle_error(e, context)
