"""Tests for TurnTakingMetric."""

import logging

import pytest

from eva.metrics.experience.turn_taking import TurnTakingMetric
from eva.models.results import MetricScore

from .conftest import make_metric_context


@pytest.fixture
def metric():
    m = TurnTakingMetric()
    m.logger = logging.getLogger("test_turn_taking")
    return m


class TestGetTurnIdsWithTurnTaking:
    def test_excludes_greeting_turn_0(self, metric):
        """Turn 0 (greeting) should be excluded."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(1.0, 2.0)], 2: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={0: [(0.1, 0.5)], 1: [(2.5, 3.0)], 2: [(6.5, 7.0)]},
        )
        turn_ids = metric._get_turn_ids_with_turn_taking(context)
        assert turn_ids == [1, 2]

    def test_only_includes_turns_in_both(self, metric):
        """Only include turns present in both user and assistant timestamp dicts."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(2.0, 3.0)], 3: [(5.0, 6.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 2.0)], 2: [(3.5, 4.0)]},
        )
        turn_ids = metric._get_turn_ids_with_turn_taking(context)
        assert turn_ids == [1, 2]


class TestComputePerTurnLatencyAndTimingLabels:
    def test_on_time_latency(self, metric):
        """Latency between 200ms and 4000ms -> On-Time."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 2.5)]},
        )
        latencies, labels = metric._compute_per_turn_latency_and_timing_labels(context, [1])
        assert latencies[1] == 0.5
        assert labels[1] == "On-Time"

    def test_early_latency(self, metric):
        """Latency < 200ms → Early / Interrupting."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(1.1, 2.0)]},
        )
        latencies, labels = metric._compute_per_turn_latency_and_timing_labels(context, [1])
        assert latencies[1] == pytest.approx(0.1, abs=1e-4)
        assert labels[1] == "Early / Interrupting"

    def test_late_latency(self, metric):
        """Latency >= 4000ms → Late."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(5.5, 6.5)]},
        )
        latencies, labels = metric._compute_per_turn_latency_and_timing_labels(context, [1])
        assert latencies[1] == pytest.approx(4.5, abs=1e-4)
        assert labels[1] == "Late"


class TestCompute:
    @pytest.mark.asyncio
    async def test_all_on_time_returns_perfect_score(self, metric):
        """All turns within 200–4000ms → On-Time → normalized_score = 1.0."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(4.0, 5.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 3.0)], 2: [(5.5, 7.0)]},
        )

        result = await metric.compute(context)

        assert isinstance(result, MetricScore)
        assert result.name == "turn_taking"
        assert result.error is None
        assert result.normalized_score == 1.0
        assert result.score == 0.0
        assert result.details["num_turns"] == 2
        assert result.details["num_evaluated"] == 2
        assert result.details["per_turn_timing_labels"] == {1: "On-Time", 2: "On-Time"}
        assert result.details["per_turn_latency"] == {1: 0.5, 2: 0.5}

    @pytest.mark.asyncio
    async def test_mixed_ratings_affect_normalized_score(self, metric):
        """One Early and one Late → abs_mean([-1, 1]) = 1.0 → normalized = 0.0."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(10.0, 11.0)]},
            # Turn 1: latency 0.1s → Early. Turn 2: latency 5.0s → Late.
            audio_timestamps_assistant_turns={1: [(1.1, 2.0)], 2: [(16.0, 17.0)]},
        )

        result = await metric.compute(context)

        assert result.error is None
        assert result.details["per_turn_timing_labels"] == {
            1: "Early / Interrupting",
            2: "Late",
        }
        assert result.normalized_score == 0.0

    @pytest.mark.asyncio
    async def test_no_timestamps_returns_skipped(self, metric):
        """No timestamp pairs → graceful skip, not an error."""
        context = make_metric_context(
            audio_timestamps_user_turns={},
            audio_timestamps_assistant_turns={},
        )

        result = await metric.compute(context)
        assert result.normalized_score is None
        assert result.error

    @pytest.mark.asyncio
    async def test_partial_timestamps_scores_only_paired_turns(self, metric):
        """A turn with only one side's timestamps is silently excluded."""
        context = make_metric_context(
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(4.0, 5.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 2.5)]},
        )

        result = await metric.compute(context)
        assert result.error is None
        assert result.details["per_turn_timing_labels"] == {1: "On-Time"}
        assert result.details["num_turns"] == 1
        assert result.details["num_evaluated"] == 1
