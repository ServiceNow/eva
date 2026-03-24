"""Tests for TurnTakingMetric."""

import json
import logging

import pytest

from eva.metrics.experience.turn_taking import TurnTakingMetric
from eva.models.results import MetricScore

from .conftest import make_judge_metric, make_metric_context


@pytest.fixture
def metric():
    return make_judge_metric(TurnTakingMetric, mock_llm=True, logger_name="test_turn_taking")


class TestGetTurnIdsWithTurnTaking:
    def test_excludes_greeting_turn_0(self, metric):
        """Turn 0 (greeting) should be excluded."""
        context = make_metric_context(
            transcribed_user_turns={0: "Hi", 1: "Help me", 2: "Thanks"},
            transcribed_assistant_turns={0: "Hello", 1: "Sure", 2: "Bye"},
        )
        turn_ids = metric._get_turn_ids_with_turn_taking(context)
        assert 0 not in turn_ids
        assert turn_ids == [1, 2]

    def test_only_includes_turns_in_both(self, metric):
        """Only include turns present in both user and assistant dicts."""
        context = make_metric_context(
            transcribed_user_turns={1: "Help me", 2: "More", 3: "Extra user"},
            transcribed_assistant_turns={1: "Sure", 2: "Ok"},
        )
        turn_ids = metric._get_turn_ids_with_turn_taking(context)
        assert turn_ids == [1, 2]


class TestComputePerTurnLatencyAndTimingLabels:
    def test_on_time_latency(self, metric):
        """Latency between 200ms and 4000ms -> On-Time."""
        context = make_metric_context(
            transcribed_user_turns={1: "Hello"},
            transcribed_assistant_turns={1: "Hi"},
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 2.5)]},
        )
        latencies, labels = metric._compute_per_turn_latency_and_timing_labels(context, [1])
        assert latencies[1] == 0.5
        assert labels[1] == "On-Time"

    def test_early_latency(self, metric):
        """Latency < 200ms → Early / Interrupting."""
        context = make_metric_context(
            transcribed_user_turns={1: "Hello"},
            transcribed_assistant_turns={1: "Hi"},
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(1.1, 2.0)]},
        )
        latencies, labels = metric._compute_per_turn_latency_and_timing_labels(context, [1])
        assert latencies[1] == pytest.approx(0.1, abs=1e-4)
        assert labels[1] == "Early / Interrupting"

    def test_late_latency(self, metric):
        """Latency >= 4000ms → Late."""
        context = make_metric_context(
            transcribed_user_turns={1: "Hello"},
            transcribed_assistant_turns={1: "Hi"},
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(5.5, 6.5)]},
        )
        latencies, labels = metric._compute_per_turn_latency_and_timing_labels(context, [1])
        assert latencies[1] == pytest.approx(4.5, abs=1e-4)
        assert labels[1] == "Late"

    def test_missing_timestamps_returns_none(self, metric):
        """Missing timestamps → None latency and label."""
        context = make_metric_context(
            transcribed_user_turns={1: "Hello"},
            transcribed_assistant_turns={1: "Hi"},
            audio_timestamps_user_turns={},
            audio_timestamps_assistant_turns={},
        )
        latencies, labels = metric._compute_per_turn_latency_and_timing_labels(context, [1])
        assert latencies[1] is None
        assert labels[1] is None

    def test_missing_timestamps_last_turn_no_warning(self, metric, caplog):
        """Last turn missing timestamps should not produce a warning."""
        context = make_metric_context(
            transcribed_user_turns={1: "Hello", 2: "Bye"},
            transcribed_assistant_turns={1: "Hi", 2: "Goodbye"},
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 2.5)]},
        )
        turn_keys = [1, 2]
        with caplog.at_level(logging.WARNING):
            latencies, labels = metric._compute_per_turn_latency_and_timing_labels(context, turn_keys)

        assert latencies[1] is not None
        assert latencies[2] is None
        # Last turn (id=2, len(turn_keys)=2) should not warn
        assert "Missing audio timestamps at turn 2" not in caplog.text


class TestNumNotApplicable:
    def test_last_turn_missing_timestamps_is_not_applicable(self, metric):
        """Last turn with missing timestamps should count as not_applicable."""
        turn_keys = [1, 2, 3]
        skipped_turn_ids = {3}  # last turn missing
        valid_ratings = [0, -1]  # 2 valid ratings from turns 1,2

        last_turn = turn_keys[-1]
        last_turn_skipped = last_turn in skipped_turn_ids
        num_not_applicable = 1 if last_turn_skipped else 0
        num_evaluated = len(valid_ratings) + num_not_applicable

        assert num_not_applicable == 1
        assert num_evaluated == 3

    def test_middle_turn_missing_timestamps_is_not_counted(self, metric):
        """Non-last turn with missing timestamps should NOT count as not_applicable."""
        turn_keys = [1, 2, 3]
        skipped_turn_ids = {2}  # middle turn missing
        valid_ratings = [0, 1]  # 2 valid ratings from turns 1,3

        last_turn = turn_keys[-1]
        last_turn_skipped = last_turn in skipped_turn_ids
        num_not_applicable = 1 if last_turn_skipped else 0
        num_evaluated = len(valid_ratings) + num_not_applicable

        assert num_not_applicable == 0
        assert num_evaluated == 2  # middle turn missing is a real gap

    def test_no_missing_timestamps(self, metric):
        """All timestamps present → num_not_applicable = 0."""
        turn_keys = [1, 2, 3]
        skipped_turn_ids = set()
        valid_ratings = [0, -1, 1]

        last_turn = turn_keys[-1]
        last_turn_skipped = last_turn in skipped_turn_ids
        num_not_applicable = 1 if last_turn_skipped else 0
        num_evaluated = len(valid_ratings) + num_not_applicable

        assert num_not_applicable == 0
        assert num_evaluated == 3

    def test_only_last_turn_and_missing(self, metric):
        """Single turn that is also last, missing timestamps → not_applicable."""
        turn_keys = [1]
        skipped_turn_ids = {1}
        valid_ratings = []

        last_turn = turn_keys[-1]
        last_turn_skipped = last_turn in skipped_turn_ids
        num_not_applicable = 1 if last_turn_skipped else 0
        num_evaluated = len(valid_ratings) + num_not_applicable

        assert num_not_applicable == 1
        assert num_evaluated == 1

    def test_empty_turn_keys(self, metric):
        """No turns → num_not_applicable = 0."""
        turn_keys = []
        skipped_turn_ids = set()

        last_turn = turn_keys[-1] if turn_keys else None
        last_turn_skipped = last_turn is not None and last_turn in skipped_turn_ids
        num_not_applicable = 1 if last_turn_skipped else 0

        assert num_not_applicable == 0


class TestFormatConversationContext:
    def test_includes_expected_and_heard_text(self, metric):
        """Context should include both what was intended and what STT transcribed."""
        context = make_metric_context(
            transcribed_user_turns={1: "Help me rebook"},
            transcribed_assistant_turns={1: "Sure, let me check"},
            intended_user_turns={1: "Help me rebook my flight"},
            intended_assistant_turns={1: "Sure, let me check your reservation"},
            audio_timestamps_user_turns={1: [(0.0, 1.5)]},
            audio_timestamps_assistant_turns={1: [(2.0, 4.0)]},
            assistant_interrupted_turns=set(),
            user_interrupted_turns=set(),
        )

        result = metric._format_conversation_context(context, [1], {1: 0.5})

        assert 'Expected: "Help me rebook my flight"' in result
        assert 'Heard: "Help me rebook"' in result
        assert 'Expected: "Sure, let me check your reservation"' in result
        assert "duration=" in result

    def test_missing_timestamps_shows_skip_marker(self, metric):
        context = make_metric_context(
            transcribed_user_turns={1: "Hello"},
            transcribed_assistant_turns={1: "Hi"},
            intended_user_turns={1: "Hello"},
            intended_assistant_turns={1: "Hi"},
            audio_timestamps_user_turns={},
            audio_timestamps_assistant_turns={},
            assistant_interrupted_turns=set(),
            user_interrupted_turns=set(),
        )

        result = metric._format_conversation_context(context, [1], {1: None})
        assert "Skip consideration due to missing timestamps" in result

    def test_assistant_interruption_annotated(self, metric):
        context = make_metric_context(
            transcribed_user_turns={1: "Wait"},
            transcribed_assistant_turns={1: "Sorry"},
            intended_user_turns={1: "Wait"},
            intended_assistant_turns={1: "Sorry"},
            audio_timestamps_user_turns={1: [(3.0, 3.5)]},
            audio_timestamps_assistant_turns={1: [(3.2, 4.0)]},
            assistant_interrupted_turns={1},
            user_interrupted_turns=set(),
        )

        result = metric._format_conversation_context(context, [1], {1: -0.3})
        assert "assistant interrupted the user" in result

    def test_user_interruption_annotated(self, metric):
        context = make_metric_context(
            transcribed_user_turns={1: "Hello"},
            transcribed_assistant_turns={1: "Hi"},
            intended_user_turns={1: "Hello"},
            intended_assistant_turns={1: "Hi"},
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 2.5)]},
            assistant_interrupted_turns=set(),
            user_interrupted_turns={1},
        )

        result = metric._format_conversation_context(context, [1], {1: 0.5})
        assert "user interrupted the assistant" in result

    def test_multi_segment_shows_transitions(self, metric):
        """Multiple audio segments per turn should list per-segment latencies."""
        context = make_metric_context(
            transcribed_user_turns={1: "Hello world"},
            transcribed_assistant_turns={1: "Hi there"},
            intended_user_turns={1: "Hello world"},
            intended_assistant_turns={1: "Hi there"},
            audio_timestamps_user_turns={1: [(0.0, 0.5), (0.7, 1.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 2.0), (2.2, 2.8)]},
            assistant_interrupted_turns=set(),
            user_interrupted_turns=set(),
        )

        result = metric._format_conversation_context(context, [1], {1: 0.5})
        assert "user_end" in result and "assistant_start" in result


class TestCompute:
    @pytest.mark.asyncio
    async def test_happy_path_returns_normalized_score(self, metric):
        context = make_metric_context(
            transcribed_user_turns={1: "Help me", 2: "Thanks"},
            transcribed_assistant_turns={1: "Sure", 2: "Bye"},
            intended_user_turns={1: "Help me", 2: "Thanks"},
            intended_assistant_turns={1: "Sure", 2: "Bye"},
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(4.0, 5.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 3.0)], 2: [(5.5, 7.0)]},
            assistant_interrupted_turns=set(),
            user_interrupted_turns=set(),
        )

        metric.llm_client.generate_text.return_value = json.dumps(
            [
                {"turn_id": 1, "rating": 0, "label": "On-Time", "explanation": "Good timing"},
                {"turn_id": 2, "rating": 0, "label": "On-Time", "explanation": "Good timing"},
            ]
        )

        result = await metric.compute(context)

        assert result.name == "turn_taking"
        assert result.error is None
        assert result.normalized_score == 1.0  # all 0 ratings → abs_mean=0 → 1-0=1
        assert result.details["num_turns"] == 2
        assert result.details["agreement_with_latency_values"] == 1.0

    @pytest.mark.asyncio
    async def test_null_judge_response_returns_error(self, metric):
        context = make_metric_context(
            transcribed_user_turns={1: "Help"},
            transcribed_assistant_turns={1: "Sure"},
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 3.0)]},
            assistant_interrupted_turns=set(),
            user_interrupted_turns=set(),
        )

        metric.llm_client.generate_text.return_value = None

        result = await metric.compute(context)
        assert result.error == "No response from judge"
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_dict_response_instead_of_list_returns_error(self, metric):
        context = make_metric_context(
            transcribed_user_turns={1: "Help"},
            transcribed_assistant_turns={1: "Sure"},
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 3.0)]},
            assistant_interrupted_turns=set(),
            user_interrupted_turns=set(),
        )

        metric.llm_client.generate_text.return_value = json.dumps({"rating": 0})

        result = await metric.compute(context)
        assert "Unexpected response format" in result.error

    @pytest.mark.asyncio
    async def test_all_timestamps_missing_returns_skipped(self, metric):
        """All turns missing timestamps → graceful skip, not an error."""
        context = make_metric_context(
            transcribed_user_turns={1: "Hello"},
            transcribed_assistant_turns={1: "Hi"},
            audio_timestamps_user_turns={},
            audio_timestamps_assistant_turns={},
            assistant_interrupted_turns=set(),
            user_interrupted_turns=set(),
        )

        metric.llm_client.generate_text.return_value = json.dumps(
            [
                {"turn_id": 1, "rating": None, "label": "N/A", "explanation": "No timestamps"},
            ]
        )

        result = await metric.compute(context)
        assert result.normalized_score is None
        assert result.error is None
        assert result.details["skipped"] is True
        assert result.details["skipped_reason"] == "All turns have missing audio timestamps"

    @pytest.mark.asyncio
    async def test_partial_timestamps_skips_affected_turns(self, metric):
        """Turn with missing timestamps should be excluded from judge evaluation."""
        context = make_metric_context(
            transcribed_user_turns={1: "Hello", 2: "Bye"},
            transcribed_assistant_turns={1: "Hi", 2: "Goodbye"},
            intended_user_turns={1: "Hello", 2: "Bye"},
            intended_assistant_turns={1: "Hi", 2: "Goodbye"},
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 2.5)]},
            assistant_interrupted_turns=set(),
            user_interrupted_turns=set(),
        )

        metric.llm_client.generate_text.return_value = json.dumps(
            [
                {"turn_id": 1, "rating": 0, "label": "On-Time", "explanation": "Good"},
                {"turn_id": 2, "rating": 0, "label": "On-Time", "explanation": "Good"},
            ]
        )

        result = await metric.compute(context)
        assert result.error is None
        assert 2 in result.details["turns_missing_timestamps"]
        assert 1 in result.details["per_turn_judge_timing_ratings"]

    @pytest.mark.asyncio
    async def test_invalid_rating_produces_error(self, metric):
        context = make_metric_context(
            transcribed_user_turns={1: "Hello"},
            transcribed_assistant_turns={1: "Hi"},
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 2.5)]},
            assistant_interrupted_turns=set(),
            user_interrupted_turns=set(),
        )

        metric.llm_client.generate_text.return_value = json.dumps(
            [
                {"turn_id": 1, "rating": 5, "label": "On-Time", "explanation": "Invalid"},
            ]
        )

        result = await metric.compute(context)
        assert result.error is not None
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_llm_exception_returns_error_score(self, metric):
        context = make_metric_context(
            transcribed_user_turns={1: "Hello"},
            transcribed_assistant_turns={1: "Hi"},
            audio_timestamps_user_turns={1: [(0.0, 1.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 2.5)]},
            assistant_interrupted_turns=set(),
            user_interrupted_turns=set(),
        )

        metric.llm_client.generate_text.side_effect = RuntimeError("API down")

        result = await metric.compute(context)
        assert result.error is not None
        assert "API down" in result.error

    @pytest.mark.asyncio
    async def test_mixed_ratings_affect_normalized_score(self, metric):
        """Negative and positive ratings should produce a score < 1.0."""
        context = make_metric_context(
            transcribed_user_turns={1: "Hello", 2: "Bye"},
            transcribed_assistant_turns={1: "Hi", 2: "Goodbye"},
            intended_user_turns={1: "Hello", 2: "Bye"},
            intended_assistant_turns={1: "Hi", 2: "Goodbye"},
            audio_timestamps_user_turns={1: [(0.0, 1.0)], 2: [(4.0, 5.0)]},
            audio_timestamps_assistant_turns={1: [(1.5, 2.5)], 2: [(5.5, 6.5)]},
            assistant_interrupted_turns=set(),
            user_interrupted_turns=set(),
        )

        metric.llm_client.generate_text.return_value = json.dumps(
            [
                {"turn_id": 1, "rating": -1, "label": "Early / Interrupting", "explanation": "Too early"},
                {"turn_id": 2, "rating": 1, "label": "Late", "explanation": "Too slow"},
            ]
        )

        result = await metric.compute(context)
        assert result.error is None
        # abs_mean of [-1, 1] = 1.0, normalized = 1 - 1 = 0.0
        assert result.normalized_score == 0.0
        assert len(result.details["per_turn_judge_timing_ratings"]) == 2

    @pytest.mark.asyncio
    async def test_greeting_only_produces_no_turns(self, metric):
        """Only turn 0 (greeting) → no turns to evaluate."""
        context = make_metric_context(
            transcribed_user_turns={0: "Hi"},
            transcribed_assistant_turns={0: "Hello"},
            audio_timestamps_user_turns={},
            audio_timestamps_assistant_turns={},
            assistant_interrupted_turns=set(),
            user_interrupted_turns=set(),
        )

        metric.llm_client.generate_text.return_value = json.dumps([])

        result = await metric.compute(context)
        assert isinstance(result, MetricScore)
