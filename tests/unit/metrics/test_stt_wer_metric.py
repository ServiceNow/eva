"""Tests for eva.metrics.debug.stt_wer compute method."""

import pytest

from eva.metrics.diagnostic.stt_wer import STTWERMetric

from .conftest import make_metric_context


class TestSTTWERMetricCompute:
    @pytest.mark.asyncio
    async def test_perfect_transcription(self):
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello world"},
            transcribed_user_turns={1: "hello world"},
        )
        result = await metric.compute(ctx)
        assert result.details["wer"] == 0.0
        assert result.details["accuracy"] == 1.0
        assert result.details["num_turns"] == 1

    @pytest.mark.asyncio
    async def test_some_errors(self):
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "the quick brown fox"},
            transcribed_user_turns={1: "the quick green fox"},
        )
        result = await metric.compute(ctx)
        assert result.details["wer"] > 0
        assert result.details["accuracy"] < 1.0
        assert result.normalized_score == result.details["accuracy"]

    @pytest.mark.asyncio
    async def test_no_common_turns(self):
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello"},
            transcribed_user_turns={2: "hello"},
        )
        result = await metric.compute(ctx)
        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_bracket_annotations_stripped(self):
        """Bracket annotations like [slow] should be removed before comparison."""
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "[slow] hello world [likely cut off]"},
            transcribed_user_turns={1: "hello world"},
        )
        result = await metric.compute(ctx)
        # After stripping brackets, both should be "hello world"
        assert result.details["wer"] == 0.0

    @pytest.mark.asyncio
    async def test_multiple_turns(self):
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "hello", 2: "goodbye", 3: "thank you"},
            transcribed_user_turns={1: "hello", 2: "goodbye", 3: "thank you"},
        )
        result = await metric.compute(ctx)
        assert result.details["num_turns"] == 3
        assert result.details["wer"] == 0.0
        assert len(result.details["per_turn_wer"]) == 3

    @pytest.mark.asyncio
    async def test_empty_turns_skipped(self):
        """Turns where reference or hypothesis is empty after stripping are skipped."""
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "[likely interruption]", 2: "hello world"},
            transcribed_user_turns={1: "", 2: "hello world"},
        )
        result = await metric.compute(ctx)
        # Turn 1 should be skipped (empty after stripping)
        assert result.details["num_turns"] == 1

    @pytest.mark.asyncio
    async def test_language_config(self):
        metric = STTWERMetric(config={"language": "en"})
        assert metric.language == "en"

    @pytest.mark.asyncio
    async def test_per_turn_errors_included(self):
        metric = STTWERMetric()
        ctx = make_metric_context(
            intended_user_turns={1: "the cat sat on the mat"},
            transcribed_user_turns={1: "the cat sit on a mat"},
        )
        result = await metric.compute(ctx)
        assert 1 in result.details["per_turn_wer"]
        assert 1 in result.details["per_turn_errors"]
        assert "error_summary" in result.details
