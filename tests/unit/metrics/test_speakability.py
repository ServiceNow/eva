"""Tests for SpeakabilityJudgeMetric."""

import json
from unittest.mock import AsyncMock

import pytest

from eva.metrics.diagnostic.speakability import SpeakabilityJudgeMetric

from .conftest import make_judge_metric, make_metric_context


@pytest.fixture
def metric():
    return make_judge_metric(SpeakabilityJudgeMetric)


@pytest.mark.asyncio
async def test_all_turns_speakable(metric):
    """All turns rated as voice-friendly."""
    context = make_metric_context(
        intended_assistant_turns={1: "Hello, how can I help?", 2: "Your flight is confirmed.", 3: "Goodbye!"},
    )

    metric.llm_client.generate_text = AsyncMock(
        return_value=(
            json.dumps(
                [
                    {"turn_id": 1, "rating": 1, "explanation": "Natural speech"},
                    {"turn_id": 2, "rating": 1, "explanation": "Natural speech"},
                    {"turn_id": 3, "rating": 1, "explanation": "Natural speech"},
                ]
            ),
            None,
        )
    )

    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_ratings"] == {1: 1, 2: 1, 3: 1}
    assert result.score == 1.0
    assert result.normalized_score == 1.0


@pytest.mark.asyncio
async def test_mixed_ratings(metric):
    """Mix of speakable and non-speakable turns."""
    context = make_metric_context(
        intended_assistant_turns={1: "Hello!", 2: "See table below:\n| Col1 | Col2 |", 3: "Goodbye!"},
    )

    metric.llm_client.generate_text = AsyncMock(
        return_value=(
            json.dumps(
                [
                    {"turn_id": 1, "rating": 1, "explanation": "Good"},
                    {"turn_id": 2, "rating": 0, "explanation": "Contains table"},
                    {"turn_id": 3, "rating": 1, "explanation": "Good"},
                ]
            ),
            None,
        )
    )

    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_ratings"] == {1: 1, 2: 0, 3: 1}
    assert result.details["num_evaluated"] == 3
    # mean of [1, 0, 1] = 0.667
    assert result.score == pytest.approx(0.667, abs=0.001)


@pytest.mark.asyncio
async def test_empty_turns_skipped(metric):
    """Empty TTS text turns should be excluded."""
    context = make_metric_context(
        intended_assistant_turns={1: "Hello!", 2: "", 3: "Goodbye!"},
    )

    metric.llm_client.generate_text = AsyncMock(
        return_value=(
            json.dumps(
                [
                    {"turn_id": 1, "rating": 1, "explanation": "Good"},
                    {"turn_id": 3, "rating": 1, "explanation": "Good"},
                ]
            ),
            None,
        )
    )

    result = await metric.compute(context)

    assert result.error is None
    assert result.details["num_turns"] == 2
    assert 2 not in result.details["per_turn_ratings"]


@pytest.mark.asyncio
async def test_turn_ids_preserved(metric):
    """Turn IDs from context should be preserved in output."""
    context = make_metric_context(
        intended_assistant_turns={3: "First response", 7: "Second response"},
    )

    metric.llm_client.generate_text = AsyncMock(
        return_value=(
            json.dumps(
                [
                    {"turn_id": 3, "rating": 1, "explanation": "Good"},
                    {"turn_id": 7, "rating": 0, "explanation": "Bad"},
                ]
            ),
            None,
        )
    )

    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_ratings"] == {3: 1, 7: 0}


@pytest.mark.asyncio
async def test_null_rating_excluded(metric):
    """Null ratings should be stored but excluded from aggregation."""
    context = make_metric_context(
        intended_assistant_turns={1: "Hello!", 2: "Help?", 3: "Bye!"},
    )

    metric.llm_client.generate_text = AsyncMock(
        return_value=(
            json.dumps(
                [
                    {"turn_id": 1, "rating": 1, "explanation": "Good"},
                    {"turn_id": 2, "rating": None, "explanation": "Not applicable"},
                    {"turn_id": 3, "rating": 1, "explanation": "Good"},
                ]
            ),
            None,
        )
    )

    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_ratings"][2] is None
    assert result.details["num_evaluated"] == 2


@pytest.mark.asyncio
async def test_no_response_from_judge(metric):
    """None response from LLM returns error."""
    context = make_metric_context(intended_assistant_turns={1: "Hello!", 2: "Bye!"})

    metric.llm_client.generate_text = AsyncMock(return_value=(None, None))

    result = await metric.compute(context)

    assert result.error == "Failed to parse judge response"
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_all_null_ratings(metric):
    """All null ratings should return error."""
    context = make_metric_context(intended_assistant_turns={1: "Hello!", 2: "Bye!"})

    metric.llm_client.generate_text = AsyncMock(
        return_value=(
            json.dumps(
                [
                    {"turn_id": 1, "rating": None, "explanation": ""},
                    {"turn_id": 2, "rating": None, "explanation": ""},
                ]
            ),
            None,
        )
    )

    result = await metric.compute(context)

    assert result.error == "All turns failed to evaluate"
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_no_turns_to_evaluate(metric):
    """No assistant turns should return error."""
    context = make_metric_context(intended_assistant_turns={})

    result = await metric.compute(context)

    assert result.error == "No turns to evaluate"
    assert result.score == 0.0
