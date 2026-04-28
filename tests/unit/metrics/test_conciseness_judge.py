"""Tests for ConcisenessJudgeMetric."""

import json
from unittest.mock import AsyncMock

import pytest

from eva.metrics.experience.conciseness import ConcisenessJudgeMetric

from .conftest import make_judge_metric, make_metric_context

SAMPLE_TURNS = [
    {"turn_id": 1, "role": "user", "content": "Hi, I need help rebooking my flight."},
    {"turn_id": 1, "role": "assistant", "content": "Sure, what's your confirmation number?"},
    {"turn_id": 2, "role": "user", "content": "ABC123"},
    {"turn_id": 2, "type": "tool_call", "tool_name": "get_reservation", "parameters": {"code": "ABC123"}},
    {"turn_id": 2, "type": "tool_response", "tool_name": "get_reservation", "tool_response": {"status": "ok"}},
    {"turn_id": 2, "role": "assistant", "content": "Found your reservation. When would you like to fly?"},
    {"turn_id": 3, "role": "user", "content": "March 25th please."},
    {"turn_id": 3, "role": "assistant", "content": "Done, you're rebooked for March 25th."},
]


@pytest.fixture
def metric():
    return make_judge_metric(ConcisenessJudgeMetric)


@pytest.mark.asyncio
async def test_all_turns_rated(metric):
    """All turns get valid ratings."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": 3, "explanation": "Concise", "failure_modes": []},
            {"turn_id": 2, "rating": 2, "explanation": "Slightly verbose", "failure_modes": ["verbosity_or_filler"]},
            {"turn_id": 3, "rating": 3, "explanation": "Concise", "failure_modes": []},
        ]
    )

    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_ratings"] == {1: 3, 2: 2, 3: 3}
    assert result.details["num_turns"] == 3
    assert result.details["num_evaluated"] == 3
    # mean of [3, 2, 3] = 2.667
    assert result.score == pytest.approx(2.667, abs=0.001)
    # normalized: [1.0, 0.5, 1.0] -> mean = 0.833
    assert result.normalized_score == pytest.approx(0.833, abs=0.001)


@pytest.mark.asyncio
async def test_surfaces_failure_mode_sub_metrics(metric):
    """Sub-metrics surface per-failure-mode rates across rated turns."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": 1, "explanation": "verbose", "failure_modes": ["verbosity_or_filler"]},
            {
                "turn_id": 2,
                "rating": 2,
                "explanation": "dense",
                "failure_modes": ["excess_information_density", "verbosity_or_filler"],
            },
            {"turn_id": 3, "rating": 3, "explanation": "clean", "failure_modes": []},
            {"turn_id": 4, "rating": None, "explanation": "user only", "failure_modes": []},
        ]
    )
    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.sub_metrics is not None
    expected_keys = {
        "verbosity_or_filler_rate",
        "excess_information_density_rate",
        "over_enumeration_or_list_exhaustion_rate",
        "contextually_disproportionate_detail_rate",
    }
    assert set(result.sub_metrics.keys()) == expected_keys
    # 2 out of 3 rated turns flagged verbosity_or_filler
    verbosity = result.sub_metrics["verbosity_or_filler_rate"]
    assert verbosity.name == "conciseness.verbosity_or_filler_rate"
    assert verbosity.score == pytest.approx(2 / 3, abs=0.001)
    assert verbosity.normalized_score == pytest.approx(2 / 3, abs=0.001)
    assert verbosity.details["count"] == 2
    assert verbosity.details["num_rated"] == 3
    assert set(verbosity.details["turn_ids"]) == {1, 2}
    # modes with zero occurrences still emitted at rate 0
    over_enum = result.sub_metrics["over_enumeration_or_list_exhaustion_rate"]
    assert over_enum.score == 0.0
    assert over_enum.details["count"] == 0


@pytest.mark.asyncio
async def test_null_rating_excluded_from_aggregation(metric):
    """Null ratings (not applicable) are stored but excluded from score."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": None, "explanation": "User-only turn", "failure_modes": []},
            {"turn_id": 2, "rating": 3, "explanation": "Concise", "failure_modes": []},
            {"turn_id": 3, "rating": 2, "explanation": "Ok", "failure_modes": ["verbosity_or_filler"]},
        ]
    )

    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_ratings"] == {1: None, 2: 3, 3: 2}
    assert result.details["num_evaluated"] == 2
    # mean of [3, 2] = 2.5
    assert result.score == 2.5
    # normalized: [1.0, 0.5] -> mean = 0.75
    assert result.normalized_score == 0.75


@pytest.mark.asyncio
async def test_invalid_rating_treated_as_none(metric):
    """Invalid ratings are stored as None with error explanation."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": 3, "explanation": "Good", "failure_modes": []},
            {"turn_id": 2, "rating": 5, "explanation": "Bad rating", "failure_modes": []},
            {"turn_id": 3, "rating": 3, "explanation": "Good", "failure_modes": []},
        ]
    )

    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_ratings"][2] is None
    assert "Invalid rating" in result.details["per_turn_explanations"][2]
    assert result.details["num_evaluated"] == 2


@pytest.mark.asyncio
async def test_string_rating_coerced_to_int(metric):
    """String ratings like '3' are coerced to int."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": "3", "explanation": "Good", "failure_modes": []},
            {"turn_id": 2, "rating": "2", "explanation": "Ok", "failure_modes": []},
            {"turn_id": 3, "rating": "1", "explanation": "Bad", "failure_modes": []},
        ]
    )

    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_ratings"] == {1: 3, 2: 2, 3: 1}


@pytest.mark.asyncio
async def test_failure_modes_cleared_for_rating_3(metric):
    """Rating 3 with failure_modes should have them cleared."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": 3, "explanation": "Good", "failure_modes": ["verbosity_or_filler"]},
            {"turn_id": 2, "rating": 3, "explanation": "Good", "failure_modes": []},
            {"turn_id": 3, "rating": 3, "explanation": "Good", "failure_modes": []},
        ]
    )

    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.details["per_turn_failure_modes"][1] == []


@pytest.mark.asyncio
async def test_unknown_turn_id_skipped(metric):
    """Turn IDs not in the transcript are skipped."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": 3, "explanation": "Good", "failure_modes": []},
            {"turn_id": 99, "rating": 3, "explanation": "Unknown", "failure_modes": []},
            {"turn_id": 3, "rating": 3, "explanation": "Good", "failure_modes": []},
        ]
    )

    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert 99 not in result.details["per_turn_ratings"]
    assert result.details["num_evaluated"] == 2


@pytest.mark.asyncio
async def test_no_response_from_judge(metric):
    """None response from LLM returns error."""
    metric.llm_client.generate_text = AsyncMock(return_value=(None, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error == "Failed to parse judge response"
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_all_null_ratings_returns_error(metric):
    """All null ratings should return an error."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": None, "explanation": "", "failure_modes": []},
            {"turn_id": 2, "rating": None, "explanation": "", "failure_modes": []},
            {"turn_id": 3, "rating": None, "explanation": "", "failure_modes": []},
        ]
    )

    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error == "All turns failed to evaluate"
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_single_dict_response_wrapped(metric):
    """Single dict response (not array) should be wrapped in list."""
    mock_response = json.dumps({"turn_id": 1, "rating": 3, "explanation": "Good", "failure_modes": []})

    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_ratings"] == {1: 3}
    assert result.details["num_evaluated"] == 1
