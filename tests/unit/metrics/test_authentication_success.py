"""Tests for AuthenticationSuccessMetric."""

import pytest

from eva.metrics.diagnostic.authentication_success import AuthenticationSuccessMetric

from .conftest import make_metric_context


@pytest.fixture
def metric():
    return AuthenticationSuccessMetric()


@pytest.mark.asyncio
async def test_no_tool_calls(metric):
    """No tool calls at all should return 0.0 with tool not found."""
    context = make_metric_context(tool_params=[], tool_responses=[])
    result = await metric.compute(context)

    assert result.score == 0.0
    assert result.normalized_score == 0.0
    assert result.details["get_reservation_found"] is False
    assert result.details["get_reservation_call_count"] == 0
    assert result.details["get_reservation_success_count"] == 0


@pytest.mark.asyncio
async def test_successful_get_reservation(metric):
    """A successful get_reservation call should return 1.0."""
    context = make_metric_context(
        tool_params=[],
        tool_responses=[
            {
                "tool_name": "get_reservation",
                "tool_response": {"status": "success", "reservation": {"confirmation_number": "ABC123"}},
            },
        ],
    )
    result = await metric.compute(context)

    assert result.score == 1.0
    assert result.normalized_score == 1.0
    assert result.details["get_reservation_found"] is True
    assert result.details["get_reservation_call_count"] == 1
    assert result.details["get_reservation_success_count"] == 1


@pytest.mark.asyncio
async def test_failed_get_reservation(metric):
    """A get_reservation call with error status should return 0.0."""
    context = make_metric_context(
        tool_params=[],
        tool_responses=[
            {
                "tool_name": "get_reservation",
                "tool_response": {"status": "error", "error_type": "not_found", "message": "No reservation found"},
            },
        ],
    )
    result = await metric.compute(context)

    assert result.score == 0.0
    assert result.details["get_reservation_found"] is True
    assert result.details["get_reservation_call_count"] == 1
    assert result.details["get_reservation_success_count"] == 0


@pytest.mark.asyncio
async def test_mixed_calls(metric):
    """One failed and one successful get_reservation plus other tools should return 1.0."""
    context = make_metric_context(
        tool_params=[],
        tool_responses=[
            {
                "tool_name": "get_reservation",
                "tool_response": {"status": "error", "error_type": "verification_failed", "message": "Bad last name"},
            },
            {
                "tool_name": "search_rebooking_options",
                "tool_response": {"status": "success", "options": []},
            },
            {
                "tool_name": "get_reservation",
                "tool_response": {"status": "success", "reservation": {}},
            },
        ],
    )
    result = await metric.compute(context)

    assert result.score == 1.0
    assert result.details["get_reservation_call_count"] == 2
    assert result.details["get_reservation_success_count"] == 1


@pytest.mark.asyncio
async def test_other_tools_only(metric):
    """Only non-get_reservation tools should return 0.0 with tool not found."""
    context = make_metric_context(
        tool_params=[],
        tool_responses=[
            {"tool_name": "search_rebooking_options", "tool_response": {"status": "success", "options": []}},
            {"tool_name": "rebook_flight", "tool_response": {"status": "success"}},
        ],
    )
    result = await metric.compute(context)

    assert result.score == 0.0
    assert result.details["get_reservation_found"] is False
    assert result.details["get_reservation_call_count"] == 0
