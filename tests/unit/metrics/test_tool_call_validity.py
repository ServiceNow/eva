"""Tests for ToolCallValidity metric."""

import pytest

from eva.metrics.diagnostic.tool_call_validity import CALL_ERROR_TYPES, ToolCallValidity

from .conftest import make_metric_context


@pytest.fixture
def metric():
    return ToolCallValidity()


@pytest.mark.asyncio
async def test_no_tool_calls(metric):
    """No tool calls should return perfect score."""
    context = make_metric_context(tool_params=[], tool_responses=[])
    result = await metric.compute(context)

    assert result.score == 1.0
    assert result.normalized_score == 1.0
    assert result.details["total_tool_calls"] == 0
    assert result.sub_metrics is not None
    assert "num_tool_calls" in result.sub_metrics
    assert result.sub_metrics["num_tool_calls"].score == 0.0
    assert result.sub_metrics["num_tool_calls"].normalized_score is None


@pytest.mark.asyncio
async def test_sub_metrics_num_tool_calls_and_error_rates(metric):
    """Sub-metrics surface num_tool_calls count and per-error-type rates."""
    tool_params = [
        {"tool_name": "get_reservation", "tool_parameters": {}},
        {"tool_name": "get_reservation", "tool_parameters": {}},
        {"tool_name": "rebook_flight", "tool_parameters": {}},
        {"tool_name": "search_rebooking_options", "tool_parameters": {}},
    ]
    tool_responses = [
        {
            "tool_name": "get_reservation",
            "tool_response": {"status": "error", "error_type": "invalid_parameter", "message": "bad"},
        },
        {
            "tool_name": "get_reservation",
            "tool_response": {"status": "error", "error_type": "invalid_parameter", "message": "bad"},
        },
        {
            "tool_name": "rebook_flight",
            "tool_response": {"status": "error", "error_type": "execution_error", "message": "boom"},
        },
        {
            "tool_name": "search_rebooking_options",
            "tool_response": {"status": "success"},
        },
    ]
    context = make_metric_context(tool_params=tool_params, tool_responses=tool_responses)
    result = await metric.compute(context)

    assert result.sub_metrics is not None
    count_sub = result.sub_metrics["num_tool_calls"]
    assert count_sub.name == "tool_call_validity.num_tool_calls"
    assert count_sub.score == 4.0
    assert count_sub.normalized_score is None

    # Every known error type has a sub-metric (rate = 0 when absent).
    for error_type in CALL_ERROR_TYPES:
        key = f"{error_type}_rate"
        assert key in result.sub_metrics

    inv = result.sub_metrics["invalid_parameter_rate"]
    assert inv.score == pytest.approx(0.5)
    assert inv.normalized_score == pytest.approx(0.5)
    assert inv.details == {"count": 2, "total_tool_calls": 4}

    exec_err = result.sub_metrics["execution_error_rate"]
    assert exec_err.score == pytest.approx(0.25)
    assert exec_err.details == {"count": 1, "total_tool_calls": 4}

    tool_not_found = result.sub_metrics["tool_not_found_rate"]
    assert tool_not_found.score == 0.0
    assert tool_not_found.details == {"count": 0, "total_tool_calls": 4}


@pytest.mark.asyncio
async def test_all_calls_correct(metric):
    """All successful tool calls should return perfect score."""
    tool_params = [
        {"tool_name": "get_reservation", "tool_parameters": {"confirmation_number": "ABC123", "last_name": "Doe"}},
        {"tool_name": "search_rebooking_options", "tool_parameters": {"origin": "JFK", "destination": "LAX"}},
    ]
    tool_responses = [
        {"tool_name": "get_reservation", "tool_response": {"status": "success", "reservation": {}}},
        {"tool_name": "search_rebooking_options", "tool_response": {"status": "success", "options": []}},
    ]
    context = make_metric_context(tool_params=tool_params, tool_responses=tool_responses)
    result = await metric.compute(context)

    assert result.score == 1.0
    assert result.details["valid_tool_calls"] == 2
    assert result.details["invalid_tool_calls"] == 0


@pytest.mark.asyncio
async def test_business_logic_errors_not_penalized(metric):
    """Errors like 'not_found' or 'verification_failed' should not count as format errors."""
    tool_params = [
        {"tool_name": "get_reservation", "tool_parameters": {"confirmation_number": "XXXXXX", "last_name": "Doe"}},
        {"tool_name": "rebook_flight", "tool_parameters": {"confirmation_number": "ABC123"}},
    ]
    tool_responses = [
        {
            "tool_name": "get_reservation",
            "tool_response": {
                "status": "error",
                "error_type": "not_found",
                "message": "No reservation found",
            },
        },
        {
            "tool_name": "rebook_flight",
            "tool_response": {
                "status": "error",
                "error_type": "no_seats_available",
                "message": "No seats",
            },
        },
    ]
    context = make_metric_context(tool_params=tool_params, tool_responses=tool_responses)
    result = await metric.compute(context)

    assert result.score == 1.0
    assert result.details["invalid_tool_calls"] == 0


@pytest.mark.asyncio
async def test_errors_penalized(metric):
    """Format validation errors should reduce the score."""
    tool_params = [
        {"tool_name": "get_reservation", "tool_parameters": {"confirmation_number": "ABC", "last_name": "Doe"}},
        {"tool_name": "get_flight_status", "tool_parameters": {"flight_number": "12345", "flight_date": "2026-03-20"}},
        {"tool_name": "get_reservation", "tool_parameters": {"confirmation_number": "ABC123", "last_name": "Doe"}},
    ]
    tool_responses = [
        {
            "tool_name": "get_reservation",
            "tool_response": {
                "status": "error",
                "error_type": "invalid_confirmation_number_format",
                "message": "Invalid confirmation_number 'ABC'",
            },
        },
        {
            "tool_name": "get_flight_status",
            "tool_response": {
                "status": "error",
                "error_type": "invalid_flight_number_format",
                "message": "Invalid flight_number '12345'",
            },
        },
        {
            "tool_name": "get_reservation",
            "tool_response": {
                "status": "success",
                "reservation": {},
            },
        },
    ]
    context = make_metric_context(tool_params=tool_params, tool_responses=tool_responses)
    result = await metric.compute(context)

    assert result.score == pytest.approx(1 / 3, abs=0.001)
    assert result.details["valid_tool_calls"] == 1
    assert result.details["invalid_tool_calls"] == 2
    assert len(result.details["errors"]) == 2
    assert result.details["errors"][0]["error_type"] == "invalid_confirmation_number_format"
    assert result.details["errors"][1]["error_type"] == "invalid_flight_number_format"


@pytest.mark.asyncio
async def test_tool_not_found_penalized(metric):
    """Calling a non-existent tool is a call correctness error."""
    tool_params = [
        {"tool_name": "nonexistent_tool", "tool_parameters": {}},
    ]
    tool_responses = [
        {
            "tool_name": "nonexistent_tool",
            "tool_response": {
                "status": "error",
                "error_type": "tool_not_found",
                "message": "Tool nonexistent_tool not found in configuration",
            },
        },
    ]
    context = make_metric_context(tool_params=tool_params, tool_responses=tool_responses)
    result = await metric.compute(context)

    assert result.score == 0.0
    assert result.details["errors"][0]["error_type"] == "tool_not_found"


@pytest.mark.asyncio
async def test_enum_errors_penalized(metric):
    """Invalid enum values (e.g. wrong rebooking_type) should be penalized."""
    tool_params = [
        {"tool_name": "rebook_flight", "tool_parameters": {"rebooking_type": "wrong_type"}},
    ]
    tool_responses = [
        {
            "tool_name": "rebook_flight",
            "tool_response": {
                "status": "error",
                "error_type": "invalid_rebooking_type",
                "message": "Invalid rebooking_type 'wrong_type'",
            },
        },
    ]
    context = make_metric_context(tool_params=tool_params, tool_responses=tool_responses)
    result = await metric.compute(context)

    assert result.score == 0.0
    assert result.details["errors"][0]["error_type"] == "invalid_rebooking_type"


@pytest.mark.asyncio
async def test_mixed_errors_and_successes(metric):
    """Mix of successes, format errors, and business errors."""
    tool_params = [
        {"tool_name": "get_reservation", "tool_parameters": {"confirmation_number": "ABC123"}},
        {"tool_name": "get_reservation", "tool_parameters": {"confirmation_number": "X"}},
        {"tool_name": "get_reservation", "tool_parameters": {"confirmation_number": "XXXXXX"}},
        {"tool_name": "search_rebooking_options", "tool_parameters": {"origin": "JFK"}},
    ]
    tool_responses = [
        {"tool_name": "get_reservation", "tool_response": {"status": "success", "reservation": {}}},
        {
            "tool_name": "get_reservation",
            "tool_response": {
                "status": "error",
                "error_type": "invalid_confirmation_number_format",
                "message": "bad",
            },
        },
        {
            "tool_name": "get_reservation",
            "tool_response": {
                "status": "error",
                "error_type": "not_found",
                "message": "No reservation found",
            },
        },
        {"tool_name": "search_rebooking_options", "tool_response": {"status": "success", "options": []}},
    ]
    context = make_metric_context(tool_params=tool_params, tool_responses=tool_responses)
    result = await metric.compute(context)

    # 1 format error out of 4 calls
    assert result.score == pytest.approx(3 / 4, abs=0.001)
    assert result.details["valid_tool_calls"] == 3
    assert result.details["invalid_tool_calls"] == 1


@pytest.mark.asyncio
async def test_error_details_include_parameters(metric):
    """Format error details should include the bad parameters."""
    tool_params = [
        {"tool_name": "get_reservation", "tool_parameters": {"confirmation_number": "X", "last_name": "Doe"}},
    ]
    tool_responses = [
        {
            "tool_name": "get_reservation",
            "tool_response": {
                "status": "error",
                "error_type": "invalid_confirmation_number_format",
                "message": "Invalid confirmation_number 'X'",
            },
        },
    ]
    context = make_metric_context(tool_params=tool_params, tool_responses=tool_responses)
    result = await metric.compute(context)

    error = result.details["errors"][0]
    assert error["tool_name"] == "get_reservation"
    assert error["parameters"] == {"confirmation_number": "X", "last_name": "Doe"}
    assert error["message"] == "Invalid confirmation_number 'X'"


@pytest.mark.asyncio
async def test_non_dict_tool_response_skipped(metric):
    """Non-dict tool_response values should be safely skipped."""
    tool_params = [
        {"tool_name": "some_tool", "tool_parameters": {}},
    ]
    tool_responses = [
        {"tool_name": "some_tool", "tool_response": "unexpected string"},
    ]
    context = make_metric_context(tool_params=tool_params, tool_responses=tool_responses)
    result = await metric.compute(context)

    assert result.score == 1.0
    assert result.details["total_tool_calls"] == 1


def test_call_error_types_includes_field_error_types():
    """CALL_ERROR_TYPES should include all error types from FIELD_ERROR_TYPES."""
    from eva.assistant.tools.airline_params import FIELD_ERROR_TYPES

    for error_type, _ in FIELD_ERROR_TYPES.values():
        assert error_type in CALL_ERROR_TYPES, f"{error_type} not in CALL_ERROR_TYPES"
