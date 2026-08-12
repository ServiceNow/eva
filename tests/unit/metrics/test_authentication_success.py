"""Tests for AuthenticationSuccessMetric."""

import pytest

from eva.metrics.diagnostic.authentication_success import AuthenticationSuccessMetric

from .conftest import make_metric_context

SUCCESS_RESPONSE = {"status": "success"}
FAILURE_RESPONSE = {"status": "error"}


@pytest.fixture
def metric():
    return AuthenticationSuccessMetric()


@pytest.mark.asyncio
async def test_session_matches_expected(metric):
    """Final session matching expected session exactly should score 1.0."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123", "last_name": "Doe"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123", "last_name": "doe"}},
    )
    result = await metric.compute(ctx)

    assert result.score == 1.0
    assert result.normalized_score == 1.0
    assert result.details["mismatches"] == {}


@pytest.mark.asyncio
async def test_session_is_superset(metric):
    """Final session with extra keys beyond expected should still score 1.0."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123", "last_name": "doe"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123", "last_name": "doe", "extra_key": "value"}},
    )
    result = await metric.compute(ctx)

    assert result.score == 1.0
    assert result.details["mismatches"] == {}


@pytest.mark.asyncio
async def test_wrong_confirmation_number(metric):
    """Final session with wrong confirmation number should score 0.0."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123", "last_name": "doe"}},
        final_scenario_db={"session": {"confirmation_number": "WRONG1", "last_name": "doe"}},
    )
    result = await metric.compute(ctx)

    assert result.score == 0.0
    assert result.normalized_score == 0.0
    assert "confirmation_number" in result.details["mismatches"]


@pytest.mark.asyncio
async def test_wrong_last_name(metric):
    """Final session with wrong last name should score 0.0."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123", "last_name": "doe"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123", "last_name": "smith"}},
    )
    result = await metric.compute(ctx)

    assert result.score == 0.0
    assert "last_name" in result.details["mismatches"]


@pytest.mark.asyncio
async def test_empty_final_session(metric):
    """No session written (agent never authenticated) should score 0.0."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123", "last_name": "doe"}},
        final_scenario_db={},
    )
    result = await metric.compute(ctx)

    assert result.score == 0.0
    assert result.details["actual_session"] == {}
    assert len(result.details["mismatches"]) == 2


@pytest.mark.asyncio
async def test_no_expected_session(metric):
    """Missing expected session key should skip auth check and score 1.0."""
    ctx = make_metric_context(
        expected_scenario_db={},
        final_scenario_db={},
    )
    result = await metric.compute(ctx)

    assert result.score is None
    assert result.skipped is True
    assert "skipping" in result.details["reason"]


@pytest.mark.asyncio
async def test_auth_first_try_success_true(metric):
    """Auth succeeding with every auth tool called exactly once should score auth_first_try_success=1.0."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123"}},
        agent_tools=[{"name": "get_reservation", "tool_type": "auth"}],
        tool_responses=[{"tool_name": "get_reservation", "tool_response": SUCCESS_RESPONSE}],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is not None
    sub = result.sub_metrics["auth_first_try_success"]
    assert sub.score == 1.0
    assert sub.normalized_score == 1.0
    assert sub.skipped is False


@pytest.mark.asyncio
async def test_auth_first_try_success_false(metric):
    """Auth succeeding but only after a retry on an auth tool should score auth_first_try_success=0.0."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123"}},
        agent_tools=[{"name": "get_reservation", "tool_type": "auth"}],
        tool_responses=[
            {"tool_name": "get_reservation", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "get_reservation", "tool_response": SUCCESS_RESPONSE},
        ],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is not None
    sub = result.sub_metrics["auth_first_try_success"]
    assert sub.score == 0.0
    assert sub.normalized_score == 0.0
    assert sub.skipped is False


@pytest.mark.asyncio
async def test_auth_first_try_success_multiple_tools_one_retried(metric):
    """Auth succeeding with one of several auth tools needing a retry should score auth_first_try_success=0.0."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123"}},
        agent_tools=[
            {"name": "get_reservation", "tool_type": "auth"},
            {"name": "verify", "tool_type": "auth"},
        ],
        tool_responses=[
            {"tool_name": "get_reservation", "tool_response": SUCCESS_RESPONSE},
            {"tool_name": "verify", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "verify", "tool_response": SUCCESS_RESPONSE},
        ],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is not None
    sub = result.sub_metrics["auth_first_try_success"]
    assert sub.score == 0.0
    assert sub.normalized_score == 0.0
    assert sub.skipped is False


@pytest.mark.asyncio
async def test_num_auth_calls_success_single_tool(metric):
    """Auth succeeding with a single auth tool called once should score num_auth_calls=1.0."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123"}},
        agent_tools=[{"name": "get_reservation", "tool_type": "auth"}],
        tool_responses=[{"tool_name": "get_reservation", "tool_response": SUCCESS_RESPONSE}],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is not None
    sub = result.sub_metrics["num_auth_calls"]
    assert sub.score == 1.0
    assert sub.normalized_score is None
    assert sub.skipped is False
    assert sub.details["num_auth_calls_per_tool"] == {"get_reservation": 1}
    assert sub.details["num_auth_tools"] == 1

    assert "num_auth_calls_on_failure" not in result.sub_metrics


@pytest.mark.asyncio
async def test_num_auth_calls_success_multiple_calls(metric):
    """Auth succeeding after multiple calls to the same tool should average those calls."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123"}},
        agent_tools=[{"name": "get_reservation", "tool_type": "auth"}],
        tool_responses=[
            {"tool_name": "get_reservation", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "get_reservation", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "get_reservation", "tool_response": SUCCESS_RESPONSE},
        ],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is not None
    sub = result.sub_metrics["num_auth_calls"]
    assert sub.score == 3.0
    assert sub.normalized_score is None
    assert sub.skipped is False
    assert sub.details["num_auth_calls_per_tool"] == {"get_reservation": 3}
    assert sub.details["num_auth_tools"] == 1

    assert "num_auth_calls_on_failure" not in result.sub_metrics


@pytest.mark.asyncio
async def test_num_auth_calls_success_multiple_tools(metric):
    """Auth succeeding with multiple distinct auth tools should average calls across tools."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123"}},
        agent_tools=[
            {"name": "get_reservation", "tool_type": "auth"},
            {"name": "verify", "tool_type": "auth"},
        ],
        tool_responses=[
            {"tool_name": "get_reservation", "tool_response": SUCCESS_RESPONSE},
            {"tool_name": "get_reservation", "tool_response": SUCCESS_RESPONSE},
            {"tool_name": "verify", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "verify", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "verify", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "verify", "tool_response": SUCCESS_RESPONSE},
        ],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is not None
    sub = result.sub_metrics["num_auth_calls"]
    assert sub.score == 3.0
    assert sub.normalized_score is None
    assert sub.skipped is False
    assert sub.details["num_auth_calls_per_tool"] == {"get_reservation": 2, "verify": 4}
    assert sub.details["num_auth_tools"] == 2

    assert "num_auth_calls_on_failure" not in result.sub_metrics


@pytest.mark.asyncio
async def test_authentication_success_rate_single_tool_first_try(metric):
    """Auth succeeding with a single tool called once should score authentication_success_rate=1.0."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123"}},
        agent_tools=[{"name": "get_reservation", "tool_type": "auth"}],
        tool_responses=[{"tool_name": "get_reservation", "tool_response": SUCCESS_RESPONSE}],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is not None
    sub = result.sub_metrics["authentication_success_rate"]
    assert sub.score == 1.0
    assert sub.normalized_score == 1.0
    assert sub.skipped is False
    assert sub.details["num_auth_calls_per_tool"] == {"get_reservation": 1}
    assert sub.details["num_auth_tools"] == 1


@pytest.mark.asyncio
async def test_authentication_success_rate_single_tool_with_retries(metric):
    """Auth succeeding after 3 calls to one tool should score authentication_success_rate=1/3."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123"}},
        agent_tools=[{"name": "get_reservation", "tool_type": "auth"}],
        tool_responses=[
            {"tool_name": "get_reservation", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "get_reservation", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "get_reservation", "tool_response": SUCCESS_RESPONSE},
        ],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is not None
    sub = result.sub_metrics["authentication_success_rate"]
    assert sub.score == pytest.approx(0.3333, abs=1e-4)
    assert sub.normalized_score == pytest.approx(0.3333, abs=1e-4)
    assert sub.skipped is False


@pytest.mark.asyncio
async def test_authentication_success_rate_multiple_tools(metric):
    """Auth succeeding across 2 tools with 6 total calls should score authentication_success_rate=2/6."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123"}},
        agent_tools=[
            {"name": "get_reservation", "tool_type": "auth"},
            {"name": "verify", "tool_type": "auth"},
        ],
        tool_responses=[
            {"tool_name": "get_reservation", "tool_response": SUCCESS_RESPONSE},
            {"tool_name": "get_reservation", "tool_response": SUCCESS_RESPONSE},
            {"tool_name": "verify", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "verify", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "verify", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "verify", "tool_response": SUCCESS_RESPONSE},
        ],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is not None
    sub = result.sub_metrics["authentication_success_rate"]
    assert sub.score == pytest.approx(0.3333, abs=1e-4)
    assert sub.normalized_score == pytest.approx(0.3333, abs=1e-4)
    assert sub.skipped is False


@pytest.mark.asyncio
async def test_authentication_success_rate_absent_on_failure(metric):
    """Auth failing should not produce an authentication_success_rate sub-metric."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "WRONG1"}},
        agent_tools=[{"name": "get_reservation", "tool_type": "auth"}],
        tool_responses=[{"tool_name": "get_reservation", "tool_response": FAILURE_RESPONSE}],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is not None
    assert "authentication_success_rate" not in result.sub_metrics


@pytest.mark.asyncio
async def test_num_auth_calls_on_failure_single_tool(metric):
    """Auth failing with a single auth tool called once should score num_auth_calls_on_failure=1.0."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "WRONG1"}},
        agent_tools=[{"name": "get_reservation", "tool_type": "auth"}],
        tool_responses=[{"tool_name": "get_reservation", "tool_response": FAILURE_RESPONSE}],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is not None
    sub = result.sub_metrics["num_auth_calls_on_failure"]
    assert sub.score == 1.0
    assert sub.normalized_score is None
    assert sub.skipped is False
    assert sub.details["num_auth_calls_per_tool"] == {"get_reservation": 1}
    assert sub.details["num_auth_tools"] == 1

    assert "num_auth_calls" not in result.sub_metrics
    assert "auth_first_try_success" not in result.sub_metrics


@pytest.mark.asyncio
async def test_num_auth_calls_on_failure_multiple_calls(metric):
    """Auth failing after multiple calls to the same tool should average those calls."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "WRONG1"}},
        agent_tools=[{"name": "get_reservation", "tool_type": "auth"}],
        tool_responses=[
            {"tool_name": "get_reservation", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "get_reservation", "tool_response": FAILURE_RESPONSE},
            {"tool_name": "get_reservation", "tool_response": FAILURE_RESPONSE},
        ],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is not None
    sub = result.sub_metrics["num_auth_calls_on_failure"]
    assert sub.score == 3.0
    assert sub.normalized_score is None
    assert sub.skipped is False
    assert sub.details["num_auth_calls_per_tool"] == {"get_reservation": 3}
    assert sub.details["num_auth_tools"] == 1

    assert "num_auth_calls" not in result.sub_metrics
    assert "auth_first_try_success" not in result.sub_metrics


@pytest.mark.asyncio
async def test_no_auth_tools_no_sub_metrics(metric):
    """No auth-type tools on the agent should produce no sub-metrics."""
    ctx = make_metric_context(
        expected_scenario_db={"session": {"confirmation_number": "ABC123"}},
        final_scenario_db={"session": {"confirmation_number": "ABC123"}},
        agent_tools=[{"name": "get_flight_status", "tool_type": "read"}],
    )
    result = await metric.compute(ctx)

    assert result.sub_metrics is None
