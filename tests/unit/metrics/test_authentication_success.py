"""Tests for AuthenticationSuccessMetric."""

import pytest

from eva.metrics.diagnostic.authentication_success import AuthenticationSuccessMetric

from .conftest import make_metric_context


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
