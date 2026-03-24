"""Tests for the ResponseSpeedMetric."""

import pytest

from eva.metrics.diagnostic.response_speed import ResponseSpeedMetric

from .conftest import make_metric_context


class TestResponseSpeedMetric:
    @pytest.mark.asyncio
    async def test_no_latencies_none(self):
        """None latencies returns error."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(response_speed_latencies=None)

        result = await metric.compute(ctx)

        assert result.name == "response_speed"
        assert result.score == 0.0
        assert result.normalized_score is None
        assert result.error is not None
        assert "No response latencies" in result.error

    @pytest.mark.asyncio
    async def test_no_latencies_empty(self):
        """Empty list returns error."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(response_speed_latencies=[])

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_valid_latencies(self):
        """Valid latencies produce correct mean, max, and per-turn details."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(response_speed_latencies=[1.0, 2.0, 3.0])

        result = await metric.compute(ctx)

        assert result.score == pytest.approx(2.0)
        assert result.normalized_score is None
        assert result.error is None
        assert result.details["mean_speed_seconds"] == pytest.approx(2.0)
        assert result.details["max_speed_seconds"] == pytest.approx(3.0)
        assert result.details["num_turns"] == 3
        assert result.details["per_turn_speeds"] == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_filters_invalid_values(self):
        """Negative and >1000s values are filtered out."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(response_speed_latencies=[-1.0, 0.5, 1500.0, 2.5, 0.0])

        result = await metric.compute(ctx)

        # Only 0.5 and 2.5 are valid (0 < x < 1000); 0.0 is excluded (not > 0)
        assert result.error is None
        assert result.details["num_turns"] == 2
        expected_mean = (0.5 + 2.5) / 2
        assert result.score == pytest.approx(expected_mean)
        assert result.details["max_speed_seconds"] == pytest.approx(2.5)
        assert result.details["per_turn_speeds"] == [0.5, 2.5]

    @pytest.mark.asyncio
    async def test_all_latencies_filtered_out(self):
        """When all values are invalid, returns error."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(response_speed_latencies=[-5.0, 0.0, 2000.0])

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.normalized_score is None
        assert result.error is not None
        assert "No valid response speeds" in result.error

    @pytest.mark.asyncio
    async def test_single_latency_value(self):
        """Single valid latency works correctly."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(response_speed_latencies=[0.75])

        result = await metric.compute(ctx)

        assert result.score == pytest.approx(0.75)
        assert result.details["mean_speed_seconds"] == pytest.approx(0.75)
        assert result.details["max_speed_seconds"] == pytest.approx(0.75)
        assert result.details["num_turns"] == 1
        assert result.details["per_turn_speeds"] == [0.75]
