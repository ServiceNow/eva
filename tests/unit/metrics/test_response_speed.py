"""Tests for the ResponseSpeedMetric."""

import pytest

from eva.metrics.diagnostic.response_speed import ResponseSpeedMetric

from .conftest import make_metric_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(tool_call_turn_ids: set[int], all_turn_ids: set[int]) -> list[dict]:
    """Build a minimal conversation_trace with the given turn structure."""
    trace = []
    for tid in sorted(all_turn_ids):
        trace.append({"turn_id": tid, "type": "transcribed", "content": "user utterance"})
        if tid in tool_call_turn_ids:
            trace.append({"turn_id": tid, "type": "tool_call", "tool_name": "some_tool"})
            trace.append({"turn_id": tid, "type": "tool_response", "tool_name": "some_tool"})
    return trace


# ---------------------------------------------------------------------------
# ResponseSpeedMetric
# ---------------------------------------------------------------------------


class TestResponseSpeedMetric:
    @pytest.mark.asyncio
    async def test_no_latencies(self):
        """Missing latency data returns error."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context()

        result = await metric.compute(ctx)

        assert result.name == "response_speed"
        assert result.score == 0.0
        assert result.normalized_score is None
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_valid_latencies(self):
        """Valid latencies produce correct mean, max, and per-turn details."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(latency_assistant_turns={1: 1.0, 2: 2.0, 3: 3.0})

        result = await metric.compute(ctx)

        assert result.score == pytest.approx(2.0)
        assert result.normalized_score is None
        assert result.error is None
        assert result.details["mean_speed_seconds"] == pytest.approx(2.0)
        assert result.details["max_speed_seconds"] == pytest.approx(3.0)
        assert result.details["num_turns"] == 3

    @pytest.mark.asyncio
    async def test_filters_invalid_values(self):
        """Negative and >1000s values are filtered out."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(latency_assistant_turns={1: -1.0, 2: 0.5, 3: 1500.0, 4: 2.5})

        result = await metric.compute(ctx)

        # Only 0.5 and 2.5 are valid (0 < x < 1000)
        assert result.error is None
        assert result.details["num_turns"] == 2
        assert result.score == pytest.approx((0.5 + 2.5) / 2)
        assert result.details["max_speed_seconds"] == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_all_latencies_filtered_out(self):
        """When all values are invalid, returns error."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(latency_assistant_turns={1: -5.0, 2: 2000.0})

        result = await metric.compute(ctx)

        assert result.score == 0.0
        assert result.normalized_score is None
        assert result.error is not None
        assert "No valid response speeds" in result.error

    @pytest.mark.asyncio
    async def test_single_latency_value(self):
        """Single valid latency works correctly."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(latency_assistant_turns={1: 0.75})

        result = await metric.compute(ctx)

        assert result.score == pytest.approx(0.75)
        assert result.details["mean_speed_seconds"] == pytest.approx(0.75)
        assert result.details["max_speed_seconds"] == pytest.approx(0.75)
        assert result.details["num_turns"] == 1
        assert result.details["per_turn_speeds"] == [0.75]

    @pytest.mark.asyncio
    async def test_no_tool_call_breakdown_without_trace(self):
        """with_tool_calls absent and no_tool_calls covers all turns when trace is absent."""
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(latency_assistant_turns={1: 1.0, 2: 2.0})

        result = await metric.compute(ctx)

        assert result.error is None
        # No trace → no tool call turn ids → all turns go into no_tool bucket
        assert result.sub_metrics is not None
        assert "with_tool_calls" not in result.sub_metrics
        no_tc = result.sub_metrics["no_tool_calls"]
        assert no_tc.details["num_turns"] == 2

    @pytest.mark.asyncio
    async def test_tool_call_breakdown_mixed_turns(self):
        """with_tool_calls and no_tool_calls sub-metrics reflect the correct split."""
        trace = _make_trace(tool_call_turn_ids={2, 4}, all_turn_ids={1, 2, 3, 4})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(
            latency_assistant_turns={1: 1.0, 2: 5.0, 3: 3.0, 4: 7.0},
            conversation_trace=trace,
        )

        result = await metric.compute(ctx)

        assert result.error is None
        assert result.sub_metrics is not None
        with_tc = result.sub_metrics["with_tool_calls"]
        no_tc = result.sub_metrics["no_tool_calls"]
        assert with_tc.details["num_turns"] == 2
        assert with_tc.details["mean_speed_seconds"] == pytest.approx((5.0 + 7.0) / 2)
        assert with_tc.details["max_speed_seconds"] == pytest.approx(7.0)
        assert with_tc.score == pytest.approx((5.0 + 7.0) / 2)
        assert no_tc.details["num_turns"] == 2
        assert no_tc.details["mean_speed_seconds"] == pytest.approx((1.0 + 3.0) / 2)
        assert no_tc.details["max_speed_seconds"] == pytest.approx(3.0)
        assert no_tc.score == pytest.approx((1.0 + 3.0) / 2)

    @pytest.mark.asyncio
    async def test_tool_call_breakdown_all_tool_turns(self):
        """no_tool_calls absent when every turn has a tool call."""
        trace = _make_trace(tool_call_turn_ids={1, 2}, all_turn_ids={1, 2})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(
            latency_assistant_turns={1: 2.0, 2: 4.0},
            conversation_trace=trace,
        )

        result = await metric.compute(ctx)

        assert result.error is None
        assert result.sub_metrics is not None
        assert result.sub_metrics["with_tool_calls"].details["num_turns"] == 2
        assert "no_tool_calls" not in result.sub_metrics

    @pytest.mark.asyncio
    async def test_tool_call_breakdown_filters_invalid_latencies(self):
        """Sanity filter (0 < x < 1000) applies within the breakdown sub-metrics."""
        trace = _make_trace(tool_call_turn_ids={1, 2, 3, 4}, all_turn_ids={1, 2, 3, 4})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(
            latency_assistant_turns={1: -1.0, 2: 5.0, 3: 2000.0, 4: 3.0},
            conversation_trace=trace,
        )

        result = await metric.compute(ctx)

        assert result.error is None
        assert result.sub_metrics is not None
        with_tc = result.sub_metrics["with_tool_calls"]
        assert with_tc.details["num_turns"] == 2  # only 5.0 and 3.0 pass the filter

    @pytest.mark.asyncio
    async def test_with_and_no_tool_split_is_exhaustive(self):
        """with_tool + no_tool latencies together cover all per_turn_latency values."""
        latencies = {1: 1.0, 2: 5.0, 3: 3.0, 4: 7.0, 5: 2.0}
        trace = _make_trace(tool_call_turn_ids={2, 4}, all_turn_ids={1, 2, 3, 4, 5})
        metric = ResponseSpeedMetric()
        ctx = make_metric_context(
            latency_assistant_turns=latencies,
            conversation_trace=trace,
        )

        result = await metric.compute(ctx)

        assert result.error is None
        assert result.sub_metrics is not None
        combined = (
            result.sub_metrics["with_tool_calls"].details["per_turn_speeds"]
            + result.sub_metrics["no_tool_calls"].details["per_turn_speeds"]
        )
        assert sorted(combined) == sorted(latencies.values())
