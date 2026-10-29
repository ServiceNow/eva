"""Tests for the DelegationRateMetric."""

import json

import pytest

from eva.metrics.diagnostic.delegation_rate import DelegationRateMetric

from .conftest import make_metric_context

MARKER = {"timestamp": 500, "type": "delegation_session", "data": {"frame": "delegation_session"}}


def _turn_start(ts: int) -> dict:
    return {"timestamp": ts, "type": "turn_start", "data": {"frame": "turn_start"}}


def _delegation(ts: int, delegation_id: str = "d") -> dict:
    return {"timestamp": ts, "type": "delegation", "data": {"frame": "delegation", "delegation_id": delegation_id}}


def _context(tmp_path, entries: list[dict] | None):
    if entries is not None:
        (tmp_path / "framework_logs.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return make_metric_context(output_dir=str(tmp_path))


class TestDelegationRateMetric:
    def setup_method(self):
        self.metric = DelegationRateMetric()

    def test_metadata(self):
        assert self.metric.name == "delegation_rate"
        assert self.metric.category == "diagnostic"
        assert self.metric.exclude_from_pass_at_k is True

    @pytest.mark.asyncio
    async def test_rate_is_delegated_user_turns_over_all_user_turns(self, tmp_path):
        # Four user turns; delegations in turns 1 and 3 (turn 3 delegates twice).
        entries = [
            MARKER,
            _turn_start(1000),
            _turn_start(2000),
            _turn_start(3000),
            _turn_start(4000),
            _delegation(1500, "a"),
            _delegation(3200, "b"),
            _delegation(3300, "c"),
        ]
        result = await self.metric.compute(_context(tmp_path, entries))

        assert result.name == "delegation_rate"
        assert result.score == pytest.approx(0.5)
        assert result.normalized_score is None
        assert result.skipped is False and result.error is None
        assert result.details["num_user_turns"] == 4
        assert result.details["num_delegated_turns"] == 2
        assert result.details["num_delegations"] == 3

    @pytest.mark.asyncio
    async def test_entries_are_ordered_by_timestamp_not_file_order(self, tmp_path):
        # The server writes a user turn's turn_start after the fact, with its original start timestamp.
        entries = [MARKER, _turn_start(1000), _delegation(2500), _turn_start(2000), _turn_start(3000)]
        result = await self.metric.compute(_context(tmp_path, entries))

        assert result.details["num_delegated_turns"] == 1  # the delegation belongs to the turn started at 2000
        assert result.score == pytest.approx(1 / 3)

    @pytest.mark.asyncio
    async def test_no_delegations_in_a_delegation_capable_run_scores_zero(self, tmp_path):
        result = await self.metric.compute(_context(tmp_path, [MARKER, _turn_start(1000), _turn_start(2000)]))

        assert result.score == 0.0
        assert result.skipped is False

    @pytest.mark.asyncio
    async def test_run_without_the_marker_is_skipped(self, tmp_path):
        """Other frameworks never write the marker, so the metric does not apply to them."""
        result = await self.metric.compute(_context(tmp_path, [_turn_start(1000), _turn_start(2000)]))

        assert result.skipped is True and result.score is None
        assert "delegation" in result.details["reason"]

    @pytest.mark.asyncio
    async def test_missing_framework_log_is_skipped(self, tmp_path):
        result = await self.metric.compute(_context(tmp_path, None))

        assert result.skipped is True and result.score is None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_no_user_turns_is_skipped(self, tmp_path):
        result = await self.metric.compute(_context(tmp_path, [MARKER, _delegation(1500)]))

        assert result.skipped is True and result.score is None

    @pytest.mark.asyncio
    async def test_delegation_before_any_user_turn_is_counted_but_not_attributed(self, tmp_path):
        result = await self.metric.compute(_context(tmp_path, [MARKER, _delegation(500), _turn_start(1000)]))

        assert result.score == 0.0
        assert result.details["num_delegations"] == 1
        assert result.details["num_unattributed_delegations"] == 1
