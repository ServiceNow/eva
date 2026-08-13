"""Tests for the TimeToCompletionMetric."""

import pytest

from eva.metrics.diagnostic.time_to_completion import TimeToCompletionMetric
from eva.utils.hash_utils import get_dict_hash

from .conftest import make_metric_context


class TestTimeToCompletionMetric:
    def setup_method(self):
        self.metric = TimeToCompletionMetric()

    @pytest.mark.asyncio
    async def test_completed_task_reports_duration(self):
        """When the task completed, reports the total duration in seconds."""
        db = {"reservations": {"ABC": {"status": "confirmed"}}}
        ctx = make_metric_context(
            expected_scenario_db=db,
            final_scenario_db=db,
            final_scenario_db_hash=get_dict_hash(db),
            duration_seconds=42.5,
        )

        result = await self.metric.compute(ctx)

        assert result.name == "time_to_completion"
        assert result.score == pytest.approx(42.5)
        assert result.normalized_score is None
        assert result.error is None
        assert result.skipped is False
        assert result.details["task_completed"] is True
        assert result.details["duration_seconds"] == pytest.approx(42.5)

    @pytest.mark.asyncio
    async def test_incomplete_task_is_skipped(self):
        """When the task did not complete (hash mismatch), the metric is skipped."""
        expected_db = {"reservations": {"ABC": {"status": "confirmed"}}}
        actual_db = {"reservations": {"ABC": {"status": "cancelled"}}}
        ctx = make_metric_context(
            expected_scenario_db=expected_db,
            final_scenario_db=actual_db,
            final_scenario_db_hash=get_dict_hash(actual_db),
            duration_seconds=42.5,
        )

        result = await self.metric.compute(ctx)

        assert result.score is None
        assert result.normalized_score is None
        assert result.error is None
        assert result.skipped is True
        assert result.details["task_completed"] is False

    @pytest.mark.asyncio
    async def test_auth_failure_is_skipped(self):
        """Auth failure means the task is not completed, so the metric is skipped."""
        db = {"reservations": {"ABC": {"status": "confirmed"}}}
        ctx = make_metric_context(
            expected_scenario_db={**db, "session": {"confirmation_number": "ABC", "last_name": "doe"}},
            final_scenario_db={**db, "session": {"confirmation_number": "ABC", "last_name": "wrong"}},
            final_scenario_db_hash=get_dict_hash(db),
            duration_seconds=42.5,
        )

        result = await self.metric.compute(ctx)

        assert result.skipped is True
        assert result.details["task_completed"] is False
        assert "Authentication failed" in result.details["reason"]

    @pytest.mark.asyncio
    async def test_completed_but_no_duration_is_skipped(self):
        """Completed task with a non-positive duration is skipped rather than scored."""
        db = {"reservations": {"ABC": {"status": "confirmed"}}}
        ctx = make_metric_context(
            expected_scenario_db=db,
            final_scenario_db=db,
            final_scenario_db_hash=get_dict_hash(db),
            duration_seconds=0.0,
        )

        result = await self.metric.compute(ctx)

        assert result.score is None
        assert result.skipped is True
        assert result.details["task_completed"] is True

    def test_metric_attributes(self):
        assert self.metric.name == "time_to_completion"
        assert self.metric.category == "diagnostic"
        assert self.metric.exclude_from_pass_at_k is True
        assert self.metric.higher_is_better is False
