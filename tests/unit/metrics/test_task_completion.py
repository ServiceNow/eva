"""Tests for TaskCompletion metric."""

import pytest

from eva.metrics.accuracy.task_completion import TaskCompletion
from eva.utils.hash_utils import get_dict_hash
from tests.unit.metrics.conftest import make_metric_context


class TestTaskCompletion:
    def setup_method(self):
        self.metric = TaskCompletion()

    @pytest.mark.asyncio
    async def test_matching_hashes_returns_pass(self):
        db = {"reservations": {"ABC": {"status": "confirmed"}}}
        expected_hash = get_dict_hash(db)

        ctx = make_metric_context(
            expected_scenario_db=db,
            final_scenario_db=db,
            final_scenario_db_hash=expected_hash,
        )
        score = await self.metric.compute(ctx)

        assert score.score == 1.0
        assert score.normalized_score == 1.0
        assert score.details["match"] is True

    @pytest.mark.asyncio
    async def test_mismatched_hashes_returns_fail(self):
        expected_db = {"reservations": {"ABC": {"status": "confirmed"}}}
        actual_db = {"reservations": {"ABC": {"status": "cancelled"}}}

        ctx = make_metric_context(
            expected_scenario_db=expected_db,
            final_scenario_db=actual_db,
            final_scenario_db_hash=get_dict_hash(actual_db),
        )
        score = await self.metric.compute(ctx)

        assert score.score == 0.0
        assert score.normalized_score == 0.0
        assert score.details["match"] is False
        assert "diff" in score.details
        assert "diff_summary" in score.details

    @pytest.mark.asyncio
    async def test_diff_shows_modified_tables(self):
        expected_db = {"flights": {"F1": {"gate": "A1"}}}
        actual_db = {"flights": {"F1": {"gate": "B2"}}}

        ctx = make_metric_context(
            expected_scenario_db=expected_db,
            final_scenario_db=actual_db,
            final_scenario_db_hash=get_dict_hash(actual_db),
        )
        score = await self.metric.compute(ctx)

        assert score.details["diff_summary"] == "1 tables modified"
        assert score.details["diff"]["tables_modified"]["flights"]["records_modified"]["F1"]["fields_modified"] == {
            "gate": {"actual": "B2", "expected": "A1", "type": "value_mismatch"}
        }

    @pytest.mark.asyncio
    async def test_diff_shows_added_tables(self):
        expected_db = {"flights": {}}
        actual_db = {"flights": {}, "extras": {"X": 1}}

        ctx = make_metric_context(
            expected_scenario_db=expected_db,
            final_scenario_db=actual_db,
            final_scenario_db_hash=get_dict_hash(actual_db),
        )
        score = await self.metric.compute(ctx)

        assert score.details["diff_summary"] == "1 tables added"
        assert score.details["diff"] == {"tables_added": ["extras"], "tables_modified": {}, "tables_removed": []}

    @pytest.mark.asyncio
    async def test_diff_shows_removed_tables(self):
        expected_db = {"flights": {}, "extras": {"X": 1}}
        actual_db = {"flights": {}}

        ctx = make_metric_context(
            expected_scenario_db=expected_db,
            final_scenario_db=actual_db,
            final_scenario_db_hash=get_dict_hash(actual_db),
        )
        score = await self.metric.compute(ctx)

        assert score.details["diff_summary"] == "1 tables removed"
        assert score.details["diff"] == {"tables_added": [], "tables_modified": {}, "tables_removed": ["extras"]}

    @pytest.mark.asyncio
    async def test_empty_dbs_match(self):
        ctx = make_metric_context(
            expected_scenario_db={},
            final_scenario_db={},
            final_scenario_db_hash=get_dict_hash({}),
        )
        score = await self.metric.compute(ctx)
        assert score.score == 1.0

    @pytest.mark.asyncio
    async def test_auth_failure_returns_zero(self):
        """If session auth fails, task_completion must return 0.0 regardless of DB state."""
        db = {"reservations": {"ABC": {"status": "confirmed"}}}
        ctx = make_metric_context(
            expected_scenario_db={**db, "session": {"confirmation_number": "ABC", "last_name": "doe"}},
            final_scenario_db={**db, "session": {"confirmation_number": "ABC", "last_name": "wrong"}},
            final_scenario_db_hash=get_dict_hash(db),
        )
        score = await self.metric.compute(ctx)

        assert score.score == 0.0
        assert score.details["auth_success"] is False
        assert "last_name" in score.details["auth_mismatches"]

    @pytest.mark.asyncio
    async def test_no_expected_session_does_not_block(self):
        """Empty expected session means no auth requirement — DB hash comparison proceeds normally."""
        db = {"reservations": {"ABC": {"status": "confirmed"}}}
        ctx = make_metric_context(
            expected_scenario_db={**db, "session": {}},
            final_scenario_db=db,
            final_scenario_db_hash=get_dict_hash(db),
        )
        score = await self.metric.compute(ctx)

        assert score.score == 1.0
        assert score.details.get("auth_failed") is not True

    def test_metric_attributes(self):
        assert self.metric.name == "task_completion"
        assert self.metric.category == "accuracy"
