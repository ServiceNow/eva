"""Unit tests for transcription_accuracy_key_entities metric."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from eva.metrics.diagnostic.transcription_accuracy_key_entities import (
    TranscriptionAccuracyKeyEntitiesMetric,
)
from eva.metrics.utils import aggregate_per_turn_scores

from .conftest import make_judge_metric, make_metric_context


class TestComputeTurnScore:
    def setup_method(self):
        with patch.object(TranscriptionAccuracyKeyEntitiesMetric, "__init__", lambda self, **kw: None):
            self.metric = TranscriptionAccuracyKeyEntitiesMetric()

    def test_with_entities_some_correct(self):
        """Entities present, some correct → returns correct ratio."""
        turn_eval = {
            "entities": [
                {"value": "ABC123", "correct": True},
                {"value": "John", "correct": False},
                {"value": "2026-03-15", "correct": True},
                {"value": "$500", "correct": False},
            ]
        }
        score, normalized = self.metric._compute_turn_score(turn_eval)
        assert score == 0.5
        assert normalized == 0.5

    def test_all_correct(self):
        """All entities correct → returns (1.0, 1.0)."""
        turn_eval = {
            "entities": [
                {"value": "ABC123", "correct": True},
                {"value": "Smith", "correct": True},
            ]
        }
        score, normalized = self.metric._compute_turn_score(turn_eval)
        assert score == 1.0
        assert normalized == 1.0

    def test_no_entities(self):
        """Empty entities list → returns (-1, -1) to signal not applicable."""
        turn_eval = {"entities": []}
        score, normalized = self.metric._compute_turn_score(turn_eval)
        assert score == -1.0
        assert normalized == -1.0

    def test_no_entities_key_missing(self):
        """Missing entities key → returns (-1, -1) to signal not applicable."""
        turn_eval = {"summary": "No key entities found."}
        score, normalized = self.metric._compute_turn_score(turn_eval)
        assert score == -1.0
        assert normalized == -1.0

    def test_no_entities_excluded_from_aggregation(self):
        """No-entity turns (-1) should not inflate the aggregated score.

        Simulates three turns:
          - Turn 1: 1/2 entities correct → 0.5
          - Turn 2: no entities → -1 (excluded)
          - Turn 3: 2/4 entities correct → 0.25

        The aggregated mean should be 0.375, not (0.5 + 0.25) / 3.
        """
        turn_evaluations = [
            {
                "entities": [
                    {"value": "ABC", "correct": True},
                    {"value": "DEF", "correct": False},
                ]
            },
            {"entities": []},  # no entities — should be excluded
            {
                "entities": [
                    {"value": "GHI", "correct": True},
                    {"value": "JKL", "correct": False},
                    {"value": "MNO", "correct": False},
                    {"value": "PQR", "correct": False},
                ]
            },
        ]

        per_turn_normalized = []
        for turn_eval in turn_evaluations:
            _, normalized = self.metric._compute_turn_score(turn_eval)
            per_turn_normalized.append(normalized)

        assert per_turn_normalized == [0.5, -1.0, 0.25]

        # Filter -1 (not applicable) before aggregation, matching metric's compute()
        applicable = [v for v in per_turn_normalized if v is not None and v != -1.0]
        aggregated = aggregate_per_turn_scores(applicable, "mean")
        assert aggregated == 0.375

    def test_skipped_entities_excluded_from_aggregation(self):
        """Skipped entities should be excluded from score."""
        turn_evaluations = [
            {
                "entities": [
                    {"value": "ABC", "correct": True, "skipped": False},
                    {"value": "DEF", "correct": False, "skipped": True},  # skipped entity, should be excluded
                    {"value": "GHI", "correct": False, "skipped": False},
                ]
            },
            {"entities": []},  # no entities — should be excluded
            {
                "entities": [
                    {"value": "GHI", "correct": True, "skipped": False},
                    {"value": "JKL", "correct": False, "skipped": True},  # skipped entity, should be excluded
                    {"value": "MNO", "correct": False, "skipped": True},  # skipped entity, should be excluded
                    {"value": "PQR", "correct": False, "skipped": True},  # skipped entity, should be excluded
                ]
            },
        ]

        per_turn_normalized = []
        for turn_eval in turn_evaluations:
            _, normalized = self.metric._compute_turn_score(turn_eval)
            per_turn_normalized.append(normalized)

        assert per_turn_normalized == [0.5, -1.0, 1.0]

        # Filter -1 (not applicable) before aggregation, matching metric's compute()
        applicable = [v for v in per_turn_normalized if v is not None and v != -1.0]
        aggregated = aggregate_per_turn_scores(applicable, "mean")
        assert aggregated == 0.75

    def test_all_turns_no_entities_aggregates_to_none(self):
        """When all turns have no entities (-1), the aggregate should be None."""
        turn_evaluations = [
            {"entities": []},
            {"entities": []},
            {"summary": "No key entities found."},
        ]

        per_turn_normalized = []
        for turn_eval in turn_evaluations:
            _, normalized = self.metric._compute_turn_score(turn_eval)
            per_turn_normalized.append(normalized)

        assert per_turn_normalized == [-1.0, -1.0, -1.0]

        applicable = [v for v in per_turn_normalized if v is not None and v != -1.0]
        assert applicable == []
        aggregated = aggregate_per_turn_scores(applicable, "mean") if applicable else None
        assert aggregated is None


def _make_judge_response(turn_evals: list[dict]) -> str:
    """Build a JSON judge response."""
    return json.dumps(turn_evals)


@pytest.fixture
def metric():
    return make_judge_metric(TranscriptionAccuracyKeyEntitiesMetric)


class TestCompute:
    @pytest.mark.asyncio
    async def test_surfaces_per_entity_type_sub_metrics(self, metric):
        """Sub-metrics aggregate accuracy per entity type across turns."""
        context = make_metric_context(
            intended_user_turns={0: "My name is John Smith", 1: "Confirmation ABC123 on Dec 15"},
            transcribed_user_turns={0: "My name is John Smith", 1: "Confirmation ABC123 on Dec 15"},
        )
        response = _make_judge_response(
            [
                {
                    "turn_id": 0,
                    "summary": "mostly correct",
                    "entities": [
                        {"type": "name", "correct": True, "skipped": False},
                        {"type": "name", "correct": False, "skipped": False},
                    ],
                },
                {
                    "turn_id": 1,
                    "summary": "one skipped",
                    "entities": [
                        {"type": "confirmation_code", "correct": True, "skipped": False},
                        {"type": "date", "correct": True, "skipped": False},
                        {"type": "date", "correct": False, "skipped": True},
                    ],
                },
            ]
        )
        metric.llm_client.generate_text = AsyncMock(return_value=response)

        result = await metric.compute(context)

        assert result.error is None
        assert result.sub_metrics is not None
        assert set(result.sub_metrics.keys()) == {
            "name_accuracy",
            "confirmation_code_accuracy",
            "date_accuracy",
        }
        name_sub = result.sub_metrics["name_accuracy"]
        assert name_sub.name == "transcription_accuracy_key_entities.name_accuracy"
        assert name_sub.score == pytest.approx(0.5)
        assert name_sub.details == {"correct": 1, "total_non_skipped": 2, "skipped": 0}
        code_sub = result.sub_metrics["confirmation_code_accuracy"]
        assert code_sub.score == 1.0
        assert code_sub.details == {"correct": 1, "total_non_skipped": 1, "skipped": 0}
        date_sub = result.sub_metrics["date_accuracy"]
        assert date_sub.score == 1.0
        assert date_sub.details == {"correct": 1, "total_non_skipped": 1, "skipped": 1}

    @pytest.mark.asyncio
    async def test_all_turns_have_entities(self, metric):
        """All turns have entities -> num_evaluated == num_turns, num_not_applicable == 0."""
        context = make_metric_context(
            intended_user_turns={0: "My name is John Smith", 1: "Confirmation ABC123"},
            transcribed_user_turns={0: "My name is John Smith", 1: "Confirmation ABC123"},
        )
        response = _make_judge_response(
            [
                {
                    "turn_id": 0,
                    "summary": "All correct",
                    "entities": [{"entity": "John Smith", "correct": True}],
                },
                {
                    "turn_id": 1,
                    "summary": "All correct",
                    "entities": [{"entity": "ABC123", "correct": True}],
                },
            ]
        )
        metric.llm_client.generate_text = AsyncMock(return_value=response)

        result = await metric.compute(context)

        assert result.error is None
        assert result.details["num_turns"] == 2
        assert result.details["num_evaluated"] == 2
        assert result.details["num_not_applicable"] == 0
        assert result.skipped is False
        assert result.score == 1.0
        assert result.normalized_score == 1.0

    @pytest.mark.asyncio
    async def test_one_turn_no_entities(self, metric):
        """One turn has no entities -> num_not_applicable == 1, excluded from score."""
        context = make_metric_context(
            intended_user_turns={0: "Yes please", 1: "Confirmation ABC123"},
            transcribed_user_turns={0: "Yes please", 1: "Confirmation ABC123"},
        )
        response = _make_judge_response(
            [
                {
                    "turn_id": 0,
                    "summary": "No entities",
                    "entities": [],
                },
                {
                    "turn_id": 1,
                    "summary": "All correct",
                    "entities": [{"entity": "ABC123", "correct": True}],
                },
            ]
        )
        metric.llm_client.generate_text = AsyncMock(return_value=response)

        result = await metric.compute(context)

        assert result.error is None
        assert result.details["num_turns"] == 2
        assert result.details["num_evaluated"] == 2
        assert result.details["num_not_applicable"] == 1
        assert result.skipped is False
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_all_turns_no_entities(self, metric):
        """All turns have no entities -> skipped, normalized_score is None."""
        context = make_metric_context(
            intended_user_turns={0: "Yes", 1: "Ok thanks"},
            transcribed_user_turns={0: "Yes", 1: "Ok thanks"},
        )
        response = _make_judge_response(
            [
                {"turn_id": 0, "summary": "No entities", "entities": []},
                {"turn_id": 1, "summary": "No entities", "entities": []},
            ]
        )
        metric.llm_client.generate_text = AsyncMock(return_value=response)

        result = await metric.compute(context)

        assert result.details["num_turns"] == 2
        assert result.details["num_evaluated"] == 2
        assert result.details["num_not_applicable"] == 2
        assert result.skipped is True
        assert result.normalized_score is None

    @pytest.mark.asyncio
    async def test_mixed_entity_correctness(self, metric):
        """Some entities correct, some not -> score reflects ratio."""
        context = make_metric_context(
            intended_user_turns={0: "John Smith on March 25th"},
            transcribed_user_turns={0: "John Smith on March 26th"},
        )
        response = _make_judge_response(
            [
                {
                    "turn_id": 0,
                    "summary": "One wrong",
                    "entities": [
                        {"entity": "John Smith", "correct": True},
                        {"entity": "March 25th", "correct": False},
                    ],
                },
            ]
        )
        metric.llm_client.generate_text = AsyncMock(return_value=response)

        result = await metric.compute(context)

        assert result.error is None
        assert result.details["num_turns"] == 1
        assert result.details["num_evaluated"] == 1
        assert result.details["num_not_applicable"] == 0
        assert result.score == 0.5
        assert result.normalized_score == 0.5

    @pytest.mark.asyncio
    async def test_all_entities_skipped_is_not_applicable(self, metric):
        """Turn where all entities are skipped -> treated as not applicable (-1)."""
        context = make_metric_context(
            intended_user_turns={0: "My name is John Smith", 1: "Confirmation ABC123"},
            transcribed_user_turns={0: "My name is John Smith", 1: "Confirmation ABC123"},
        )
        response = _make_judge_response(
            [
                {
                    "turn_id": 0,
                    "summary": "Entities skipped",
                    "entities": [{"entity": "John Smith", "correct": True, "skipped": True}],
                },
                {
                    "turn_id": 1,
                    "summary": "All correct",
                    "entities": [{"entity": "ABC123", "correct": True}],
                },
            ]
        )
        metric.llm_client.generate_text = AsyncMock(return_value=response)

        result = await metric.compute(context)

        assert result.error is None
        assert result.details["num_turns"] == 2
        assert result.details["num_evaluated"] == 2
        assert result.details["num_not_applicable"] == 1
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_response_from_judge(self, metric):
        """None response from LLM returns error."""
        context = make_metric_context(
            intended_user_turns={0: "Hello"},
            transcribed_user_turns={0: "Hello"},
        )
        metric.llm_client.generate_text = AsyncMock(return_value=None)

        result = await metric.compute(context)

        assert result.error == "No response from judge"
        assert result.score == 0.0
