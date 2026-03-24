"""Tests for ValidationRunner."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from eva.metrics.runner import MetricsRunResult
from eva.models.results import MetricScore, RecordMetrics
from eva.orchestrator.validation_runner import ValidationResult, ValidationRunner
from tests.unit.conftest import make_evaluation_record
from tests.unit.metrics.conftest import make_metric_score


def _make_record(record_id: str):
    return make_evaluation_record(record_id)


def _make_score(name: str, score: float, error: str | None = None, details: dict | None = None) -> MetricScore:
    return make_metric_score(name, score=score, error=error, details=details or {})


@pytest.fixture
def sample_records():
    return [_make_record("record_1"), _make_record("record_2")]


@pytest.fixture
def validation_runner(temp_dir, sample_records):
    return ValidationRunner(
        run_dir=temp_dir,
        dataset=sample_records,
        thresholds={"conversation_finished": 1.0, "user_behavioral_fidelity": 1.0},
    )


@pytest.fixture
def runner_skip_cf(temp_dir, sample_records):
    return ValidationRunner(
        run_dir=temp_dir,
        dataset=sample_records,
        thresholds={"conversation_finished": 1.0, "user_behavioral_fidelity": 1.0},
        skip_conversation_finished=True,
    )


class TestValidationResult:
    def test_passed_defaults(self):
        vr = ValidationResult(passed=True)
        assert vr.passed is True
        assert vr.failed_metrics == []
        assert vr.failure_category == "passed"
        assert vr.scores == {}

    def test_failed_fields(self):
        vr = ValidationResult(
            passed=False,
            failed_metrics=["conversation_finished"],
            failure_category="not_finished",
            scores={"conversation_finished": 0.0},
        )
        assert vr.passed is False
        assert vr.failure_category == "not_finished"
        assert len(vr.failed_metrics) == 1


class TestEvaluateRecord:
    def _runner(self, thresholds: dict | None = None) -> ValidationRunner:
        return ValidationRunner(run_dir=Path("/tmp/fake"), dataset=[], thresholds=thresholds or {})

    def _metrics(self, **scores) -> RecordMetrics:
        return RecordMetrics(
            record_id="rec-0",
            metrics={name: _make_score(name, score) for name, score in scores.items()},
        )

    def test_all_pass(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(conversation_finished=1.0, user_behavioral_fidelity=1.0),
            ["conversation_finished", "user_behavioral_fidelity"],
        )
        assert result.passed is True
        assert result.failed_metrics == []

    def test_one_below_threshold(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(conversation_finished=1.0, user_behavioral_fidelity=0.5),
            ["conversation_finished", "user_behavioral_fidelity"],
        )
        assert result.passed is False
        assert "user_behavioral_fidelity" in result.failed_metrics

    def test_at_threshold(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(conversation_finished=1.0, user_behavioral_fidelity=1.0),
            ["conversation_finished", "user_behavioral_fidelity"],
        )
        assert result.passed is True

    def test_just_below_threshold(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(conversation_finished=1.0, user_behavioral_fidelity=0.99),
            ["conversation_finished", "user_behavioral_fidelity"],
        )
        assert result.passed is False
        assert "user_behavioral_fidelity" in result.failed_metrics

    def test_multiple_failures(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(conversation_finished=0.5, user_behavioral_fidelity=0.5),
            ["conversation_finished", "user_behavioral_fidelity"],
        )
        assert result.passed is False
        assert "conversation_finished" in result.failed_metrics
        assert "user_behavioral_fidelity" in result.failed_metrics
        assert result.scores["conversation_finished"] == 0.5
        assert result.scores["user_behavioral_fidelity"] == 0.5

    def test_metric_error_fails(self, validation_runner):
        record_metrics = RecordMetrics(
            record_id="record_1",
            metrics={
                "conversation_finished": _make_score("conversation_finished", 1.0),
                "user_behavioral_fidelity": _make_score("user_behavioral_fidelity", 0.0, error="Computation failed"),
            },
        )
        result = validation_runner._evaluate_record(
            "record_1", record_metrics, ["conversation_finished", "user_behavioral_fidelity"]
        )
        assert result.passed is False
        assert "user_behavioral_fidelity" in result.failed_metrics

    def test_missing_metric_fails(self, validation_runner):
        record_metrics = RecordMetrics(
            record_id="record_1",
            metrics={"conversation_finished": _make_score("conversation_finished", 1.0)},
        )
        result = validation_runner._evaluate_record(
            "record_1", record_metrics, ["conversation_finished", "user_behavioral_fidelity"]
        )
        assert result.passed is False
        assert "user_behavioral_fidelity" in result.failed_metrics

    # --- user_speech_fidelity per-turn logic (special-cased in _evaluate_record) ---

    def test_user_speech_fidelity_per_turn_rating_1_fails(self):
        """Any per-turn rating == 1 causes failure regardless of normalized_score."""
        runner = self._runner()
        record_metrics = RecordMetrics(
            record_id="rec-0",
            metrics={
                "user_speech_fidelity": MetricScore(
                    name="user_speech_fidelity",
                    score=2.5,
                    normalized_score=0.9,
                    details={"per_turn_ratings": {"turn_0": 3, "turn_1": 1, "turn_2": 3}},
                )
            },
        )
        result = runner._evaluate_record("rec-0", record_metrics, ["user_speech_fidelity"])
        assert not result.passed
        assert "user_speech_fidelity" in result.failed_metrics

    def test_user_speech_fidelity_all_ratings_ge_2_passes(self):
        """All per-turn ratings >= 2 passes."""
        runner = self._runner()
        record_metrics = RecordMetrics(
            record_id="rec-0",
            metrics={
                "user_speech_fidelity": MetricScore(
                    name="user_speech_fidelity",
                    score=2.5,
                    normalized_score=0.8,
                    details={"per_turn_ratings": {"turn_0": 3, "turn_1": 2, "turn_2": 3}},
                )
            },
        )
        result = runner._evaluate_record("rec-0", record_metrics, ["user_speech_fidelity"])
        assert result.passed
        assert result.failed_metrics == []

    def test_user_speech_fidelity_empty_details_falls_through_to_threshold(self):
        """Empty details → falls through to score threshold check."""
        runner = self._runner(thresholds={"user_speech_fidelity": 0.7})

        above = RecordMetrics(
            record_id="rec-0",
            metrics={
                "user_speech_fidelity": MetricScore(
                    name="user_speech_fidelity", score=2.0, normalized_score=0.8, details={}
                )
            },
        )
        assert runner._evaluate_record("rec-0", above, ["user_speech_fidelity"]).passed

        below = RecordMetrics(
            record_id="rec-0",
            metrics={
                "user_speech_fidelity": MetricScore(
                    name="user_speech_fidelity", score=1.0, normalized_score=0.5, details={}
                )
            },
        )
        result = runner._evaluate_record("rec-0", below, ["user_speech_fidelity"])
        assert not result.passed
        assert "user_speech_fidelity" in result.failed_metrics


class TestRunValidation:
    def test_initialization(self, validation_runner, temp_dir, sample_records):
        assert validation_runner.run_dir == temp_dir
        assert validation_runner.dataset == sample_records
        assert validation_runner.skip_conversation_finished is False
        assert validation_runner.VALIDATION_METRICS == [
            "conversation_finished",
            "user_behavioral_fidelity",
            "user_speech_fidelity",
        ]

    def test_skip_cf_initialization(self, runner_skip_cf):
        assert runner_skip_cf.skip_conversation_finished is True

    @pytest.mark.asyncio
    async def test_skip_cf_all_pass(self, runner_skip_cf):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        mock_results = {
            "record_1": RecordMetrics(
                record_id="record_1",
                metrics={
                    "user_behavioral_fidelity": _make_score("user_behavioral_fidelity", 1.0),
                    "user_speech_fidelity": _make_score("user_speech_fidelity", 0.8, details=tts_pass),
                },
            ),
            "record_2": RecordMetrics(
                record_id="record_2",
                metrics={
                    "user_behavioral_fidelity": _make_score("user_behavioral_fidelity", 1.0),
                    "user_speech_fidelity": _make_score("user_speech_fidelity", 0.9, details=tts_pass),
                },
            ),
        }
        with patch("eva.orchestrator.validation_runner.MetricsRunner") as Mock:
            mock_runner = AsyncMock()
            mock_runner.run.return_value = MetricsRunResult(all_metrics=mock_results, total_records=2)
            Mock.return_value = mock_runner

            results = await runner_skip_cf.run_validation()

        assert results["record_1"].passed is True
        assert results["record_2"].passed is True
        assert results["record_1"].failure_category == "passed"
        assert "conversation_finished" not in Mock.call_args[1]["metric_names"]

    @pytest.mark.asyncio
    async def test_skip_cf_some_fail(self, runner_skip_cf):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        mock_results = {
            "record_1": RecordMetrics(
                record_id="record_1",
                metrics={
                    "user_behavioral_fidelity": _make_score("user_behavioral_fidelity", 0.5),
                    "user_speech_fidelity": _make_score("user_speech_fidelity", 0.8, details=tts_pass),
                },
            ),
            "record_2": RecordMetrics(
                record_id="record_2",
                metrics={
                    "user_behavioral_fidelity": _make_score("user_behavioral_fidelity", 1.0),
                    "user_speech_fidelity": _make_score("user_speech_fidelity", 0.9, details=tts_pass),
                },
            ),
        }
        with patch("eva.orchestrator.validation_runner.MetricsRunner") as Mock:
            Mock.return_value.run = AsyncMock(return_value=MetricsRunResult(all_metrics=mock_results, total_records=2))
            results = await runner_skip_cf.run_validation()

        assert results["record_1"].passed is False
        assert results["record_1"].failure_category == "validation_failed"
        assert "user_behavioral_fidelity" in results["record_1"].failed_metrics
        assert results["record_2"].passed is True

    @pytest.mark.asyncio
    async def test_full_with_short_circuit(self, validation_runner, temp_dir):
        records_dir = temp_dir / "records"
        record_1_dir = records_dir / "record_1"
        record_1_dir.mkdir(parents=True)
        (record_1_dir / "elevenlabs_events.jsonl").write_text(
            '{"type": "connection_state", "data": {"details": {"reason": "goodbye"}}}\n'
        )
        (records_dir / "record_2").mkdir(parents=True)

        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        mock_results = {
            "record_1": RecordMetrics(
                record_id="record_1",
                metrics={
                    "user_behavioral_fidelity": _make_score("user_behavioral_fidelity", 1.0),
                    "user_speech_fidelity": _make_score("user_speech_fidelity", 0.8, details=tts_pass),
                },
            ),
        }
        with patch("eva.orchestrator.validation_runner.MetricsRunner") as Mock:
            Mock.return_value.run = AsyncMock(return_value=MetricsRunResult(all_metrics=mock_results, total_records=1))
            results = await validation_runner.run_validation()

        assert results["record_2"].passed is False
        assert results["record_2"].failure_category == "not_finished"
        assert "conversation_finished" in results["record_2"].failed_metrics
        assert results["record_1"].passed is True
        assert len(Mock.call_args[1]["dataset"]) == 1
        assert Mock.call_args[1]["dataset"][0].id == "record_1"

    @pytest.mark.asyncio
    async def test_metric_error(self, runner_skip_cf):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        mock_results = {
            "record_1": RecordMetrics(
                record_id="record_1",
                metrics={
                    "user_behavioral_fidelity": _make_score("user_behavioral_fidelity", 0.0, error="Failed to compute"),
                    "user_speech_fidelity": _make_score("user_speech_fidelity", 0.8, details=tts_pass),
                },
            ),
        }
        with patch("eva.orchestrator.validation_runner.MetricsRunner") as Mock:
            Mock.return_value.run = AsyncMock(return_value=MetricsRunResult(all_metrics=mock_results, total_records=1))
            results = await runner_skip_cf.run_validation()

        assert results["record_1"].passed is False
        assert "user_behavioral_fidelity" in results["record_1"].failed_metrics

    @pytest.mark.asyncio
    async def test_missing_metric(self, runner_skip_cf):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        mock_results = {
            "record_1": RecordMetrics(
                record_id="record_1",
                metrics={
                    "user_speech_fidelity": _make_score("user_speech_fidelity", 0.8, details=tts_pass),
                },
            ),
        }
        with patch("eva.orchestrator.validation_runner.MetricsRunner") as Mock:
            Mock.return_value.run = AsyncMock(return_value=MetricsRunResult(all_metrics=mock_results, total_records=1))
            results = await runner_skip_cf.run_validation()

        assert results["record_1"].passed is False
        assert "user_behavioral_fidelity" in results["record_1"].failed_metrics

    @pytest.mark.asyncio
    async def test_output_ids_passed_to_metrics_runner(self):
        """output_ids are forwarded as record_ids to MetricsRunner."""
        output_ids = ["rec-0/trial_0", "rec-0/trial_1"]
        runner = ValidationRunner(
            run_dir=Path("/tmp/fake"),
            dataset=[_make_record("rec-0")],
            thresholds={},
            skip_conversation_finished=True,
            output_ids=output_ids,
        )
        captured = {}
        with patch("eva.orchestrator.validation_runner.MetricsRunner") as Mock:
            mock_instance = AsyncMock()
            mock_instance.run.return_value = MetricsRunResult(all_metrics={})
            Mock.return_value = mock_instance
            Mock.side_effect = lambda **kwargs: (captured.update(kwargs), mock_instance)[1]

            await runner.run_validation()

        assert captured["record_ids"] == output_ids
