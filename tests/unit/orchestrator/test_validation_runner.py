"""Tests for ValidationRunner."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eva.metrics.runner import MetricsRunResult
from eva.models.results import MetricScore, RecordMetrics
from eva.orchestrator.validation_runner import GATE_METRIC, ValidationResult, ValidationRunner
from tests.unit.conftest import make_evaluation_record
from tests.unit.metrics.conftest import make_metric_score


def _make_record(record_id: str):
    return make_evaluation_record(record_id)


def _make_score(name: str, score: float, error: str | None = None, details: dict | None = None) -> MetricScore:
    return make_metric_score(name, score=score, error=error, details=details or {})


def _gate_score(
    score: float = 1.0,
    *,
    agent_timeout: bool = False,
    error: str | None = None,
) -> MetricScore:
    details = {"reason": "agent_timeout_on_user_turn"} if agent_timeout else {}
    return MetricScore(
        name=GATE_METRIC,
        score=score,
        normalized_score=score,
        details=details,
        error=error,
    )


def _gate_result(per_record: dict[str, MetricScore]) -> dict[str, RecordMetrics]:
    return {rid: RecordMetrics(record_id=rid, metrics={GATE_METRIC: ms}) for rid, ms in per_record.items()}


def _mock_runner(contexts: dict, all_metrics: dict) -> MagicMock:
    instance = MagicMock()
    instance.process_records.return_value = contexts
    instance.run = AsyncMock(return_value=MetricsRunResult(all_metrics=all_metrics, total_records=len(all_metrics)))
    return instance


def _patch_runners(gate_all_metrics: dict, downstream_all_metrics: dict, gate_contexts: dict | None = None):
    gate_instance = _mock_runner(gate_contexts or {}, gate_all_metrics)
    downstream_instance = _mock_runner({}, downstream_all_metrics)
    captured_calls: list[dict] = []

    def _ctor(**kwargs):
        captured_calls.append(kwargs)
        return gate_instance if len(captured_calls) == 1 else downstream_instance

    return patch("eva.orchestrator.validation_runner.MetricsRunner", side_effect=_ctor), captured_calls


@pytest.fixture
def sample_records():
    return [_make_record("record_1"), _make_record("record_2")]


@pytest.fixture
def validation_runner(temp_dir, sample_records):
    return ValidationRunner(
        run_dir=temp_dir,
        dataset=sample_records,
        thresholds={GATE_METRIC: 1.0, "user_behavioral_fidelity": 1.0},
    )


class TestValidationResult:
    def test_passed_defaults(self):
        vr = ValidationResult(passed=True)
        assert vr.passed is True
        assert vr.failed_metrics == []
        assert vr.scores == {}

    def test_not_finished_has_empty_failed_metrics(self):
        vr = ValidationResult(passed=False)
        assert vr.passed is False
        assert vr.failed_metrics == []

    def test_validation_failed_has_populated_failed_metrics(self):
        vr = ValidationResult(passed=False, failed_metrics=["user_behavioral_fidelity"])
        assert vr.passed is False
        assert vr.failed_metrics == ["user_behavioral_fidelity"]


class TestPartition:
    """Gate decision is driven entirely off the conversation_valid_end metric result."""

    def test_goodbye_passes(self):
        metrics = _gate_result({"r1": _gate_score(1.0)})
        gp, nf, at = ValidationRunner._partition(["r1"], metrics)
        assert gp == ["r1"]
        assert nf == []
        assert at == set()

    def test_agent_timeout_passes_and_flagged(self):
        metrics = _gate_result({"r1": _gate_score(1.0, agent_timeout=True)})
        gp, nf, at = ValidationRunner._partition(["r1"], metrics)
        assert gp == ["r1"]
        assert nf == []
        assert at == {"r1"}

    def test_score_below_one_is_not_finished(self):
        metrics = _gate_result({"r1": _gate_score(0.0)})
        gp, nf, at = ValidationRunner._partition(["r1"], metrics)
        assert gp == []
        assert nf == ["r1"]
        assert at == set()

    def test_metric_error_is_not_finished(self):
        metrics = _gate_result({"r1": _gate_score(0.0, error="boom")})
        gp, nf, at = ValidationRunner._partition(["r1"], metrics)
        assert gp == []
        assert nf == ["r1"]

    def test_missing_metric_is_not_finished(self):
        gp, nf, at = ValidationRunner._partition(["r1"], {})
        assert gp == []
        assert nf == ["r1"]
        assert at == set()

    def test_mixed_set(self):
        metrics = _gate_result(
            {
                "a": _gate_score(1.0),
                "b": _gate_score(1.0, agent_timeout=True),
                "c": _gate_score(0.0),
            }
        )
        gp, nf, at = ValidationRunner._partition(["a", "b", "c", "d"], metrics)
        assert set(gp) == {"a", "b"}
        assert set(nf) == {"c", "d"}
        assert at == {"b"}


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
            self._metrics(user_behavioral_fidelity=1.0),
            ["user_behavioral_fidelity"],
        )
        assert result.passed is True
        assert result.failed_metrics == []

    def test_one_below_threshold(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(user_behavioral_fidelity=0.5),
            ["user_behavioral_fidelity"],
        )
        assert result.passed is False
        assert "user_behavioral_fidelity" in result.failed_metrics

    def test_at_threshold(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(user_behavioral_fidelity=1.0),
            ["user_behavioral_fidelity"],
        )
        assert result.passed is True

    def test_just_below_threshold(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(user_behavioral_fidelity=0.99),
            ["user_behavioral_fidelity"],
        )
        assert result.passed is False
        assert "user_behavioral_fidelity" in result.failed_metrics

    def test_multiple_failures(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(user_behavioral_fidelity=0.5),
            ["user_behavioral_fidelity"],
        )
        assert result.passed is False
        assert "user_behavioral_fidelity" in result.failed_metrics

    def test_metric_not_in_thresholds_defaults_to_1(self, validation_runner):
        record_metrics = RecordMetrics(
            record_id="rec-0",
            metrics={"user_speech_fidelity": _make_score("user_speech_fidelity", 1.0)},
        )
        result = validation_runner._evaluate_record("record_1", record_metrics, ["user_speech_fidelity"])
        assert result.passed is True

    def test_missing_metric_fails(self, validation_runner):
        record_metrics = RecordMetrics(record_id="rec-0", metrics={})
        result = validation_runner._evaluate_record("record_1", record_metrics, ["user_behavioral_fidelity"])
        assert result.passed is False
        assert "user_behavioral_fidelity" in result.failed_metrics

    def test_user_speech_fidelity_per_turn_rating_1_fails(self):
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
        assert validation_runner.VALIDATION_METRICS == [
            GATE_METRIC,
            "user_behavioral_fidelity",
            "user_speech_fidelity",
        ]

    @pytest.mark.asyncio
    async def test_all_pass(self, validation_runner):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        gate_metrics = _gate_result(
            {
                "record_1": _gate_score(1.0),
                "record_2": _gate_score(1.0),
            }
        )
        downstream = {
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
        patcher, calls = _patch_runners(gate_metrics, downstream)
        with patcher:
            results = await validation_runner.run_validation()

        assert results["record_1"].passed is True
        assert results["record_2"].passed is True
        assert results["record_1"].failed_metrics == []
        assert calls[0]["metric_names"] == [GATE_METRIC]
        assert GATE_METRIC not in calls[1]["metric_names"]

    @pytest.mark.asyncio
    async def test_some_fail(self, validation_runner):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        gate_metrics = _gate_result(
            {
                "record_1": _gate_score(1.0),
                "record_2": _gate_score(1.0),
            }
        )
        downstream = {
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
        patcher, _ = _patch_runners(gate_metrics, downstream)
        with patcher:
            results = await validation_runner.run_validation()

        assert results["record_1"].passed is False
        assert "user_behavioral_fidelity" in results["record_1"].failed_metrics
        assert results["record_2"].passed is True

    @pytest.mark.asyncio
    async def test_gate_rejection_short_circuits_downstream(self, validation_runner):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        gate_metrics = _gate_result(
            {
                "record_1": _gate_score(1.0),
                "record_2": _gate_score(0.0),
            }
        )
        downstream = {
            "record_1": RecordMetrics(
                record_id="record_1",
                metrics={
                    "user_behavioral_fidelity": _make_score("user_behavioral_fidelity", 1.0),
                    "user_speech_fidelity": _make_score("user_speech_fidelity", 0.8, details=tts_pass),
                },
            ),
        }
        patcher, calls = _patch_runners(gate_metrics, downstream)
        with patcher:
            results = await validation_runner.run_validation()

        assert results["record_2"].passed is False
        assert results["record_2"].failed_metrics == []
        assert results["record_1"].passed is True
        assert calls[1]["record_ids"] == ["record_1"]

    @pytest.mark.asyncio
    async def test_agent_timeout_still_fails_on_bad_downstream_metrics(self, validation_runner):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        gate_metrics = _gate_result(
            {
                "record_1": _gate_score(1.0, agent_timeout=True),
                "record_2": _gate_score(1.0),
            }
        )
        downstream = {
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
        patcher, _ = _patch_runners(gate_metrics, downstream)
        with patcher:
            results = await validation_runner.run_validation()

        assert results["record_1"].passed is False
        assert "user_behavioral_fidelity" in results["record_1"].failed_metrics
        assert results["record_2"].passed is True

    @pytest.mark.asyncio
    async def test_metric_error(self, validation_runner):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        gate_metrics = _gate_result({"record_1": _gate_score(1.0)})
        downstream = {
            "record_1": RecordMetrics(
                record_id="record_1",
                metrics={
                    "user_behavioral_fidelity": _make_score("user_behavioral_fidelity", 0.0, error="Failed to compute"),
                    "user_speech_fidelity": _make_score("user_speech_fidelity", 0.8, details=tts_pass),
                },
            ),
        }
        patcher, _ = _patch_runners(gate_metrics, downstream)
        with patcher:
            results = await validation_runner.run_validation()

        assert results["record_1"].passed is False
        assert "user_behavioral_fidelity" in results["record_1"].failed_metrics

    @pytest.mark.asyncio
    async def test_missing_metric(self, validation_runner):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        gate_metrics = _gate_result({"record_1": _gate_score(1.0)})
        downstream = {
            "record_1": RecordMetrics(
                record_id="record_1",
                metrics={
                    "user_speech_fidelity": _make_score("user_speech_fidelity", 0.8, details=tts_pass),
                },
            ),
        }
        patcher, _ = _patch_runners(gate_metrics, downstream)
        with patcher:
            results = await validation_runner.run_validation()

        assert results["record_1"].passed is False
        assert "user_behavioral_fidelity" in results["record_1"].failed_metrics

    @pytest.mark.asyncio
    async def test_output_ids_passed_to_metrics_runner(self, temp_dir):
        output_ids = ["rec-0/trial_0", "rec-0/trial_1"]
        gate_metrics = _gate_result(
            {
                "rec-0/trial_0": _gate_score(1.0),
                "rec-0/trial_1": _gate_score(1.0),
            }
        )
        runner = ValidationRunner(
            run_dir=temp_dir,
            dataset=[_make_record("rec-0")],
            thresholds={},
            output_ids=output_ids,
        )
        patcher, calls = _patch_runners(gate_metrics, {})
        with patcher:
            await runner.run_validation()

        assert calls[0]["record_ids"] == output_ids
