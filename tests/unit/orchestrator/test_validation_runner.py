"""Tests for ValidationRunner."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


def _ctx(
    conversation_finished: bool = False,
    conversation_ended_reason: str | None = None,
    audio_timestamps_user_turns: dict | None = None,
    audio_timestamps_assistant_turns: dict | None = None,
):
    """Duck-typed stand-in for _ProcessorContext exposing the fields the gate reads."""
    return SimpleNamespace(
        conversation_finished=conversation_finished,
        conversation_ended_reason=conversation_ended_reason,
        audio_timestamps_user_turns=audio_timestamps_user_turns or {},
        audio_timestamps_assistant_turns=audio_timestamps_assistant_turns or {},
    )


def _ctx_agent_timeout():
    """Duck-typed context that satisfies is_agent_timeout_on_user_turn."""
    return _ctx(
        conversation_ended_reason="inactivity_timeout",
        audio_timestamps_user_turns={0: [(0.0, 5.0)]},
        audio_timestamps_assistant_turns={0: [(1.0, 2.0)]},
    )


@pytest.fixture
def sample_records():
    return [_make_record("record_1"), _make_record("record_2")]


@pytest.fixture
def validation_runner(temp_dir, sample_records):
    return ValidationRunner(
        run_dir=temp_dir,
        dataset=sample_records,
        thresholds={"conversation_valid_end": 1.0, "user_behavioral_fidelity": 1.0},
    )


def _mock_metrics_runner(contexts: dict, mock_results: dict) -> MagicMock:
    """Build a MagicMock that matches MetricsRunner's shape: sync process_records + async run."""
    instance = MagicMock()
    instance.process_records.return_value = contexts
    instance.run = AsyncMock(return_value=MetricsRunResult(all_metrics=mock_results, total_records=len(mock_results)))
    return instance


class TestValidationResult:
    def test_passed_defaults(self):
        vr = ValidationResult(passed=True)
        assert vr.passed is True
        assert vr.failed_metrics == []
        assert vr.scores == {}

    def test_not_finished_has_empty_failed_metrics(self):
        """Gate-rejected records have empty failed_metrics — the signal for "not_finished"."""
        vr = ValidationResult(passed=False)
        assert vr.passed is False
        assert vr.failed_metrics == []

    def test_validation_failed_has_populated_failed_metrics(self):
        """Metric-threshold failures carry the names of violating metrics."""
        vr = ValidationResult(passed=False, failed_metrics=["user_behavioral_fidelity"])
        assert vr.passed is False
        assert vr.failed_metrics == ["user_behavioral_fidelity"]


class TestClassify:
    """Direct coverage of the gate's classification rule."""

    def test_goodbye_passes(self):
        contexts = {"r1": _ctx(conversation_finished=True)}
        gp, nf, at = ValidationRunner._classify(contexts, ["r1"])
        assert gp == ["r1"]
        assert nf == []
        assert at == set()

    def test_agent_timeout_passes_and_flagged(self):
        contexts = {"r1": _ctx_agent_timeout()}
        gp, nf, at = ValidationRunner._classify(contexts, ["r1"])
        assert gp == ["r1"]
        assert nf == []
        assert at == {"r1"}

    def test_neither_is_not_finished(self):
        contexts = {"r1": _ctx()}
        gp, nf, at = ValidationRunner._classify(contexts, ["r1"])
        assert gp == []
        assert nf == ["r1"]
        assert at == set()

    def test_missing_context_is_not_finished(self):
        gp, nf, at = ValidationRunner._classify({}, ["r1"])
        assert gp == []
        assert nf == ["r1"]
        assert at == set()

    def test_goodbye_takes_precedence_over_flag(self):
        # Conversation-finished + would-be agent-timeout markers: goodbye short-circuits.
        contexts = {
            "r1": _ctx(
                conversation_finished=True,
                conversation_ended_reason="inactivity_timeout",
                audio_timestamps_user_turns={0: [(0.0, 5.0)]},
                audio_timestamps_assistant_turns={0: [(1.0, 2.0)]},
            )
        }
        gp, nf, at = ValidationRunner._classify(contexts, ["r1"])
        assert gp == ["r1"]
        assert at == set()

    def test_mixed_set(self):
        contexts = {
            "a": _ctx(conversation_finished=True),
            "b": _ctx_agent_timeout(),
            "c": _ctx(),
        }
        gp, nf, at = ValidationRunner._classify(contexts, ["a", "b", "c", "d"])
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
            self._metrics(conversation_valid_end=1.0, user_behavioral_fidelity=1.0),
            ["conversation_valid_end", "user_behavioral_fidelity"],
        )
        assert result.passed is True
        assert result.failed_metrics == []

    def test_one_below_threshold(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(conversation_valid_end=1.0, user_behavioral_fidelity=0.5),
            ["conversation_valid_end", "user_behavioral_fidelity"],
        )
        assert result.passed is False
        assert "user_behavioral_fidelity" in result.failed_metrics

    def test_at_threshold(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(conversation_valid_end=1.0, user_behavioral_fidelity=1.0),
            ["conversation_valid_end", "user_behavioral_fidelity"],
        )
        assert result.passed is True

    def test_just_below_threshold(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(conversation_valid_end=1.0, user_behavioral_fidelity=0.99),
            ["conversation_valid_end", "user_behavioral_fidelity"],
        )
        assert result.passed is False
        assert "user_behavioral_fidelity" in result.failed_metrics

    def test_multiple_failures(self, validation_runner):
        result = validation_runner._evaluate_record(
            "record_1",
            self._metrics(conversation_valid_end=0.5, user_behavioral_fidelity=0.5),
            ["conversation_valid_end", "user_behavioral_fidelity"],
        )
        assert result.passed is False
        assert "conversation_valid_end" in result.failed_metrics
        assert "user_behavioral_fidelity" in result.failed_metrics
        assert result.scores["conversation_valid_end"] == 0.5
        assert result.scores["user_behavioral_fidelity"] == 0.5

    def test_metric_error_fails(self, validation_runner):
        record_metrics = RecordMetrics(
            record_id="record_1",
            metrics={
                "conversation_valid_end": _make_score("conversation_valid_end", 1.0),
                "user_behavioral_fidelity": _make_score("user_behavioral_fidelity", 0.0, error="Computation failed"),
            },
        )
        result = validation_runner._evaluate_record(
            "record_1", record_metrics, ["conversation_valid_end", "user_behavioral_fidelity"]
        )
        assert result.passed is False
        assert "user_behavioral_fidelity" in result.failed_metrics

    def test_missing_metric_fails(self, validation_runner):
        record_metrics = RecordMetrics(
            record_id="record_1",
            metrics={"conversation_valid_end": _make_score("conversation_valid_end", 1.0)},
        )
        result = validation_runner._evaluate_record(
            "record_1", record_metrics, ["conversation_valid_end", "user_behavioral_fidelity"]
        )
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
            "conversation_valid_end",
            "user_behavioral_fidelity",
            "user_speech_fidelity",
        ]

    @pytest.mark.asyncio
    async def test_all_pass(self, validation_runner):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        contexts = {
            "record_1": _ctx(conversation_finished=True),
            "record_2": _ctx(conversation_finished=True),
        }
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
            Mock.return_value = _mock_metrics_runner(contexts, mock_results)
            results = await validation_runner.run_validation()

        assert results["record_1"].passed is True
        assert results["record_2"].passed is True
        assert results["record_1"].failed_metrics == []
        assert "conversation_valid_end" not in Mock.call_args[1]["metric_names"]

    @pytest.mark.asyncio
    async def test_some_fail(self, validation_runner):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        contexts = {
            "record_1": _ctx(conversation_finished=True),
            "record_2": _ctx(conversation_finished=True),
        }
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
            Mock.return_value = _mock_metrics_runner(contexts, mock_results)
            results = await validation_runner.run_validation()

        assert results["record_1"].passed is False
        assert "user_behavioral_fidelity" in results["record_1"].failed_metrics
        assert results["record_2"].passed is True

    @pytest.mark.asyncio
    async def test_not_finished_short_circuits(self, validation_runner):
        """A record with no context is marked not_finished and metrics aren't run on it."""
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        contexts = {
            "record_1": _ctx(conversation_finished=True),
        }
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
            Mock.return_value = _mock_metrics_runner(contexts, mock_results)
            results = await validation_runner.run_validation()

        assert results["record_2"].passed is False
        # Gate rejected: no metrics ran → empty failed_metrics is the "not_finished" signal.
        assert results["record_2"].failed_metrics == []
        assert results["record_1"].passed is True

    @pytest.mark.asyncio
    async def test_agent_timeout_passes_even_if_metric_scores_fail(self, validation_runner):
        """Agent-timeout records always pass the validation layer.

        The agent failure is surfaced via the ``agent_turn_response`` diagnostic metric
        in ``metrics.json``, not as a validation failure. Even when a validation metric
        dropped below its threshold for this record, ``vr.passed`` is forced True and
        ``failed_metrics`` is cleared — the metric signal is distorted by the agent's
        missing final turn, not a true validation failure.
        """
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        contexts = {
            "record_1": _ctx_agent_timeout(),
            "record_2": _ctx(conversation_finished=True),
        }
        mock_results = {
            "record_1": RecordMetrics(
                record_id="record_1",
                # Below threshold — would fail validation normally.
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
            Mock.return_value = _mock_metrics_runner(contexts, mock_results)
            results = await validation_runner.run_validation()

        assert results["record_1"].passed is True
        assert results["record_1"].failed_metrics == []
        assert results["record_2"].passed is True

    @pytest.mark.asyncio
    async def test_metric_error(self, validation_runner):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        contexts = {"record_1": _ctx(conversation_finished=True)}
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
            Mock.return_value = _mock_metrics_runner(contexts, mock_results)
            results = await validation_runner.run_validation()

        assert results["record_1"].passed is False
        assert "user_behavioral_fidelity" in results["record_1"].failed_metrics

    @pytest.mark.asyncio
    async def test_missing_metric(self, validation_runner):
        tts_pass = {"per_turn_ratings": {"turn_0": 3, "turn_1": 2}}
        contexts = {"record_1": _ctx(conversation_finished=True)}
        mock_results = {
            "record_1": RecordMetrics(
                record_id="record_1",
                metrics={
                    "user_speech_fidelity": _make_score("user_speech_fidelity", 0.8, details=tts_pass),
                },
            ),
        }
        with patch("eva.orchestrator.validation_runner.MetricsRunner") as Mock:
            Mock.return_value = _mock_metrics_runner(contexts, mock_results)
            results = await validation_runner.run_validation()

        assert results["record_1"].passed is False
        assert "user_behavioral_fidelity" in results["record_1"].failed_metrics

    @pytest.mark.asyncio
    async def test_output_ids_passed_to_metrics_runner(self, temp_dir):
        """output_ids are forwarded as record_ids to MetricsRunner."""
        output_ids = ["rec-0/trial_0", "rec-0/trial_1"]
        contexts = {
            "rec-0/trial_0": _ctx(conversation_finished=True),
            "rec-0/trial_1": _ctx(conversation_finished=True),
        }
        runner = ValidationRunner(
            run_dir=temp_dir,
            dataset=[_make_record("rec-0")],
            thresholds={},
            output_ids=output_ids,
        )
        captured = {}
        with patch("eva.orchestrator.validation_runner.MetricsRunner") as Mock:
            instance = _mock_metrics_runner(contexts, {})
            Mock.side_effect = lambda **kwargs: (captured.update(kwargs), instance)[1]

            await runner.run_validation()

        assert captured["record_ids"] == output_ids
