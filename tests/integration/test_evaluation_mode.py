"""Integration test for validation mode.

Tests the complete evaluation pipeline:
1. Run conversations per-record (pipelined)
2. Check conversation_finished
3. Run validation metrics per-record
4. Archive and rerun failures (single flat loop)
5. Generate final summary
"""

import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from eva.models.config import PipelineConfig, RunConfig
from eva.models.record import EvaluationRecord, GroundTruth
from eva.models.results import ConversationResult
from eva.orchestrator.runner import BenchmarkRunner
from eva.orchestrator.validation_runner import ValidationResult

_TEST_MODEL_LIST = [
    {"model_name": "gpt-4", "litellm_params": {"model": "openai/gpt-4", "api_key": "test-key"}},
]


@pytest.fixture
def mock_dataset():
    """Create a mock dataset with records that will pass/fail validation."""
    return [
        EvaluationRecord(
            id="pass_record_1",
            user_goal="Test goal 1",
            user_config={
                "name": "Robert White",
                "gender": "man",
                "user_persona_id": 2,
                "user_persona": "You're direct and to the point.",
            },
            current_date_time="2024-01-15T10:00:00Z",
            scenario_context={},
            ground_truth=GroundTruth(
                expected_scenario_db={},
            ),
            category="test",
        ),
        EvaluationRecord(
            id="fail_record_1",
            user_goal="Test goal 2",
            user_config={
                "name": "Robert White",
                "gender": "man",
                "user_persona_id": 2,
                "user_persona": "You're direct and to the point.",
            },
            current_date_time="2024-01-15T10:00:00Z",
            scenario_context={},
            ground_truth=GroundTruth(
                expected_scenario_db={},
            ),
            category="test",
        ),
    ]


@pytest.fixture
@patch.dict(os.environ, {}, clear=True)
def eval_config(tmp_path):
    """Create a test config for validation mode."""
    return RunConfig(
        run_id="test_eval_run",
        model_list=_TEST_MODEL_LIST,
        model=PipelineConfig(
            llm="gpt-4",
            stt="deepgram",
            tts="cartesia",
            stt_params={"api_key": "test-key", "model": "nova-2"},
            tts_params={"api_key": "test-key", "model": "sonic-english"},
        ),
        max_rerun_attempts=3,
        validation_thresholds={
            "conversation_valid_end": 1.0,
            "user_behavioral_fidelity": 1.0,
        },
        max_concurrent_conversations=2,
        output_dir=tmp_path / "output",
    )


def create_mock_conversation_result(record_id: str, completed: bool = True, output_dir: str = "") -> ConversationResult:
    """Helper to create a mock ConversationResult."""
    return ConversationResult(
        record_id=record_id,
        completed=completed,
        error=None if completed else "Validation failed",
        started_at=datetime.now(),
        ended_at=datetime.now(),
        duration_seconds=10.0,
        output_dir=output_dir or f"output/records/{record_id}",
        num_turns=5,
        num_tool_calls=2,
        tools_called=["tool1"],
        conversation_ended_reason="goodbye" if completed else "error",
        initial_scenario_db_hash="abc123",
        final_scenario_db_hash="abc123",
    )


def create_mock_validation_results(pass_ids: list[str], fail_ids: list[str]) -> dict[str, ValidationResult]:
    """Helper to create mock validation results keyed by output_id."""
    results = {}
    for record_id in pass_ids:
        results[record_id] = ValidationResult(passed=True)
    for record_id in fail_ids:
        results[record_id] = ValidationResult(
            passed=False,
            failed_metrics=["user_behavioral_fidelity"],
        )
    return results


def _mock_run_conversation_helper(runner, call_counts=None, completed_fn=None):
    """Create a mock for _run_conversation that creates result dirs and returns (result, None).

    Args:
        runner: The BenchmarkRunner instance (for output_dir).
        call_counts: Optional dict tracking per-output_id call counts (updated in-place).
        completed_fn: Optional function(record_id, per_record_attempt) -> bool.

    Returns:
        Async function matching the _run_conversation(record, output_id) signature.
    """

    async def mock_conversation(record, output_id):
        if call_counts is not None:
            call_counts[output_id] = call_counts.get(output_id, 0) + 1
            per_record_attempt = call_counts[output_id]
        else:
            per_record_attempt = 1

        record_dir = runner.output_dir / "records" / output_id
        record_dir.mkdir(parents=True, exist_ok=True)

        completed = True
        if completed_fn is not None:
            completed = completed_fn(record.id, per_record_attempt)

        result = create_mock_conversation_result(
            record_id=record.id,
            completed=completed,
            output_dir=str(record_dir),
        )
        return result, None  # (ConversationResult, deferred_audio_task=None)

    return mock_conversation


def _make_validate_one_side_effect(validation_results_list: list[dict[str, ValidationResult]]):
    """Create a validate_one side_effect that returns results per per-record attempt.

    Args:
        validation_results_list: List of result dicts, one per attempt. The first dict
            is used on the first call for each output_id, the second on the second call, etc.

    Returns:
        Async callable suitable for AsyncMock.side_effect.
    """
    call_counts: dict[str, int] = {}

    async def side_effect(output_id: str) -> ValidationResult:
        call_counts[output_id] = call_counts.get(output_id, 0) + 1
        attempt_idx = call_counts[output_id] - 1  # 0-indexed
        attempt_idx = min(attempt_idx, len(validation_results_list) - 1)
        return validation_results_list[attempt_idx].get(output_id, ValidationResult(passed=True))

    return side_effect


@pytest.mark.asyncio
async def test_evaluation_mode_all_pass_first_attempt(eval_config, mock_dataset):
    """Test validation mode when all records pass validation on first attempt."""
    runner = BenchmarkRunner(eval_config)

    call_counts: dict[str, int] = {}
    validation_results = create_mock_validation_results(
        pass_ids=["pass_record_1", "fail_record_1"],
        fail_ids=[],
    )

    with patch.object(runner, "_run_conversation", side_effect=_mock_run_conversation_helper(runner, call_counts)):
        with patch("eva.orchestrator.runner.ValidationRunner") as MockValidationRunner:
            mock_val_runner = AsyncMock()
            MockValidationRunner.return_value = mock_val_runner
            mock_val_runner.validate_one = AsyncMock(side_effect=_make_validate_one_side_effect([validation_results]))

            summary = await runner.run(mock_dataset)

            assert summary.total_records == 2
            assert summary.successful_records == 2
            assert summary.failed_records == 0
            # Each record run exactly once (one loop iteration)
            assert call_counts == {"pass_record_1": 1, "fail_record_1": 1}

            eval_summary_path = runner.output_dir / "evaluation_summary.json"
            assert eval_summary_path.exists()

            with open(eval_summary_path) as f:
                eval_summary = json.load(f)
                sim = eval_summary["simulation"]
                assert sim["total_records"] == 2
                assert sim["successful_records"] == 2
                assert sim["failed_records"] == 0
                assert sim["total_attempts"] == 1


@pytest.mark.asyncio
async def test_evaluation_mode_rerun_failures(eval_config, mock_dataset):
    """Test validation mode with reruns for failed records."""
    runner = BenchmarkRunner(eval_config)

    call_counts: dict[str, int] = {}

    # Attempt 1: fail_record_1 fails validation
    # Attempt 2: fail_record_1 passes validation
    validation_attempts = [
        create_mock_validation_results(
            pass_ids=["pass_record_1"],
            fail_ids=["fail_record_1"],
        ),
        create_mock_validation_results(
            pass_ids=["fail_record_1"],
            fail_ids=[],
        ),
    ]

    with patch.object(runner, "_run_conversation", side_effect=_mock_run_conversation_helper(runner, call_counts)):
        with patch("eva.orchestrator.runner.ValidationRunner") as MockValidationRunner:
            mock_val_runner = AsyncMock()
            MockValidationRunner.return_value = mock_val_runner
            mock_val_runner.validate_one = AsyncMock(side_effect=_make_validate_one_side_effect(validation_attempts))

            summary = await runner.run(mock_dataset)

            assert summary.total_records == 2
            assert summary.successful_records == 2
            assert summary.failed_records == 0
            # pass_record_1 ran once, fail_record_1 ran twice
            assert call_counts.get("pass_record_1") == 1
            assert call_counts.get("fail_record_1") == 2

            eval_summary_path = runner.output_dir / "evaluation_summary.json"
            with open(eval_summary_path) as f:
                eval_summary = json.load(f)
                sim = eval_summary["simulation"]
                assert sim["total_attempts"] == 2
                assert sim["successful_records"] == 2

            # Check that failed attempt was archived
            archive_dir = runner.output_dir / "records" / "fail_record_1_failed_attempt_1"
            assert archive_dir.exists()


@pytest.mark.asyncio
async def test_evaluation_mode_max_reruns_reached(eval_config, mock_dataset):
    """Test validation mode when max reruns is reached with persistent failures."""
    runner = BenchmarkRunner(eval_config)

    call_counts: dict[str, int] = {}

    # pass_record_1 always passes, fail_record_1 always fails
    always_fail_results = {
        "pass_record_1": ValidationResult(passed=True),
        "fail_record_1": ValidationResult(
            passed=False,
            failed_metrics=["user_behavioral_fidelity"],
        ),
    }

    with patch.object(runner, "_run_conversation", side_effect=_mock_run_conversation_helper(runner, call_counts)):
        with patch("eva.orchestrator.runner.ValidationRunner") as MockValidationRunner:
            mock_val_runner = AsyncMock()
            MockValidationRunner.return_value = mock_val_runner
            mock_val_runner.validate_one = AsyncMock(
                side_effect=lambda oid: always_fail_results.get(oid, ValidationResult(passed=True))
            )

            summary = await runner.run(mock_dataset)

            assert summary.total_records == 2
            assert summary.successful_records == 1  # Only pass_record_1
            assert summary.failed_records == 1  # fail_record_1 never passed

            # fail_record_1 ran max_rerun_attempts times (3)
            assert call_counts.get("fail_record_1") == 3

            eval_summary_path = runner.output_dir / "evaluation_summary.json"
            with open(eval_summary_path) as f:
                eval_summary = json.load(f)
                sim = eval_summary["simulation"]
                assert sim["total_attempts"] == 3
                assert sim["successful_records"] == 1
                assert sim["failed_records"] == 1
                assert "fail_record_1" in sim["failed_record_ids"]
                assert len(eval_summary["rerun_history"]["fail_record_1"]) == 3
                # Each entry is a dict with structured failure info
                for entry in eval_summary["rerun_history"]["fail_record_1"]:
                    assert "attempt" in entry
                    assert "reason" in entry

            # All attempts are archived, including the final one — the original
            # record dir is moved into _failed_attempt_3 so downstream tools see
            # the failure via the directory suffix.
            for attempt in [1, 2, 3]:
                archive_dir = runner.output_dir / "records" / f"fail_record_1_failed_attempt_{attempt}"
                assert archive_dir.exists()

            final_record_dir = runner.output_dir / "records" / "fail_record_1"
            assert not final_record_dir.exists()


@pytest.mark.asyncio
async def test_archive_failed_attempt(eval_config):
    """Test _archive_failed_attempt helper method."""
    runner = BenchmarkRunner(eval_config)

    record_id = "test_record"
    record_dir = runner.output_dir / "records" / record_id
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "result.json").write_text("{}")

    runner._archive_failed_attempt(record_id, 1)

    assert not record_dir.exists()

    archive_dir = runner.output_dir / "records" / f"{record_id}_failed_attempt_1"
    assert archive_dir.exists()
    assert (archive_dir / "result.json").exists()


@pytest.mark.asyncio
async def test_archive_failed_failed_attempt_nested_output_id(eval_config):
    """_archive_failed_attempt works with nested output IDs like rec-0/trial_0."""
    runner = BenchmarkRunner(eval_config)

    nested_id = "rec-0/trial_0"
    record_dir = runner.output_dir / "records" / nested_id
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "result.json").write_text("{}")

    runner._archive_failed_attempt(nested_id, 1)

    assert not record_dir.exists()

    archive_dir = runner.output_dir / "records" / "rec-0" / "trial_0_failed_attempt_1"
    assert archive_dir.exists()
    assert (archive_dir / "result.json").exists()


@pytest.mark.asyncio
async def test_evaluation_mode_conversation_not_finished_retries(eval_config, mock_dataset):
    """Test that not_finished failures from the gate trigger retries in the flat loop."""
    runner = BenchmarkRunner(eval_config)

    call_counts: dict[str, int] = {}

    # Attempt 1: fail_record_1 fails the gate (not_finished) — validate_one returns
    # ValidationResult(passed=False) with empty failed_metrics (gate-rejection convention).
    # Attempt 2: both pass.
    validation_attempts = [
        {
            "pass_record_1": ValidationResult(passed=True),
            "fail_record_1": ValidationResult(passed=False),  # empty failed_metrics = not_finished
        },
        create_mock_validation_results(pass_ids=["fail_record_1"], fail_ids=[]),
    ]

    with patch.object(runner, "_run_conversation", side_effect=_mock_run_conversation_helper(runner, call_counts)):
        with patch("eva.orchestrator.runner.ValidationRunner") as MockValidationRunner:
            mock_val_runner = AsyncMock()
            MockValidationRunner.return_value = mock_val_runner
            mock_val_runner.validate_one = AsyncMock(side_effect=_make_validate_one_side_effect(validation_attempts))

            summary = await runner.run(mock_dataset)

            assert summary.total_records == 2
            assert summary.successful_records == 2
            assert summary.failed_records == 0
            # fail_record_1 ran twice (once gate-rejected, once passed)
            assert call_counts.get("fail_record_1") == 2


@pytest.mark.asyncio
async def test_evaluation_mode_with_unresolved_errors(eval_config, mock_dataset):
    """Test validation mode with records that have unresolved errors (completed=False)."""
    runner = BenchmarkRunner(eval_config)

    call_counts: dict[str, int] = {}

    # fail_record_1 always fails to complete
    def completed_fn(record_id, per_record_attempt):
        return record_id != "fail_record_1"

    validation_results = create_mock_validation_results(
        pass_ids=["pass_record_1"],
        fail_ids=[],
    )

    with patch.object(
        runner, "_run_conversation", side_effect=_mock_run_conversation_helper(runner, call_counts, completed_fn)
    ):
        with patch("eva.orchestrator.runner.ValidationRunner") as MockValidationRunner:
            mock_val_runner = AsyncMock()
            MockValidationRunner.return_value = mock_val_runner
            mock_val_runner.validate_one = AsyncMock(side_effect=_make_validate_one_side_effect([validation_results]))

            summary = await runner.run(mock_dataset)

            # fail_record_1 should fail due to completed=False
            assert summary.successful_records == 1
            assert summary.failed_records == 1

            # Should reach max attempts
            assert call_counts.get("fail_record_1") == 3
