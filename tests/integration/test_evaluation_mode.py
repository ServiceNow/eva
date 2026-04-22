"""Integration test for validation mode.

Tests the complete evaluation pipeline:
1. Run targeted conversations
2. Check conversation_finished
3. Run validation metrics
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
            subflow_in_depth={},
            expected_flow="test_flow",
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
            subflow_in_depth={},
            expected_flow="test_flow",
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
    """Helper to create mock validation results."""
    results = {}
    for record_id in pass_ids:
        results[record_id] = ValidationResult(passed=True)
    for record_id in fail_ids:
        results[record_id] = ValidationResult(
            passed=False,
            failed_metrics=["user_behavioral_fidelity"],
        )
    return results


def _mock_run_targeted(runner, attempt_counter=None, completed_fn=None):
    """Create a mock _run_targeted that creates result files and returns results.

    Args:
        runner: The BenchmarkRunner instance.
        attempt_counter: Optional dict with "count" key to track attempts.
        completed_fn: Optional function(record_id, attempt) -> bool to control completion.
    """

    async def mock_targeted(tasks):
        if attempt_counter is not None:
            attempt_counter["count"] += 1

        results = {}
        for record, output_id in tasks:
            record_dir = runner.output_dir / "records" / output_id
            record_dir.mkdir(parents=True, exist_ok=True)

            completed = True
            if completed_fn is not None:
                completed = completed_fn(record.id, attempt_counter["count"] if attempt_counter else 1)

            result = create_mock_conversation_result(
                record_id=record.id,
                completed=completed,
                output_dir=str(record_dir),
            )
            (record_dir / "result.json").write_text(result.model_dump_json(indent=2))
            results[output_id] = result

        return results

    return mock_targeted


@pytest.mark.asyncio
async def test_evaluation_mode_all_pass_first_attempt(eval_config, mock_dataset):
    """Test validation mode when all records pass validation on first attempt."""
    runner = BenchmarkRunner(eval_config)

    attempt_counter = {"count": 0}

    mock_validation_results = create_mock_validation_results(
        pass_ids=["pass_record_1", "fail_record_1"],
        fail_ids=[],
    )

    with patch.object(runner, "_run_targeted", side_effect=_mock_run_targeted(runner, attempt_counter)):
        with patch("eva.orchestrator.runner.check_conversation_finished", return_value=True):
            with patch("eva.orchestrator.runner.ValidationRunner") as MockValidationRunner:
                mock_val_runner = AsyncMock()
                mock_val_runner.run_validation.return_value = mock_validation_results
                MockValidationRunner.return_value = mock_val_runner

                summary = await runner.run(mock_dataset)

                assert summary.total_records == 2
                assert summary.successful_records == 2
                assert summary.failed_records == 0
                assert attempt_counter["count"] == 1

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

    attempt_counter = {"count": 0}

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

    with patch.object(runner, "_run_targeted", side_effect=_mock_run_targeted(runner, attempt_counter)):
        with patch("eva.orchestrator.runner.check_conversation_finished", return_value=True):
            with patch("eva.orchestrator.runner.ValidationRunner") as MockValidationRunner:
                mock_val_runner = AsyncMock()
                mock_val_runner.run_validation.side_effect = validation_attempts
                MockValidationRunner.return_value = mock_val_runner

                summary = await runner.run(mock_dataset)

                assert summary.total_records == 2
                assert summary.successful_records == 2
                assert summary.failed_records == 0
                assert attempt_counter["count"] == 2

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

    attempt_counter = {"count": 0}

    # Validation always fails for fail_record_1
    def mock_validation_results_always_fail(*args, **kwargs):
        return create_mock_validation_results(
            pass_ids=["pass_record_1"],
            fail_ids=["fail_record_1"],
        )

    with patch.object(runner, "_run_targeted", side_effect=_mock_run_targeted(runner, attempt_counter)):
        with patch("eva.orchestrator.runner.check_conversation_finished", return_value=True):
            with patch("eva.orchestrator.runner.ValidationRunner") as MockValidationRunner:
                mock_val_runner = AsyncMock()
                mock_val_runner.run_validation.side_effect = mock_validation_results_always_fail
                MockValidationRunner.return_value = mock_val_runner

                summary = await runner.run(mock_dataset)

                assert summary.total_records == 2
                assert summary.successful_records == 1  # Only pass_record_1
                assert summary.failed_records == 1  # fail_record_1 never passed

                # Should have run max_rerun_attempts times (3)
                assert attempt_counter["count"] == 3

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

                # Archiving happens for attempts 1 and 2, not 3 (final stays)
                for attempt in [1, 2]:
                    archive_dir = runner.output_dir / "records" / f"fail_record_1_failed_attempt_{attempt}"
                    assert archive_dir.exists()

                final_failed_attempt_archive = runner.output_dir / "records" / "fail_record_1_failed_attempt_3"
                assert not final_failed_attempt_archive.exists()

                final_record_dir = runner.output_dir / "records" / "fail_record_1"
                assert final_record_dir.exists()


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

    attempt_counter = {"count": 0}

    # Attempt 1: fail_record_1 fails the gate (not_finished); pass_record_1 passes.
    # Attempt 2: both pass. ValidationRunner owns the gate, so we emit not_finished
    # directly in its results instead of patching check_conversation_finished.
    attempt_1_results = {
        "pass_record_1": ValidationResult(passed=True),
        # not_finished: gate rejected, no metrics ran → empty failed_metrics.
        "fail_record_1": ValidationResult(passed=False),
    }
    attempt_2_results = create_mock_validation_results(
        pass_ids=["fail_record_1"],
        fail_ids=[],
    )

    with patch.object(runner, "_run_targeted", side_effect=_mock_run_targeted(runner, attempt_counter)):
        with patch("eva.orchestrator.runner.ValidationRunner") as MockValidationRunner:
            mock_val_runner = AsyncMock()
            mock_val_runner.run_validation.side_effect = [attempt_1_results, attempt_2_results]
            MockValidationRunner.return_value = mock_val_runner

            summary = await runner.run(mock_dataset)

            assert summary.total_records == 2
            assert summary.successful_records == 2
            assert summary.failed_records == 0
            assert attempt_counter["count"] == 2


@pytest.mark.asyncio
async def test_evaluation_mode_with_unresolved_errors(eval_config, mock_dataset):
    """Test validation mode with records that have unresolved errors (completed=False)."""
    runner = BenchmarkRunner(eval_config)

    attempt_counter = {"count": 0}

    # fail_record_1 always fails to complete
    def completed_fn(record_id, attempt):
        return record_id != "fail_record_1"

    mock_validation_results = create_mock_validation_results(
        pass_ids=["pass_record_1"],
        fail_ids=[],
    )

    with patch.object(runner, "_run_targeted", side_effect=_mock_run_targeted(runner, attempt_counter, completed_fn)):
        with patch("eva.orchestrator.runner.check_conversation_finished", return_value=True):
            with patch("eva.orchestrator.runner.ValidationRunner") as MockValidationRunner:
                mock_val_runner = AsyncMock()
                mock_val_runner.run_validation.return_value = mock_validation_results
                MockValidationRunner.return_value = mock_val_runner

                summary = await runner.run(mock_dataset)

                # fail_record_1 should fail due to completed=False
                assert summary.successful_records == 1
                assert summary.failed_records == 1

                # Should reach max attempts
                assert attempt_counter["count"] == 3
