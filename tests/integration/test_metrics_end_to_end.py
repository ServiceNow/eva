"""Integration test for metrics system end-to-end flow.

Tests the complete pipeline: postprocessor → metrics computation → aggregation.
Uses real conversation artifacts from tests/artifacts/records/1.1.2/.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from eva.metrics.runner import MetricsRunner
from eva.models.record import EvaluationRecord, GroundTruth

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "records" / "1.1.2"
RECORD_ID = "1.1.2"


@pytest.fixture
def mock_dataset():
    """Create a dataset with one record matching the real artifacts."""
    record = EvaluationRecord(
        id=RECORD_ID,
        user_goal="Change flight to March 25th",
        user_config={
            "user_persona": "Traveler rebooking a flight",
        },
        current_date_time="2026-03-01T22:22:03Z",
        scenario_context={"steps": ["get_reservation", "search_rebooking_options", "rebook_flight"]},
        ground_truth=GroundTruth(
            expected_scenario_db=json.loads((ARTIFACTS_DIR / "final_scenario_db.json").read_text()),
        ),
        category="airline",
    )
    return [record]


@pytest.fixture
def mock_run_dir(tmp_path):
    """Create a run directory using real artifact files."""
    run_dir = tmp_path / "output" / "test_run"
    record_dir = run_dir / "records" / RECORD_ID
    record_dir.mkdir(parents=True)

    # Copy all artifact files into the temp record directory
    for src_file in ARTIFACTS_DIR.iterdir():
        if src_file.is_file():
            shutil.copy2(src_file, record_dir / src_file.name)

    # Rewrite result.json with corrected paths (originals point to the run output dir)
    result_data = json.loads((record_dir / "result.json").read_text())
    result_data["output_dir"] = str(record_dir)
    result_data["pipecat_logs_path"] = str(record_dir / "pipecat_logs.jsonl")
    result_data["elevenlabs_logs_path"] = str(record_dir / "elevenlabs_events.jsonl")
    # Fix audio paths if present
    for audio_key in ("audio_mixed_path", "audio_assistant_path", "audio_user_path"):
        if result_data.get(audio_key):
            filename = Path(result_data[audio_key]).name
            result_data[audio_key] = str(record_dir / filename)
    (record_dir / "result.json").write_text(json.dumps(result_data, indent=2))

    # Create agent config YAML
    agent_config = {
        "id": "airline_agent",
        "name": "Airline Agent",
        "description": "Handles airline requests",
        "role": "You are a airline assistant.",
        "instructions": "Help users rebook their flights.",
        "tools": [
            {"id": "get_reservation", "name": "get_reservation", "description": "Get reservation details"},
            {
                "id": "search_rebooking_options",
                "name": "search_rebooking_options",
                "description": "Search rebooking options",
            },
            {"id": "rebook_flight", "name": "rebook_flight", "description": "Rebook a flight"},
            {"id": "assign_seat", "name": "assign_seat", "description": "Assign a seat"},
        ],
    }
    agent_config_path = tmp_path / "agent_config.yaml"
    with open(agent_config_path, "w") as f:
        yaml.dump(agent_config, f)

    # Create config.json pointing to agent config
    config = {"agent_config_path": str(agent_config_path)}
    (run_dir / "config.json").write_text(json.dumps(config))

    return run_dir


@pytest.mark.asyncio
async def test_code_metrics_compute(mock_run_dir, mock_dataset):
    """Test that code-based metrics can compute successfully."""
    with patch("eva.utils.llm_client.LLMClient.generate_text", new_callable=AsyncMock):
        runner = MetricsRunner(
            run_dir=mock_run_dir,
            dataset=mock_dataset,
            metric_names=["task_completion"],
            metric_configs={},
        )

        metrics = await runner.run()

        assert RECORD_ID in metrics.all_metrics
        record_metrics = metrics.all_metrics[RECORD_ID]
        assert "task_completion" in record_metrics.metrics


@pytest.mark.asyncio
async def test_judge_metrics_with_mock_llm(mock_run_dir, mock_dataset):
    """Test that judge metrics work with mocked LLM responses."""

    # Conciseness judge expects a list with one item per assistant turn, rating 1-3
    # Each item must include turn_id matching the turn numbers from the conversation
    async def mock_generate_text(messages, **kwargs):
        # Generate mock responses with turn_ids 0..9; only those matching
        # actual assistant turn_ids in the conversation will be accepted.
        mock_response = json.dumps(
            [{"turn_id": i, "rating": 3, "explanation": "Response was concise", "failure_modes": []} for i in range(10)]
        )
        return mock_response, None

    with patch("eva.utils.llm_client.LLMClient.generate_text", side_effect=mock_generate_text):
        runner = MetricsRunner(
            run_dir=mock_run_dir,
            dataset=mock_dataset,
            metric_names=["conciseness"],
            metric_configs={
                "conciseness": {"judge_model": "gpt-5-turbo"},
            },
        )

        metrics = await runner.run()

        assert RECORD_ID in metrics.all_metrics
        record_metrics = metrics.all_metrics[RECORD_ID]

        assert "conciseness" in record_metrics.metrics

        conc_score = record_metrics.metrics["conciseness"]
        assert conc_score.error is None
        assert conc_score.score == 3.0
        assert conc_score.normalized_score == 1.0


@pytest.mark.asyncio
async def test_metrics_runner_aggregation(mock_run_dir, mock_dataset):
    """Test that MetricsRunner correctly computes and saves results."""
    with patch("eva.utils.llm_client.LLMClient.generate_text", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = (json.dumps({"rating": 3, "explanation": "Good"}), None)

        runner = MetricsRunner(
            run_dir=mock_run_dir,
            dataset=mock_dataset,
            metric_names=["task_completion"],
            metric_configs={},
        )

        metrics = await runner.run()
        record_metrics = metrics.all_metrics[RECORD_ID]

        assert len(record_metrics.metrics) > 0


@pytest.mark.asyncio
async def test_validation_metrics(mock_run_dir, mock_dataset):
    """Test validation metrics for conversation quality."""
    with patch("eva.utils.llm_client.LLMClient.generate_text", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps({"rating": 3, "explanation": "User dialogue was very plausible"})

        runner = MetricsRunner(
            run_dir=mock_run_dir,
            dataset=mock_dataset,
            metric_names=["conversation_valid_end"],
            metric_configs={},
        )

        metrics = await runner.run()
        record_metrics = metrics.all_metrics[RECORD_ID]

        assert "conversation_valid_end" in record_metrics.metrics


@pytest.mark.asyncio
async def test_metrics_summary_generation(mock_run_dir, mock_dataset):
    """Test that summary files are generated correctly."""
    with patch("eva.utils.llm_client.LLMClient.generate_text", new_callable=AsyncMock):
        runner = MetricsRunner(
            run_dir=mock_run_dir,
            dataset=mock_dataset,
            metric_names=["task_completion"],
            metric_configs={},
        )

        await runner.run()

        summary_json = mock_run_dir / "metrics_summary.json"
        assert summary_json.exists()

        summary_data = json.loads(summary_json.read_text())
        assert "total_records" in summary_data
        assert "per_metric" in summary_data
        assert "overall_scores" in summary_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
