"""Tests for BenchmarkRunner."""

import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eva.models.config import ModelConfig, RunConfig
from eva.models.results import ConversationResult
from eva.orchestrator.runner import BenchmarkRunner
from tests.unit.conftest import make_evaluation_record

_MODEL_LIST = [{"model_name": "test", "litellm_params": {"model": "test"}}]
_BASE_ENV = {"EVA_MODEL_LIST": json.dumps(_MODEL_LIST)}


def _make_record(record_id: str):
    return make_evaluation_record(record_id)


@patch.dict(os.environ, _BASE_ENV, clear=True)
def _make_config(tmp_path: Path, max_concurrent: int = 3) -> RunConfig:
    """Create a minimal RunConfig for testing."""
    return RunConfig(
        model=ModelConfig(
            llm="test-model",
            stt="deepgram",
            tts="cartesia",
            stt_params={"api_key": "k", "model": "nova-2"},
            tts_params={"api_key": "k", "model": "sonic"},
        ),
        max_concurrent_conversations=max_concurrent,
        output_dir=tmp_path / "output",
        run_id="test-run",
    )


def _make_runner(config: RunConfig) -> BenchmarkRunner:
    """Create a BenchmarkRunner with mocked agent loading."""
    with patch.object(BenchmarkRunner, "_load_agent_config", return_value=MagicMock()):
        return BenchmarkRunner(config)


class TestFilterRecords:
    def test_debug_mode_returns_one_record(self, tmp_path):
        """Debug mode returns only the first record."""
        config = _make_config(tmp_path)
        config = config.model_copy(update={"debug": True})
        runner = _make_runner(config)

        records = [_make_record(f"rec-{i}") for i in range(5)]
        filtered = runner._filter_records(records)

        assert len(filtered) == 1
        assert filtered[0].id == "rec-0"

    def test_record_ids_filter(self, tmp_path):
        """Filtering by specific record IDs."""
        config = _make_config(tmp_path)
        config = config.model_copy(update={"record_ids": ["rec-1", "rec-3"]})
        runner = _make_runner(config)

        records = [_make_record(f"rec-{i}") for i in range(5)]
        filtered = runner._filter_records(records)

        assert len(filtered) == 2
        assert {r.id for r in filtered} == {"rec-1", "rec-3"}

    def test_no_filter_returns_all(self, tmp_path):
        """No filters returns all records."""
        config = _make_config(tmp_path)
        runner = _make_runner(config)

        records = [_make_record(f"rec-{i}") for i in range(5)]
        filtered = runner._filter_records(records)

        assert len(filtered) == 5

    def test_missing_ids_still_returns_found(self, tmp_path):
        """Requesting IDs that don't exist should return only the ones found."""
        config = _make_config(tmp_path)
        config = config.model_copy(update={"record_ids": ["rec-0", "rec-99"]})
        runner = _make_runner(config)

        records = [_make_record(f"rec-{i}") for i in range(3)]
        filtered = runner._filter_records(records)

        assert len(filtered) == 1
        assert filtered[0].id == "rec-0"


class TestArchiveFailedAttempt:
    def test_moves_record_dir_to_archive(self, tmp_path):
        runner = _make_runner(_make_config(tmp_path))
        runner.output_dir = tmp_path

        record_dir = tmp_path / "records" / "rec-1"
        record_dir.mkdir(parents=True)
        (record_dir / "result.json").write_text('{"completed": false}')
        (record_dir / "audit_log.json").write_text("[]")

        runner._archive_failed_attempt("rec-1", 1)

        assert not record_dir.exists()
        archive = tmp_path / "records" / "rec-1_failed_attempt_1"
        assert archive.exists()
        assert (archive / "result.json").exists()
        assert (archive / "audit_log.json").exists()

    def test_noop_when_record_dir_missing(self, tmp_path):
        runner = _make_runner(_make_config(tmp_path))
        runner.output_dir = tmp_path
        (tmp_path / "records").mkdir(parents=True)
        runner._archive_failed_attempt("nonexistent", 1)

    def test_collision_increments_attempt_number(self, tmp_path):
        """If attempt_1 archive already exists, should use attempt_2."""
        runner = _make_runner(_make_config(tmp_path))
        runner.output_dir = tmp_path

        record_dir = tmp_path / "records" / "rec-1"
        record_dir.mkdir(parents=True)
        (record_dir / "marker.txt").write_text("run2")

        (tmp_path / "records" / "rec-1_failed_attempt_1").mkdir(parents=True)

        runner._archive_failed_attempt("rec-1", 1)

        assert not record_dir.exists()
        assert (tmp_path / "records" / "rec-1_failed_attempt_2" / "marker.txt").exists()


class TestSaveResultsCsv:
    def test_csv_format_and_content(self, tmp_path):
        runner = _make_runner(_make_config(tmp_path))
        runner.output_dir = tmp_path

        result = ConversationResult(
            record_id="rec-1",
            completed=True,
            started_at=datetime(2026, 1, 1, 10, 0),
            ended_at=datetime(2026, 1, 1, 10, 0, 10),
            duration_seconds=10.567,
            output_dir=str(tmp_path),
            num_turns=3,
            num_tool_calls=2,
            conversation_ended_reason="goodbye",
        )

        runner._save_results_csv(
            successful=[("rec-1", result)],
            failed_ids=["rec-2"],
        )

        csv = (tmp_path / "results.csv").read_text()
        lines = csv.strip().split("\n")

        assert lines[0] == "record_id,completed,duration_seconds,num_turns,num_tool_calls,ended_reason,error"

        fields = lines[1].split(",")
        assert fields[0] == "rec-1"
        assert fields[1] == "true"
        assert fields[2] == "10.57"
        assert fields[3] == "3"
        assert fields[4] == "2"
        assert fields[5] == "goodbye"

        assert lines[2].startswith("rec-2,false")

    def test_empty_csv_has_only_header(self, tmp_path):
        runner = _make_runner(_make_config(tmp_path))
        runner.output_dir = tmp_path

        runner._save_results_csv([], [])

        lines = (tmp_path / "results.csv").read_text().strip().split("\n")
        assert len(lines) == 1
        assert "record_id" in lines[0]


class TestFromExistingRun:
    @patch.dict(os.environ, _BASE_ENV, clear=True)
    def test_sets_output_dir_to_run_dir(self, tmp_path):
        config = _make_config(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(config.model_dump_json(indent=2))

        with patch.object(BenchmarkRunner, "_load_agent_config", return_value=MagicMock()):
            runner = BenchmarkRunner.from_existing_run(run_dir)

        assert runner.output_dir == run_dir

    def test_ignores_env_vars_when_loading_saved_config(self, tmp_path):
        """from_existing_run loads the saved config without env var contamination.

        If the current environment has a different pipeline mode set (e.g. EVA_MODEL__LLM)
        but the saved run used S2S, the saved config should load without conflicts.
        """
        # Create a saved S2S config on disk (in a clean env to avoid conflicts)
        with patch.dict(os.environ, {}, clear=True):
            s2s_config = RunConfig(
                model={"s2s": "gpt-realtime-mini", "s2s_params": {"api_key": "k", "model": "rt"}},
                model_list=_MODEL_LIST,
                run_id="s2s-run",
                output_dir=tmp_path / "output",
            )
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(s2s_config.model_dump_json(indent=2))

        # Set env vars for a *different* pipeline mode — these must be ignored
        conflicting_env = _BASE_ENV | {
            "EVA_MODEL__LLM": "gpt-5.2",
            "EVA_MODEL__STT": "deepgram",
            "EVA_MODEL__TTS": "cartesia",
            "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "k", "model": "nova-2"}),
            "EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "k", "model": "sonic"}),
        }
        with patch.dict(os.environ, conflicting_env, clear=True):
            with patch.object(BenchmarkRunner, "_load_agent_config", return_value=MagicMock()):
                runner = BenchmarkRunner.from_existing_run(run_dir)

        assert runner.config.model.s2s == "gpt-realtime-mini"
        assert runner.output_dir == run_dir

    def test_from_existing_run_then_apply_env_overrides_restores_secrets(self, tmp_path):
        """Full flow: save S2S config → from_existing_run → apply_env_overrides with a live config built from an env where cascade vars also leak in.

        The live config is the one that carries fresh secrets from the current
        environment.  If that environment has both S2S *and* cascade vars set
        (e.g. because .env contains both), max_rerun_attempts=0 must allow
        constructing the live config and apply_env_overrides must still restore
        the S2S secrets.
        """
        # Create a saved S2S config with a real secret
        with patch.dict(os.environ, {}, clear=True):
            saved = RunConfig(
                model={"s2s": "gpt-realtime-mini", "s2s_params": {"api_key": "real_secret", "model": "rt"}},
                model_list=_MODEL_LIST,
                run_id="s2s-run",
                output_dir=tmp_path / "output",
            )
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(saved.model_dump_json(indent=2))

        # Load via from_existing_run (env doesn't matter — _StoredRunConfig ignores it)
        with patch.dict(os.environ, _BASE_ENV, clear=True):
            with patch.object(BenchmarkRunner, "_load_agent_config", return_value=MagicMock()):
                runner = BenchmarkRunner.from_existing_run(run_dir)

        assert runner.config.model.s2s_params["api_key"] == "***"

        # Build the live config from an env that has both S2S and cascade vars.
        # max_rerun_attempts=0 suppresses the conflict error.
        conflicting_env = _BASE_ENV | {
            "EVA_MODEL__S2S": "gpt-realtime-mini",
            "EVA_MODEL__S2S_PARAMS": json.dumps({"api_key": "fresh_secret", "model": "rt"}),
            "EVA_MODEL__LLM": "gpt-5.2",
            "EVA_MODEL__STT": "deepgram",
            "EVA_MODEL__TTS": "cartesia",
            "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "k", "model": "nova-2"}),
            "EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "k", "model": "sonic"}),
            "EVA_MODEL__AUDIO_LLM": "whatever",
        }
        with patch.dict(os.environ, conflicting_env, clear=True):
            live = RunConfig(max_rerun_attempts=0, _cli_parse_args=[])

        # apply_env_overrides restores secrets from the live config
        runner.config.apply_env_overrides(live, strict_llm=False)
        assert runner.config.model.s2s_params["api_key"] == "fresh_secret"

    def test_missing_config_json_raises_file_not_found(self, tmp_path):
        run_dir = tmp_path / "no_config"
        run_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="config.json not found"):
            BenchmarkRunner.from_existing_run(run_dir)
