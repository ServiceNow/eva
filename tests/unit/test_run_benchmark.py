"""Tests for the existing-run branch of run_benchmark(), in particular --ignore-previous-config."""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eva.models.config import ModelConfig, RunConfig
from eva.models.results import RunResult
from eva.run_benchmark import run_benchmark

_MODEL_LIST = [{"model_name": "test", "litellm_params": {"model": "test"}}]
_BASE_ENV = {"EVA_MODEL_LIST": json.dumps(_MODEL_LIST)}


def _make_config(tmp_path: Path, **overrides) -> RunConfig:
    kwargs = {
        "model": ModelConfig(
            llm="test-model",
            stt="deepgram",
            tts="cartesia",
            stt_params={"api_key": "k", "model": "nova-2"},
            tts_params={"api_key": "k", "model": "sonic"},
        ),
        "output_dir": tmp_path / "output",
        "run_id": "test-run",
        **overrides,
    }
    return RunConfig(**kwargs)


def _write_existing_run(tmp_path: Path, config: RunConfig) -> None:
    run_dir = tmp_path / "output" / config.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))


def _passing_summary(run_id: str) -> RunResult:
    return RunResult(run_id=run_id, total_records=1, successful_records=1, failed_records=0, duration_seconds=1.0)


@pytest.mark.asyncio
@patch.dict(os.environ, _BASE_ENV, clear=True)
@patch("eva.run_benchmark.setup_logging")
@patch("eva.run_benchmark.router")
@patch("eva.run_benchmark.EvaluationRecord")
@patch("eva.run_benchmark.BenchmarkRunner")
class TestExistingRunBranch:
    async def test_default_rerun_loads_saved_config(
        self, mock_runner_cls, mock_record, mock_router, mock_setup_logging, tmp_path
    ):
        """Without --ignore-previous-config, the saved config.json is loaded and merged."""
        mock_record.load_dataset.return_value = [MagicMock()]
        saved_config = _make_config(tmp_path)
        _write_existing_run(tmp_path, saved_config)
        live_config = _make_config(tmp_path)

        runner = mock_runner_cls.from_existing_run.return_value
        runner.config = saved_config.model_copy()
        runner.validate_existing = AsyncMock(return_value=_passing_summary(live_config.run_id))

        exit_code = await run_benchmark(live_config)

        assert exit_code == 0
        mock_runner_cls.from_existing_run.assert_called_once()
        mock_runner_cls.assert_not_called()  # constructor not used directly on this path

    async def test_ignore_previous_config_bypasses_saved_config(
        self, mock_runner_cls, mock_record, mock_router, mock_setup_logging, tmp_path
    ):
        """--ignore-previous-config skips from_existing_run entirely and builds the runner from the live config."""
        mock_record.load_dataset.return_value = [MagicMock()]
        saved_config = _make_config(tmp_path)
        _write_existing_run(tmp_path, saved_config)
        live_config = _make_config(tmp_path, ignore_previous_config=True, domain="itsm")

        runner = mock_runner_cls.return_value
        runner.validate_existing = AsyncMock(return_value=_passing_summary(live_config.run_id))

        exit_code = await run_benchmark(live_config)

        assert exit_code == 0
        mock_runner_cls.from_existing_run.assert_not_called()
        mock_runner_cls.assert_called_once_with(live_config)

    async def test_ignore_previous_config_uses_dataset_from_live_config(
        self, mock_runner_cls, mock_record, mock_router, mock_setup_logging, tmp_path
    ):
        """The dataset loaded should follow the live config's domain, not the saved one's."""
        mock_record.load_dataset.return_value = [MagicMock()]
        saved_config = _make_config(tmp_path, domain="airline")
        _write_existing_run(tmp_path, saved_config)
        live_config = _make_config(tmp_path, ignore_previous_config=True, domain="itsm")

        runner = mock_runner_cls.return_value
        runner.config = live_config
        runner.validate_existing = AsyncMock(return_value=_passing_summary(live_config.run_id))

        await run_benchmark(live_config)

        mock_record.load_dataset.assert_called_once_with(live_config.dataset_path)
        assert "itsm" in str(live_config.dataset_path)
