"""Unit tests for RunConfig model."""

import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsError

from eva.models.config import RunConfig, SpeechToSpeechConfig

MODEL_LIST = [
    {
        "model_name": "gpt-5.2",
        "litellm_params": {
            "model": "azure/gpt-5.2",
            "api_key": "must_be_redacted",
            "max_parallel_requests": 5,
            "max_tokens": 10000,
            "reasoning_effort": "low",
            "temperature": 0.7,
            "top_p": 0.9,
            "custom_param": "must_be_preserved",
        },
        "model_info": {"base_model": "gpt-5.2"},
    },
    {
        "model_name": "gemini-3-pro",
        "litellm_params": {
            "model": "vertex_ai/gemini-3-pro",
            "vertex_project": "my-gcp-project",
            "vertex_location": "global",
            "vertex_credentials": "must_be_redacted",
            "max_parallel_requests": 5,
        },
    },
    {
        "model_name": "us.anthropic.claude-opus-4-6",
        "litellm_params": {
            "model": "bedrock/us.anthropic.claude-opus-4-6-v1",
            "aws_access_key_id": "must_be_redacted",
            "aws_secret_access_key": "must_be_redacted",
            "max_parallel_requests": 5,
        },
    },
]

_EVA_MODEL_LIST_ENV = {"EVA_MODEL_LIST": json.dumps(MODEL_LIST)}
_BASE_ENV = _EVA_MODEL_LIST_ENV | {
    "EVA_MODEL__LLM": "gpt-5.2",
    "EVA_MODEL__STT": "deepgram",
    "EVA_MODEL__TTS": "cartesia",
    "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "test_key", "model": "nova-2"}),
    "EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "test_key", "model": "sonic"}),
}


def _config(
    *,
    env_file: Path | None = None,
    env_file_vars: dict[str, str] | None = None,
    env_vars: dict[str, str] | None = None,
    cli_args: list[str] | None = None,
    **kwargs,
):
    if env_file_vars:
        assert env_file is not None, "Please pass `env_file=tmp_path / '.env'` along with `env_file_vars`."
        env_file.write_text("".join(f"{key}='{value}'\n" for key, value in env_file_vars.items()))

    with patch.dict(os.environ, env_vars or {}, clear=True):
        return RunConfig(_env_file=env_file, _cli_parse_args=cli_args, **kwargs)


class TestRunConfig:
    def test_create_minimal_config(self):
        """Test creating a minimal RunConfig."""
        config = _config(env_vars=_BASE_ENV | {"EVA_DOMAIN": "airline", "EVA_MODEL__LLM": "gpt-5.2"})

        assert config.dataset_path == Path("data/airline_dataset.jsonl")
        assert config.tool_mocks_path == Path("data/airline_scenarios")
        assert datetime.strptime(config.run_id, "%Y-%m-%d_%H-%M-%S.%f")
        assert config.max_concurrent_conversations == 1
        assert config.conversation_timeout_seconds == 360

    def test_create_full_config(self, temp_dir: Path):
        """Test creating a RunConfig with all options."""
        config = _config(
            env_vars=_EVA_MODEL_LIST_ENV
            | {
                "EVA_MODEL__LLM": "gemini",
                "EVA_MODEL__STT": "deepgram",
                "EVA_MODEL__TTS": "cartesia",
                "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "test_key", "model": "nova-2"}),
                "EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "test_key", "model": "sonic"}),
                "EVA_RUN_ID": "test_run_001",
                "EVA_MAX_CONCURRENT_CONVERSATIONS": "50",
                "EVA_CONVERSATION_TIMEOUT_SECONDS": "180",
                "EVA_OUTPUT_DIR": str(temp_dir / "output"),
                "EVA_BASE_PORT": "8000",
                "EVA_PORT_POOL_SIZE": "200",
            }
        )

        assert config.run_id == "test_run_001"
        assert config.model.llm == "gemini"
        assert config.model.stt == "deepgram"
        assert config.model.tts == "cartesia"
        assert config.max_concurrent_conversations == 50
        assert config.base_port == 8000
        assert config.port_pool_size == 200

    def test_yaml_roundtrip(self, temp_dir: Path):
        """Test saving and loading config from YAML."""
        original = _config(
            env_vars=_BASE_ENV
            | {
                "EVA_RUN_ID": "yaml_test",
                "EVA_MAX_CONCURRENT_CONVERSATIONS": "25",
            }
        )

        yaml_path = temp_dir / "config.yaml"
        original.to_yaml(yaml_path)

        assert yaml_path.exists()

        with patch.dict(os.environ, _BASE_ENV, clear=True):
            loaded = RunConfig.from_yaml(yaml_path)
        assert loaded.run_id == "yaml_test"
        assert loaded.max_concurrent_conversations == 25
        assert loaded.model.llm == "gpt-5.2"

    def test_validation_bounds(self):
        """Test that values are validated within bounds."""
        # max_concurrent_conversations too low
        with pytest.raises(ValueError):
            _config(env_vars=_BASE_ENV | {"EVA_MAX_CONCURRENT_CONVERSATIONS": "0"})

        # conversation_timeout_seconds too low
        with pytest.raises(ValueError):
            _config(env_vars=_BASE_ENV | {"EVA_CONVERSATION_TIMEOUT_SECONDS": "10"})

    @pytest.mark.parametrize("indent", (None, 2))
    @pytest.mark.parametrize("vars_location", ("env_vars", "env_file_vars"))
    def test_indentation_in_model_list(self, tmp_path: Path, vars_location: str, indent: int | None):
        """Multiple deployments are parsed correctly."""
        env = {
            "EVA_MODEL_LIST": json.dumps(MODEL_LIST, indent=indent),
            "EVA_MODEL__LLM": "gpt-5.2",
            "EVA_MODEL__STT": "deepgram",
            "EVA_MODEL__TTS": "cartesia",
            "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "test_key", "model": "nova-2"}),
            "EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "test_key", "model": "sonic"}),
        }
        config = _config(env_file=tmp_path / ".env", **{vars_location: env})

        assert config.model_list == MODEL_LIST

    def test_secrets_redacted(self):
        """Secrets are redacted in model_list."""
        config = _config(env_vars=_BASE_ENV)
        dumped = config.model_dump(mode="json")
        assert dumped["model_list"][0]["litellm_params"]["api_key"] == "***"
        assert dumped["model_list"][1]["litellm_params"]["vertex_credentials"] == "***"
        assert dumped["model_list"][2]["litellm_params"]["aws_access_key_id"] == "***"
        assert dumped["model_list"][2]["litellm_params"]["aws_secret_access_key"] == "***"

    @pytest.mark.parametrize(
        "environ, expected_exception, expected_message",
        (
            (
                {"EVA_MODEL__LLM": "gpt-5.2"},
                ValidationError,
                r"model_list\s+Field required",
            ),
            (
                {"EVA_MODEL_LIST": "invalid json", "EVA_MODEL__LLM": "gpt-5.2"},
                SettingsError,
                r'error parsing value for field "model_list"',
            ),
            (
                {"EVA_MODEL_LIST": "[]", "EVA_MODEL__LLM": "gpt-5.2"},
                ValidationError,
                r"model_list\s+List should have at least 1 item",
            ),
            (
                {"EVA_MODEL_LIST": '[{"litellm_params": {"model": "azure/gpt-5.2"}}]', "EVA_MODEL__LLM": "gpt-5.2"},
                ValidationError,
                r"model_name\s+Field required",
            ),
            (
                {"EVA_MODEL_LIST": '[{"model_name": "gpt-5.2"}]', "EVA_MODEL__LLM": "gpt-5.2"},
                ValidationError,
                r"litellm_params\s+Field required",
            ),
        ),
    )
    def test_invalid_model_list(self, environ, expected_exception, expected_message):
        """Missing EVA_MODEL_LIST env var raises a ValidationError."""
        with pytest.raises(expected_exception, match=expected_message):
            _config(env_vars=environ)

    @pytest.mark.parametrize(
        "environ, expected_message",
        (
            (
                {},
                r"model\s+Field required",
            ),
            (
                {"EVA_MODEL": "{}"},
                # Discriminator defaults to PipelineConfig when no unique field present
                r"model\.pipeline\.llm\s+Field required",
            ),
            (
                {"EVA_MODEL__LLM": "a", "EVA_MODEL__S2S": "b"},
                "Multiple pipeline modes set",
            ),
            (
                {"EVA_MODEL__LLM": "a", "EVA_MODEL__AUDIO_LLM": "ultravox"},
                "Multiple pipeline modes set",
            ),
            (
                {"EVA_MODEL__S2S": "a", "EVA_MODEL__AUDIO_LLM": "ultravox"},
                "Multiple pipeline modes set",
            ),
            (
                {"EVA_MODEL__LLM": "a", "EVA_MODEL__S2S": "b", "EVA_MODEL__AUDIO_LLM": "ultravox"},
                "Multiple pipeline modes set",
            ),
            (
                {"EVA_MODEL__LLM": "gpt-5.2", "EVA_MODEL__TTS": "cartesia"},
                r"model\.pipeline\.stt\s+Field required",
            ),
            (
                {"EVA_MODEL__LLM": "gpt-5.2", "EVA_MODEL__STT": "deepgram"},
                r"model\.pipeline\.tts\s+Field required",
            ),
            (
                {"EVA_MODEL__AUDIO_LLM": "ultravox"},
                r"model\.audio_llm\.tts\s+Field required",
            ),
        ),
        ids=(
            "Missing",
            "Empty",
            "Mixed LLM + S2S",
            "Mixed LLM + Audio LLM",
            "Mixed S2S + Audio LLM",
            "Mixed all three",
            "LLM without STT",
            "LLM without TTS",
            "Audio LLM without TTS",
        ),
    )
    def test_invalid_model_pipeline(self, environ, expected_message):
        with pytest.raises(ValidationError, match=expected_message):
            _config(env_vars=_EVA_MODEL_LIST_ENV | environ)

    def test_missing_stt_tts_params(self):
        """Missing api_key or model in STT/TTS params causes a clear error."""
        base = _EVA_MODEL_LIST_ENV | {
            "EVA_MODEL__LLM": "gpt-5.2",
            "EVA_MODEL__STT": "deepgram",
            "EVA_MODEL__TTS": "cartesia",
        }
        # Empty params → missing both api_key and model
        with pytest.raises(ValueError, match=r'"api_key" and "model" required in EVA_MODEL__STT_PARAMS'):
            _config(
                env_vars=base
                | {
                    "EVA_MODEL__STT_PARAMS": json.dumps({}),
                    "EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "k", "model": "sonic"}),
                }
            )

        # api_key present but model missing
        with pytest.raises(ValueError, match=r'"model" required in EVA_MODEL__TTS_PARAMS'):
            _config(
                env_vars=base
                | {
                    "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "k", "model": "nova-2"}),
                    "EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "k"}),
                }
            )

    def test_nvidia_stt_skips_params_validation(self):
        """NVIDIA STT skips api_key/model validation (uses url-based config)."""
        config = _config(
            env_vars=_EVA_MODEL_LIST_ENV
            | {
                "EVA_MODEL__LLM": "gpt-5.2",
                "EVA_MODEL__STT": "nvidia",
                "EVA_MODEL__TTS": "cartesia",
                "EVA_MODEL__STT_PARAMS": json.dumps({"url": "ws://localhost:8000"}),
                "EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "k", "model": "sonic"}),
            }
        )
        assert config.model.stt == "nvidia"


class TestDefaults:
    """Verify default values match expectations."""

    def test_defaults(self):
        c = _config(env_vars=_BASE_ENV)
        assert c.domain == "airline"
        assert c.dataset_path == Path("data/airline_dataset.jsonl")
        assert c.tool_mocks_path == Path("data/airline_scenarios")
        assert c.agent_config_path == Path("configs/agents/airline_agent.yaml")
        assert c.output_dir == Path("output")
        assert c.model.llm == "gpt-5.2"
        assert c.model.stt == "deepgram"
        assert c.model.tts == "cartesia"
        assert c.max_concurrent_conversations == 1
        assert c.conversation_timeout_seconds == 360
        assert c.base_port == 10000
        assert c.port_pool_size == 150
        assert c.max_rerun_attempts == 3
        assert c.num_trials == 1
        assert c.metrics is None
        assert c.debug is False
        assert c.record_ids is None
        assert c.log_level == "INFO"
        assert c.dry_run is False


class TestDeprecatedEnvVars:
    """Deprecated env vars cause validation errors."""

    @pytest.mark.parametrize(
        "environ, old_var, new_var, value, accessor",
        (
            (
                _BASE_ENV,
                "LLM_MODEL",
                "EVA_MODEL__LLM",
                "test-model",
                lambda c: c.model.llm,
            ),
            (
                _BASE_ENV,
                "STT_MODEL",
                "EVA_MODEL__STT",
                "test-model",
                lambda c: c.model.stt,
            ),
            (
                _BASE_ENV,
                "TTS_MODEL",
                "EVA_MODEL__TTS",
                "test-model",
                lambda c: c.model.tts,
            ),
            (
                _EVA_MODEL_LIST_ENV,
                "REALTIME_MODEL",
                "EVA_MODEL__S2S",
                "test-model",
                lambda c: c.model.s2s,
            ),
            (
                _EVA_MODEL_LIST_ENV,
                "EVA_MODEL__REALTIME_MODEL",
                "EVA_MODEL__S2S",
                "test-model",
                lambda c: c.model.s2s,
            ),
            (
                _BASE_ENV,
                "STT_PARAMS",
                "EVA_MODEL__STT_PARAMS",
                {"api_key": "k", "model": "nova-2", "foo": "bar"},
                lambda c: c.model.stt_params,
            ),
            (
                _BASE_ENV,
                "TTS_PARAMS",
                "EVA_MODEL__TTS_PARAMS",
                {"api_key": "k", "model": "sonic", "foo": "bar"},
                lambda c: c.model.tts_params,
            ),
            (
                _EVA_MODEL_LIST_ENV | {"EVA_MODEL__S2S": "test-model"},
                "REALTIME_MODEL_PARAMS",
                "EVA_MODEL__S2S_PARAMS",
                {"foo": "bar"},
                lambda c: c.model.s2s_params,
            ),
            (
                _EVA_MODEL_LIST_ENV | {"EVA_MODEL__S2S": "test-model"},
                "EVA_MODEL__REALTIME_MODEL_PARAMS",
                "EVA_MODEL__S2S_PARAMS",
                {"foo": "bar"},
                lambda c: c.model.s2s_params,
            ),
            (
                _BASE_ENV,
                "EVA_MAX_CONCURRENT",
                "EVA_MAX_CONCURRENT_CONVERSATIONS",
                1,
                lambda c: c.max_concurrent_conversations,
            ),
            (
                _BASE_ENV,
                "EVA_CONVERSATION_TIMEOUT",
                "EVA_CONVERSATION_TIMEOUT_SECONDS",
                360,
                lambda c: c.conversation_timeout_seconds,
            ),
        ),
    )
    @pytest.mark.parametrize("vars_location", ("env_vars", "env_file_vars"))
    def test_old_env_var_errors(
        self, tmp_path: Path, vars_location: str, environ: dict, old_var: str, new_var: str, value: str, accessor
    ):
        """Using an old env var raises a ValueError."""
        env_value = json.dumps(value) if isinstance(value, dict) else str(value)

        with pytest.raises(ValueError, match=f"{old_var} -> use {new_var}"):
            _config(env_file=tmp_path / ".env", **{vars_location: environ | {old_var: env_value}})

        config = _config(env_file=tmp_path / ".env", **{vars_location: environ | {new_var: env_value}})
        assert accessor(config) == value


class TestExpandMetricsAll:
    """Tests for _expand_metrics_all validator that expands 'all' to non-validation metrics."""

    def test_all_excludes_validation_metrics(self):
        """'all' expands to all registered metrics minus validation metrics."""
        all_metrics = [
            "task_completion",
            "conciseness",
            "conversation_finished",
            "user_behavioral_fidelity",
            "user_speech_fidelity",
            "stt_wer",
            "response_speed",
        ]

        mock_registry = MagicMock()
        mock_registry.list_metrics.return_value = all_metrics

        with patch("eva.metrics.registry._global_registry", mock_registry):
            c = _config(env_vars=_BASE_ENV | {"EVA_METRICS": "all"})

        assert set(c.metrics) == {"task_completion", "conciseness", "stt_wer", "response_speed"}

    def test_all_case_insensitive(self):
        """'ALL' and 'All' also expand."""
        mock_registry = MagicMock()
        mock_registry.list_metrics.return_value = ["stt_wer"]

        with patch("eva.metrics.registry._global_registry", mock_registry):
            c = _config(env_vars=_BASE_ENV | {"EVA_METRICS": "ALL"})

        assert c.metrics == ["stt_wer"]

    def test_explicit_names_not_expanded(self):
        """Comma-separated metric names pass through without registry lookup."""
        c = _config(env_vars=_BASE_ENV | {"EVA_METRICS": "task_completion, conciseness"})
        assert c.metrics == ["task_completion", "conciseness"]


class TestCommaSeparatedFields:
    """Comma-separated env vars are parsed into lists."""

    def test_metrics_parsed(self):
        c = _config(env_vars=_BASE_ENV | {"EVA_METRICS": "task_completion_judge,stt_wer, response_speed"})
        assert c.metrics == ["task_completion_judge", "stt_wer", "response_speed"]

    def test_record_ids_parsed(self):
        c = _config(env_vars=_BASE_ENV | {"EVA_RECORD_IDS": "1.2.1, 1.2.2, 1.3.1"})
        assert c.record_ids == ["1.2.1", "1.2.2", "1.3.1"]

    def test_empty_string_becomes_none(self):
        c = _config(env_vars=_BASE_ENV | {"EVA_METRICS": ""})
        assert c.metrics is None

    def test_whitespace_only_becomes_none(self):
        c = _config(env_vars=_BASE_ENV | {"EVA_RECORD_IDS": " , , "})
        assert c.record_ids is None


class TestDomainResolution:
    """EVA_DOMAIN derives default paths."""

    def test_domain_sets_paths(self):
        c = _config(env_vars=_BASE_ENV | {"EVA_DOMAIN": "airline"})
        assert c.dataset_path == Path("data/airline_dataset.jsonl")
        assert c.agent_config_path == Path("configs/agents/airline_agent.yaml")
        assert c.tool_mocks_path == Path("data/airline_scenarios")


class TestExecutionSettings:
    """EVA_ prefixed execution settings."""

    def test_max_concurrent_conversations(self):
        c = _config(env_vars=_BASE_ENV | {"EVA_MAX_CONCURRENT_CONVERSATIONS": "20"})
        assert c.max_concurrent_conversations == 20

    def test_conversation_timeout_seconds(self):
        c = _config(env_vars=_BASE_ENV | {"EVA_CONVERSATION_TIMEOUT_SECONDS": "600"})
        assert c.conversation_timeout_seconds == 600

    def test_base_port(self):
        c = _config(env_vars=_BASE_ENV | {"EVA_BASE_PORT": "8000"})
        assert c.base_port == 8000

    def test_debug(self):
        c = _config(env_vars=_BASE_ENV | {"EVA_DEBUG": "true"})
        assert c.debug is True

    def test_dry_run(self):
        c = _config(env_vars=_BASE_ENV | {"EVA_DRY_RUN": "true"})
        assert c.dry_run is True

    def test_max_rerun_attempts(self):
        c = _config(env_vars=_BASE_ENV | {"EVA_MAX_RERUN_ATTEMPTS": "5"})
        assert c.max_rerun_attempts == 5

    def test_num_trials(self):
        c = _config(env_vars=_BASE_ENV | {"EVA_NUM_TRIALS": "10"})
        assert c.num_trials == 10

    def test_validation_thresholds(self):
        thresholds = {"conversation_finished": 0.9, "user_behavioral_fidelity": 0.8}
        c = _config(env_vars=_BASE_ENV | {"EVA_VALIDATION_THRESHOLDS": json.dumps(thresholds)})
        assert c.validation_thresholds == thresholds

    def test_stt_params(self):
        params = {"api_key": "k", "model": "nova-2", "language": "en", "punctuate": True}
        c = _config(env_vars=_BASE_ENV | {"EVA_MODEL__STT_PARAMS": json.dumps(params)})
        assert c.model.stt_params == params

    def test_tts_params(self):
        params = {"api_key": "k", "model": "sonic", "voice": "alloy", "speed": 1.2}
        c = _config(env_vars=_BASE_ENV | {"EVA_MODEL__TTS_PARAMS": json.dumps(params)})
        assert c.model.tts_params == params


class TestCliArgs:
    """Backward-compat: every old argparse flag from run_benchmark.py and run_benchmark_with_reruns.py still works."""

    # ── run_benchmark.py old flags ───────────────────────────────

    def test_output_short_flag(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["-o", "/tmp/out"])
        assert c.output_dir == Path("/tmp/out")

    def test_output_long_flag(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--output", "/tmp/out"])
        assert c.output_dir == Path("/tmp/out")

    def test_max_concurrent_short_flag(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["-n", "20"])
        assert c.max_concurrent_conversations == 20

    def test_max_concurrent_long_flag(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--max-concurrent", "20"])
        assert c.max_concurrent_conversations == 20

    def test_timeout(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--timeout", "600"])
        assert c.conversation_timeout_seconds == 600

    def test_llm_model(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--llm-model", "gpt-4o"])
        assert c.model.llm == "gpt-4o"

    def test_stt_model(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--stt-model", "deepgram"])
        assert c.model.stt == "deepgram"

    def test_tts_model(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--tts-model", "cartesia"])
        assert c.model.tts == "cartesia"

    def test_realtime_model(self):
        config = _config(env_vars=_EVA_MODEL_LIST_ENV, cli_args=["--realtime-model", "test-model"])
        assert config.model.s2s == "test-model"

    def test_run_id(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--run-id", "my-run"])
        assert c.run_id == "my-run"

    def test_log_level(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--log-level", "DEBUG"])
        assert c.log_level == "DEBUG"

    def test_dry_run(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--dry-run"])
        assert c.dry_run is True

    def test_debug(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--debug"])
        assert c.debug is True

    def test_ids(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--ids", "1.2.1,1.2.2"])
        assert c.record_ids == ["1.2.1", "1.2.2"]

    def test_num_trials(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--num-trials", "3"])
        assert c.num_trials == 3

    # ── run_benchmark_with_reruns.py old flags ───────────────────

    def test_max_reruns(self):
        c = _config(env_vars=_BASE_ENV, cli_args=["--max-reruns", "5"])
        assert c.max_rerun_attempts == 5

    # ── priority and composition ─────────────────────────────────

    def test_cli_overrides_env(self):
        """CLI args take priority over environment variables."""
        c = _config(env_vars=_BASE_ENV, cli_args=["--llm-model", "gpt-4o"])
        assert c.model.llm == "gpt-4o"

    def test_multiple_old_flags_together(self):
        """Combination of old flags from run_benchmark.py works."""
        c = _config(
            env_vars=_BASE_ENV,
            cli_args=[
                "-o",
                "out",
                "-n",
                "10",
                "--llm-model",
                "gpt-4o",
                "--timeout",
                "600",
                "--debug",
                "--ids",
                "1.2.1",
            ],
        )
        assert c.output_dir == Path("out")
        assert c.max_concurrent_conversations == 10
        assert c.model.llm == "gpt-4o"
        assert c.conversation_timeout_seconds == 600
        assert c.debug is True
        assert c.record_ids == ["1.2.1"]


class TestSpeechToSpeechConfig:
    """Tests for SpeechToSpeechConfig discriminated union."""

    def test_s2s_config_from_env(self):
        """EVA_MODEL__S2S selects SpeechToSpeechConfig."""
        config = _config(env_vars=_EVA_MODEL_LIST_ENV | {"EVA_MODEL__S2S": "gpt-realtime-mini"})
        assert isinstance(config.model, SpeechToSpeechConfig)
        assert config.model.s2s == "gpt-realtime-mini"

    def test_s2s_config_from_cli(self):
        """--s2s-model selects SpeechToSpeechConfig."""
        config = _config(env_vars=_EVA_MODEL_LIST_ENV, cli_args=["--model.s2s", "gemini_live"])
        assert isinstance(config.model, SpeechToSpeechConfig)
        assert config.model.s2s == "gemini_live"

    def test_s2s_config_with_params(self):
        """S2S params are passed through."""
        config = _config(
            env_vars=_EVA_MODEL_LIST_ENV, model={"s2s": "gpt-realtime-mini", "s2s_params": {"voice": "alloy"}}
        )
        assert isinstance(config.model, SpeechToSpeechConfig)
        assert config.model.s2s_params == {"voice": "alloy"}
