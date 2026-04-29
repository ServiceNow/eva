"""Unit tests for RunConfig model."""

import json
import os
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
_S2S_ENV = _EVA_MODEL_LIST_ENV | {
    "EVA_MODEL__S2S": "gpt-realtime-mini",
    "EVA_MODEL__S2S_PARAMS": json.dumps({"api_key": "", "model": "test"}),
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

    with patch.dict(os.environ, {"PATH": os.environ["PATH"]} | (env_vars or {}), clear=True):
        return RunConfig(_env_file=env_file, _cli_parse_args=cli_args, **kwargs)


def _load_json_into_runconfig(json_str: str) -> RunConfig:
    """Load RunConfig from JSON with isolated environment (no real env vars)."""
    with patch.dict(os.environ, {"PATH": os.environ["PATH"]}, clear=True):
        return RunConfig.model_validate_json(json_str)


class TestRunConfig:
    def test_create_minimal_config(self):
        """Test creating a minimal RunConfig."""
        config = _config(env_vars=_BASE_ENV | {"EVA_DOMAIN": "airline", "EVA_MODEL__LLM": "gpt-5.2"})

        assert config.dataset_path == Path("data/airline_dataset.jsonl")
        assert config.tool_mocks_path == Path("data/airline_scenarios")
        # run_id = timestamp + model suffix (e.g. "2024-01-15_14-30-45.123456_nova-2_gpt-5.2_sonic")
        assert config.run_id.endswith("nova-2_gpt-5.2_sonic")
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
        """Secrets are redacted in model_list and STT/TTS params."""
        config = _config(env_vars=_BASE_ENV)
        dumped = config.model_dump(mode="json")
        assert dumped["model_list"][0]["litellm_params"]["api_key"] == "***"
        assert dumped["model_list"][1]["litellm_params"]["vertex_credentials"] == "***"
        assert dumped["model_list"][2]["litellm_params"]["aws_access_key_id"] == "***"
        assert dumped["model_list"][2]["litellm_params"]["aws_secret_access_key"] == "***"
        # STT/TTS params api_key must also be redacted
        assert dumped["model"]["stt_params"]["api_key"] == "***"
        assert dumped["model"]["tts_params"]["api_key"] == "***"
        # Non-secret fields preserved
        assert dumped["model"]["stt_params"]["model"] == "nova-2"
        assert dumped["model"]["tts_params"]["model"] == "sonic"

    def test_secrets_redaction_does_not_mutate_live_config(self):
        """Serializing must not corrupt the in-memory config objects."""
        config = _config(env_vars=_BASE_ENV)
        config.model_dump(mode="json")
        # model_list keys must still hold real values
        assert config.model_list[0]["litellm_params"]["api_key"] == "must_be_redacted"
        assert config.model_list[1]["litellm_params"]["vertex_credentials"] == "must_be_redacted"
        # STT/TTS params must still hold real values
        assert config.model.stt_params["api_key"] == "test_key"
        assert config.model.tts_params["api_key"] == "test_key"

    def test_apply_env_overrides(self):
        """Redacted secrets are restored from a live config for both model and model_list."""
        config = _config(env_vars=_BASE_ENV)
        dumped_json = config.model_dump_json()
        loaded = _load_json_into_runconfig(dumped_json)

        # Everything is redacted after round-trip
        assert loaded.model.stt_params["api_key"] == "***"
        assert loaded.model.tts_params["api_key"] == "***"
        assert loaded.model_list[0]["litellm_params"]["api_key"] == "***"
        assert loaded.model_list[1]["litellm_params"]["vertex_credentials"] == "***"
        assert loaded.model_list[2]["litellm_params"]["aws_access_key_id"] == "***"

        loaded.apply_env_overrides(config)

        # STT/TTS params restored
        assert loaded.model.stt_params["api_key"] == "test_key"
        assert loaded.model.tts_params["api_key"] == "test_key"
        assert loaded.model.stt_params["model"] == "nova-2"
        # model_list restored
        assert loaded.model_list[0]["litellm_params"]["api_key"] == "must_be_redacted"
        assert loaded.model_list[1]["litellm_params"]["vertex_credentials"] == "must_be_redacted"
        assert loaded.model_list[2]["litellm_params"]["aws_access_key_id"] == "must_be_redacted"
        assert loaded.model_list[2]["litellm_params"]["aws_secret_access_key"] == "must_be_redacted"

    def test_apply_env_overrides_provider_mismatch(self, caplog):
        """Restoring secrets warns (but succeeds) if the STT/TTS provider changed."""
        config = _config(env_vars=_BASE_ENV)
        dumped_json = config.model_dump_json()
        loaded = _load_json_into_runconfig(dumped_json)

        live = _config(
            env_vars=_BASE_ENV
            | {
                "EVA_MODEL__STT": "openai_whisper",
                "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "k", "model": "whisper-1"}),
            }
        )
        with caplog.at_level("WARNING", logger="eva.models.config"):
            loaded.apply_env_overrides(live)
        assert "Provider mismatch for stt_params" in caplog.text
        assert "deepgram" in caplog.text
        assert "openai_whisper" in caplog.text

    def test_apply_env_overrides_alias_mismatch(self):
        """Restoring secrets fails if the alias changed."""
        config = _config(
            env_vars=_BASE_ENV
            | {
                "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "k", "model": "nova-2", "alias": "stt-v1"}),
            }
        )
        dumped_json = config.model_dump_json()
        loaded = _load_json_into_runconfig(dumped_json)

        live = _config(
            env_vars=_BASE_ENV
            | {
                "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "k", "model": "nova-2", "alias": "stt-v2"}),
            }
        )
        with pytest.raises(
            ValueError,
            match=r"saved stt_params\[alias\]='stt-v1'.*current environment has stt_params\[alias\]='stt-v2'",
        ):
            loaded.apply_env_overrides(live)

    def test_apply_env_overrides_model_mismatch_warns(self, caplog):
        """Restoring secrets warns (but succeeds) if the STT/TTS model changed."""
        config = _config(env_vars=_BASE_ENV)
        dumped_json = config.model_dump_json()
        loaded = _load_json_into_runconfig(dumped_json)

        live = _config(env_vars=_BASE_ENV | {"EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "k", "model": "sonic-2"})})
        with caplog.at_level("WARNING", logger="eva.models.config"):
            loaded.apply_env_overrides(live)
        assert "sonic" in caplog.text
        assert "sonic-2" in caplog.text
        assert loaded.model.tts_params["api_key"] == "k"

    def test_apply_env_overrides_url_from_env(self, caplog):
        """Url is always taken from the live env, with a warning if it differs."""
        saved_env = _BASE_ENV | {
            "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "k", "model": "nova-2", "url": "wss://old-host/stt"}),
        }
        config = _config(env_vars=saved_env)
        dumped_json = config.model_dump_json()
        loaded = _load_json_into_runconfig(dumped_json)

        # Live env has a different url
        live_env = _BASE_ENV | {
            "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "k", "model": "nova-2", "url": "wss://new-host/stt"}),
        }
        live = _config(env_vars=live_env)

        with caplog.at_level("WARNING", logger="eva.models.config"):
            loaded.apply_env_overrides(live)

        assert loaded.model.stt_params["url"] == "wss://new-host/stt"
        assert "wss://old-host/stt" in caplog.text
        assert "wss://new-host/stt" in caplog.text

    def test_apply_env_overrides_url_added_from_env(self):
        """Url from live env is added even if the saved config didn't have one."""
        config = _config(env_vars=_BASE_ENV)
        dumped_json = config.model_dump_json()
        loaded = _load_json_into_runconfig(dumped_json)

        live_env = _BASE_ENV | {
            "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "k", "model": "nova-2", "url": "wss://new-host/stt"}),
        }
        live = _config(env_vars=live_env)
        loaded.apply_env_overrides(live)

        assert loaded.model.stt_params["url"] == "wss://new-host/stt"

    def test_apply_env_overrides_llm_deployment_mismatch(self):
        """Restoring secrets fails if the active LLM deployment is missing from the live model_list."""
        config = _config(env_vars=_BASE_ENV)
        dumped_json = config.model_dump_json()
        loaded = _load_json_into_runconfig(dumped_json)

        # Live config has a different model_list (only one deployment, different name)
        different_model_list = [
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "real_key"},
            }
        ]
        live = _config(
            env_vars={
                "EVA_MODEL_LIST": json.dumps(different_model_list),
                "EVA_MODEL__LLM": "gpt-4o",
                "EVA_MODEL__STT": "deepgram",
                "EVA_MODEL__TTS": "cartesia",
                "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "k", "model": "nova-2"}),
                "EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "k", "model": "sonic"}),
            }
        )
        with pytest.raises(ValueError, match=r"deployment 'gpt-5.2' not found in current EVA_MODEL_LIST"):
            loaded.apply_env_overrides(live)

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
        assert isinstance(c.metrics, list)
        assert len(c.metrics) > 0
        assert c.debug is False
        assert c.record_ids is None
        assert c.log_level == "INFO"
        assert c.dry_run is False


class TestDeprecatedEnvVars:
    """Deprecated env vars cause validation errors."""


class TestExpandMetricsAll:
    """Tests for _expand_metrics_all validator that expands 'all' to non-validation metrics."""

    @pytest.mark.parametrize(
        "env_vars, cli_args",
        (
            pytest.param({"EVA_METRICS": "all"}, [], id="EVA_METRICS=all"),
            pytest.param({"EVA_METRICS": "ALL"}, [], id="EVA_METRICS=ALL"),
            pytest.param({"EVA_METRICS": "All"}, [], id="EVA_METRICS=All"),
            pytest.param({}, ["--metrics", "all"], id="--metrics=all"),
            pytest.param({}, ["--metrics", "ALL"], id="--metrics=ALL"),
            pytest.param({}, ["--metrics", "All"], id="--metrics=All"),
            pytest.param({}, [], id="default"),
        ),
    )
    def test_all_excludes_validation_metrics(self, env_vars, cli_args):
        """'all' expands to all registered metrics minus validation metrics."""
        all_metrics = [
            "task_completion",
            "conciseness",
            "conversation_valid_end",
            "user_behavioral_fidelity",
            "user_speech_fidelity",
            "stt_wer",
            "response_speed",
        ]

        mock_registry = MagicMock()
        mock_registry.list_metrics.return_value = all_metrics

        with patch("eva.metrics.registry._global_registry", mock_registry):
            c = _config(env_vars=_BASE_ENV | env_vars, cli_args=cli_args)

        assert set(c.metrics) == {"task_completion", "conciseness", "stt_wer", "response_speed"}

    def test_explicit_names_not_expanded(self):
        """Comma-separated metric names pass through without registry lookup."""
        c = _config(env_vars=_BASE_ENV | {"EVA_METRICS": "task_completion, conciseness"})
        assert c.metrics == ["task_completion", "conciseness"]


class TestCommaSeparatedFields:
    """Comma-separated env vars are parsed into lists."""

    @pytest.mark.parametrize(
        "env_vars, cli_args",
        (
            pytest.param({"EVA_METRICS": "task_completion_judge,stt_wer, response_speed"}, [], id="EVA_METRICS"),
            pytest.param({}, ["--metrics", "task_completion_judge,stt_wer, response_speed"], id="single --metrics"),
            pytest.param(
                {},
                ["--metrics", "task_completion_judge", "--metrics", "stt_wer", "--metrics", "response_speed"],
                id="multiple --metrics",
            ),
            pytest.param(
                {},
                ["--metrics", "task_completion_judge", "--metrics", "stt_wer, response_speed"],
                id="mixed --metrics",
            ),
        ),
    )
    def test_metrics_parsed(self, env_vars, cli_args):
        c = _config(env_vars=_BASE_ENV | env_vars, cli_args=cli_args)
        assert c.metrics == ["task_completion_judge", "stt_wer", "response_speed"]

    @pytest.mark.parametrize(
        "env_vars, cli_args",
        (
            pytest.param({"EVA_RECORD_IDS": "1.2.1, 1.2.2, 1.3.1"}, [], id="EVA_RECORD_IDS"),
            pytest.param({}, ["--record-ids", "1.2.1, 1.2.2, 1.3.1"], id="single --record-ids"),
            pytest.param(
                {},
                ["--record-ids", "1.2.1", "--record-ids", "1.2.2", "--record-ids", "1.3.1"],
                id="multiple --record-ids",
            ),
            pytest.param({}, ["--record-ids", "1.2.1", "--record-ids", "1.2.2, 1.3.1"], id="mixed --record-ids"),
        ),
    )
    def test_record_ids_parsed(self, env_vars, cli_args):
        c = _config(env_vars=_BASE_ENV | env_vars, cli_args=cli_args)
        assert c.record_ids == ["1.2.1", "1.2.2", "1.3.1"]

    @pytest.mark.parametrize(
        "env_vars, cli_args",
        (
            pytest.param({"EVA_METRICS": ""}, [], id="EVA_METRICS"),
            pytest.param({"EVA_METRICS": "[]"}, [], id="EVA_METRICS"),
            pytest.param({}, ["--metrics", ""], id='--metrics ""'),
            pytest.param({}, ["--metrics="], id="--metrics="),
            pytest.param({}, ["--metrics", "[]"], id="--metrics []"),
        ),
    )
    def test_empty_string_becomes_none(self, env_vars, cli_args):
        c = _config(env_vars=_BASE_ENV | env_vars, cli_args=cli_args)
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
        thresholds = {"conversation_valid_end": 0.9, "user_behavioral_fidelity": 0.8}
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


class TestTurnStrategyConfig:
    """Tests for configurable turn start/stop strategy fields."""

    def test_pipeline_config_turn_strategy_defaults(self):
        """PipelineConfig has expected defaults for turn strategy fields."""
        config = _config(env_vars=_BASE_ENV)
        assert config.model.turn_start_strategy == "vad"
        assert config.model.turn_start_strategy_params == {}
        assert config.model.turn_stop_strategy == "turn_analyzer"
        assert config.model.turn_stop_strategy_params == {}
        assert config.model.vad == "silero"
        assert config.model.vad_params == {}

    def test_pipeline_config_turn_start_strategy_from_env(self):
        """EVA_MODEL__TURN_START_STRATEGY sets turn_start_strategy."""
        config = _config(env_vars=_BASE_ENV | {"EVA_MODEL__TURN_START_STRATEGY": "external"})
        assert config.model.turn_start_strategy == "external"

    def test_pipeline_config_turn_stop_strategy_from_env(self):
        """EVA_MODEL__TURN_STOP_STRATEGY sets turn_stop_strategy."""
        config = _config(env_vars=_BASE_ENV | {"EVA_MODEL__TURN_STOP_STRATEGY": "speech_timeout"})
        assert config.model.turn_stop_strategy == "speech_timeout"

    def test_pipeline_config_turn_start_strategy_params_from_env(self):
        """EVA_MODEL__TURN_START_STRATEGY_PARAMS sets turn_start_strategy_params."""
        params = {"some_param": True}
        config = _config(env_vars=_BASE_ENV | {"EVA_MODEL__TURN_START_STRATEGY_PARAMS": json.dumps(params)})
        assert config.model.turn_start_strategy_params == params

    def test_pipeline_config_turn_stop_strategy_params_from_env(self):
        """EVA_MODEL__TURN_STOP_STRATEGY_PARAMS sets turn_stop_strategy_params."""
        params = {"user_speech_timeout": 1.5}
        config = _config(env_vars=_BASE_ENV | {"EVA_MODEL__TURN_STOP_STRATEGY_PARAMS": json.dumps(params)})
        assert config.model.turn_stop_strategy_params == params

    def test_pipeline_config_vad_from_env(self):
        """EVA_MODEL__VAD sets vad."""
        config = _config(env_vars=_BASE_ENV | {"EVA_MODEL__VAD": "silero"})
        assert config.model.vad == "silero"

    def test_pipeline_config_vad_params_from_env(self):
        """EVA_MODEL__VAD_PARAMS sets vad_params."""
        params = {"stop_secs": 0.5, "confidence": 0.8}
        config = _config(env_vars=_BASE_ENV | {"EVA_MODEL__VAD_PARAMS": json.dumps(params)})
        assert config.model.vad_params == params

    def test_s2s_config_turn_strategy_defaults(self):
        """SpeechToSpeechConfig has expected defaults for turn strategy fields."""
        config = _config(env_vars=_S2S_ENV)
        assert config.model.turn_start_strategy == "vad"
        assert config.model.turn_start_strategy_params == {}
        assert config.model.turn_stop_strategy == "turn_analyzer"
        assert config.model.turn_stop_strategy_params == {}
        assert config.model.vad == "silero"
        assert config.model.vad_params == {}

    def test_s2s_config_turn_strategy_from_env(self):
        """S2S turn strategies can be overridden via env."""
        config = _config(
            env_vars=_S2S_ENV
            | {
                "EVA_MODEL__TURN_START_STRATEGY": "transcription",
                "EVA_MODEL__TURN_STOP_STRATEGY": "external",
            }
        )
        assert config.model.turn_start_strategy == "transcription"
        assert config.model.turn_stop_strategy == "external"

    def test_audio_llm_config_turn_strategy_defaults(self):
        """AudioLLMConfig has expected defaults for turn strategy fields."""
        config = _config(
            env_vars=_EVA_MODEL_LIST_ENV
            | {
                "EVA_MODEL__AUDIO_LLM": "vllm",
                "EVA_MODEL__AUDIO_LLM_PARAMS": json.dumps(
                    {"api_key": "k", "model": "ultravox", "base_url": "http://localhost:8000"}
                ),
                "EVA_MODEL__TTS": "cartesia",
                "EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "k", "model": "sonic"}),
            }
        )
        assert config.model.turn_start_strategy == "vad"
        assert config.model.turn_stop_strategy == "turn_analyzer"
        assert config.model.vad == "silero"
        assert config.model.vad_params == {}


class TestApiKeyRedactionInPipelineModels:
    """api_key redaction works for all three pipeline config types via RunConfig serialization."""

    def test_pipeline_config_stt_tts_params_api_key_redacted(self):
        """RunConfig redacts api_key in stt_params and tts_params on serialization."""
        config = _config(env_vars=_BASE_ENV)
        dumped = config.model_dump(mode="json")["model"]
        assert dumped["stt_params"]["api_key"] == "***"
        assert dumped["tts_params"]["api_key"] == "***"
        # Non-secret fields survive
        assert dumped["stt_params"]["model"] == "nova-2"
        assert dumped["tts_params"]["model"] == "sonic"

    def test_pipeline_config_redaction_does_not_mutate(self):
        """Serializing RunConfig does not mutate live stt_params/tts_params."""
        config = _config(env_vars=_BASE_ENV)
        config.model_dump(mode="json")
        assert config.model.stt_params["api_key"] == "test_key"
        assert config.model.tts_params["api_key"] == "test_key"

    def test_s2s_config_s2s_params_api_key_redacted(self):
        """RunConfig redacts api_key in s2s_params on serialization."""
        config = _config(
            env_vars=_EVA_MODEL_LIST_ENV
            | {
                "EVA_MODEL__S2S": "gpt-realtime-mini",
                "EVA_MODEL__S2S_PARAMS": json.dumps({"api_key": "secret", "model": "gpt-realtime-mini"}),
            }
        )
        dumped = config.model_dump(mode="json")["model"]
        assert dumped["s2s_params"]["api_key"] == "***"
        # Non-secret fields survive
        assert dumped["s2s_params"]["model"] == "gpt-realtime-mini"

    def test_s2s_config_redaction_does_not_mutate(self):
        """Serializing RunConfig does not mutate live s2s_params."""
        config = _config(
            env_vars=_EVA_MODEL_LIST_ENV
            | {
                "EVA_MODEL__S2S": "gpt-realtime-mini",
                "EVA_MODEL__S2S_PARAMS": json.dumps({"api_key": "secret", "model": "gpt-realtime-mini"}),
            }
        )
        config.model_dump(mode="json")
        assert config.model.s2s_params["api_key"] == "secret"

    def test_audio_llm_config_params_api_key_redacted(self):
        """RunConfig redacts api_key in audio_llm_params and tts_params on serialization."""
        config = _config(
            env_vars=_EVA_MODEL_LIST_ENV
            | {
                "EVA_MODEL__AUDIO_LLM": "vllm",
                "EVA_MODEL__AUDIO_LLM_PARAMS": json.dumps(
                    {"api_key": "secret", "base_url": "http://localhost:8000", "model": "ultravox"}
                ),
                "EVA_MODEL__TTS": "cartesia",
                "EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "tts_secret", "model": "sonic"}),
            }
        )
        dumped = config.model_dump(mode="json")["model"]
        assert dumped["audio_llm_params"]["api_key"] == "***"
        assert dumped["tts_params"]["api_key"] == "***"
        # Non-secret fields survive
        assert dumped["audio_llm_params"]["base_url"] == "http://localhost:8000"
        assert dumped["tts_params"]["model"] == "sonic"

    def test_non_secret_params_not_affected_by_redaction(self):
        """Non-api_key fields in params pass through serialization unchanged."""
        config = _config(
            env_vars=_BASE_ENV
            | {
                "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "k", "model": "nova-2", "language": "en"}),
                "EVA_MODEL__TTS_PARAMS": json.dumps({"api_key": "k", "model": "sonic", "speed": 1.0}),
            }
        )
        dumped = config.model_dump(mode="json")["model"]
        # api_key is redacted
        assert dumped["stt_params"]["api_key"] == "***"
        assert dumped["tts_params"]["api_key"] == "***"
        # Extra non-secret fields survive unchanged
        assert dumped["stt_params"]["language"] == "en"
        assert dumped["tts_params"]["speed"] == 1.0


class TestParamAlias:
    """Tests for the _param_alias helper used to build run_id suffixes."""

    def test_alias_takes_priority_over_model(self):
        """When alias is present it is returned."""
        config = _config(
            env_vars=_BASE_ENV
            | {
                "EVA_MODEL__STT_PARAMS": json.dumps({"api_key": "k", "model": "nova-2", "alias": "my-stt"}),
            }
        )
        # run_id suffix uses alias for STT component
        assert "my-stt" in config.run_id

    def test_model_used_when_no_alias(self):
        """When no alias, model is used for the suffix."""
        config = _config(env_vars=_BASE_ENV)
        assert "nova-2" in config.run_id


class TestSpeechToSpeechConfig:
    """Tests for SpeechToSpeechConfig discriminated union."""

    def test_s2s_config_from_env(self):
        """EVA_MODEL__S2S selects SpeechToSpeechConfig."""
        config = _config(
            env_vars=_EVA_MODEL_LIST_ENV
            | {
                "EVA_MODEL__S2S": "gpt-realtime-mini",
                "EVA_MODEL__S2S_PARAMS": json.dumps({"api_key": "", "model": "gpt-realtime-mini"}),
            }
        )
        assert isinstance(config.model, SpeechToSpeechConfig)
        assert config.model.s2s == "gpt-realtime-mini"

    def test_s2s_config_from_cli(self):
        """--s2s-model selects SpeechToSpeechConfig."""
        config = _config(
            env_vars=_EVA_MODEL_LIST_ENV,
            cli_args=[
                "--model.s2s",
                "gemini_live",
                "--model.s2s-params",
                '{"api_key": "test-key", "model": "gemini_live"}',
            ],
        )
        assert isinstance(config.model, SpeechToSpeechConfig)
        assert config.model.s2s == "gemini_live"
        assert config.model.s2s_params == {"api_key": "test-key", "model": "gemini_live"}

    def test_s2s_config_with_params(self):
        """S2S params are passed through."""
        config = _config(
            env_vars=_EVA_MODEL_LIST_ENV,
            model={
                "s2s": "gpt-realtime-mini",
                "s2s_params": {"voice": "alloy", "api_key": "key_1", "model": "gpt-realtime-mini"},
            },
        )
        assert isinstance(config.model, SpeechToSpeechConfig)
        assert config.model.s2s_params == {"voice": "alloy", "api_key": "key_1", "model": "gpt-realtime-mini"}
