"""Unit tests for apps/config_io.py (annotation-aware env parser/serializer)."""

from __future__ import annotations

from pathlib import Path

from apps.config_io import load_env, parse_env_example, serialize_env
from apps.config_schema import GROUP_MISC

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def test_parses_active_var() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    spec = parsed.by_name["ELEVENLABS_API_KEY"]
    assert spec.is_active is True
    assert spec.widget == "secret"
    assert "ElevenLabs" in spec.info


def test_parses_inactive_var() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    spec = parsed.by_name["EVA_DOMAIN"]
    assert spec.is_active is False
    assert spec.widget == "enum"
    assert "airline" in spec.options


def test_parses_enum_options() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    spec = parsed.by_name["EVA_MODEL__STT"]
    assert "deepgram" in spec.options
    assert "cartesia" in spec.options


def test_parses_range() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    spec = parsed.by_name["EVA_MAX_CONCURRENT_CONVERSATIONS"]
    assert spec.range is not None
    assert spec.range[0] == 1.0
    assert spec.range[1] == 100.0


def test_parses_condition() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    spec = parsed.by_name["EVA_MODEL__STT"]
    assert ("pipeline_mode", "LLM") in spec.conditions


def test_parses_multi_condition() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    spec = parsed.by_name["EVA_FRENCH_ACCENT_USER_F"]
    assert ("perturbation_mode", "Accent") in spec.conditions
    assert ("EVA_PERTURBATION__ACCENT", "french") in spec.conditions


def test_group_from_section_header() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    assert parsed.by_name["ELEVENLABS_API_KEY"].group == "API Configs"
    assert parsed.by_name["EVA_MODEL__LLM"].group == "Voice Pipeline"


def test_dedupes_repeated_names() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    occurrences = [v for v in parsed.vars if v.name == "EVA_METRICS"]
    assert len(occurrences) == 1


def test_multiline_deployment_list() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    spec = parsed.by_name["EVA_MODEL_LIST"]
    assert spec.widget == "json_deployment_list"
    assert spec.line_end > spec.line_start


def test_serialize_with_no_values_is_byte_identical_to_example() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    rendered = serialize_env({}, parsed)
    original = ENV_EXAMPLE.read_text()
    if not original.endswith("\n"):
        original += "\n"
    assert rendered == original


def test_serialize_overrides_active_var(tmp_path: Path) -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    rendered = serialize_env({"OPENAI_API_KEY": "sk-test-123"}, parsed)
    assert "OPENAI_API_KEY=sk-test-123" in rendered
    assert "your_openai_api_key_here" not in rendered


def test_serialize_activates_inactive_var() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    rendered = serialize_env({"EVA_DOMAIN": "airline"}, parsed)
    lines = rendered.splitlines()
    assert "EVA_DOMAIN=airline" in lines
    assert "#v EVA_DOMAIN=airline" not in lines


def test_serialize_json_blob_single_quoted() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    deployments = [{"model_name": "x", "litellm_params": {"model": "openai/x"}}]
    rendered = serialize_env({"EVA_MODEL_LIST": deployments}, parsed)
    assert "EVA_MODEL_LIST='" in rendered


def test_serialize_bool_lowercases() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    rendered = serialize_env({"EVA_DEBUG": True}, parsed)
    assert "EVA_DEBUG=true" in rendered


def test_serialize_appends_misc_section_for_unknown_vars() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    rendered = serialize_env({"EVA_TOTALLY_NEW_VAR": "hello"}, parsed)
    assert GROUP_MISC in rendered
    assert "EVA_TOTALLY_NEW_VAR=hello" in rendered


def test_serialize_disabled_var_uses_current_value() -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    rendered = serialize_env(
        {"EVA_MODEL__STT": "deepgram"},
        parsed,
        disabled={"EVA_MODEL__STT"},
    )
    assert "#v EVA_MODEL__STT=deepgram" in rendered


def test_load_env_reads_existing_file(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("FOO=bar\n#v COMMENTED=skipme\nQUOTED='hello world'\nJSON='[{\"a\": 1}]'\n")
    out = load_env(p)
    assert out == {"FOO": "bar", "QUOTED": "hello world", "JSON": '[{"a": 1}]'}


def test_load_env_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_env(tmp_path / "does-not-exist") == {}


def test_round_trip_through_load_env(tmp_path: Path) -> None:
    parsed = parse_env_example(ENV_EXAMPLE)
    written = serialize_env(
        {
            "OPENAI_API_KEY": "sk-abc",
            "EVA_DEBUG": True,
            "EVA_MAX_CONCURRENT_CONVERSATIONS": 8,
        },
        parsed,
    )
    p = tmp_path / ".env"
    p.write_text(written)
    loaded = load_env(p)
    assert loaded["OPENAI_API_KEY"] == "sk-abc"
    assert loaded["EVA_DEBUG"] == "true"
    assert loaded["EVA_MAX_CONCURRENT_CONVERSATIONS"] == "8"
