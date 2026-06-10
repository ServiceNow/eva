"""End-to-end check.

A config built via the editor's serializer must construct a valid RunConfig
for each pipeline mode.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from apps.config_io import compute_disabled, parse_env_example, serialize_env
from eva.models.config import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

_MODEL_LIST = [
    {
        "model_name": "gpt-5.2",
        "litellm_params": {"model": "openai/gpt-5.2", "api_key": "sk-test", "max_parallel_requests": 5},
        "model_info": {"base_model": "gpt-5.2"},
    },
    {
        "model_name": "gemini-3.1-pro-preview",
        "litellm_params": {
            "model": "vertex_ai/gemini-3.1-pro-preview",
            "vertex_project": "p",
            "vertex_location": "global",
            "vertex_credentials": "/tmp/x.json",
            "max_parallel_requests": 5,
        },
    },
    {
        "model_name": "us.anthropic.claude-opus-4-6",
        "litellm_params": {
            "model": "bedrock/us.anthropic.claude-opus-4-6-v1",
            "aws_access_key_id": "k",
            "aws_secret_access_key": "s",
            "max_parallel_requests": 5,
        },
    },
]


def _serialize(values: dict, parsed, pipeline_mode: str = "LLM", perturbation_mode: str = "None") -> str:
    disabled = compute_disabled(parsed, pipeline_mode=pipeline_mode, perturbation_mode=perturbation_mode)
    return serialize_env(values, parsed, disabled=disabled)


def _load_isolated(env_file: Path) -> RunConfig:
    with patch.dict(os.environ, {"PATH": os.environ["PATH"]}, clear=True):
        return RunConfig(_env_file=env_file, _cli_parse_args=False)


def test_llm_pipeline_serialization_constructs_runconfig(tmp_path: Path) -> None:
    parsed = parse_env_example(REPO_ROOT / ".env.example")
    values = {
        "EVA_MODEL_LIST": _MODEL_LIST,
        "EVA_MODEL__LLM": "gpt-5.2",
        "EVA_MODEL__STT": "deepgram",
        "EVA_MODEL__TTS": "cartesia",
        "EVA_MODEL__STT_PARAMS": {"api_key": "k", "model": "nova-2"},
        "EVA_MODEL__TTS_PARAMS": {"api_key": "k", "model": "sonic"},
        "EVA_DOMAIN": "airline",
    }
    env_file = tmp_path / ".env"
    env_file.write_text(_serialize(values, parsed))
    config = _load_isolated(env_file)
    assert config.model.llm == "gpt-5.2"
    assert config.model.stt == "deepgram"
    assert config.model.tts == "cartesia"
    assert config.domain == "airline"


def test_s2s_pipeline_serialization_constructs_runconfig(tmp_path: Path) -> None:
    parsed = parse_env_example(REPO_ROOT / ".env.example")
    values = {
        "EVA_MODEL_LIST": _MODEL_LIST,
        "EVA_MODEL__S2S": "gpt-realtime-mini",
        "EVA_MODEL__S2S_PARAMS": {"api_key": "k", "model": "gpt-realtime-mini"},
        "EVA_DOMAIN": "airline",
    }
    env_file = tmp_path / ".env"
    env_file.write_text(_serialize(values, parsed, pipeline_mode="S2S"))
    config = _load_isolated(env_file)
    assert config.model.s2s == "gpt-realtime-mini"


def test_perturbation_accent_serialization_constructs_runconfig(tmp_path: Path) -> None:
    parsed = parse_env_example(REPO_ROOT / ".env.example")
    values = {
        "EVA_MODEL_LIST": _MODEL_LIST,
        "EVA_MODEL__LLM": "gpt-5.2",
        "EVA_MODEL__STT": "deepgram",
        "EVA_MODEL__TTS": "cartesia",
        "EVA_MODEL__STT_PARAMS": {"api_key": "k", "model": "nova-2"},
        "EVA_MODEL__TTS_PARAMS": {"api_key": "k", "model": "sonic"},
        "EVA_DOMAIN": "airline",
        "EVA_PERTURBATION__ACCENT": "french",
    }
    env_file = tmp_path / ".env"
    env_file.write_text(_serialize(values, parsed, perturbation_mode="Accent"))
    config = _load_isolated(env_file)
    assert config.perturbation is not None
    assert config.perturbation.accent == "french"
    assert config.perturbation.behavior is None
