"""Tests for orchestrator/preflight.py — preflight model probes."""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from eva.models.config import ModelConfig, RunConfig, S2SSimulatorConfig
from eva.orchestrator import preflight
from eva.orchestrator.preflight import (
    PreflightError,
    _run_preflight,
    run_preflight,
)

_MODEL_LIST = [{"model_name": "test-model", "litellm_params": {"model": "test"}}]
# Preserve PATH so pydub's lazy ffmpeg lookup doesn't KeyError under clear=True.
_BASE_ENV = {"EVA_MODEL_LIST": json.dumps(_MODEL_LIST), "PATH": os.environ.get("PATH", "")}


class _GoodTTS(TTSService):
    async def run_tts(self, text, context_id=None) -> AsyncGenerator[Frame | None, None]:
        yield TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=24000, num_channels=1)


class _BadTTS(TTSService):
    async def run_tts(self, text, context_id=None) -> AsyncGenerator[Frame | None, None]:
        await self.push_error(ErrorFrame("401 Unauthorized", fatal=True))
        return
        yield  # pragma: no cover — makes this an async generator


class _GoodSTT(STTService):
    """STT that connects fine and emits nothing (silence in → no transcript)."""

    async def run_stt(self, audio) -> AsyncGenerator[Frame | None, None]:
        yield None


def _cascade_config(tmp_path) -> RunConfig:
    with patch.dict(os.environ, _BASE_ENV, clear=True):
        return RunConfig(
            model=ModelConfig(
                llm="test-model",
                stt="deepgram",
                tts="cartesia",
                stt_params={"api_key": "k", "model": "nova-2", "alias": "deepgram-prod"},
                tts_params={"api_key": "k", "model": "sonic"},
            ),
            output_dir=tmp_path / "out",
            run_id="r",
        )


@pytest.mark.asyncio
async def test_cascade_all_pass(tmp_path):
    cfg = _cascade_config(tmp_path)
    with (
        patch.object(preflight, "create_stt_service", return_value=_GoodSTT()),
        patch.object(preflight, "create_tts_service", return_value=_GoodTTS()),
        patch.object(preflight, "LiteLLMClient") as llm,
    ):
        llm.return_value.complete = AsyncMock(return_value=("pong", {}))
        results = await _run_preflight(cfg)

    assert {r.model_type for r in results} == {"LLM", "STT", "TTS"}
    assert all(r.ok for r in results)
    # Display name uses the params alias when present.
    assert next(r for r in results if r.model_type == "STT").model_alias == "deepgram-prod"


@pytest.mark.asyncio
async def test_llm_failure_is_reported(tmp_path):
    cfg = _cascade_config(tmp_path)
    with (
        patch.object(preflight, "create_stt_service", return_value=_GoodSTT()),
        patch.object(preflight, "create_tts_service", return_value=_GoodTTS()),
        patch.object(preflight, "LiteLLMClient") as llm,
    ):
        llm.return_value.complete = AsyncMock(side_effect=Exception("401 invalid api key"))
        results = await _run_preflight(cfg)

    llm_result = next(r for r in results if r.model_type == "LLM")
    assert not llm_result.ok
    assert "401" in llm_result.detail


@pytest.mark.asyncio
async def test_tts_error_frame_fails(tmp_path):
    cfg = _cascade_config(tmp_path)
    with (
        patch.object(preflight, "create_stt_service", return_value=_GoodSTT()),
        patch.object(preflight, "create_tts_service", return_value=_BadTTS()),
        patch.object(preflight, "LiteLLMClient") as llm,
    ):
        llm.return_value.complete = AsyncMock(return_value=("pong", {}))
        results = await _run_preflight(cfg)

    tts_result = next(r for r in results if r.model_type == "TTS")
    assert not tts_result.ok
    assert "Unauthorized" in tts_result.detail


@pytest.mark.asyncio
async def test_s2s_is_skipped(tmp_path):
    with patch.dict(os.environ, _BASE_ENV, clear=True):
        cfg = RunConfig(
            model=ModelConfig(s2s="gpt-realtime", s2s_params={"api_key": "k", "model": "gpt-realtime"}),
            framework="openai_realtime",
            output_dir=tmp_path / "out",
            run_id="r",
        )
    results = await _run_preflight(cfg)
    assert results == []


# ── Cheap backend-construction validation (_preflight_backends) ──────────────


def test_backend_construction_validates_s2s_assistant_missing_key(tmp_path):
    # S2S key is no longer validated at config load (removed); construction catches it.
    with patch.dict(os.environ, _BASE_ENV, clear=True):  # no OPENAI_API_KEY
        cfg = RunConfig(
            model=ModelConfig(s2s="gpt-realtime", s2s_params={"model": "gpt-realtime"}),
            framework="openai_realtime",
            output_dir=tmp_path / "out",
            run_id="r",
        )
        with pytest.raises(PreflightError, match="assistant framework 'openai_realtime'"):
            preflight._preflight_backends(cfg)


def test_backend_construction_passes_with_key(tmp_path):
    with patch.dict(os.environ, _BASE_ENV | {"OPENAI_API_KEY": "k"}, clear=True):
        cfg = RunConfig(
            model=ModelConfig(s2s="gpt-realtime", s2s_params={"api_key": "k", "model": "gpt-realtime"}),
            framework="openai_realtime",
            output_dir=tmp_path / "out",
            run_id="r",
        )
        preflight._preflight_backends(cfg)  # no raise


def test_backend_construction_skips_legacy_user_sim(tmp_path):
    # ElevenLabs (the default user sim) isn't factory-backed -> skipped, no raise.
    with patch.dict(os.environ, _BASE_ENV, clear=True):
        preflight._preflight_backends(_cascade_config(tmp_path))


def test_backend_construction_validates_factory_user_sim(tmp_path):
    with patch.dict(os.environ, _BASE_ENV, clear=True):  # no OPENAI_API_KEY
        cfg = _cascade_config(tmp_path)
        cfg.user_simulator = S2SSimulatorConfig(provider="openai_realtime")
        with pytest.raises(PreflightError, match="user simulator 'openai_realtime'"):
            preflight._preflight_backends(cfg)


@pytest.mark.asyncio
async def test_backend_construction_runs_even_when_preflight_disabled(tmp_path):
    # --no-preflight skips only the live model probes; the cheap construction check still
    # runs. Use the user-sim provider, whose key isn't validated at config load.
    with patch.dict(os.environ, _BASE_ENV, clear=True):  # no OPENAI_API_KEY
        cfg = _cascade_config(tmp_path)
        cfg.user_simulator = S2SSimulatorConfig(provider="openai_realtime")
        cfg.preflight = False
        with pytest.raises(PreflightError, match="user simulator 'openai_realtime'"):
            await run_preflight(cfg)


@pytest.mark.asyncio
async def test_guard_times_out():
    async def hang():
        await asyncio.sleep(10)
        return True, ""

    result = await preflight._guard("LLM", "m", hang(), timeout=0.05)
    assert not result.ok
    assert "timed out" in result.detail


@pytest.mark.asyncio
async def test_or_raise_passes_when_all_ok(tmp_path):
    cfg = _cascade_config(tmp_path)
    with (
        patch.object(preflight, "create_stt_service", return_value=_GoodSTT()) as stt,
        patch.object(preflight, "create_tts_service", return_value=_GoodTTS()) as tts,
        patch.object(preflight, "LiteLLMClient") as llm,
    ):
        llm.return_value.complete = AsyncMock(return_value=("pong", {}))
        await run_preflight(cfg)
        stt.assert_called_once()
        tts.assert_called_once()
        llm.assert_called_once()


@pytest.mark.asyncio
async def test_or_raise_raises_on_failure(tmp_path):
    cfg = _cascade_config(tmp_path)
    with (
        patch.object(preflight, "create_stt_service", return_value=_GoodSTT()),
        patch.object(preflight, "create_tts_service", return_value=_GoodTTS()),
        patch.object(preflight, "LiteLLMClient") as llm,
    ):
        llm.return_value.complete = AsyncMock(side_effect=Exception("401"))
        with pytest.raises(PreflightError, match="LLM"):
            await run_preflight(cfg)


@pytest.mark.asyncio
async def test_or_raise_noop_when_disabled(tmp_path):
    cfg = _cascade_config(tmp_path)
    cfg.preflight = False
    with patch.object(preflight, "run_preflight") as probe:
        await run_preflight(cfg)
        probe.assert_not_called()
