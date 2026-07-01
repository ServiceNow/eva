"""Tests for the Smallest Hydra S2S server helpers and transcription plumbing."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from eva.assistant.audio_bridge import (
    mulaw_8k_to_pcm16_48k,
    pcm16_48k_to_mulaw_8k,
    pcm16_to_wav_bytes,
)
from eva.assistant.s2s_transcription import BatchTranscriber, create_transcriber
from eva.assistant.smallest_hydra_server import _agent_tools_to_hydra
from eva.models.agents import AgentConfig, AgentTool, AgentToolParameter


def _agent_with_tools() -> AgentConfig:
    return AgentConfig(
        id="a1",
        name="Test Agent",
        description="desc",
        role="role",
        instructions="be helpful",
        tool_module_path="eva.assistant.tools.airline_tools",
        tools=[
            AgentTool(
                id="t1",
                name="Lookup Booking",
                description="Look up a booking",
                required_parameters=[AgentToolParameter(name="booking_id", type="str", description="The booking id")],
                optional_parameters=[AgentToolParameter(name="verbose", type="bool")],
            )
        ],
    )


class TestToolConversion:
    def test_no_tools_returns_none(self):
        agent = MagicMock()
        agent.tools = []
        assert _agent_tools_to_hydra(agent) is None

    def test_openai_style_function_declaration(self):
        functions = _agent_tools_to_hydra(_agent_with_tools())
        assert functions is not None and len(functions) == 1
        fn = functions[0]
        assert fn["type"] == "function"
        assert fn["name"]  # function_name derived from the tool
        assert "Lookup Booking" in fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "booking_id" in params["properties"]
        assert params["required"] == ["booking_id"]


class TestAudioHelpers:
    def test_mulaw_round_trip_preserves_sample_count(self):
        # One 20 ms mulaw chunk: 160 samples @ 8 kHz.
        mulaw = b"\xff" * 160
        pcm_48k = mulaw_8k_to_pcm16_48k(mulaw)
        assert len(pcm_48k) == 160 * 6 * 2  # 6x upsample, 16-bit
        back = pcm16_48k_to_mulaw_8k(pcm_48k)
        assert len(back) == 160  # exact inverse sample count

    def test_pcm16_to_wav_has_riff_header(self):
        wav = pcm16_to_wav_bytes(b"\x00\x00" * 100, 48000)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"


class TestCreateTranscriber:
    def test_default_is_smallest_pulse_pro_reusing_s2s_key(self):
        t = create_transcriber({"api_key": "SMALL"}, "en")
        assert (t.provider, t.model, t.api_key, t.language) == ("smallest", "pulse-pro", "SMALL", "en")

    def test_smallest_non_english_falls_back_to_multilingual_pulse(self):
        t = create_transcriber({"api_key": "SMALL", "transcription": {"language": "de"}}, "de")
        assert t.model == "pulse"

    def test_explicit_provider_and_model(self):
        t = create_transcriber(
            {"api_key": "X", "transcription": {"provider": "openai", "model": "gpt-4o-transcribe", "api_key": "OAI"}},
            "en",
        )
        assert (t.provider, t.model, t.api_key) == ("openai", "gpt-4o-transcribe", "OAI")

    def test_openai_default_model(self):
        t = create_transcriber({"transcription": {"provider": "openai", "api_key": "OAI"}}, "en")
        assert t.model == "whisper-1"

    def test_missing_key_for_non_smallest_provider_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key"):
            create_transcriber({"api_key": "SMALL", "transcription": {"provider": "openai"}}, "en")

    def test_openai_falls_back_to_env_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "ENVOAI")
        t = create_transcriber({"transcription": {"provider": "openai"}}, "en")
        assert t.api_key == "ENVOAI"

    def test_deepgram_falls_back_to_env_key(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "ENVDG")
        t = create_transcriber({"transcription": {"provider": "deepgram"}}, "en")
        assert t.api_key == "ENVDG"


def _mock_response(json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=json_body)
    return resp


class TestBatchTranscribe:
    @pytest.mark.asyncio
    async def test_smallest_parses_transcription_field(self):
        t = BatchTranscriber("smallest", "pulse-pro", "K", "en")
        t._client.post = AsyncMock(return_value=_mock_response({"transcription": " hello "}))
        assert await t.transcribe(b"\x00\x00" * 10, 16000) == "hello"
        await t.aclose()

    @pytest.mark.asyncio
    async def test_openai_parses_text_field(self):
        t = BatchTranscriber("openai", "whisper-1", "K", "en")
        t._client.post = AsyncMock(return_value=_mock_response({"text": "hi there"}))
        assert await t.transcribe(b"\x00\x00" * 10, 16000) == "hi there"
        await t.aclose()

    @pytest.mark.asyncio
    async def test_deepgram_parses_nested_transcript(self):
        t = BatchTranscriber("deepgram", "nova-3", "K", "en")
        body = {"results": {"channels": [{"alternatives": [{"transcript": "deep text"}]}]}}
        t._client.post = AsyncMock(return_value=_mock_response(body))
        assert await t.transcribe(b"\x00\x00" * 10, 16000) == "deep text"
        await t.aclose()

    @pytest.mark.asyncio
    async def test_empty_pcm_skips_request(self):
        t = BatchTranscriber("smallest", "pulse-pro", "K", "en")
        t._client.post = AsyncMock()
        assert await t.transcribe(b"", 16000) == ""
        t._client.post.assert_not_called()
        await t.aclose()

    @pytest.mark.asyncio
    async def test_failure_is_fail_soft(self):
        t = BatchTranscriber("smallest", "pulse-pro", "K", "en")
        t._client.post = AsyncMock(side_effect=RuntimeError("boom"))
        assert await t.transcribe(b"\x00\x00" * 10, 16000) == ""  # no exception propagates
        await t.aclose()
