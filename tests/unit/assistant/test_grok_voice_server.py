"""Tests for GrokVoiceAssistantServer hook overrides."""

from unittest.mock import MagicMock

from openai import AsyncOpenAI

from eva.assistant.grok_voice_server import GrokVoiceAssistantServer


def _bare_server() -> GrokVoiceAssistantServer:
    srv = object.__new__(GrokVoiceAssistantServer)
    srv.pipeline_config = MagicMock()
    srv.pipeline_config.s2s_params = {
        "api_key": "xai-test-key",
        "model": "grok-voice-latest",
    }
    srv._model = "grok-voice-latest"
    srv._system_prompt = "you are a helpful assistant"
    srv._realtime_tools = []
    return srv


class TestCreateClient:
    def test_uses_xai_base_url(self):
        srv = _bare_server()
        client = srv._create_client()
        assert isinstance(client, AsyncOpenAI)
        assert client.api_key == "xai-test-key"
        assert "api.x.ai" in str(client.base_url)

    def test_raises_when_api_key_missing(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {}
        try:
            srv._create_client()
        except ValueError as e:
            assert "API key required" in str(e)
            assert "Grok Voice" in str(e)
        else:
            raise AssertionError("expected ValueError")


class TestDefaultVoice:
    def test_default_voice_is_eve(self):
        srv = _bare_server()
        assert srv._default_voice() == "eve"


class TestBuildSessionConfig:
    def test_voice_defaults_to_eve(self):
        srv = _bare_server()
        cfg = srv._build_session_config()
        assert cfg["audio"]["output"]["voice"] == "eve"

    def test_explicit_voice_passes_through(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {
            "api_key": "xai-test-key",
            "model": "grok-voice-latest",
            "voice": "rex",
        }
        cfg = srv._build_session_config()
        assert cfg["audio"]["output"]["voice"] == "rex"


class TestServiceLabels:
    def test_service_name(self):
        assert GrokVoiceAssistantServer._service_name == "Grok Voice"

    def test_metrics_processor_name(self):
        assert GrokVoiceAssistantServer._metrics_processor_name == "grok_voice"
