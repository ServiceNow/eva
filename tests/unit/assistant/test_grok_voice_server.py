"""Tests for GrokVoiceAssistantServer hook overrides."""

from unittest.mock import MagicMock

from openai import AsyncOpenAI

from eva.assistant.grok_voice_server import GrokVoiceAssistantServer
from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer


def _bare_server() -> GrokVoiceAssistantServer:
    srv = object.__new__(GrokVoiceAssistantServer)
    srv.pipeline_config = MagicMock()
    srv.pipeline_config.s2s_params = {
        "api_key": "xai-test-key",
        "model": "grok-voice-latest",
    }
    srv.pipeline_config.parallel_tool_calls = None
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

    def test_custom_websocket_base_url_passes_through(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params["websocket_base_url"] = "wss://custom.x.ai/v1"

        client = srv._create_client()

        assert client.websocket_base_url == "wss://custom.x.ai/v1"


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

    def test_does_not_send_openai_transcription_selector(self):
        srv = _bare_server()

        cfg = srv._build_session_config()

        assert "transcription" not in cfg["audio"]["input"]


class TestServiceLabels:
    def test_service_name(self):
        assert GrokVoiceAssistantServer._service_name == "Grok Voice"

    def test_metrics_processor_name(self):
        assert GrokVoiceAssistantServer._metrics_processor_name == "grok_voice"

    def test_inherits_response_scoped_tool_batching(self):
        assert GrokVoiceAssistantServer._on_function_call_done is OpenAIRealtimeAssistantServer._on_function_call_done
        assert GrokVoiceAssistantServer._finalize_tool_response is OpenAIRealtimeAssistantServer._finalize_tool_response
