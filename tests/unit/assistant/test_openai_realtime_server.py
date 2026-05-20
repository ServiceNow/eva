"""Tests for OpenAIRealtimeAssistantServer extension hooks.

These tests verify behavior of the hooks that GrokVoiceAssistantServer overrides.
They guard against regressions when refactoring shared logic.
"""

from unittest.mock import MagicMock

from openai import AsyncOpenAI

from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer


def _bare_server() -> OpenAIRealtimeAssistantServer:
    """Construct an instance without running __init__ (skips PromptManager + tool building)."""
    srv = object.__new__(OpenAIRealtimeAssistantServer)
    srv.pipeline_config = MagicMock()
    srv.pipeline_config.s2s_params = {"api_key": "sk-test", "model": "gpt-realtime-mini"}
    srv._model = "gpt-realtime-mini"
    srv._system_prompt = "you are a helpful assistant"
    srv._realtime_tools = []
    return srv


class TestCreateClient:
    def test_returns_async_openai_with_api_key(self):
        srv = _bare_server()
        client = srv._create_client()
        assert isinstance(client, AsyncOpenAI)
        # Verify api_key was passed
        assert client.api_key == "sk-test"

    def test_default_base_url_is_openai(self):
        srv = _bare_server()
        client = srv._create_client()
        # Default OpenAI base URL (do not override)
        assert "openai.com" in str(client.base_url)

    def test_raises_when_api_key_missing(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {}
        try:
            srv._create_client()
        except ValueError as e:
            assert "API key required" in str(e)
        else:
            raise AssertionError("expected ValueError")


class TestDefaultVoice:
    def test_default_voice_is_marin(self):
        srv = _bare_server()
        assert srv._default_voice() == "marin"
