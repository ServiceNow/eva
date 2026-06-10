"""Verify framework dispatcher returns the right server class."""

import pytest

from eva.assistant.grok_voice_server import GrokVoiceAssistantServer
from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer
from eva.orchestrator.worker import _get_server_class


def test_grok_voice_dispatch_returns_grok_class():
    cls = _get_server_class("grok_voice")
    assert cls is GrokVoiceAssistantServer


def test_grok_voice_is_subclass_of_openai_realtime():
    assert issubclass(GrokVoiceAssistantServer, OpenAIRealtimeAssistantServer)


def test_unknown_framework_error_lists_grok_voice():
    with pytest.raises(ValueError) as exc_info:
        _get_server_class("nope")
    assert "grok_voice" in str(exc_info.value)
