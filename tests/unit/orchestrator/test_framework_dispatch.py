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


def test_cartesia_dispatch_returns_cartesia_class():
    from eva.assistant.cartesia_server import CartesiaAssistantServer

    assert _get_server_class("cartesia") is CartesiaAssistantServer


def test_openai_live_dispatch_returns_live_class():
    from eva.assistant.openai_live_server import OpenAILiveAssistantServer

    assert _get_server_class("openai_live") is OpenAILiveAssistantServer


def test_unknown_framework_error_lists_openai_live():
    with pytest.raises(ValueError) as exc_info:
        _get_server_class("nope")
    assert "openai_live" in str(exc_info.value)
