"""Unit tests for GrokVoiceBackend (no network).

Covers what Grok changes vs the OpenAI Realtime backend it subclasses: xAI
defaults, the required api key, factory dispatch, and the buffered/deferred
input-transcription behavior. Everything else is inherited and covered by
``test_openai_realtime_backend``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from eva.backend.base import BackendEventType
from eva.backend.factory import BackendFactory
from eva.backend.grok_voice import DEFAULT_VOICE, XAI_REALTIME_BASE_URL, GrokVoiceBackend, GrokVoiceSession
from eva.backend.openai_realtime import OpenAIRealtimeSession


def _backend(**overrides) -> GrokVoiceBackend:
    return GrokVoiceBackend(config={"model": "grok-voice", "api_key": "xai-key", **overrides})


def _session() -> GrokVoiceSession:
    return GrokVoiceSession(client=None, conn_cm=None, conn=None)  # type: ignore[arg-type]


def _map(session, event):
    return GrokVoiceBackend._map_event(session, event)


def test_api_key_falls_back_to_xai_env(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-from-env")
    b = GrokVoiceBackend(config={"model": "grok-voice"})
    assert b._api_key == "xai-from-env"


def test_openai_env_does_not_satisfy_grok(monkeypatch):
    # An OpenAI key is the wrong key for x.ai: it must NOT be used as a fallback.
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")
    with pytest.raises(ValueError, match="XAI_API_KEY"):
        GrokVoiceBackend(config={"model": "grok-voice"})


def test_xai_defaults_applied():
    b = _backend()
    assert b._base_url == XAI_REALTIME_BASE_URL
    assert b._session_config["audio"]["output"]["voice"] == DEFAULT_VOICE


def test_explicit_config_wins_over_defaults():
    b = _backend(base_url="https://custom/v1", voice="ara")
    assert b._base_url == "https://custom/v1"
    assert b._session_config["audio"]["output"]["voice"] == "ara"


def test_open_uses_grok_session_class():
    assert GrokVoiceBackend._SESSION_CLS is GrokVoiceSession
    assert issubclass(GrokVoiceSession, OpenAIRealtimeSession)


def test_factory_dispatch():
    b = BackendFactory().create("grok_voice", {"model": "grok-voice", "api_key": "xai-key"})
    assert isinstance(b, GrokVoiceBackend)


def test_incremental_transcription_is_buffered_not_emitted():
    s = _session()
    # Progressive completed events accumulate but emit nothing.
    assert _map(s, SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript="I")) == []
    assert (
        _map(s, SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript="I want"))
        == []
    )
    assert s.pending_input_transcript == "I want"


def test_buffered_transcript_flushes_on_speech_started():
    s = _session()
    _map(s, SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript="hello there"))
    events = _map(s, SimpleNamespace(type="input_audio_buffer.speech_started"))
    # First event is the flushed final input transcript, then the inherited speech-start.
    assert events[0].event_type == BackendEventType.TRANSCRIPT
    assert events[0].transcript == "hello there"
    assert events[0].metadata == {"stream": "input", "final": True}
    assert events[-1].event_type == BackendEventType.INPUT_SPEECH_STARTED
    assert s.pending_input_transcript == ""


def test_buffered_transcript_flushes_on_response_done():
    s = _session()
    _map(s, SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript="final text"))
    events = _map(s, SimpleNamespace(type="response.done", response=None))
    assert events[0].event_type == BackendEventType.TRANSCRIPT
    assert events[0].transcript == "final text"
    assert any(e.event_type == BackendEventType.TURN_END for e in events)


def test_no_pending_flush_is_noop():
    s = _session()
    # speech_started with nothing buffered -> just the inherited event, no TRANSCRIPT.
    events = _map(s, SimpleNamespace(type="input_audio_buffer.speech_started"))
    assert all(e.event_type != BackendEventType.TRANSCRIPT for e in events)


def test_other_events_delegate_to_parent():
    # AUDIO_OUTPUT still normalized by the inherited handler.
    (be,) = _map(_session(), SimpleNamespace(type="response.output_audio.delta", delta="AQIDBA=="))
    assert be.event_type == BackendEventType.AUDIO_OUTPUT
