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
    b = _backend(base_url="https://custom/v1", speaker_id="ara")
    assert b._base_url == "https://custom/v1"
    assert b._session_config["audio"]["output"]["voice"] == "ara"


def test_open_uses_grok_session_class():
    assert GrokVoiceBackend._SESSION_CLS is GrokVoiceSession
    assert issubclass(GrokVoiceSession, OpenAIRealtimeSession)


def test_factory_dispatch():
    b = BackendFactory().create("grok_voice", {"model": "grok-voice", "api_key": "xai-key"})
    assert isinstance(b, GrokVoiceBackend)


def _completed(text=""):
    return SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript=text)


def _speech_started():
    return SimpleNamespace(type="input_audio_buffer.speech_started")


def _response_done(status="completed"):
    response = SimpleNamespace(status=status, usage=None, output=[])
    return SimpleNamespace(type="response.done", response=response)


def _transcripts(events):
    return [e.transcript for e in events if e.event_type == BackendEventType.TRANSCRIPT]


def test_cumulative_completed_is_buffered_not_emitted():
    # xAI re-sends the whole cumulative transcript each time; nothing is emitted until
    # the turn boundary flush, so the role logs one user turn (not one per fragment).
    s = _session()
    assert _map(s, _completed("Hi, I need to change my")) == []
    assert _map(s, _completed("Hi, I need to change my flight to March 20")) == []
    assert _map(s, _completed("Hi, I need to change my flight to March 25th.")) == []
    assert s.pending_input_transcript == "Hi, I need to change my flight to March 25th."


def test_flush_on_response_done_emits_latest_cumulative_once():
    s = _session()
    _map(s, _completed("Hi, I need to change my"))
    _map(s, _completed("Hi, I need to change my flight to March 25th."))
    events = _map(s, _response_done())
    # The buffered transcript is flushed once, before the parent's TURN_END.
    assert _transcripts(events) == ["Hi, I need to change my flight to March 25th."]
    assert events[0].event_type == BackendEventType.TRANSCRIPT
    assert events[-1].event_type == BackendEventType.TURN_END
    assert s.pending_input_transcript == ""  # cleared after flush


def test_flush_on_speech_started_emits_and_precedes_speech_signal():
    s = _session()
    _map(s, _completed("First utterance."))
    events = _map(s, _speech_started())
    assert _transcripts(events) == ["First utterance."]
    assert events[0].event_type == BackendEventType.TRANSCRIPT
    assert events[-1].event_type == BackendEventType.INPUT_SPEECH_STARTED


def test_flush_without_buffer_emits_no_transcript():
    s = _session()
    assert _transcripts(_map(s, _response_done())) == []


def test_flush_is_not_repeated_after_clear():
    s = _session()
    _map(s, _completed("only once"))
    assert _transcripts(_map(s, _response_done())) == ["only once"]
    assert _transcripts(_map(s, _speech_started())) == []  # buffer already cleared


def test_empty_completed_buffers_nothing():
    s = _session()
    _map(s, _completed(""))
    assert s.pending_input_transcript == ""


def test_other_events_delegate_to_parent():
    # AUDIO_OUTPUT still normalized by the inherited handler.
    (be,) = _map(_session(), SimpleNamespace(type="response.output_audio.delta", delta="AQIDBA=="))
    assert be.event_type == BackendEventType.AUDIO_OUTPUT
