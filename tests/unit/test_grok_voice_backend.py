"""Unit tests for GrokVoiceBackend (no network).

Covers what Grok changes vs the OpenAI Realtime backend it subclasses: xAI
defaults, the required api key, factory dispatch, and the buffered/deferred
input-transcription behavior. Everything else is inherited and covered by
``test_openai_realtime_backend``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

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


def test_grok_caller_widens_silence_window():
    from eva.backend.grok_voice import GROK_CALLER_MIN_SILENCE_MS

    # As a caller (manual_turn_taking) grok's VAD silence is widened so it doesn't
    # segment mid-turn; a shorter configured value is raised to the minimum.
    td = _backend(manual_turn_taking=True, vad_settings={"silence_duration_ms": 500})._session_config["audio"]["input"][
        "turn_detection"
    ]
    assert td["silence_duration_ms"] == GROK_CALLER_MIN_SILENCE_MS
    # As an assistant (no manual_turn_taking) its own tuning is kept.
    td_asst = _backend(vad_settings={"silence_duration_ms": 200})._session_config["audio"]["input"]["turn_detection"]
    assert td_asst["silence_duration_ms"] == 200


def test_grok_declares_no_manual_response_support():
    # xAI ignores manual turn-taking, so it declares supports_manual_response=False,
    # routing a UserRole to the native-VAD path. OpenAI keeps manual support.
    from eva.backend.openai_realtime import OpenAIRealtimeBackend

    assert _backend().capabilities.supports_manual_response is False
    assert OpenAIRealtimeBackend(config={"model": "rt", "api_key": "k"}).capabilities.supports_manual_response is True


def test_grok_inherits_role_declared_interruptibility():
    # Grok does not force the flag (xAI ignores interrupt_response anyway); it honors the
    # role-declared value like any backend. Non-interruptibility is enforced at the audio
    # layer instead (see the suppression tests below).
    td = _backend(manual_turn_taking=True, interruptible=False)._session_config["audio"]["input"]["turn_detection"]
    assert td["create_response"] is False
    assert td["interrupt_response"] is False


@pytest.mark.asyncio
async def test_grok_suppresses_input_audio_while_responding():
    # Non-interruptibility enforced in the backend: while a response is in flight, inbound
    # audio is dropped so xAI's VAD can't re-trigger and wedge the response.
    b = _backend()
    conn = SimpleNamespace(input_audio_buffer=SimpleNamespace(append=AsyncMock()))
    s = GrokVoiceSession(client=None, conn_cm=None, conn=conn)  # type: ignore[arg-type]

    await b.send(s, audio=b"\x00\x01")  # not responding -> forwarded
    assert conn.input_audio_buffer.append.await_count == 1

    s.responding = True
    await b.send(s, audio=b"\x00\x01")  # responding -> dropped
    assert conn.input_audio_buffer.append.await_count == 1


@pytest.mark.asyncio
async def test_grok_trigger_response_marks_responding():
    b = _backend()
    conn = SimpleNamespace(response=SimpleNamespace(create=AsyncMock()))
    s = GrokVoiceSession(client=None, conn_cm=None, conn=conn)  # type: ignore[arg-type]
    await b.trigger_response(s)
    assert s.responding is True
    conn.response.create.assert_awaited_once()


def test_interruptible_is_a_uniform_config_flag_on_the_base():
    # The mechanism isn't grok-specific: the base backend reads the role-declared
    # `interruptible` flag. OpenAI honors False (its proven caller behavior) and, if a
    # role ever declared True, would honor that too.
    from eva.backend.openai_realtime import OpenAIRealtimeBackend

    off = OpenAIRealtimeBackend(config={"model": "rt", "api_key": "k", "manual_turn_taking": True})
    on = OpenAIRealtimeBackend(
        config={"model": "rt", "api_key": "k", "manual_turn_taking": True, "interruptible": True}
    )
    assert off._session_config["audio"]["input"]["turn_detection"]["interrupt_response"] is False
    assert on._session_config["audio"]["input"]["turn_detection"]["interrupt_response"] is True


def test_caller_role_declares_non_interruptible_intent():
    from eva.role.user import UserRole

    assert UserRole.CALLER_BACKEND_DEFAULTS["interruptible"] is False


def test_open_uses_grok_session_class():
    assert GrokVoiceBackend._SESSION_CLS is GrokVoiceSession
    assert issubclass(GrokVoiceSession, OpenAIRealtimeSession)


def test_factory_dispatch():
    b = BackendFactory().create("grok_voice", {"model": "grok-voice", "api_key": "xai-key"})
    assert isinstance(b, GrokVoiceBackend)


def _completed(text=""):
    return SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript=text)


def _transcripts(events):
    return [e.transcript for e in events if e.event_type == BackendEventType.TRANSCRIPT]


def test_cumulative_completed_emits_only_new_suffix():
    # xAI re-sends the whole cumulative transcript each time; we emit only the new part
    # so pieces are mutually exclusive and the postprocessor re-joins them with a space.
    s = _session()
    assert _transcripts(_map(s, _completed("Yes, but only show options."))) == ["Yes, but only show options."]
    assert _transcripts(_map(s, _completed("Yes, but only show options. And the total cost."))) == [
        "And the total cost."
    ]
    assert _transcripts(_map(s, _completed("Yes, but only show options. And the total cost. By 4 PM."))) == ["By 4 PM."]
    assert s.last_input_transcript == "Yes, but only show options. And the total cost. By 4 PM."


def test_repeated_identical_completed_emits_nothing():
    s = _session()
    assert _transcripts(_map(s, _completed("hello"))) == ["hello"]
    assert _map(s, _completed("hello")) == []  # no new text -> no duplicate


def test_fresh_utterance_not_extending_is_emitted_whole():
    s = _session()
    _map(s, _completed("First utterance."))
    assert _transcripts(_map(s, _completed("Completely different."))) == ["Completely different."]


def test_empty_completed_emits_nothing():
    s = _session()
    assert _map(s, _completed("")) == []


def test_other_events_delegate_to_parent():
    # AUDIO_OUTPUT still normalized by the inherited handler.
    (be,) = _map(_session(), SimpleNamespace(type="response.output_audio.delta", delta="AQIDBA=="))
    assert be.event_type == BackendEventType.AUDIO_OUTPUT
