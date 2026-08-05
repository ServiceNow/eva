"""Unit tests for the OpenAI Realtime ``Backend`` (no network).

Covers the pure surfaces: ``session.update`` assembly + tool translation, the
stateful provider-event -> normalized ``BackendEvent`` mapping (including
final-transcript selection and interruption), capability flags, sample-rate
exposure, factory dispatch, and ``send()`` validation. The live session
(open/receive over a real connection) is out of scope here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from eva.backend.base import BackendEventType, ToolCallResult
from eva.backend.default_factory import DefaultBackendFactory
from eva.backend.openai_realtime import OpenAIRealtimeBackend, OpenAIRealtimeSession


def _backend(**overrides) -> OpenAIRealtimeBackend:
    config = {"model": "gpt-realtime", "api_key": "test-key", "input_format": "pcm", **overrides}
    return OpenAIRealtimeBackend(config=config)


def _session() -> OpenAIRealtimeSession:
    return OpenAIRealtimeSession(client=None, conn_cm=None, conn=None)  # type: ignore[arg-type]


def _map(session, event):
    return OpenAIRealtimeBackend._map_event(session, event)


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAIRealtimeBackend(config={"model": "gpt-realtime"})


def test_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    # No api_key in config -> backend picks it up from the environment (no raise).
    backend = OpenAIRealtimeBackend(config={"model": "gpt-realtime"})
    assert backend.output_sample_rate == 24000


def test_accent_is_rejected():
    with pytest.raises(ValueError):
        OpenAIRealtimeBackend(config={"model": "gpt-realtime", "api_key": "k", "accent": "british"})


def test_capabilities():
    caps = _backend().capabilities
    assert caps.emits_continuous_audio is True
    assert caps.supports_streaming_interruption is True
    assert caps.owns_playout_clock is False


def test_sample_rates_from_input_format():
    b = _backend()  # pcm
    assert b.output_sample_rate == 24000
    assert b.input_sample_rate == 24000
    b2 = _backend(input_format="pcmu")  # telephony/caller input
    assert b2.output_sample_rate == 24000
    assert b2.input_sample_rate == 8000


def test_assemble_assistant_session_defaults():
    # pcm input, auto turn-taking: no manual create_response fields; whisper, no language.
    sc = _backend(voice="marin", vad_settings={})._session_config
    assert sc["output_modalities"] == ["audio"]
    assert sc["audio"]["output"] == {"voice": "marin", "format": {"type": "audio/pcm", "rate": 24000}}
    assert sc["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert sc["audio"]["input"]["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 200,
    }
    assert sc["audio"]["input"]["transcription"] == {"model": "whisper-1"}


def test_assemble_caller_session_manual_turn_taking():
    sc = _backend(
        input_format="pcmu",
        voice="ballad",
        vad_settings={"threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 500},
        manual_turn_taking=True,
        transcription_language="en",
        parallel_tool_calls=False,
    )._session_config
    assert sc["audio"]["input"]["format"] == {"type": "audio/pcmu"}
    assert sc["audio"]["input"]["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
        "create_response": False,
        "interrupt_response": False,
        "idle_timeout_ms": 15000,
    }
    assert sc["audio"]["input"]["transcription"] == {"model": "whisper-1", "language": "en"}
    assert sc["parallel_tool_calls"] is False


def test_build_session_update_stamps_owned_fields_and_translates_tools():
    b = _backend()
    tools = [{"name": "get_reservation", "description": "look up", "parameters": {"type": "object", "properties": {}}}]
    session = b._build_session_update("SYSTEM PROMPT", tools)

    assert session["type"] == "realtime"
    assert session["instructions"] == "SYSTEM PROMPT"
    assert session["output_modalities"] == ["audio"]
    # Generic tool spec -> OpenAI schema (type: function added).
    assert session["tools"] == [
        {
            "type": "function",
            "name": "get_reservation",
            "description": "look up",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_build_session_update_none_tools_becomes_empty_list():
    assert _backend()._build_session_update("p", None)["tools"] == []


def test_map_audio_delta():
    (be,) = _map(_session(), SimpleNamespace(type="response.output_audio.delta", delta="AQIDBA=="))
    assert be.event_type == BackendEventType.AUDIO_OUTPUT
    assert be.audio == b"\x01\x02\x03\x04"


def test_map_empty_audio_delta_dropped():
    assert _map(_session(), SimpleNamespace(type="response.output_audio.delta", delta="")) == []


def test_map_output_transcript_delta_accumulates_silently_then_done_emits():
    s = _session()
    assert _map(s, SimpleNamespace(type="response.output_audio_transcript.delta", delta="hel")) == []
    assert _map(s, SimpleNamespace(type="response.output_audio_transcript.delta", delta="lo")) == []
    (be,) = _map(s, SimpleNamespace(type="response.output_audio_transcript.done", transcript="hello"))
    assert be.event_type == BackendEventType.TRANSCRIPT
    assert be.transcript == "hello"
    assert be.metadata == {"stream": "output", "final": True}


def test_map_input_transcription_completed_and_failed():
    (done,) = _map(
        _session(), SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript="hi")
    )
    assert done.event_type == BackendEventType.TRANSCRIPT
    assert done.transcript == "hi" and done.metadata == {"stream": "input", "final": True}

    # Empty completed transcription is dropped.
    assert (
        _map(_session(), SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript=""))
        == []
    )

    (failed,) = _map(_session(), SimpleNamespace(type="conversation.item.input_audio_transcription.failed", error="x"))
    assert failed.event_type == BackendEventType.TRANSCRIPT
    assert failed.metadata == {"stream": "input", "failed": True}


def test_map_speech_boundaries_no_interruption():
    (be,) = _map(_session(), SimpleNamespace(type="input_audio_buffer.speech_started"))
    assert be.event_type == BackendEventType.INPUT_SPEECH_STARTED
    (be2,) = _map(_session(), SimpleNamespace(type="input_audio_buffer.speech_stopped"))
    assert be2.event_type == BackendEventType.INPUT_SPEECH_STOPPED


def test_map_interruption_flushes_partial_turn_then_speech_started():
    s = _session()
    _map(s, SimpleNamespace(type="response.created"))
    _map(s, SimpleNamespace(type="response.output_audio_transcript.delta", delta="I was sa"))
    events = _map(s, SimpleNamespace(type="input_audio_buffer.speech_started"))
    # Interrupted TURN_END (with the partial) precedes the speech-started signal.
    assert [e.event_type for e in events] == [BackendEventType.TURN_END, BackendEventType.INPUT_SPEECH_STARTED]
    turn_end = events[0]
    assert turn_end.transcript == "I was sa"
    assert turn_end.metadata["interrupted"] is True and turn_end.metadata["cancelled"] is False
    # State reset: a second speech_started does not re-flush.
    assert [e.event_type for e in _map(s, SimpleNamespace(type="input_audio_buffer.speech_started"))] == [
        BackendEventType.INPUT_SPEECH_STARTED
    ]


def test_map_function_call():
    s = _session()
    (be,) = _map(
        s,
        SimpleNamespace(
            type="response.function_call_arguments.done",
            call_id="c1",
            name="get_reservation",
            arguments='{"n": "ABC"}',
        ),
    )
    assert be.event_type == BackendEventType.TOOL_CALL_REQUEST
    assert be.tool_call_request.call_id == "c1"
    assert be.tool_call_request.arguments == {"n": "ABC"}
    assert s.has_function_calls is True


def test_map_function_call_bad_arguments_becomes_empty():
    (be,) = _map(
        _session(),
        SimpleNamespace(type="response.function_call_arguments.done", call_id="c", name="f", arguments="nope"),
    )
    assert be.tool_call_request.arguments == {}


def test_map_output_audio_done():
    (be,) = _map(_session(), SimpleNamespace(type="response.output_audio.done"))
    assert be.event_type == BackendEventType.OUTPUT_AUDIO_DONE


def test_map_turn_started():
    (be,) = _map(_session(), SimpleNamespace(type="response.created"))
    assert be.event_type == BackendEventType.OUTPUT_TURN_STARTED


def test_map_response_done_selects_final_transcript_and_usage():
    s = _session()
    _map(s, SimpleNamespace(type="response.created"))
    _map(s, SimpleNamespace(type="response.output_audio_transcript.done", transcript="all done"))
    usage = SimpleNamespace(input_tokens=11, output_tokens=7)
    response = SimpleNamespace(status="completed", usage=usage, output=[])
    (be,) = _map(s, SimpleNamespace(type="response.done", response=response))
    assert be.event_type == BackendEventType.TURN_END
    assert be.transcript == "all done"
    assert be.metadata["cancelled"] is False and be.metadata["interrupted"] is False
    assert be.metadata["usage"] == {"prompt_tokens": 11, "completion_tokens": 7}


def test_map_response_done_cancelled():
    response = SimpleNamespace(status="cancelled", usage=None, output=[])
    (be,) = _map(_session(), SimpleNamespace(type="response.done", response=response))
    assert be.event_type == BackendEventType.TURN_END
    assert be.metadata["cancelled"] is True and be.metadata["usage"] is None


def test_map_response_done_has_function_calls_from_output_items():
    response = SimpleNamespace(status="completed", usage=None, output=[SimpleNamespace(type="function_call")])
    (be,) = _map(_session(), SimpleNamespace(type="response.done", response=response))
    assert be.metadata["has_function_calls"] is True


def test_map_error_carries_code():
    (be,) = _map(
        _session(), SimpleNamespace(type="error", error=SimpleNamespace(code="rate_limit", message="slow down"))
    )
    assert be.event_type == BackendEventType.ERROR
    assert be.metadata["code"] == "rate_limit"


def test_map_unhandled_event_dropped():
    assert _map(_session(), SimpleNamespace(type="session.updated")) == []
    assert _map(_session(), SimpleNamespace(type="conversation.item.created")) == []


@pytest.mark.asyncio
async def test_send_requires_exactly_one_arg():
    b, s = _backend(), _session()
    with pytest.raises(ValueError):
        await b.send(s)
    with pytest.raises(ValueError):
        await b.send(s, audio=b"x", text="y")


@pytest.mark.asyncio
async def test_send_wrong_session_type_raises():
    with pytest.raises(TypeError):
        await _backend().send(object(), tool_result=ToolCallResult(call_id="c", result={}))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_close_is_idempotent_and_network_free():
    b, s = _backend(), _session()
    await b.close(s)
    await b.close(s)


def test_factory_dispatch_builds_backend_from_flat_config():
    backend = DefaultBackendFactory().create(
        "openai_realtime", {"model": "gpt-realtime", "api_key": "k", "input_format": "pcmu"}
    )
    assert isinstance(backend, OpenAIRealtimeBackend)
    assert backend.output_sample_rate == 24000
    assert backend.input_sample_rate == 8000


def test_factory_unknown_provider_raises():
    with pytest.raises(ValueError):
        DefaultBackendFactory().create("does_not_exist", {})
