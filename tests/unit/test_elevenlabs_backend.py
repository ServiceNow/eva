"""Unit tests for ElevenLabsBackend (no network, no live SDK session).

Covers the deterministic surfaces: config/api-key validation with the
``ELEVENLABS_API_KEY`` fallback, factory dispatch, sample-rate/capability
exposure, the SDK-callback -> normalized ``BackendEvent`` mapping, the
greeting-unwrap, and the role-side tool-call bridge (a ``ClientTools`` handler
surfaces a ``TOOL_CALL_REQUEST`` and blocks until the role resolves it via
``send(tool_result=...)``). The live ``AsyncConversation`` (open/start over a
real WS) is out of scope.
"""

from __future__ import annotations

import asyncio

import pytest

from eva.backend.base import BackendEventType, ToolCallResult
from eva.backend.elevenlabs import (
    ElevenLabsBackend,
    ElevenLabsSession,
    _unwrap_greeting,
)
from eva.backend.factory import BackendFactory


def _backend(**overrides) -> ElevenLabsBackend:
    config = {"model": "elevenlabs", "api_key": "el-key", "speaker_id": "ag_1", **overrides}
    return ElevenLabsBackend(config=config)


def _session() -> ElevenLabsSession:
    return ElevenLabsSession(client=None, bridge=None, system_prompt="p", client_tools=None)


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        ElevenLabsBackend(config={"speaker_id": "ag_1"})


def test_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "env-key")
    b = ElevenLabsBackend(config={"speaker_id": "ag_1"})
    assert b._api_key == "env-key"


def test_requires_speaker_id():
    with pytest.raises(ValueError, match="speaker_id"):
        ElevenLabsBackend(config={"api_key": "k"})


def test_multiple_config_errors_are_cumulative(monkeypatch):
    # Missing api_key AND speaker_id -> both reported in one raise, not just the first.
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ValueError) as exc:
        ElevenLabsBackend(config={})
    message = str(exc.value)
    assert "api_key" in message
    assert "speaker_id" in message


def test_model_defaults_to_elevenlabs():
    assert ElevenLabsBackend(config={"api_key": "k", "speaker_id": "a"})._model == "elevenlabs"
    assert _backend(model="custom-label")._model == "custom-label"


def test_sample_rates_are_role_rate():
    b = _backend()
    # The SDK's 8k/16k rates are hidden; the role sees a uniform 24 kHz in/out.
    assert b.input_sample_rate == 24000
    assert b.output_sample_rate == 24000


def test_capabilities():
    caps = _backend().capabilities
    assert caps.emits_continuous_audio is True
    assert caps.supports_streaming_interruption is False
    assert caps.owns_playout_clock is False


def test_factory_dispatch():
    b = BackendFactory().create("elevenlabs", {"api_key": "k", "speaker_id": "ag_1"})
    assert isinstance(b, ElevenLabsBackend)


# ── Greeting unwrap ──────────────────────────────────────────────────────


def test_unwrap_greeting_strips_role_wrapper():
    assert _unwrap_greeting("Say: 'Hello there!'") == "Hello there!"


def test_unwrap_greeting_passthrough_when_unwrapped():
    assert _unwrap_greeting("Hello there!") == "Hello there!"


# ── Callback -> normalized event mapping ─────────────────────────────────


def test_agent_response_event_is_turn_end():
    be = ElevenLabsBackend._agent_response_event("  all done  ")
    assert be.event_type == BackendEventType.TURN_END
    assert be.transcript == "all done"
    assert be.metadata == {"interrupted": False, "cancelled": False, "has_function_calls": False, "usage": None}


def test_correction_event_is_interrupted_turn_end():
    be = ElevenLabsBackend._correction_event("partial reply")
    assert be.event_type == BackendEventType.TURN_END
    assert be.transcript == "partial reply"
    assert be.metadata["interrupted"] is True


def test_user_transcript_event_is_input_transcript():
    be = ElevenLabsBackend._user_transcript_event("hi there")
    assert be.event_type == BackendEventType.TRANSCRIPT
    assert be.transcript == "hi there"
    assert be.metadata == {"stream": "input", "final": True}


# ── Tool-call bridge (role-side execution) ───────────────────────────────


@pytest.mark.asyncio
async def test_client_tools_bound_to_running_loop():
    # The SDK must run tool handlers on OUR loop, not a separate thread loop -- otherwise
    # the handler touches the main-loop event queue / result future cross-loop, raises,
    # and the SDK reports the tool as failed to the agent.
    b, s = _backend(), _session()
    client_tools = b._build_client_tools(s, [{"name": "get_reservation", "description": "d", "parameters": {}}])
    assert client_tools is not None
    assert client_tools._custom_loop is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_tool_call_bridges_to_role_and_awaits_result():
    b, s = _backend(), _session()

    async def _role_resolves() -> None:
        # Wait for the handler to enqueue the request, then resolve it like the role.
        event = await s.events.get()
        assert event.event_type == BackendEventType.TOOL_CALL_REQUEST
        req = event.tool_call_request
        assert req.name == "get_reservation"
        assert req.arguments == {"confirmation_number": "ABC"}  # tool_call_id stripped
        await b.send(s, tool_result=ToolCallResult(call_id=req.call_id, result={"status": "ok"}))

    resolver = asyncio.create_task(_role_resolves())
    out = await b._bridge_tool_call(s, "get_reservation", {"confirmation_number": "ABC", "tool_call_id": "call-42"})
    await resolver
    assert out == '{"status": "ok"}'
    assert s.pending_tools == {}  # popped on resolution


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
async def test_receive_returns_when_ended_and_drained():
    b, s = _backend(), _session()
    s.ended.set()
    events = [e async for e in b.receive(s)]
    assert events == []


@pytest.mark.asyncio
async def test_close_is_idempotent_and_network_free():
    b, s = _backend(), _session()
    await b.close(s)
    await b.close(s)
