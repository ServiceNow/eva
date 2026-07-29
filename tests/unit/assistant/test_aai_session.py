"""Tests for the aai host-mode session: framing, handshake, and event mapping."""

import json

import pytest

from eva.assistant.aai_events import (
    AAIAgentTranscriptEvent,
    AAIAudioDoneEvent,
    AAICustomEvent,
    AAIErrorEvent,
    AAIIdleTimeoutEvent,
    AAIReplyDoneEvent,
    AAISpeechStartedEvent,
    AAISpeechStoppedEvent,
    AAIToolCallEvent,
    AAIUserTranscriptEvent,
)
from eva.assistant.aai_session import (
    AAIHostSession,
    AAIHostSessionError,
    build_host_config_message,
    map_aai_event,
    with_host_flag,
)
from eva.assistant.bridge_events import (
    AssistantTranscript,
    BackendError,
    SpeechStarted,
    ToolCall,
    TurnDone,
    UserTranscript,
)


class TestWithHostFlag:
    def test_appends_host_flag(self):
        assert with_host_flag("ws://localhost:3000/websocket") == "ws://localhost:3000/websocket?host=1"

    def test_merges_with_existing_query_params(self):
        result = with_host_flag("ws://localhost:3000/websocket?sessionId=abc")
        assert "sessionId=abc" in result
        assert "host=1" in result

    def test_overwrites_existing_host_param(self):
        assert with_host_flag("ws://h/websocket?host=0").endswith("host=1")


class TestBuildHostConfigMessage:
    def test_builds_handshake_frame(self):
        msg = build_host_config_message(
            system_prompt="You are an airline agent.",
            tools=[{"type": "function", "name": "t", "description": "d", "parameters": {"type": "object"}}],
            greeting="Thank you for calling.",
            input_rate=16000,
            output_rate=24000,
        )
        assert msg["type"] == "config"
        assert msg["audioFormat"] == "pcm16"
        assert msg["sampleRate"] == 16000
        assert msg["ttsSampleRate"] == 24000
        assert msg["host"]["systemPrompt"] == "You are an airline agent."
        assert msg["host"]["greeting"] == "Thank you for calling."
        assert len(msg["host"]["tools"]) == 1

    def test_omits_greeting_when_empty(self):
        msg = build_host_config_message(system_prompt="p", tools=[], greeting=None, input_rate=16000, output_rate=24000)
        assert "greeting" not in msg["host"]

    def test_is_json_serializable(self):
        msg = build_host_config_message(system_prompt="p", tools=[], greeting="g", input_rate=16000, output_rate=24000)
        assert json.loads(json.dumps(msg)) == msg


class TestMapAaiEvent:
    def test_maps_agent_transcript(self):
        assert map_aai_event(AAIAgentTranscriptEvent(text="hi")) == AssistantTranscript(text="hi")

    def test_maps_user_transcript(self):
        assert map_aai_event(AAIUserTranscriptEvent(text="hello")) == UserTranscript(text="hello")

    def test_maps_tool_call(self):
        event = AAIToolCallEvent(toolCallId="c1", toolName="get_reservation", args={"x": 1})
        assert map_aai_event(event) == ToolCall(call_id="c1", name="get_reservation", arguments={"x": 1})

    def test_maps_speech_started_to_barge_in_signal(self):
        assert map_aai_event(AAISpeechStartedEvent()) == SpeechStarted()

    def test_maps_both_turn_end_signals(self):
        assert map_aai_event(AAIReplyDoneEvent()) == TurnDone()
        assert map_aai_event(AAIAudioDoneEvent()) == TurnDone()

    def test_maps_error_with_code_and_message(self):
        result = map_aai_event(AAIErrorEvent(code="bad_config", message="nope"))
        assert isinstance(result, BackendError)
        assert "bad_config" in result.message
        assert "nope" in result.message
        assert result.fatal is False

    def test_idle_timeout_is_fatal(self):
        result = map_aai_event(AAIIdleTimeoutEvent())
        assert isinstance(result, BackendError)
        assert result.fatal is True

    def test_unmapped_events_return_none(self):
        assert map_aai_event(AAISpeechStoppedEvent()) is None
        assert map_aai_event(AAICustomEvent(event="metric", data={"a": 1})) is None


class _FakeWebSocket:
    """Stands in for a websockets client connection."""

    def __init__(self, inbound=()):
        self.sent: list[object] = []
        self._inbound = list(inbound)
        self.closed = False

    async def send(self, data) -> None:
        self.sent.append(data)

    async def recv(self):
        if not self._inbound:
            raise AssertionError("recv() called with no inbound frames left")
        return self._inbound.pop(0)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._inbound:
            raise StopAsyncIteration
        return self._inbound.pop(0)

    async def close(self) -> None:
        self.closed = True


class TestSessionFraming:
    async def test_send_audio_sends_raw_binary(self):
        ws = _FakeWebSocket()
        session = AAIHostSession(ws, input_rate=16000, output_rate=24000)

        await session.send_audio(b"\x01\x02\x03\x04")

        assert ws.sent == [b"\x01\x02\x03\x04"]

    async def test_send_tool_result_json_encodes_the_result(self):
        ws = _FakeWebSocket()
        session = AAIHostSession(ws, input_rate=16000, output_rate=24000)

        await session.send_tool_result("call_1", {"status": "ok", "seat": "12A"})

        frame = json.loads(ws.sent[0])
        assert frame["type"] == "tool_result"
        assert frame["toolCallId"] == "call_1"
        # aai unwraps a JSON string on the wire, so result must be a string.
        assert json.loads(frame["result"]) == {"status": "ok", "seat": "12A"}

    async def test_send_tool_result_survives_non_serializable_values(self):
        ws = _FakeWebSocket()
        session = AAIHostSession(ws, input_rate=16000, output_rate=24000)

        await session.send_tool_result("call_2", {"when": object()})

        frame = json.loads(ws.sent[0])
        assert frame["toolCallId"] == "call_2"

    async def test_events_yields_audio_for_binary_frames(self):
        ws = _FakeWebSocket([b"\x01\x02", json.dumps({"type": "reply_done"})])
        session = AAIHostSession(ws, input_rate=16000, output_rate=24000)

        events = [event async for event in session.events()]

        assert events[0].pcm == b"\x01\x02"
        assert events[1] == TurnDone()

    async def test_events_skips_unmapped_and_malformed_frames(self):
        ws = _FakeWebSocket(
            [
                "not json at all",
                json.dumps({"type": "speech_stopped"}),
                json.dumps({"type": "agent_transcript", "text": "ok"}),
            ]
        )
        session = AAIHostSession(ws, input_rate=16000, output_rate=24000)

        events = [event async for event in session.events()]

        assert events == [AssistantTranscript(text="ok")]

    async def test_aclose_closes_the_socket(self):
        ws = _FakeWebSocket()
        session = AAIHostSession(ws, input_rate=16000, output_rate=24000)

        await session.aclose()

        assert ws.closed is True

    def test_declares_backend_rates(self):
        session = AAIHostSession(_FakeWebSocket(), input_rate=16000, output_rate=24000)
        assert session.backend_input_rate == 16000
        assert session.backend_output_rate == 24000


class TestHandshake:
    async def test_await_config_ack_accepts_config_event(self):
        ws = _FakeWebSocket([json.dumps({"type": "config"})])
        await AAIHostSession._await_config_ack(ws, timeout_s=1.0)

    async def test_await_config_ack_skips_binary_frames(self):
        ws = _FakeWebSocket([b"\x00\x00", json.dumps({"type": "config"})])
        await AAIHostSession._await_config_ack(ws, timeout_s=1.0)

    async def test_await_config_ack_raises_on_error_event(self):
        ws = _FakeWebSocket([json.dumps({"type": "error", "code": "host_disabled", "message": "AAI_ALLOW_HOST"})])

        with pytest.raises(AAIHostSessionError) as excinfo:
            await AAIHostSession._await_config_ack(ws, timeout_s=1.0)

        assert "AAI_ALLOW_HOST" in str(excinfo.value)


class _FakeResponse:
    """Carries the status code websockets attaches to an InvalidStatus."""

    def __init__(self, status_code: int):
        self.status_code = status_code


class TestConnectAuth:
    """Host mode on a deployed agent authenticates with the owner's API key."""

    async def _connect(self, monkeypatch, *, api_key=None, raises=None):
        captured: dict = {}

        async def fake_connect(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            if raises is not None:
                raise raises
            return _FakeWebSocket([json.dumps({"type": "config"})])

        monkeypatch.setattr("eva.assistant.aai_session.websockets.connect", fake_connect)
        session = await AAIHostSession.connect(
            ws_url="wss://host.example/agent/websocket",
            system_prompt="be helpful",
            tools=[],
            api_key=api_key,
        )
        return session, captured

    async def test_api_key_is_sent_as_a_bearer_header(self, monkeypatch):
        _, captured = await self._connect(monkeypatch, api_key="secret-key")

        assert captured["additional_headers"] == {"Authorization": "Bearer secret-key"}

    async def test_api_key_never_travels_in_the_url(self, monkeypatch):
        _, captured = await self._connect(monkeypatch, api_key="secret-key")

        assert "secret-key" not in captured["url"]

    async def test_no_api_key_sends_no_headers(self, monkeypatch):
        """A local `aai dev` host gates on AAI_ALLOW_HOST and takes no key."""
        _, captured = await self._connect(monkeypatch)

        assert captured["additional_headers"] is None

    async def test_401_explains_the_missing_owner_key(self, monkeypatch):
        error = Exception("server rejected WebSocket connection: HTTP 401")
        error.response = _FakeResponse(401)  # type: ignore[attr-defined]

        with pytest.raises(AAIHostSessionError) as excinfo:
            await self._connect(monkeypatch, raises=error)

        assert "owner's API key" in str(excinfo.value)

    async def test_403_explains_the_key_does_not_own_the_slug(self, monkeypatch):
        error = Exception("server rejected WebSocket connection: HTTP 403")
        error.response = _FakeResponse(403)  # type: ignore[attr-defined]

        with pytest.raises(AAIHostSessionError) as excinfo:
            await self._connect(monkeypatch, api_key="wrong-owner", raises=error)

        assert "does not own this agent slug" in str(excinfo.value)

    async def test_other_connect_errors_pass_through_unembellished(self, monkeypatch):
        with pytest.raises(AAIHostSessionError) as excinfo:
            await self._connect(monkeypatch, raises=OSError("connection refused"))

        assert "connection refused" in str(excinfo.value)
