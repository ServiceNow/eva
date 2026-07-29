# EVA aai Assistant Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let EVA evaluate the aai voice-agent framework by adding an assistant server that bridges EVA's user simulator to an aai host-mode WebSocket, executing all tool calls inside EVA so the scenario database mutates.

**Architecture:** A new `WebSocketBridgeAssistantServer` base owns everything the EVA assistant-server contract requires (Twilio framing, 20 ms output pacing, time-aligned recording buffers, turn bookkeeping, audit and metrics writes). A subclass supplies only a `VoiceBackendSession` — for aai, a WebSocket client that performs the `config.host` handshake and relays `tool_call` → `tool_result`. All five existing servers are left untouched.

**Tech Stack:** Python 3.11–3.13, FastAPI + uvicorn (WebSocket server), `websockets` (client, already a dependency), pydantic v2, pytest with `asyncio_mode = "auto"`.

**Spec:** `docs/superpowers/specs/2026-07-28-eva-aai-assistant-server-design.md`

## Global Constraints

- **Additive only.** Do not modify `pipecat_server.py`, `openai_realtime_server.py`, `gemini_live_server.py`, `elevenlabs_server.py`, `grok_voice_server.py`, or `base_server.py`. Their numbers are on the published leaderboard.
- **`_shutdown()`, not `stop()`.** `AbstractAssistantServer.stop()` is a concrete template method (`base_server.py:138`); `_shutdown()` is the abstract hook (`base_server.py:197`). The contract doc at `docs/assistant_server_contract.md` is stale on this point — follow the code.
- **`_build_system_prompt()` already exists** on the base class (`base_server.py:341`). Do not reimplement it.
- **Recording sample rate is 24000 Hz** for both audio buffers regardless of backend wire rates. Convert inbound μ-law twice — once for the wire, once for the recording buffer — exactly as `gemini_live_server.py:405-410` does.
- **Never call `self.tool_handler.execute()` directly.** Always `await self.execute_tool(...)`, which writes the audit-log entries.
- **Do not write token usage.** aai emits no usage events.
- **Logging** uses `from eva.utils.logging import get_logger`, never `loguru` (tau2's source files use loguru; swap it on port).
- **Line length 120**, matching the existing codebase. Run `uv run ruff check` and `uv run ruff format` before each commit.
- Tests need no `@pytest.mark.asyncio` decorator — `asyncio_mode = "auto"` is set in `pyproject.toml:89`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/eva/assistant/bridge_events.py` *(new)* | Backend-agnostic event dataclasses + the `VoiceBackendSession` protocol. No I/O, no EVA imports. |
| `src/eva/assistant/ws_bridge_server.py` *(new)* | `WebSocketBridgeAssistantServer` — the EVA contract over an abstract backend session. |
| `src/eva/assistant/aai_events.py` *(new)* | aai wire-event pydantic models + `parse_aai_event`. Pure parsing. |
| `src/eva/assistant/aai_session.py` *(new)* | `AAIHostSession` — host handshake, audio frames, `tool_result` relay, wire→bridge event mapping. |
| `src/eva/assistant/aai_server.py` *(new)* | `AAIAssistantServer` — tool-schema construction and backend wiring. Thin. |
| `src/eva/models/config.py:506` *(modify)* | Add `"aai"` to the `framework` Literal. |
| `src/eva/orchestrator/worker.py:26-56` *(modify)* | Register the server in `_get_server_class()`. |
| `.env.example` *(modify)* | Document `AAI_WS_URL`. |
| `docs/aai_integration.md` *(new)* | Setup, run command, known gaps. |
| `tests/unit/assistant/test_aai_events.py` *(new)* | Event parsing. |
| `tests/unit/assistant/test_ws_bridge_server.py` *(new)* | Base-class turn/latency/barge-in/pacing behavior. |
| `tests/unit/assistant/test_aai_session.py` *(new)* | Handshake framing, tool_result framing, event mapping. |
| `tests/unit/assistant/test_aai_server.py` *(new)* | Tool schemas + registration. |
| `tests/integration/test_aai_fake_backend.py` *(new)* | Full session against an in-process fake aai server. |

This splits the seam types into their own module (`bridge_events.py`), a refinement of the spec's four-module list: it keeps `ws_bridge_server.py` focused and lets a future backend import the seam without importing the server.

---

### Task 1: aai wire-event models

**Files:**
- Create: `src/eva/assistant/aai_events.py`
- Test: `tests/unit/assistant/test_aai_events.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `parse_aai_event(data: dict) -> BaseAAIEvent`, and the models
  `AAIConfigEvent`, `AAISpeechStartedEvent`, `AAISpeechStoppedEvent`,
  `AAIUserTranscriptEvent(text, turn_order)`, `AAIAgentTranscriptEvent(text)`,
  `AAIToolCallEvent(tool_call_id, tool_name, args)`,
  `AAIToolCallDoneEvent(tool_call_id, result)`, `AAIReplyDoneEvent`,
  `AAIAudioDoneEvent`, `AAICancelledEvent`, `AAIResetEvent`,
  `AAIIdleTimeoutEvent`, `AAIErrorEvent(code, message)`,
  `AAICustomEvent(event, data)`, `AAIUnknownEvent(type, raw)`.

This is a port of `~/Code/tau2-bench/src/tau2/voice/audio_native/aai/events.py` with three changes: loguru → `eva.utils.logging`, drop `AAITimeoutEvent` and `AAIAudioChunkEvent` (both were artifacts of tau2's tick loop; binary audio is handled out-of-band here), and drop `_prepare_log_data` (no base64 fields exist on this protocol — aai audio is binary frames).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/assistant/test_aai_events.py`:

```python
"""Tests for aai wire-event parsing."""

from eva.assistant.aai_events import (
    AAIAgentTranscriptEvent,
    AAIErrorEvent,
    AAIIdleTimeoutEvent,
    AAIReplyDoneEvent,
    AAISpeechStartedEvent,
    AAIToolCallEvent,
    AAIUnknownEvent,
    AAIUserTranscriptEvent,
    parse_aai_event,
)


class TestParseAaiEvent:
    def test_parses_tool_call_camel_case_aliases(self):
        event = parse_aai_event(
            {
                "type": "tool_call",
                "toolCallId": "call_abc",
                "toolName": "get_reservation",
                "args": {"confirmation_number": "DJ3LPO"},
            }
        )
        assert isinstance(event, AAIToolCallEvent)
        assert event.tool_call_id == "call_abc"
        assert event.tool_name == "get_reservation"
        assert event.args == {"confirmation_number": "DJ3LPO"}

    def test_tool_call_args_default_to_empty_dict(self):
        event = parse_aai_event({"type": "tool_call", "toolCallId": "c1", "toolName": "list_flights"})
        assert isinstance(event, AAIToolCallEvent)
        assert event.args == {}

    def test_parses_agent_transcript(self):
        event = parse_aai_event({"type": "agent_transcript", "text": "How can I help?"})
        assert isinstance(event, AAIAgentTranscriptEvent)
        assert event.text == "How can I help?"

    def test_parses_user_transcript_with_turn_order(self):
        event = parse_aai_event({"type": "user_transcript", "text": "hello", "turnOrder": 3})
        assert isinstance(event, AAIUserTranscriptEvent)
        assert event.text == "hello"
        assert event.turn_order == 3

    def test_parses_zero_field_events(self):
        assert isinstance(parse_aai_event({"type": "speech_started"}), AAISpeechStartedEvent)
        assert isinstance(parse_aai_event({"type": "reply_done"}), AAIReplyDoneEvent)
        assert isinstance(parse_aai_event({"type": "idle_timeout"}), AAIIdleTimeoutEvent)

    def test_parses_error_with_code_and_message(self):
        event = parse_aai_event({"type": "error", "code": "host_disabled", "message": "AAI_ALLOW_HOST"})
        assert isinstance(event, AAIErrorEvent)
        assert event.code == "host_disabled"
        assert event.message == "AAI_ALLOW_HOST"

    def test_ignores_unknown_extra_fields(self):
        event = parse_aai_event({"type": "agent_transcript", "text": "hi", "futureField": 1})
        assert isinstance(event, AAIAgentTranscriptEvent)

    def test_unrecognized_type_returns_unknown_event(self):
        event = parse_aai_event({"type": "brand_new_thing", "x": 1})
        assert isinstance(event, AAIUnknownEvent)
        assert event.type == "brand_new_thing"
        assert event.raw == {"type": "brand_new_thing", "x": 1}

    def test_missing_type_returns_unknown_event(self):
        event = parse_aai_event({"text": "orphan"})
        assert isinstance(event, AAIUnknownEvent)
        assert event.type == "unknown"

    def test_malformed_known_event_returns_unknown_rather_than_raising(self):
        # `text` is required on agent_transcript; a malformed frame must never abort a conversation.
        event = parse_aai_event({"type": "agent_transcript"})
        assert isinstance(event, AAIUnknownEvent)
        assert event.type == "agent_transcript"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/assistant/test_aai_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eva.assistant.aai_events'`

- [ ] **Step 3: Write the implementation**

Create `src/eva/assistant/aai_events.py`:

```python
"""Pydantic models for aai voice-agent wire events.

The aai host sends camelCase JSON frames over the session WebSocket; binary
frames carry PCM16 audio and are handled by the session, not here.

Ported from the tau2-bench aai provider. Parsing never raises: an unrecognized
or malformed frame becomes an ``AAIUnknownEvent`` so a single bad frame cannot
abort a conversation mid-evaluation.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class BaseAAIEvent(BaseModel):
    """Base class for all aai events.

    ``extra="ignore"`` so that new server-side fields never break parsing.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    type: str


class AAIConfigEvent(BaseAAIEvent):
    """Server acknowledgment of the client's config frame — the handshake completing."""

    type: Literal["config"] = "config"


class AAISpeechStartedEvent(BaseAAIEvent):
    """aai's VAD detected the start of user speech."""

    type: Literal["speech_started"] = "speech_started"


class AAISpeechStoppedEvent(BaseAAIEvent):
    """aai's VAD detected the end of user speech."""

    type: Literal["speech_stopped"] = "speech_stopped"


class AAIUserTranscriptEvent(BaseAAIEvent):
    """Transcript of a user turn, as heard by aai."""

    type: Literal["user_transcript"] = "user_transcript"
    text: str
    turn_order: int | None = Field(default=None, alias="turnOrder")


class AAIAgentTranscriptEvent(BaseAAIEvent):
    """Transcript of the agent's reply. Carries the full text, not a delta."""

    type: Literal["agent_transcript"] = "agent_transcript"
    text: str


class AAIToolCallEvent(BaseAAIEvent):
    """A relayed tool call awaiting a ``tool_result`` from this client."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    args: dict = Field(default_factory=dict)


class AAIToolCallDoneEvent(BaseAAIEvent):
    """Acknowledgment that a relayed tool call was resolved."""

    type: Literal["tool_call_done"] = "tool_call_done"
    tool_call_id: str = Field(alias="toolCallId")
    result: str = ""


class AAIReplyDoneEvent(BaseAAIEvent):
    """The agent's reply is complete."""

    type: Literal["reply_done"] = "reply_done"


class AAIAudioDoneEvent(BaseAAIEvent):
    """The agent's audio output is complete."""

    type: Literal["audio_done"] = "audio_done"


class AAICancelledEvent(BaseAAIEvent):
    """An in-flight reply was cancelled."""

    type: Literal["cancelled"] = "cancelled"


class AAIResetEvent(BaseAAIEvent):
    """Session state was reset."""

    type: Literal["reset"] = "reset"


class AAIIdleTimeoutEvent(BaseAAIEvent):
    """The session was closed for inactivity."""

    type: Literal["idle_timeout"] = "idle_timeout"


class AAIErrorEvent(BaseAAIEvent):
    """An error reported by the aai host."""

    type: Literal["error"] = "error"
    code: str | None = None
    message: str | None = None


class AAICustomEvent(BaseAAIEvent):
    """An application-defined event emitted by the agent."""

    type: Literal["custom_event"] = "custom_event"
    event: str
    data: Any | None = None


class AAIUnknownEvent(BaseAAIEvent):
    """An unrecognized or unparseable frame, preserved for logging."""

    type: str
    raw: dict | None = None


_EVENT_TYPE_MAP: dict[str, type[BaseAAIEvent]] = {
    "config": AAIConfigEvent,
    "speech_started": AAISpeechStartedEvent,
    "speech_stopped": AAISpeechStoppedEvent,
    "user_transcript": AAIUserTranscriptEvent,
    "agent_transcript": AAIAgentTranscriptEvent,
    "tool_call": AAIToolCallEvent,
    "tool_call_done": AAIToolCallDoneEvent,
    "reply_done": AAIReplyDoneEvent,
    "audio_done": AAIAudioDoneEvent,
    "cancelled": AAICancelledEvent,
    "reset": AAIResetEvent,
    "idle_timeout": AAIIdleTimeoutEvent,
    "error": AAIErrorEvent,
    "custom_event": AAICustomEvent,
}


def parse_aai_event(data: dict) -> BaseAAIEvent:
    """Parse a raw aai frame into a typed event.

    Never raises. Unknown types and validation failures both yield
    ``AAIUnknownEvent`` with the original payload attached.
    """
    event_type = data.get("type", "unknown")
    event_class = _EVENT_TYPE_MAP.get(event_type)

    if event_class is None:
        return AAIUnknownEvent(type=event_type, raw=data)

    try:
        return event_class.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to parse aai event {event_type}: {e}")
        return AAIUnknownEvent(type=event_type, raw=data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/assistant/test_aai_events.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Lint and commit**

```bash
cd ~/Code/eva
uv run ruff format src/eva/assistant/aai_events.py tests/unit/assistant/test_aai_events.py
uv run ruff check src/eva/assistant/aai_events.py tests/unit/assistant/test_aai_events.py
git add src/eva/assistant/aai_events.py tests/unit/assistant/test_aai_events.py
git commit -m "feat(aai): wire-event models and tolerant parser"
```

---

### Task 2: Backend seam + WebSocket bridge base

**Files:**
- Create: `src/eva/assistant/bridge_events.py`
- Create: `src/eva/assistant/ws_bridge_server.py`
- Test: `tests/unit/assistant/test_ws_bridge_server.py`

**Interfaces:**
- Consumes: `AbstractAssistantServer` (`base_server.py`), the `audio_bridge` helpers.
- Produces:
  - `bridge_events.AudioChunk(pcm: bytes)`, `AssistantTranscript(text: str)`,
    `UserTranscript(text: str)`, `ToolCall(call_id: str, name: str, arguments: dict)`,
    `SpeechStarted()`, `TurnDone()`, `BackendError(message: str, fatal: bool = False)`,
    the `BridgeEvent` union, and the `VoiceBackendSession` protocol.
  - `ws_bridge_server.WebSocketBridgeAssistantServer`, an abstract subclass of
    `AbstractAssistantServer` requiring `model_name: str` (property) and
    `async _open_backend(self) -> VoiceBackendSession`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/assistant/test_ws_bridge_server.py`:

```python
"""Tests for WebSocketBridgeAssistantServer turn, latency, and pacing behavior.

The server is built with ``object.__new__`` and hand-set attributes (the pattern
used by test_openai_realtime_server.py) so tests need no ToolExecutor, no
scenario database, and no live sockets.
"""

import asyncio
from unittest.mock import MagicMock

from eva.assistant.bridge_events import (
    AssistantTranscript,
    AudioChunk,
    BackendError,
    SpeechStarted,
    ToolCall,
    TurnDone,
    UserTranscript,
)
from eva.assistant.ws_bridge_server import WebSocketBridgeAssistantServer


class _FakeSession:
    """In-memory VoiceBackendSession."""

    backend_input_rate = 16000
    backend_output_rate = 24000

    def __init__(self, events=()):
        self._events = list(events)
        self.sent_audio: list[bytes] = []
        self.tool_results: list[tuple[str, object]] = []
        self.closed = False

    async def send_audio(self, pcm: bytes) -> None:
        self.sent_audio.append(pcm)

    async def send_tool_result(self, call_id: str, result: object) -> None:
        self.tool_results.append((call_id, result))

    async def events(self):
        for event in self._events:
            yield event

    async def aclose(self) -> None:
        self.closed = True


class _Server(WebSocketBridgeAssistantServer):
    @property
    def model_name(self) -> str:
        return "test-model"

    async def _open_backend(self):
        return _FakeSession()


def _bare_server(session: _FakeSession | None = None) -> _Server:
    srv = object.__new__(_Server)
    srv._session = session or _FakeSession()
    srv._fw_log = MagicMock()
    srv._metrics_log = MagicMock()
    srv.audit_log = MagicMock()
    srv._audio_out = asyncio.Queue()
    srv._stream_sid = "conv-1"
    srv._running = True
    srv._user_speech_start_ms = None
    srv._user_speech_stop_ms = None
    srv._user_speaking = False
    srv._assistant_speaking = False
    srv._turn_first_audio_ms = None
    srv._assistant_text = ""
    srv._tasks = []
    srv.user_audio_buffer = bytearray()
    srv.assistant_audio_buffer = bytearray()
    srv._audio_sample_rate = 24000
    srv._to_backend = None
    srv._from_backend = lambda pcm: b"\xff" * (len(pcm) // 6)  # 24k PCM16 -> 8k mulaw
    return srv


class TestModelResponseLatency:
    def test_writes_latency_on_first_audio_chunk_of_turn(self):
        srv = _bare_server()
        srv._user_speech_stop_ms = 1_000_000
        srv._now_ms = lambda: 1_000_450

        srv._on_audio_chunk(b"\x00\x00" * 240)

        srv._metrics_log.write_latency.assert_called_once_with("model_response", 0.45, "test-model")

    def test_writes_latency_only_once_per_turn(self):
        srv = _bare_server()
        srv._user_speech_stop_ms = 1_000_000
        srv._now_ms = lambda: 1_000_450

        srv._on_audio_chunk(b"\x00\x00" * 240)
        srv._on_audio_chunk(b"\x00\x00" * 240)

        assert srv._metrics_log.write_latency.call_count == 1

    def test_skips_latency_when_no_user_speech_stop(self):
        """The opening greeting is model-initiated: there is no user turn to measure from."""
        srv = _bare_server()
        srv._now_ms = lambda: 1_000_450

        srv._on_audio_chunk(b"\x00\x00" * 240)

        srv._metrics_log.write_latency.assert_not_called()
        srv._fw_log.turn_start.assert_called_once()

    def test_skips_implausible_latency(self):
        srv = _bare_server()
        srv._user_speech_stop_ms = 1_000_000
        srv._now_ms = lambda: 1_000_000 + 45_000  # 45s, over the 30s ceiling

        srv._on_audio_chunk(b"\x00\x00" * 240)

        srv._metrics_log.write_latency.assert_not_called()

    def test_skips_negative_latency(self):
        srv = _bare_server()
        srv._user_speech_stop_ms = 1_000_000
        srv._now_ms = lambda: 999_000

        srv._on_audio_chunk(b"\x00\x00" * 240)

        srv._metrics_log.write_latency.assert_not_called()


class TestAudioBuffers:
    def test_assistant_audio_appended_and_queued_for_output(self):
        srv = _bare_server()
        srv._now_ms = lambda: 1_000

        srv._on_audio_chunk(b"\x01\x02" * 240)

        assert len(srv.assistant_audio_buffer) == 480
        assert srv._audio_out.qsize() == 1

    def test_assistant_audio_pads_user_track_to_stay_aligned(self):
        srv = _bare_server()
        srv._now_ms = lambda: 1_000
        srv.assistant_audio_buffer.extend(b"\x00" * 960)

        srv._on_audio_chunk(b"\x01\x02" * 240)

        # User track padded up to where the assistant track started.
        assert len(srv.user_audio_buffer) == 960

    def test_no_padding_while_user_is_speaking(self):
        """During overlap both tracks advance on their own; padding would double-count."""
        srv = _bare_server()
        srv._now_ms = lambda: 1_000
        srv._user_speaking = True
        srv.assistant_audio_buffer.extend(b"\x00" * 960)

        srv._on_audio_chunk(b"\x01\x02" * 240)

        assert len(srv.user_audio_buffer) == 0


class TestTurnCompletion:
    def test_finish_turn_writes_transcript_and_turn_end(self):
        srv = _bare_server()
        srv._now_ms = lambda: 5_000
        srv._turn_first_audio_ms = 4_000
        srv._assistant_text = "Your flight is confirmed."

        srv._finish_turn(interrupted=False)

        srv.audit_log.append_assistant_output.assert_called_once_with(
            "Your flight is confirmed.", timestamp_ms="4000"
        )
        srv._fw_log.llm_response.assert_called_once_with("Your flight is confirmed.")
        srv._fw_log.turn_end.assert_called_once_with(was_interrupted=False)

    def test_finish_turn_resets_state_for_next_turn(self):
        srv = _bare_server()
        srv._now_ms = lambda: 5_000
        srv._turn_first_audio_ms = 4_000
        srv._assistant_text = "text"

        srv._finish_turn(interrupted=False)

        assert srv._assistant_text == ""
        assert srv._turn_first_audio_ms is None
        assert srv._assistant_speaking is False

    def test_finish_turn_is_idempotent(self):
        """reply_done and audio_done both map to TurnDone; the second must be a no-op."""
        srv = _bare_server()
        srv._now_ms = lambda: 5_000
        srv._turn_first_audio_ms = 4_000
        srv._assistant_text = "text"

        srv._finish_turn(interrupted=False)
        srv._finish_turn(interrupted=False)

        assert srv.audit_log.append_assistant_output.call_count == 1
        assert srv._fw_log.turn_end.call_count == 1


class TestBargeIn:
    def test_barge_in_drains_queued_audio_and_marks_turn_interrupted(self):
        srv = _bare_server()
        srv._now_ms = lambda: 5_000
        srv._turn_first_audio_ms = 4_000
        srv._assistant_text = "As I was saying"
        srv._audio_out.put_nowait(b"\xff" * 160)
        srv._audio_out.put_nowait(b"\xff" * 160)

        srv._handle_barge_in()

        assert srv._audio_out.qsize() == 0
        srv._fw_log.turn_end.assert_called_once_with(was_interrupted=True)
        srv.audit_log.append_assistant_output.assert_called_once_with(
            "As I was saying [interrupted]", timestamp_ms="4000"
        )

    def test_speech_started_while_assistant_silent_is_not_barge_in(self):
        srv = _bare_server()
        srv._now_ms = lambda: 5_000

        srv._handle_barge_in()

        srv._fw_log.turn_end.assert_not_called()


class TestToolCalls:
    async def test_tool_result_relayed_to_backend(self):
        session = _FakeSession()
        srv = _bare_server(session)
        srv.execute_tool = MagicMock(return_value=_async_value({"status": "ok", "seat": "12A"}))

        await srv._handle_tool_call(ToolCall(call_id="c1", name="assign_seat", arguments={"seat": "12A"}))

        srv.execute_tool.assert_called_once_with("assign_seat", {"seat": "12A"})
        assert session.tool_results == [("c1", {"status": "ok", "seat": "12A"})]

    async def test_tool_failure_relays_error_result_so_the_turn_continues(self):
        session = _FakeSession()
        srv = _bare_server(session)
        srv.execute_tool = MagicMock(side_effect=RuntimeError("db offline"))

        await srv._handle_tool_call(ToolCall(call_id="c2", name="broken", arguments={}))

        assert len(session.tool_results) == 1
        call_id, result = session.tool_results[0]
        assert call_id == "c2"
        assert result["status"] == "error"
        assert "db offline" in result["message"]


class TestEventDispatch:
    async def test_dispatches_each_event_kind(self):
        session = _FakeSession(
            [
                UserTranscript(text="I need to rebook"),
                AudioChunk(pcm=b"\x01\x02" * 240),
                AssistantTranscript(text="Sure, let me look."),
                TurnDone(),
            ]
        )
        srv = _bare_server(session)
        srv._now_ms = lambda: 7_000
        srv._user_speech_start_ms = 6_000

        await srv._process_backend_events()

        srv.audit_log.append_user_input.assert_called_once_with("I need to rebook", timestamp_ms="6000")
        srv.audit_log.append_assistant_output.assert_called_once_with("Sure, let me look.", timestamp_ms="7000")

    async def test_assistant_transcript_overwrites_rather_than_appends(self):
        """aai sends full text per frame, not deltas."""
        session = _FakeSession([AssistantTranscript(text="Hello"), AssistantTranscript(text="Hello there")])
        srv = _bare_server(session)
        srv._now_ms = lambda: 7_000

        await srv._process_backend_events()

        assert srv._assistant_text == "Hello there"

    async def test_fatal_backend_error_stops_dispatch(self):
        session = _FakeSession([BackendError(message="idle_timeout", fatal=True), AssistantTranscript(text="never")])
        srv = _bare_server(session)
        srv._now_ms = lambda: 7_000

        await srv._process_backend_events()

        assert srv._assistant_text == ""


class TestPacing:
    async def test_rechunks_output_to_160_byte_frames(self):
        """Backend audio does not arrive in 160-byte multiples; the simulator requires 20ms frames."""
        srv = _bare_server()
        sent: list[str] = []

        class _WS:
            async def send_text(self, msg: str) -> None:
                sent.append(msg)

        srv._audio_out.put_nowait(b"\xff" * 400)
        task = asyncio.create_task(srv._pace_audio_output(_WS()))
        await asyncio.sleep(0.08)
        srv._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # 400 bytes -> two full 160-byte frames, 80 bytes held back for the next chunk.
        assert len(sent) == 2


def _async_value(value):
    async def _coro():
        return value

    return _coro()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/assistant/test_ws_bridge_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eva.assistant.bridge_events'`

- [ ] **Step 3: Write the seam types**

Create `src/eva/assistant/bridge_events.py`:

```python
"""Backend-agnostic events and the session protocol used by the bridge server.

A voice backend (aai host mode, a hosted realtime API, …) speaks its own wire
protocol. ``WebSocketBridgeAssistantServer`` never sees that protocol: an
adapter translates it into the small event vocabulary below, and the bridge
turns those events into what EVA's evaluation contract requires.

Keeping this module free of EVA imports means a new backend adapter can be
written and unit-tested without pulling in the server.
"""

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class AudioChunk:
    """PCM16 audio from the backend, at its declared ``backend_output_rate``."""

    pcm: bytes


@dataclass(frozen=True)
class AssistantTranscript:
    """The assistant's reply text.

    Full-text semantics: each event supersedes the previous one for the current
    turn rather than appending to it.
    """

    text: str


@dataclass(frozen=True)
class UserTranscript:
    """A user turn as transcribed by the backend."""

    text: str


@dataclass(frozen=True)
class ToolCall:
    """A tool the backend wants executed. The bridge runs it and replies."""

    call_id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class SpeechStarted:
    """Backend VAD detected user speech. Triggers barge-in if the assistant is speaking."""


@dataclass(frozen=True)
class TurnDone:
    """The assistant's turn is complete. Safe to deliver more than once per turn."""


@dataclass(frozen=True)
class BackendError:
    """A backend-reported problem. ``fatal`` ends the session."""

    message: str
    fatal: bool = False


BridgeEvent = (
    AudioChunk | AssistantTranscript | UserTranscript | ToolCall | SpeechStarted | TurnDone | BackendError
)


@runtime_checkable
class VoiceBackendSession(Protocol):
    """One live conversation with a voice backend.

    Implementations own their wire protocol and nothing else: no Twilio framing,
    no recording buffers, no audit logging.
    """

    #: Sample rate of the PCM16 audio this session expects from ``send_audio``.
    backend_input_rate: int
    #: Sample rate of the PCM16 audio this session emits in ``AudioChunk``.
    backend_output_rate: int

    async def send_audio(self, pcm: bytes) -> None:
        """Send PCM16 user audio at ``backend_input_rate``."""
        ...

    async def send_tool_result(self, call_id: str, result: Any) -> None:
        """Resolve the relayed tool call identified by ``call_id``."""
        ...

    def events(self) -> AsyncIterator[BridgeEvent]:
        """Yield events until the backend closes the session."""
        ...

    async def aclose(self) -> None:
        """Close the session. Must be safe to call more than once."""
        ...
```

- [ ] **Step 4: Write the bridge server**

Create `src/eva/assistant/ws_bridge_server.py`:

```python
"""Shared base for assistant servers that bridge EVA to a WebSocket voice backend.

EVA's user simulator dials in over a Twilio-framed WebSocket; a voice backend
speaks its own protocol on its own socket. Every such integration needs the same
middle layer, and getting any part of it wrong corrupts the evaluation rather
than merely degrading audio:

* **Output pacing.** The simulator infers turn boundaries from arrival timing,
  so assistant audio must leave at wall-clock 20 ms / 160-byte cadence.
* **Track alignment.** The two recording buffers form a shared timeline; each
  must be padded to the other's position before it advances, or the mixed WAV
  is skewed and every audio judge metric reads the wrong thing.
* **Recording rate.** Both buffers are written at 24 kHz regardless of the
  backend's wire rates, so inbound mu-law is converted twice: once for the wire,
  once for the recording buffer.

Subclasses supply only a ``VoiceBackendSession`` and a model identifier.

See docs/assistant_server_contract.md for the contract this satisfies. Note that
the doc is stale on shutdown: ``AbstractAssistantServer.stop()`` is a concrete
template method and ``_shutdown()`` is the hook implemented here.
"""

import asyncio
import contextlib
import json
import time
from abc import abstractmethod
from typing import Callable

import uvicorn
from fastapi import FastAPI, WebSocket

from eva.assistant.audio_bridge import (
    FrameworkLogWriter,
    MetricsLogWriter,
    create_twilio_media_message,
    mulaw_8k_to_pcm16_16k,
    mulaw_8k_to_pcm16_24k,
    parse_twilio_media_message,
    pcm16_24k_to_mulaw_8k,
    sync_buffer_to_position,
)
from eva.assistant.base_server import AbstractAssistantServer
from eva.assistant.bridge_events import (
    AssistantTranscript,
    AudioChunk,
    BackendError,
    SpeechStarted,
    ToolCall,
    TurnDone,
    UserTranscript,
    VoiceBackendSession,
)
from eva.utils.logging import get_logger

logger = get_logger(__name__)

#: Both recording buffers are written at this rate, whatever the backend uses.
RECORDING_SAMPLE_RATE = 24000

#: mu-law bytes per 20 ms frame at 8 kHz — one Twilio media message.
MULAW_FRAME_BYTES = 160
FRAME_INTERVAL_S = 0.02

#: A model_response latency outside this bound indicates a bookkeeping bug, not a slow model.
MAX_PLAUSIBLE_LATENCY_MS = 30_000

#: mu-law 8 kHz -> PCM16 at the backend's input rate.
_TO_BACKEND: dict[int, Callable[[bytes], bytes]] = {
    16000: mulaw_8k_to_pcm16_16k,
    24000: mulaw_8k_to_pcm16_24k,
}

#: PCM16 at the backend's output rate -> mu-law 8 kHz.
_FROM_BACKEND: dict[int, Callable[[bytes], bytes]] = {
    24000: pcm16_24k_to_mulaw_8k,
}


class WebSocketBridgeAssistantServer(AbstractAssistantServer):
    """Bridges the user simulator's Twilio socket to a ``VoiceBackendSession``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._audio_sample_rate = RECORDING_SAMPLE_RATE

        self._session: VoiceBackendSession | None = None
        self._to_backend: Callable[[bytes], bytes] | None = None
        self._from_backend: Callable[[bytes], bytes] | None = None

        self._stream_sid: str = self.conversation_id
        self._audio_out: asyncio.Queue[bytes] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

        # Turn bookkeeping
        self._user_speech_start_ms: int | None = None
        self._user_speech_stop_ms: int | None = None
        self._user_speaking = False
        self._assistant_speaking = False
        self._turn_first_audio_ms: int | None = None
        self._assistant_text = ""

    # ── Subclass contract ─────────────────────────────────────────────

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier recorded in the metrics log."""
        ...

    @abstractmethod
    async def _open_backend(self) -> VoiceBackendSession:
        """Open a session with the backend. Raise to abort the conversation."""
        ...

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)

        self._app = FastAPI()

        @self._app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        @self._app.websocket("/")
        async def ws_root(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        config = uvicorn.Config(self._app, host="0.0.0.0", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            await asyncio.sleep(0.05)
        self._running = True
        logger.info(f"{type(self).__name__} listening on ws://localhost:{self.port}/ws")

    async def _shutdown(self) -> None:
        self._running = False

        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._session.aclose()
            self._session = None

        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            self._server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_task

    # ── Session ───────────────────────────────────────────────────────

    async def _handle_session(self, websocket: WebSocket) -> None:
        """Run one conversation: simulator socket in, backend socket out."""
        try:
            self._session = await self._open_backend()
        except Exception as e:
            # Loud failure: a silent fallback would yield a scored but meaningless run.
            logger.error(f"Failed to open backend session, aborting conversation: {e}", exc_info=True)
            with contextlib.suppress(Exception):
                await websocket.close()
            return

        self._to_backend = self._converter(_TO_BACKEND, self._session.backend_input_rate, "input")
        self._from_backend = self._converter(_FROM_BACKEND, self._session.backend_output_rate, "output")

        self._tasks = [
            asyncio.create_task(self._forward_user_audio(websocket)),
            asyncio.create_task(self._process_backend_events()),
            asyncio.create_task(self._pace_audio_output(websocket)),
        ]
        done, pending = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            if task.exception() is not None:
                logger.error(f"Bridge task failed: {task.exception()}")

    @staticmethod
    def _converter(table: dict[int, Callable[[bytes], bytes]], rate: int, direction: str) -> Callable[[bytes], bytes]:
        converter = table.get(rate)
        if converter is None:
            raise ValueError(
                f"No mu-law converter for backend {direction} rate {rate} Hz "
                f"(available: {sorted(table)}). Add one to audio_bridge.py."
            )
        return converter

    async def _forward_user_audio(self, websocket: WebSocket) -> None:
        """Simulator -> backend, plus the user recording track."""
        while self._running:
            try:
                raw = await websocket.receive_text()
            except Exception:
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event = msg.get("event")

            if event == "start":
                self._stream_sid = msg.get("start", {}).get("streamSid", self._stream_sid)
            elif event == "stop":
                break
            elif event == "user_speech_start":
                self._user_speech_start_ms = _as_int_ms(msg.get("timestamp_ms"))
                self._user_speaking = True
            elif event == "user_speech_stop":
                self._user_speech_stop_ms = _as_int_ms(msg.get("timestamp_ms"))
                self._user_speaking = False
            elif event == "media":
                mulaw = parse_twilio_media_message(raw)
                if not mulaw:
                    continue
                await self._session.send_audio(self._to_backend(mulaw))
                self._record_user_audio(mulaw)

    def _record_user_audio(self, mulaw: bytes) -> None:
        """Append to the user track at the recording rate, keeping tracks aligned."""
        pcm = mulaw_8k_to_pcm16_24k(mulaw)
        if not self._assistant_speaking:
            sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
        self.user_audio_buffer.extend(pcm)

    async def _process_backend_events(self) -> None:
        """Backend -> audit log, metrics, recording track, and output queue."""
        async for event in self._session.events():
            if isinstance(event, AudioChunk):
                self._on_audio_chunk(event.pcm)
            elif isinstance(event, AssistantTranscript):
                self._assistant_text = event.text
            elif isinstance(event, UserTranscript):
                timestamp = str(self._user_speech_start_ms or self._now_ms())
                self.audit_log.append_user_input(event.text, timestamp_ms=timestamp)
            elif isinstance(event, ToolCall):
                await self._handle_tool_call(event)
            elif isinstance(event, SpeechStarted):
                self._handle_barge_in()
            elif isinstance(event, TurnDone):
                self._finish_turn(interrupted=False)
            elif isinstance(event, BackendError):
                logger.error(f"Backend error: {event.message}")
                if event.fatal:
                    break

    def _on_audio_chunk(self, pcm: bytes) -> None:
        """Record assistant audio, start the turn if needed, and queue for paced send."""
        if self._turn_first_audio_ms is None:
            self._turn_first_audio_ms = self._now_ms()
            self._fw_log.turn_start(timestamp_ms=self._turn_first_audio_ms)
            self._write_model_response_latency(self._turn_first_audio_ms)

        self._assistant_speaking = True
        if not self._user_speaking:
            sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
        self.assistant_audio_buffer.extend(pcm)
        self._audio_out.put_nowait(self._from_backend(pcm))

    def _write_model_response_latency(self, first_audio_ms: int) -> None:
        """Time from the simulator's user_speech_stop to our first audio byte.

        The simulator's VAD is the contract-specified source. The backend's own
        speech events drive barge-in only; mixing the two skews this metric.
        """
        if self._user_speech_stop_ms is None:
            return  # Model-initiated turn, e.g. the opening greeting.

        latency_ms = first_audio_ms - self._user_speech_stop_ms
        if 0 < latency_ms < MAX_PLAUSIBLE_LATENCY_MS:
            self._metrics_log.write_latency("model_response", latency_ms / 1000, self.model_name)
        else:
            logger.warning(f"Discarding implausible model_response latency: {latency_ms} ms")
        self._user_speech_stop_ms = None

    def _finish_turn(self, interrupted: bool) -> None:
        """Close out the assistant turn. Idempotent — both reply_done and audio_done arrive."""
        if self._turn_first_audio_ms is None and not self._assistant_text:
            return

        text = self._assistant_text
        if interrupted and text:
            text = f"{text} [interrupted]"

        if text:
            timestamp = str(self._turn_first_audio_ms or self._now_ms())
            self.audit_log.append_assistant_output(text, timestamp_ms=timestamp)
            self._fw_log.llm_response(text)
            self._fw_log.s2s_transcript(text)
        self._fw_log.turn_end(was_interrupted=interrupted)

        self._assistant_text = ""
        self._turn_first_audio_ms = None
        self._assistant_speaking = False

    def _handle_barge_in(self) -> None:
        """Drop undelivered assistant audio and close the turn as interrupted."""
        if self._turn_first_audio_ms is None and not self._assistant_text:
            return  # Assistant was not speaking: an ordinary user turn starting.

        dropped = 0
        while not self._audio_out.empty():
            self._audio_out.get_nowait()
            dropped += 1
        if dropped:
            logger.debug(f"Barge-in: dropped {dropped} queued audio frames")
        self._finish_turn(interrupted=True)

    async def _handle_tool_call(self, event: ToolCall) -> None:
        """Execute against the scenario database and relay the result back."""
        try:
            result = await self.execute_tool(event.name, event.arguments)
        except Exception as e:
            logger.error(f"Tool {event.name} raised: {e}", exc_info=True)
            result = {"status": "error", "message": str(e)}
        await self._session.send_tool_result(event.call_id, result)

    async def _pace_audio_output(self, websocket: WebSocket) -> None:
        """Drain the output queue at 20 ms per 160-byte frame.

        Backend audio does not arrive in 160-byte multiples once converted, so
        frames are re-chunked here. Sending faster or slower than real time makes
        the simulator misjudge turn boundaries.
        """
        pending = bytearray()
        next_send = time.monotonic()

        while self._running:
            chunk = await self._audio_out.get()
            pending.extend(chunk)

            while len(pending) >= MULAW_FRAME_BYTES:
                frame = bytes(pending[:MULAW_FRAME_BYTES])
                del pending[:MULAW_FRAME_BYTES]
                try:
                    await websocket.send_text(create_twilio_media_message(self._stream_sid, frame))
                except Exception:
                    return

                next_send += FRAME_INTERVAL_S
                sleep_s = next_send - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                else:
                    # Fell behind (or the queue was idle): resync rather than burst.
                    next_send = time.monotonic()

    @staticmethod
    def _now_ms() -> int:
        """Wall-clock milliseconds. Overridden in tests."""
        return int(time.time() * 1000)


def _as_int_ms(value: object) -> int | None:
    """Coerce a simulator timestamp to int milliseconds, tolerating str or float."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/assistant/test_ws_bridge_server.py -v`
Expected: PASS (17 tests)

- [ ] **Step 6: Lint and commit**

```bash
cd ~/Code/eva
uv run ruff format src/eva/assistant/bridge_events.py src/eva/assistant/ws_bridge_server.py tests/unit/assistant/test_ws_bridge_server.py
uv run ruff check src/eva/assistant/bridge_events.py src/eva/assistant/ws_bridge_server.py tests/unit/assistant/test_ws_bridge_server.py
git add src/eva/assistant/bridge_events.py src/eva/assistant/ws_bridge_server.py tests/unit/assistant/test_ws_bridge_server.py
git commit -m "feat(assistant): shared WebSocket bridge base for voice backends"
```

---

### Task 3: aai host session

**Files:**
- Create: `src/eva/assistant/aai_session.py`
- Test: `tests/unit/assistant/test_aai_session.py`

**Interfaces:**
- Consumes: `parse_aai_event` and the event models from Task 1; the bridge event
  dataclasses from Task 2.
- Produces:
  - `DEFAULT_AAI_WS_URL = "ws://localhost:3000/websocket"`,
    `DEFAULT_AAI_INPUT_SAMPLE_RATE = 16000`, `DEFAULT_AAI_OUTPUT_SAMPLE_RATE = 24000`,
    `DEFAULT_AAI_MODEL = "aai-host"`.
  - `with_host_flag(url: str) -> str`
  - `build_host_config_message(*, system_prompt: str, tools: list[dict], greeting: str | None, input_rate: int, output_rate: int) -> dict`
  - `map_aai_event(event: BaseAAIEvent) -> BridgeEvent | None`
  - `AAIHostSessionError(Exception)`
  - `AAIHostSession` with `connect(...)` classmethod, satisfying `VoiceBackendSession`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/assistant/test_aai_session.py`:

```python
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
        msg = build_host_config_message(
            system_prompt="p", tools=[], greeting=None, input_rate=16000, output_rate=24000
        )
        assert "greeting" not in msg["host"]

    def test_is_json_serializable(self):
        msg = build_host_config_message(
            system_prompt="p", tools=[], greeting="g", input_rate=16000, output_rate=24000
        )
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
        from eva.assistant.aai_session import AAIHostSessionError

        ws = _FakeWebSocket([json.dumps({"type": "error", "code": "host_disabled", "message": "AAI_ALLOW_HOST"})])

        with pytest.raises(AAIHostSessionError) as excinfo:
            await AAIHostSession._await_config_ack(ws, timeout_s=1.0)

        assert "AAI_ALLOW_HOST" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/assistant/test_aai_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eva.assistant.aai_session'`

- [ ] **Step 3: Write the implementation**

Create `src/eva/assistant/aai_session.py`:

```python
"""aai host-mode session: the wire protocol behind AAIAssistantServer.

aai agents normally execute their tools server-side in a sandbox. Host mode
inverts that: this client supplies the system prompt, greeting, and tool
*schemas* in its first ``config`` frame, and aai relays every tool *call* back
as a ``tool_call`` frame that resolves when the matching ``tool_result``
arrives. That inversion is what lets EVA's ToolExecutor mutate the scenario
database — without it the accuracy metrics would score an unchanged database.

Host mode is selected by the ``?host=1`` query parameter and gated server-side
by ``AAI_ALLOW_HOST``.
"""

import asyncio
import contextlib
import json
from typing import Any, AsyncIterator
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import websockets

from eva.assistant.aai_events import (
    AAIAgentTranscriptEvent,
    AAIAudioDoneEvent,
    AAIConfigEvent,
    AAIErrorEvent,
    AAIIdleTimeoutEvent,
    AAIReplyDoneEvent,
    AAISpeechStartedEvent,
    AAIToolCallEvent,
    AAIUserTranscriptEvent,
    BaseAAIEvent,
    parse_aai_event,
)
from eva.assistant.bridge_events import (
    AssistantTranscript,
    AudioChunk,
    BackendError,
    BridgeEvent,
    SpeechStarted,
    ToolCall,
    TurnDone,
    UserTranscript,
)
from eva.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_AAI_WS_URL = "ws://localhost:3000/websocket"
DEFAULT_AAI_INPUT_SAMPLE_RATE = 16000
DEFAULT_AAI_OUTPUT_SAMPLE_RATE = 24000

#: aai host mode is endpoint-determined: the model is whatever the host runs.
DEFAULT_AAI_MODEL = "aai-host"

CONFIG_ACK_TIMEOUT_S = 15.0


class AAIHostSessionError(Exception):
    """The aai host refused or failed the host-mode handshake."""


def with_host_flag(url: str) -> str:
    """Return *url* with ``host=1``, activating host mode on the aai server."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    params["host"] = ["1"]
    return urlunparse(parsed._replace(query=urlencode(params, doseq=True)))


def build_host_config_message(
    *,
    system_prompt: str,
    tools: list[dict],
    greeting: str | None,
    input_rate: int,
    output_rate: int,
) -> dict:
    """Build the host-mode handshake frame.

    Mirrors ``HostConfigMessageSchema`` in the aai SDK: the audio negotiation
    fields ride alongside the ``host`` block in a single ``config`` frame.
    """
    host: dict[str, Any] = {"systemPrompt": system_prompt, "tools": list(tools)}
    if greeting:
        host["greeting"] = greeting
    return {
        "type": "config",
        "audioFormat": "pcm16",
        "sampleRate": input_rate,
        "ttsSampleRate": output_rate,
        "host": host,
    }


def map_aai_event(event: BaseAAIEvent) -> BridgeEvent | None:
    """Translate an aai wire event into a bridge event, or None to ignore it.

    Both ``reply_done`` and ``audio_done`` map to ``TurnDone``: audio completion
    is the truer end-of-turn signal for a voice call, but a text-only reply
    produces no ``audio_done``, so honoring both avoids a stuck turn. The
    bridge's turn completion is idempotent, making the duplicate harmless.
    """
    if isinstance(event, AAIAgentTranscriptEvent):
        return AssistantTranscript(text=event.text)
    if isinstance(event, AAIUserTranscriptEvent):
        return UserTranscript(text=event.text)
    if isinstance(event, AAIToolCallEvent):
        return ToolCall(call_id=event.tool_call_id, name=event.tool_name, arguments=event.args)
    if isinstance(event, AAISpeechStartedEvent):
        return SpeechStarted()
    if isinstance(event, (AAIReplyDoneEvent, AAIAudioDoneEvent)):
        return TurnDone()
    if isinstance(event, AAIIdleTimeoutEvent):
        return BackendError(message="aai session idle timeout", fatal=True)
    if isinstance(event, AAIErrorEvent):
        return BackendError(message=f"{event.code or 'error'}: {event.message or ''}".strip())
    return None


class AAIHostSession:
    """One host-mode conversation with an aai voice agent.

    Satisfies ``VoiceBackendSession``. Audio is raw binary PCM16 frames;
    everything else is JSON.
    """

    def __init__(self, ws, input_rate: int, output_rate: int):
        self._ws = ws
        self.backend_input_rate = input_rate
        self.backend_output_rate = output_rate

    @classmethod
    async def connect(
        cls,
        *,
        ws_url: str,
        system_prompt: str,
        tools: list[dict],
        greeting: str | None = None,
        input_rate: int = DEFAULT_AAI_INPUT_SAMPLE_RATE,
        output_rate: int = DEFAULT_AAI_OUTPUT_SAMPLE_RATE,
    ) -> "AAIHostSession":
        """Open a host-mode session and complete the handshake.

        Raises:
            AAIHostSessionError: the host rejected the handshake, or did not
                acknowledge it within ``CONFIG_ACK_TIMEOUT_S``.
        """
        url = with_host_flag(ws_url)
        logger.info(f"Connecting to aai host at {url} ({len(tools)} tools)")
        try:
            # max_size=None: TTS audio frames can exceed the 1 MiB default.
            ws = await websockets.connect(url, max_size=None)
        except Exception as e:
            raise AAIHostSessionError(f"Could not connect to aai host at {url}: {e}") from e

        try:
            await ws.send(
                json.dumps(
                    build_host_config_message(
                        system_prompt=system_prompt,
                        tools=tools,
                        greeting=greeting,
                        input_rate=input_rate,
                        output_rate=output_rate,
                    )
                )
            )
            await cls._await_config_ack(ws, timeout_s=CONFIG_ACK_TIMEOUT_S)
        except Exception:
            with contextlib.suppress(Exception):
                await ws.close()
            raise

        logger.info("aai host-mode handshake complete")
        return cls(ws, input_rate=input_rate, output_rate=output_rate)

    @staticmethod
    async def _await_config_ack(ws, timeout_s: float) -> None:
        """Block until the host acknowledges the config frame."""
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            except asyncio.TimeoutError as e:
                raise AAIHostSessionError(
                    f"aai host did not acknowledge the host-mode config within {timeout_s}s "
                    f"(is AAI_ALLOW_HOST set?)"
                ) from e

            if isinstance(raw, (bytes, bytearray)):
                continue  # Audio can precede the ack; ignore it.

            try:
                event = parse_aai_event(json.loads(raw))
            except json.JSONDecodeError:
                continue

            if isinstance(event, AAIConfigEvent):
                return
            if isinstance(event, AAIErrorEvent):
                raise AAIHostSessionError(f"aai host rejected host mode: {event.code}: {event.message}")

    async def send_audio(self, pcm: bytes) -> None:
        await self._ws.send(pcm)

    async def send_tool_result(self, call_id: str, result: Any) -> None:
        """Relay a tool result. ``result`` travels as a JSON string, which aai unwraps."""
        await self._ws.send(
            json.dumps(
                {
                    "type": "tool_result",
                    "toolCallId": call_id,
                    "result": json.dumps(result, default=str),
                }
            )
        )

    async def events(self) -> AsyncIterator[BridgeEvent]:
        async for raw in self._ws:
            if isinstance(raw, (bytes, bytearray)):
                yield AudioChunk(pcm=bytes(raw))
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Discarding non-JSON text frame from aai host")
                continue
            mapped = map_aai_event(parse_aai_event(data))
            if mapped is not None:
                yield mapped

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self._ws.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/assistant/test_aai_session.py -v`
Expected: PASS (22 tests)

- [ ] **Step 5: Lint and commit**

```bash
cd ~/Code/eva
uv run ruff format src/eva/assistant/aai_session.py tests/unit/assistant/test_aai_session.py
uv run ruff check src/eva/assistant/aai_session.py tests/unit/assistant/test_aai_session.py
git add src/eva/assistant/aai_session.py tests/unit/assistant/test_aai_session.py
git commit -m "feat(aai): host-mode session client with tool relay"
```

---

### Task 4: aai assistant server and registration

**Files:**
- Create: `src/eva/assistant/aai_server.py`
- Modify: `src/eva/models/config.py:506`
- Modify: `src/eva/orchestrator/worker.py:32-56`
- Test: `tests/unit/assistant/test_aai_server.py`

**Interfaces:**
- Consumes: `WebSocketBridgeAssistantServer` (Task 2), `AAIHostSession` and the
  `DEFAULT_AAI_*` constants (Task 3), `AgentConfig.tools` items which expose
  `function_name`, `name`, `description`, `get_parameter_properties()`, and
  `get_required_param_names()`.
- Produces: `AAIAssistantServer`, selected by `framework: "aai"`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/assistant/test_aai_server.py`:

```python
"""Tests for AAIAssistantServer configuration, tool schemas, and registration."""

from unittest.mock import MagicMock

from eva.assistant.aai_server import AAIAssistantServer
from eva.assistant.aai_session import DEFAULT_AAI_MODEL, DEFAULT_AAI_WS_URL


def _bare_server(s2s_params: dict | None = None) -> AAIAssistantServer:
    """Construct without __init__ (skips PromptManager and ToolExecutor setup)."""
    srv = object.__new__(AAIAssistantServer)
    srv.pipeline_config = MagicMock()
    srv.pipeline_config.s2s_params = s2s_params if s2s_params is not None else {}
    srv.agent = MagicMock()
    srv.agent.tools = []
    srv._model = (s2s_params or {}).get("model", DEFAULT_AAI_MODEL)
    srv._ws_url = (s2s_params or {}).get("ws_url", DEFAULT_AAI_WS_URL)
    return srv


def _tool(function_name: str, name: str, description: str, properties: dict, required: list[str]) -> MagicMock:
    tool = MagicMock()
    tool.function_name = function_name
    tool.name = name
    tool.description = description
    tool.get_parameter_properties.return_value = properties
    tool.get_required_param_names.return_value = required
    return tool


class TestModelName:
    def test_defaults_to_aai_host(self):
        assert _bare_server().model_name == DEFAULT_AAI_MODEL

    def test_honors_configured_model(self):
        assert _bare_server({"model": "aai-host-custom"}).model_name == "aai-host-custom"


class TestBuildAaiTools:
    def test_returns_empty_list_when_agent_has_no_tools(self):
        assert _bare_server()._build_aai_tools() == []

    def test_builds_flat_function_schema(self):
        srv = _bare_server()
        srv.agent.tools = [
            _tool(
                "get_reservation",
                "Get Reservation",
                "Look up a booking",
                {"confirmation_number": {"type": "string"}},
                ["confirmation_number"],
            )
        ]

        tools = srv._build_aai_tools()

        assert tools == [
            {
                "type": "function",
                "name": "get_reservation",
                "description": "Get Reservation: Look up a booking",
                "parameters": {
                    "type": "object",
                    "properties": {"confirmation_number": {"type": "string"}},
                    "required": ["confirmation_number"],
                },
            }
        ]

    def test_description_is_never_empty(self):
        """aai's ToolSchema requires a description of at least one character."""
        srv = _bare_server()
        srv.agent.tools = [_tool("f", "", "", {}, [])]

        description = srv._build_aai_tools()[0]["description"]

        assert len(description) >= 1


class TestWebSocketUrl:
    def test_defaults_to_localhost(self):
        assert _bare_server()._ws_url == DEFAULT_AAI_WS_URL

    def test_honors_configured_url(self):
        assert _bare_server({"ws_url": "ws://aai.internal/websocket"})._ws_url == "ws://aai.internal/websocket"


class TestRegistration:
    def test_framework_literal_accepts_aai(self):
        from eva.models.config import RunConfig

        assert "aai" in RunConfig.model_fields["framework"].annotation.__args__

    def test_worker_resolves_the_aai_server_class(self):
        from eva.orchestrator.worker import _get_server_class

        assert _get_server_class("aai") is AAIAssistantServer

    def test_unknown_framework_error_lists_aai(self):
        from eva.orchestrator.worker import _get_server_class

        try:
            _get_server_class("nope")
        except ValueError as e:
            assert "aai" in str(e)
        else:
            raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/assistant/test_aai_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eva.assistant.aai_server'`

- [ ] **Step 3: Write the server**

Create `src/eva/assistant/aai_server.py`:

```python
"""EVA assistant server for the aai voice-agent framework.

Bridges EVA's user simulator to an aai host-mode session. All the contract
plumbing lives in ``WebSocketBridgeAssistantServer``; this class only builds the
per-session agent definition — system prompt, greeting, and tool schemas — and
opens the backend.

Requires a running aai host with ``AAI_ALLOW_HOST`` enabled. See
docs/aai_integration.md.
"""

import os

from eva.assistant.aai_session import (
    DEFAULT_AAI_INPUT_SAMPLE_RATE,
    DEFAULT_AAI_MODEL,
    DEFAULT_AAI_OUTPUT_SAMPLE_RATE,
    DEFAULT_AAI_WS_URL,
    AAIHostSession,
)
from eva.assistant.bridge_events import VoiceBackendSession
from eva.assistant.ws_bridge_server import WebSocketBridgeAssistantServer
from eva.utils.logging import get_logger

logger = get_logger(__name__)


class AAIAssistantServer(WebSocketBridgeAssistantServer):
    """Runs an EVA conversation against an aai host-mode agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        s2s_params = self.pipeline_config.s2s_params or {}
        self._model: str = s2s_params.get("model") or DEFAULT_AAI_MODEL
        self._ws_url: str = s2s_params.get("ws_url") or os.environ.get("AAI_WS_URL") or DEFAULT_AAI_WS_URL
        self._input_rate: int = int(s2s_params.get("input_sample_rate") or DEFAULT_AAI_INPUT_SAMPLE_RATE)
        self._output_rate: int = int(s2s_params.get("output_sample_rate") or DEFAULT_AAI_OUTPUT_SAMPLE_RATE)

    @property
    def model_name(self) -> str:
        return self._model

    def _build_aai_tools(self) -> list[dict]:
        """Convert the agent's tools to aai's flat ToolSchema shape.

        aai validates ``{type, name, description, parameters}`` with a non-empty
        description and a JSON Schema object for parameters.
        """
        tools: list[dict] = []
        for tool in self.agent.tools or []:
            description = f"{tool.name}: {tool.description}".strip(": ").strip() or tool.function_name
            tools.append(
                {
                    "type": "function",
                    "name": tool.function_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.get_parameter_properties(),
                        "required": tool.get_required_param_names(),
                    },
                }
            )
        return tools

    async def _open_backend(self) -> VoiceBackendSession:
        return await AAIHostSession.connect(
            ws_url=self._ws_url,
            system_prompt=self._build_system_prompt(),
            tools=self._build_aai_tools(),
            greeting=self.initial_message,
            input_rate=self._input_rate,
            output_rate=self._output_rate,
        )
```

- [ ] **Step 4: Register the framework**

In `src/eva/models/config.py`, replace the `framework` field at line 506:

```python
    framework: Literal["pipecat", "openai_realtime", "gemini_live", "elevenlabs", "grok_voice", "aai"] = Field(
        "pipecat",
        description=(
            "Agent framework to use for the assistant server."
            "'pipecat' (default): Pipecat pipeline."
            "'openai_realtime': OpenAI Realtime API directly."
            "'gemini_live': Gemini Live API via google-genai."
            "'elevenlabs': ElevenLabs Conversational AI API."
            "'grok_voice': xAI Grok voice realtime API."
            "'aai': aai voice-agent framework in host mode."
        ),
```

In `src/eva/orchestrator/worker.py`, add a branch before the `else` in `_get_server_class()` and update the error message:

```python
    elif framework == "aai":
        from eva.assistant.aai_server import AAIAssistantServer

        return AAIAssistantServer
    else:
        raise ValueError(
            f"Unknown framework: {framework!r}. "
            "Supported: pipecat, openai_realtime, gemini_live, elevenlabs, grok_voice, aai"
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/assistant/test_aai_server.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Verify nothing else regressed**

Run: `uv run pytest tests/unit -q`
Expected: PASS — no pre-existing test changes behavior.

- [ ] **Step 7: Lint and commit**

```bash
cd ~/Code/eva
uv run ruff format src/eva/assistant/aai_server.py tests/unit/assistant/test_aai_server.py
uv run ruff check src/eva/assistant/aai_server.py src/eva/models/config.py src/eva/orchestrator/worker.py
git add src/eva/assistant/aai_server.py src/eva/models/config.py src/eva/orchestrator/worker.py tests/unit/assistant/test_aai_server.py
git commit -m "feat(aai): assistant server and framework registration"
```

---

### Task 5: Full-session test against a fake aai backend

**Files:**
- Test: `tests/integration/test_aai_fake_backend.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: no source changes. This is the verification gate — it proves the two
  hops connect, a tool call round-trips, and audio flows both ways, with no paid
  APIs and no Node process.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_aai_fake_backend.py`:

```python
"""End-to-end bridge test against an in-process fake aai host.

Exercises both WebSocket hops without any paid API or Node process:
a fake user simulator sends Twilio-framed mu-law audio to AAIAssistantServer,
which speaks host mode to a fake aai server that requests a tool and replies
with audio.
"""

import asyncio
import audioop
import json
from unittest.mock import MagicMock

import pytest
import websockets

from eva.assistant.aai_server import AAIAssistantServer
from eva.assistant.audio_bridge import create_twilio_media_message

FAKE_AAI_PORT = 38999
BRIDGE_PORT = 38998


class FakeAaiHost:
    """Minimal aai host: completes the handshake, calls one tool, speaks, ends the turn."""

    def __init__(self):
        self.handshake: dict | None = None
        self.tool_results: list[dict] = []
        self.received_audio_bytes = 0
        self._server = None

    async def start(self) -> None:
        self._server = await websockets.serve(self._handle, "localhost", FAKE_AAI_PORT)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws) -> None:
        # 1. Handshake: first text frame carries the host config.
        raw = await ws.recv()
        self.handshake = json.loads(raw)
        await ws.send(json.dumps({"type": "config"}))

        # 2. Ask the client to run a tool.
        await ws.send(
            json.dumps(
                {
                    "type": "tool_call",
                    "toolCallId": "call_1",
                    "toolName": "probe_tool",
                    "args": {"query": "status"},
                }
            )
        )

        # 3. Wait for the relayed result, tolerating interleaved user audio.
        while True:
            frame = await ws.recv()
            if isinstance(frame, (bytes, bytearray)):
                self.received_audio_bytes += len(frame)
                continue
            message = json.loads(frame)
            if message.get("type") == "tool_result":
                self.tool_results.append(message)
                break

        # 4. Speak: transcript, 100 ms of 24 kHz PCM16, then end the turn.
        await ws.send(json.dumps({"type": "agent_transcript", "text": "Your status is green."}))
        await ws.send(b"\x00\x01" * 2400)
        await ws.send(json.dumps({"type": "reply_done"}))
        await ws.send(json.dumps({"type": "audio_done"}))

        # Keep the socket open so the bridge can finish draining.
        await asyncio.sleep(1.0)


def _make_server() -> AAIAssistantServer:
    """Build a server with the heavy collaborators stubbed."""
    srv = object.__new__(AAIAssistantServer)

    srv.pipeline_config = MagicMock()
    srv.pipeline_config.s2s_params = {
        "model": "aai-host",
        "ws_url": f"ws://localhost:{FAKE_AAI_PORT}/websocket",
    }
    srv.agent = MagicMock()
    srv.agent.tools = []
    srv.conversation_id = "conv-test"
    srv.port = BRIDGE_PORT
    srv.initial_message = "Thank you for calling."
    srv.audit_log = MagicMock()
    srv.tool_handler = MagicMock()

    async def _fake_execute(name, args):
        return {"status": "ok", "echo": args}

    srv.execute_tool = _fake_execute
    srv._build_system_prompt = lambda: "You are a test agent."

    # Attributes normally set by the two __init__ methods.
    srv._model = "aai-host"
    srv._ws_url = f"ws://localhost:{FAKE_AAI_PORT}/websocket"
    srv._input_rate = 16000
    srv._output_rate = 24000
    srv._session = None
    srv._to_backend = None
    srv._from_backend = None
    srv._stream_sid = "conv-test"
    srv._audio_out = asyncio.Queue()
    srv._tasks = []
    srv._user_speech_start_ms = None
    srv._user_speech_stop_ms = None
    srv._user_speaking = False
    srv._assistant_speaking = False
    srv._turn_first_audio_ms = None
    srv._assistant_text = ""
    srv.user_audio_buffer = bytearray()
    srv.assistant_audio_buffer = bytearray()
    srv._audio_buffer = bytearray()
    srv._audio_sample_rate = 24000
    srv._app = None
    srv._server = None
    srv._server_task = None
    srv._running = False
    return srv


@pytest.fixture
async def fake_host():
    host = FakeAaiHost()
    await host.start()
    yield host
    await host.stop()


async def test_full_session_round_trip(fake_host, tmp_path):
    srv = _make_server()
    srv.output_dir = tmp_path
    await srv.start()

    received_audio = bytearray()
    try:
        async with websockets.connect(f"ws://localhost:{BRIDGE_PORT}/ws") as sim:
            await sim.send(json.dumps({"event": "start", "start": {"streamSid": "stream-1"}}))

            # 200 ms of silence as mu-law 8 kHz, in 20 ms Twilio frames.
            silence_mulaw = audioop.lin2ulaw(b"\x00\x00" * 1600, 2)
            for offset in range(0, len(silence_mulaw), 160):
                await sim.send(create_twilio_media_message("stream-1", silence_mulaw[offset : offset + 160]))
                await asyncio.sleep(0.005)

            await sim.send(json.dumps({"event": "user_speech_stop", "timestamp_ms": str(_now_ms())}))

            # Collect whatever the bridge paces back.
            deadline = asyncio.get_running_loop().time() + 3.0
            while asyncio.get_running_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(sim.recv(), timeout=0.3)
                except asyncio.TimeoutError:
                    if received_audio:
                        break
                    continue
                message = json.loads(raw)
                if message.get("event") == "media":
                    import base64

                    received_audio.extend(base64.b64decode(message["media"]["payload"]))
    finally:
        await srv.stop()

    # Handshake carried the injected agent definition.
    assert fake_host.handshake is not None
    assert fake_host.handshake["type"] == "config"
    assert fake_host.handshake["audioFormat"] == "pcm16"
    assert fake_host.handshake["sampleRate"] == 16000
    assert fake_host.handshake["ttsSampleRate"] == 24000
    assert fake_host.handshake["host"]["systemPrompt"] == "You are a test agent."
    assert fake_host.handshake["host"]["greeting"] == "Thank you for calling."

    # User audio reached the backend.
    assert fake_host.received_audio_bytes > 0

    # The tool round-tripped through EVA, not the backend's sandbox.
    assert len(fake_host.tool_results) == 1
    assert fake_host.tool_results[0]["toolCallId"] == "call_1"
    assert json.loads(fake_host.tool_results[0]["result"])["status"] == "ok"

    # Assistant audio came back as 160-byte mu-law frames.
    assert len(received_audio) > 0
    assert len(received_audio) % 160 == 0

    # The turn was recorded.
    srv.audit_log.append_assistant_output.assert_called()
    assert srv.audit_log.append_assistant_output.call_args[0][0] == "Your status is green."


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)
```

- [ ] **Step 2: Run test to verify it fails or reveals integration bugs**

Run: `uv run pytest tests/integration/test_aai_fake_backend.py -v`
Expected: This is the first time all four modules run together. If it fails,
the failure is a real integration bug — fix the source, not the test. The most
likely causes, in order:

1. `stop()` calling `_save_scenario_dbs()` against a `MagicMock` tool handler —
   if `save_outputs` raises, stub `srv.get_initial_scenario_db` and
   `srv.get_final_scenario_db` to return `{}` in `_make_server()`.
2. The pacing task exiting before draining, because `_running` flipped false —
   confirm `_shutdown()` cancels tasks only after the assertion window.
3. `websockets.serve` handler signature differing across versions — if the
   installed `websockets` passes `(ws, path)`, accept a second optional arg in
   `FakeAaiHost._handle`.

- [ ] **Step 3: Run the whole suite**

Run: `uv run pytest tests/unit tests/integration/test_aai_fake_backend.py -q`
Expected: PASS

- [ ] **Step 4: Lint and commit**

```bash
cd ~/Code/eva
uv run ruff format tests/integration/test_aai_fake_backend.py
uv run ruff check tests/integration/test_aai_fake_backend.py
git add tests/integration/test_aai_fake_backend.py
git commit -m "test(aai): full-session bridge test against a fake aai host"
```

---

### Task 6: Documentation

**Files:**
- Create: `docs/aai_integration.md`
- Modify: `.env.example`

**Interfaces:**
- Consumes: the finished integration.
- Produces: the run instructions the user needs for the live end-to-end pass.

- [ ] **Step 1: Write the integration doc**

Create `docs/aai_integration.md`:

```markdown
# Evaluating the aai Voice Agent

EVA can evaluate the [aai](https://github.com/alexkroman/aai) voice-agent
framework via aai's **host mode**, in which EVA supplies the domain system
prompt, greeting, and tool schemas per session and aai relays each tool call
back to EVA for execution.

Tool execution must happen inside EVA: the accuracy metrics diff
`initial_scenario_db.json` against `final_scenario_db.json`, so a backend that
ran tools internally would score as if nothing happened.

## Prerequisites

1. **A running aai host with host mode enabled.** From your aai checkout:

   ```bash
   AAI_ALLOW_HOST=1 pnpm dev
   ```

   The default endpoint is `ws://localhost:3000/websocket`; EVA appends
   `?host=1` to select host mode. Without `AAI_ALLOW_HOST`, the host rejects the
   handshake and EVA aborts the conversation with a logged error.

2. **EVA's usual keys** in `.env` — a user simulator (ElevenLabs or OpenAI
   Realtime) plus the judge-model credentials the metrics need. See the main
   README.

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| `EVA_FRAMEWORK=aai` | — | Selects this server. |
| `EVA_MODEL__S2S=aai-host` | — | Marks the run as speech-to-speech. |
| `AAI_WS_URL` | `ws://localhost:3000/websocket` | aai host endpoint. |
| `s2s_params.ws_url` | — | Per-run override, takes precedence over `AAI_WS_URL`. |
| `s2s_params.model` | `aai-host` | Label recorded in the metrics log. |
| `s2s_params.input_sample_rate` | `16000` | PCM16 rate sent to aai. |
| `s2s_params.output_sample_rate` | `24000` | PCM16 rate received from aai. |

## Running

```bash
AAI_ALLOW_HOST=1 EVA_FRAMEWORK=aai EVA_MODEL__S2S=aai-host \
EVA_USER_SIMULATOR__PROVIDER=openai_realtime EVA_DOMAIN=itsm \
EVA_RECORD_IDS=15 EVA_MAX_CONCURRENT_CONVERSATIONS=1 eva
```

`EVA_MAX_CONCURRENT_CONVERSATIONS` above 1 works: host mode builds a fresh
single-use runtime per connection, so each conversation gets its own session.

## Architecture

```
user simulator ──Twilio μ-law 8k──▶ AAIAssistantServer ──PCM16 16k───▶ aai host
                ◀──Twilio μ-law 8k── localhost:{port}/ws ◀──PCM16 24k── ?host=1
                                             │
                                       execute_tool()
                                             ▼
                                      scenario database
```

| Module | Responsibility |
|---|---|
| `assistant/ws_bridge_server.py` | Shared bridge: Twilio framing, 20 ms pacing, recording buffers, turn bookkeeping, metrics. |
| `assistant/bridge_events.py` | Backend-agnostic event vocabulary and the session protocol. |
| `assistant/aai_session.py` | aai host handshake, audio frames, `tool_result` relay. |
| `assistant/aai_events.py` | aai wire-event models. |
| `assistant/aai_server.py` | Prompt and tool-schema construction. |

Adding another WebSocket voice backend means writing one session adapter plus a
thin server subclass — no changes to the bridge.

## Known gaps

- **No token usage.** aai emits no usage events, so `pipecat_metrics.jsonl`
  contains latency entries only. `model_response_latency` is unaffected;
  token-based cost reporting is unavailable for aai runs.
- **Backend output must be 24 kHz.** Only a 24 kHz → μ-law converter exists in
  `audio_bridge.py`. Another rate raises at session start with a clear message
  rather than producing distorted audio.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `aai host did not acknowledge the host-mode config` | `AAI_ALLOW_HOST` not set, or the host is not listening. |
| `aai host rejected host mode` | Host mode disabled server-side, or the tool schemas failed validation (each tool needs a non-empty description). |
| `Could not connect to aai host` | Wrong `AAI_WS_URL`, or the host is not running. |
| Empty transcript, conversation scored as failure | Check the log for `Failed to open backend session` — the handshake aborted. |
```

- [ ] **Step 2: Document the environment variable**

Append to `.env.example`:

```bash
# ── aai framework (EVA_FRAMEWORK=aai) ────────────────────────────────
# WebSocket endpoint of a running aai host. EVA appends ?host=1 to select
# host mode; the host must be started with AAI_ALLOW_HOST=1.
# See docs/aai_integration.md
AAI_WS_URL=ws://localhost:3000/websocket
```

- [ ] **Step 3: Verify the docs match reality**

Run: `uv run pytest tests/unit/assistant/test_aai_server.py tests/unit/assistant/test_aai_session.py -q`

Then confirm by inspection that every default named in the doc table matches the
constants in `aai_session.py` and the `s2s_params` keys read in
`aai_server.py.__init__`. A doc that drifts from the code is worse than no doc.

- [ ] **Step 4: Commit**

```bash
cd ~/Code/eva
git add docs/aai_integration.md .env.example
git commit -m "docs(aai): integration setup, configuration, and known gaps"
```

---

## Final verification

- [ ] **Run the full suite**

Run: `uv run pytest tests/unit tests/integration/test_aai_fake_backend.py -q`
Expected: PASS, with no pre-existing test newly failing.

- [ ] **Confirm the additive constraint held**

Run: `git diff --stat main -- src/eva/assistant/`
Expected: only `aai_events.py`, `aai_server.py`, `aai_session.py`,
`bridge_events.py`, and `ws_bridge_server.py` appear. If any of the five
existing servers or `base_server.py` shows up, revert that change.

- [ ] **Lint the whole change**

Run: `uv run ruff check src/eva tests` and `uv run ruff format --check src/eva tests`
Expected: clean.

- [ ] **Report honestly**

State plainly that the live end-to-end run was not performed, that it requires
`.env` credentials plus a running aai host, and give the command from
`docs/aai_integration.md`. Do not describe the integration as verified
end-to-end on the strength of the fake-backend test alone.
