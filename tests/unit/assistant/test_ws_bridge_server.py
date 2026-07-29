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

        srv.audit_log.append_assistant_output.assert_called_once_with("Your flight is confirmed.", timestamp_ms="4000")
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
        """The aai backend sends full text per frame, not deltas."""
        session = _FakeSession([AssistantTranscript(text="Hello"), AssistantTranscript(text="Hello there")])
        srv = _bare_server(session)
        srv._now_ms = lambda: 7_000

        await srv._process_backend_events()

        assert srv._assistant_text == "Hello there"

    async def test_speech_started_event_triggers_barge_in(self):
        session = _FakeSession([SpeechStarted()])
        srv = _bare_server(session)
        srv._now_ms = lambda: 7_000
        srv._turn_first_audio_ms = 6_000
        srv._assistant_text = "mid sentence"

        await srv._process_backend_events()

        srv._fw_log.turn_end.assert_called_once_with(was_interrupted=True)

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
