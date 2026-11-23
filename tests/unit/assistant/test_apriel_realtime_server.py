"""Apriel-only session, tool continuation and playback regression tests."""

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI

from eva.assistant.apriel_realtime_server import (
    AprielRealtimeAssistantServer,
    _AssistantResponseState,
    _AudioPlayback,
    _UserTurnRecord,
)


def _bare_server() -> AprielRealtimeAssistantServer:
    """Construct an instance without running __init__ (skips PromptManager + tool building)."""
    srv = object.__new__(AprielRealtimeAssistantServer)
    srv.pipeline_config = MagicMock()
    srv.pipeline_config.s2s_params = {"api_key": "sk-test", "model": "gpt-realtime-mini"}
    srv.pipeline_config.parallel_tool_calls = None
    srv._model = "gpt-realtime-mini"
    srv._system_prompt = "you are a helpful assistant"
    srv._realtime_tools = []
    srv._assistant_state = _AssistantResponseState()
    srv._metrics_log = None
    srv._fw_log = None
    srv.audit_log = MagicMock()
    srv.user_audio_buffer = bytearray()
    srv.assistant_audio_buffer = bytearray()
    srv._bot_speaking = False
    srv._user_speaking = False
    srv._user_turn = None
    srv._user_turns_by_item_id = {}
    srv._audio_interface_speech_start_ts = None
    srv._last_user_audio_mono = 0.0
    srv._stream_sid = "test-stream"
    srv._reset_tool_tracking()
    srv._reset_audio_playback()
    return srv


def _function_item(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def _output_item_event(response_id: str, item: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(response_id=response_id, item=item)


def _arguments_done_event(response_id: str, call_id: str, arguments: str, **extra) -> SimpleNamespace:
    # The OpenAI SDK event intentionally has no name field. Compatible providers
    # may add one, which the shared handler also accepts through **extra.
    return SimpleNamespace(
        response_id=response_id,
        call_id=call_id,
        arguments=arguments,
        **extra,
    )


def _response_done_event(
    response_id: str,
    output: list[SimpleNamespace],
    *,
    status: str = "completed",
) -> SimpleNamespace:
    return SimpleNamespace(
        response=SimpleNamespace(
            id=response_id,
            status=status,
            output=output,
            usage=None,
        )
    )


def _connection() -> MagicMock:
    conn = MagicMock()
    conn.conversation.item.create = AsyncMock()
    conn.conversation.item.truncate = AsyncMock()
    conn.response.create = AsyncMock()
    return conn


class TestCreateClient:
    def test_returns_async_openai_with_api_key(self):
        srv = _bare_server()
        client = srv._create_client()
        assert isinstance(client, AsyncOpenAI)
        # Verify api_key was passed
        assert client.api_key == "sk-test"

    def test_default_base_url_is_openai(self):
        srv = _bare_server()
        client = srv._create_client()
        # Default OpenAI base URL (do not override)
        assert "openai.com" in str(client.base_url)

    def test_custom_http_and_websocket_base_urls_pass_through(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {
            "api_key": "apriel-test-key",
            "model": "apriel-endpoint",
            "base_url": "http://apriel.example/v1",
            "websocket_base_url": "ws://apriel.example/v1",
        }

        client = srv._create_client()

        assert str(client.base_url) == "http://apriel.example/v1/"
        assert client.websocket_base_url == "ws://apriel.example/v1"

    def test_raises_when_api_key_missing(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {}
        try:
            srv._create_client()
        except ValueError as e:
            assert "API key required" in str(e)
        else:
            raise AssertionError("expected ValueError")


class TestDefaultVoice:
    def test_no_default_voice(self):
        srv = _bare_server()
        srv._model = "apriel-realtime-v1"
        assert srv._default_voice() is None


class TestBuildSessionConfig:
    def test_includes_instructions_voice_and_tools(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {
            "api_key": "sk-test",
            "model": "gpt-realtime-mini",
            "voice": "marin",
        }
        cfg = srv._build_session_config()
        assert cfg["type"] == "realtime"
        assert cfg["instructions"] == "you are a helpful assistant"
        assert cfg["audio"]["output"]["voice"] == "marin"
        assert cfg["tools"] == []

    def test_voice_omitted_when_unset(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {"api_key": "sk-test", "model": "apriel-realtime-v1"}
        cfg = srv._build_session_config()
        assert "voice" not in cfg["audio"]["output"]

    def test_includes_whisper_transcription_model_by_default(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {"api_key": "sk-test", "model": "gpt-realtime-mini"}
        cfg = srv._build_session_config()
        assert cfg["audio"]["input"]["transcription"] == {"model": "whisper-1"}

    def test_explicit_null_disables_input_transcription(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {
            "api_key": "apriel-test-key",
            "model": "apriel-endpoint",
            "transcription_model": None,
        }

        cfg = srv._build_session_config()

        assert cfg["audio"]["input"]["transcription"] is None

    def test_apriel_session_controls_pass_through(self):
        srv = _bare_server()
        apriel = {"tts_input_streaming": "token"}
        srv.pipeline_config.s2s_params = {
            "api_key": "apriel-test-key",
            "model": "apriel-endpoint",
            "voice": "auto",
            "enable_thinking": True,
            "pre_tool_speech": False,
            "apriel": apriel,
        }

        cfg = srv._build_session_config()

        assert cfg["audio"]["output"]["voice"] == "auto"
        assert cfg["enable_thinking"] is True
        assert cfg["pre_tool_speech"] is False
        assert cfg["apriel"] == apriel

    def test_reasoning_effort_optional(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {
            "api_key": "sk-test",
            "model": "gpt-realtime-mini",
            "reasoning_effort": "low",
        }
        cfg = srv._build_session_config()
        assert cfg["reasoning"] == {"effort": "low"}

    def test_reasoning_effort_omitted_when_unset(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {"api_key": "sk-test", "model": "gpt-realtime-mini"}
        cfg = srv._build_session_config()
        assert "reasoning" not in cfg


class TestUserTurnLifecycle:
    @pytest.mark.asyncio
    async def test_item_ids_roll_turns_without_provider_transcription(self):
        srv = _bare_server()
        srv._audio_interface_speech_start_ts = "1000"

        await srv._on_speech_started(SimpleNamespace(item_id="item_1"))
        first_turn = srv._user_turn
        await srv._on_speech_stopped(SimpleNamespace(item_id="item_1"))

        srv._audio_interface_speech_start_ts = "2000"
        await srv._on_speech_started(SimpleNamespace(item_id="item_2"))

        assert first_turn is not None
        assert srv._user_turn is not first_turn
        assert first_turn.item_id == "item_1"
        assert first_turn.speech_started_wall_ms == "1000"
        assert srv._user_turn.item_id == "item_2"
        assert srv._user_turn.speech_started_wall_ms == "2000"

    @pytest.mark.asyncio
    async def test_late_transcript_is_attached_to_its_original_item(self):
        srv = _bare_server()
        srv._audio_interface_speech_start_ts = "1000"
        await srv._on_speech_started(SimpleNamespace(item_id="item_1"))
        first_turn = srv._user_turn
        await srv._on_speech_stopped(SimpleNamespace(item_id="item_1"))

        srv._audio_interface_speech_start_ts = "2000"
        await srv._on_speech_started(SimpleNamespace(item_id="item_2"))
        second_turn = srv._user_turn
        await srv._on_transcription_completed(SimpleNamespace(item_id="item_1", transcript="the first utterance"))

        assert first_turn is not None
        assert second_turn is not None
        assert first_turn.transcript == "the first utterance"
        assert first_turn.flushed is True
        assert second_turn.transcript == ""
        srv.audit_log.append_user_input.assert_called_once_with(
            "the first utterance",
            timestamp_ms="1000",
        )

    @pytest.mark.asyncio
    async def test_speech_stop_rolls_turn_when_provider_omits_item_ids(self):
        srv = _bare_server()
        await srv._on_speech_started(SimpleNamespace())
        first_turn = srv._user_turn
        await srv._on_speech_stopped(SimpleNamespace())
        await srv._on_speech_started(SimpleNamespace())

        assert srv._user_turn is not first_turn


class TestToolCallSequencing:
    @pytest.mark.asyncio
    async def test_tool_runs_only_at_response_done_then_returns_result_and_continuation(self):
        srv = _bare_server()
        srv.execute_tool = AsyncMock(return_value={"status": "success", "value": "sunny"})
        conn = _connection()
        item = _function_item("call_1", "get_weather", '{"city":"Paris"}')

        srv._on_response_output_item(_output_item_event("resp_1", item))
        srv._on_function_call_done(_arguments_done_event("resp_1", "call_1", '{"city":"Paris"}'))
        await asyncio.sleep(0)

        srv.execute_tool.assert_not_awaited()
        conn.conversation.item.create.assert_not_awaited()
        conn.response.create.assert_not_awaited()

        await srv._on_response_done(_response_done_event("resp_1", [item]), conn)

        srv.execute_tool.assert_awaited_once_with("get_weather", {"city": "Paris"})
        output_item = conn.conversation.item.create.await_args.kwargs["item"]
        assert output_item["call_id"] == "call_1"
        assert '"value": "sunny"' in output_item["output"]
        conn.response.create.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_name_from_arguments_done_is_used_when_response_omits_output(self):
        srv = _bare_server()
        srv.execute_tool = AsyncMock(return_value={"status": "success"})
        conn = _connection()

        srv._on_function_call_done(_arguments_done_event("resp_1", "call_1", '{"id":"42"}', name="lookup"))
        await srv._on_response_done(_response_done_event("resp_1", []), conn)

        srv.execute_tool.assert_awaited_once_with("lookup", {"id": "42"})
        conn.conversation.item.create.assert_awaited_once()
        conn.response.create.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_missing_name_returns_error_output_without_running_a_tool(self):
        srv = _bare_server()
        srv.execute_tool = AsyncMock()
        conn = _connection()

        srv._on_function_call_done(_arguments_done_event("resp_1", "call_1", "{}"))
        await srv._on_response_done(_response_done_event("resp_1", []), conn)

        srv.execute_tool.assert_not_awaited()
        output = json.loads(conn.conversation.item.create.await_args.kwargs["item"]["output"])
        assert output["error_type"] == "function_not_found"
        conn.response.create.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_cancelled_tool_completes_the_audit_pair(self):
        srv = _bare_server()
        started = asyncio.Event()

        async def execute_tool(name: str, arguments: dict) -> dict:
            started.set()
            await asyncio.Event().wait()

        srv.execute_tool = AsyncMock(side_effect=execute_tool)
        conn = _connection()
        item = _function_item("call_1", "lookup", "{}")
        srv._on_response_output_item(_output_item_event("resp_1", item))

        task = asyncio.create_task(srv._on_response_done(_response_done_event("resp_1", [item]), conn))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        srv.audit_log.append_tool_response.assert_called_once()
        assert srv.audit_log.append_tool_response.call_args.args[1]["error_type"] == "cancelled"
        conn.conversation.item.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_parallel_results_are_batched_in_response_order_with_one_continuation(self):
        srv = _bare_server()

        async def execute_tool(name: str, arguments: dict) -> dict:
            return {"status": "success", "tool": name, "arguments": arguments}

        srv.execute_tool = AsyncMock(side_effect=execute_tool)
        conn = _connection()
        first = _function_item("call_1", "first_tool", '{"n":1}')
        second = _function_item("call_2", "second_tool", '{"n":2}')

        for item in (first, second):
            srv._on_response_output_item(_output_item_event("resp_1", item))
            srv._on_function_call_done(_arguments_done_event("resp_1", item.call_id, item.arguments))

        await srv._on_response_done(_response_done_event("resp_1", [second, first]), conn)

        sent_call_ids = [entry.kwargs["item"]["call_id"] for entry in conn.conversation.item.create.await_args_list]
        assert sent_call_ids == ["call_2", "call_1"]
        assert conn.conversation.item.create.await_count == 2
        conn.response.create.assert_awaited_once_with()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["cancelled", "incomplete", "failed"])
    async def test_unfinished_response_does_not_run_tools_or_continue(self, status):
        srv = _bare_server()
        srv.execute_tool = AsyncMock(return_value={"status": "success"})
        conn = _connection()
        item = _function_item("call_1", "lookup", "{}")

        srv._on_response_output_item(_output_item_event("resp_1", item))
        srv._on_function_call_done(_arguments_done_event("resp_1", "call_1", "{}"))
        await srv._on_response_done(
            _response_done_event("resp_1", [item], status=status),
            conn,
        )

        srv.execute_tool.assert_not_awaited()
        conn.conversation.item.create.assert_not_awaited()
        conn.response.create.assert_not_awaited()
        assert not srv._pending_tool_calls


def test_apriel_proxy_auth_is_constructed_outside_serialized_websocket_options():
    srv = _bare_server()
    srv.pipeline_config = SimpleNamespace(
        s2s_params={
            "apriel_api_key": "test-realtime-secret",
            "websocket_base_url": "wss://test_job.example/v1",
        }
    )
    options = srv._websocket_options()
    assert options["subprotocols"] == ["realtime", "openai-insecure-api-key.test-realtime-secret"]
    assert options["server_hostname"] == "test-job.example"
    assert "websocket_connection_options" not in srv.pipeline_config.s2s_params


class TestPlaybackInterruption:
    @staticmethod
    async def dispatch(srv, queue, conn, event_type, **fields):
        await srv._handle_openai_event(
            SimpleNamespace(type=event_type, **fields),
            conn,
            AsyncMock(),
            queue,
        )

    async def start_audio(self, srv, queue, conn, response_id="old", frames=250):
        await self.dispatch(srv, queue, conn, "response.created", response=SimpleNamespace(id=response_id))
        await self.audio(srv, queue, conn, response_id, frames)

    async def audio(self, srv, queue, conn, response_id, frames=1):
        await self.dispatch(
            srv,
            queue,
            conn,
            "response.output_audio.delta",
            response_id=response_id,
            item_id=f"item-{response_id}",
            content_index=0,
            delta=base64.b64encode(b"\x10\x10" * 480 * frames).decode(),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("signal", ["speech_started", "cleared", "cancelled"])
    async def test_interruption_discards_five_seconds_and_late_audio(self, signal):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        await self.start_audio(srv, queue, conn)
        assert queue.qsize() == 250
        if signal == "speech_started":
            await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")
        elif signal == "cleared":
            await self.dispatch(srv, queue, conn, "output_audio_buffer.cleared", response_id="old")
        else:
            event = _response_done_event("old", [], status="cancelled")
            await self.dispatch(srv, queue, conn, "response.done", response=event.response)
        assert queue.empty()
        await self.audio(srv, queue, conn, "old")
        assert queue.empty(), "Late cancelled audio must not refill playback"
        assert not srv.assistant_audio_buffer, "Unplayed audio must not remain in the recording"
        await self.start_audio(srv, queue, conn, "next", frames=1)
        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_completed_generation_is_still_interruptible_while_playback_is_queued(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        await self.start_audio(srv, queue, conn)
        event = _response_done_event("old", [])
        await self.dispatch(srv, queue, conn, "response.done", response=event.response)
        assert srv._active_response_id is None
        assert not queue.empty()
        await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_pacer_stops_old_audio_and_records_only_sent_frames(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        sent = []
        first_sent, resumed = asyncio.Event(), asyncio.Event()

        async def send(message):
            sent.append(json.loads(message))
            first_sent.set()
            if len(sent) == 2:
                resumed.set()

        socket = SimpleNamespace(send_text=send)
        await self.start_audio(srv, queue, conn)
        pacer = asyncio.create_task(srv._pace_audio_output(socket, queue))
        try:
            await asyncio.wait_for(first_sent.wait(), 1)
            await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")
            await self.audio(srv, queue, conn, "old")
            await asyncio.sleep(0.06)
            assert len(sent) == 1
            assert len(srv.assistant_audio_buffer) == 960
            await self.start_audio(srv, queue, conn, "next", frames=1)
            await asyncio.wait_for(resumed.wait(), 1)
            assert len(sent) == 2
            assert len(srv.assistant_audio_buffer) == 1920
            assert all(frame["event"] == "media" for frame in sent)
        finally:
            pacer.cancel()
            await pacer

    @pytest.mark.asyncio
    async def test_delayed_old_clear_cannot_discard_new_response(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        await self.start_audio(srv, queue, conn)
        await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")
        await self.start_audio(srv, queue, conn, "next", frames=1)
        await self.dispatch(srv, queue, conn, "output_audio_buffer.cleared", response_id="old")
        await self.dispatch(
            srv, queue, conn, "response.output_audio_transcript.delta", response_id="old", delta="stale"
        )
        assert queue.qsize() == 1
        assert srv._assistant_state.transcript_parts == []

    @pytest.mark.asyncio
    async def test_clear_during_socket_send_waits_for_that_frame_then_stops(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        sending, release = asyncio.Event(), asyncio.Event()
        sent = []

        async def send(message):
            sending.set()
            await release.wait()
            sent.append(message)

        await self.start_audio(srv, queue, conn)
        pacer = asyncio.create_task(srv._pace_audio_output(SimpleNamespace(send_text=send), queue))
        try:
            await asyncio.wait_for(sending.wait(), 1)
            interruption = asyncio.create_task(
                self.dispatch(
                    srv,
                    queue,
                    conn,
                    "input_audio_buffer.speech_started",
                    item_id="user-1",
                )
            )
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(interruption, 1)
            await asyncio.sleep(0.06)
            assert len(sent) == 1
            assert queue.empty()
            await asyncio.wait_for(queue.join(), 1)
        finally:
            release.set()
            pacer.cancel()
            await pacer

    @pytest.mark.asyncio
    async def test_completed_tool_continues_after_playback_interruption(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        srv.execute_tool = AsyncMock(return_value={"booking": "found"})
        await self.start_audio(srv, queue, conn)
        item = _function_item("lookup-1", "lookup_booking", '{"id":"123"}')
        await self.dispatch(srv, queue, conn, "response.output_item.added", response_id="old", item=item)
        await self.dispatch(
            srv,
            queue,
            conn,
            "response.function_call_arguments.done",
            response_id="old",
            call_id=item.call_id,
            arguments=item.arguments,
        )
        await asyncio.sleep(0)
        event = _response_done_event("old", [item])
        await self.dispatch(srv, queue, conn, "response.done", response=event.response)
        await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")
        assert queue.empty()
        srv.execute_tool.assert_awaited_once_with("lookup_booking", {"id": "123"})
        conn.conversation.item.create.assert_awaited_once()
        conn.response.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_does_not_record_unsent_audio(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        await self.start_audio(srv, queue, conn, frames=2)
        socket = SimpleNamespace(send_text=AsyncMock(side_effect=OSError("disconnected")))
        await asyncio.wait_for(srv._pace_audio_output(socket, queue), 1)
        socket.send_text.assert_awaited_once()
        assert not srv.assistant_audio_buffer

    @pytest.mark.asyncio
    async def test_new_session_does_not_inherit_interrupted_response(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        await self.start_audio(srv, queue, conn, frames=1)
        await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")
        srv._reset_audio_playback()
        await self.start_audio(srv, queue, conn, frames=1)
        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_uninterrupted_output_drains_and_preserves_audio(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        await self.start_audio(srv, queue, conn, frames=5)
        socket = SimpleNamespace(send_text=AsyncMock())
        pacer = asyncio.create_task(srv._pace_audio_output(socket, queue))
        try:
            await asyncio.wait_for(queue.join(), 1)
            assert socket.send_text.await_count == 5
            assert srv.assistant_audio_buffer == b"\x10\x10" * 480 * 5
            assert not srv._bot_speaking
        finally:
            pacer.cancel()
            await pacer

    @pytest.mark.asyncio
    async def test_provider_without_response_ids_can_start_next_response(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        await self.start_audio(srv, queue, conn, response_id="", frames=1)
        await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")
        await self.audio(srv, queue, conn, response_id="")
        assert queue.empty()
        await self.start_audio(srv, queue, conn, response_id="", frames=1)
        assert queue.qsize() == 1

    async def finish(self, srv, queue, conn, response_id, transcript):
        await self.dispatch(
            srv,
            queue,
            conn,
            "response.output_audio_transcript.done",
            response_id=response_id,
            transcript=transcript,
        )
        event = _response_done_event(response_id, [])
        await self.dispatch(srv, queue, conn, "response.done", response=event.response)

    @pytest.mark.asyncio
    async def test_output_is_held_until_playback_drains(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        srv._fw_log = MagicMock()
        await self.start_audio(srv, queue, conn, frames=3)
        await self.finish(srv, queue, conn, "old", "Your flight is booked.")
        srv.audit_log.append_assistant_output.assert_not_called()

        socket = SimpleNamespace(send_text=AsyncMock())
        pacer = asyncio.create_task(srv._pace_audio_output(socket, queue))
        try:
            await asyncio.wait_for(queue.join(), 1)
        finally:
            pacer.cancel()
            await pacer
        srv.audit_log.append_assistant_output.assert_called_once()
        assert srv.audit_log.append_assistant_output.call_args.args[0] == "Your flight is booked."
        srv._fw_log.turn_end.assert_called_once_with(was_interrupted=False)

    @pytest.mark.asyncio
    async def test_barge_in_after_response_done_logs_output_as_interrupted(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        srv._fw_log = MagicMock()
        await self.start_audio(srv, queue, conn)
        await self.finish(srv, queue, conn, "old", "Your confirmation code is ABC123.")
        await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")

        srv.audit_log.append_assistant_output.assert_called_once()
        # No audio had been sent yet, so none of the text was heard.
        assert srv.audit_log.append_assistant_output.call_args.args[0] == "[interrupted]"
        srv._fw_log.turn_end.assert_called_once_with(was_interrupted=True)

    @pytest.mark.asyncio
    async def test_interruption_during_generation_is_logged_once(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        await self.start_audio(srv, queue, conn)
        await self.dispatch(
            srv, queue, conn, "response.output_audio_transcript.delta", response_id="old", delta="Hello"
        )
        await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")
        # A provider that does not cancel on barge-in still reports completion.
        event = _response_done_event(
            "old",
            [SimpleNamespace(type="message", content=[SimpleNamespace(type="audio", transcript="Hello there")])],
        )
        await self.dispatch(srv, queue, conn, "response.done", response=event.response)
        srv.audit_log.append_assistant_output.assert_called_once()
        assert srv.audit_log.append_assistant_output.call_args.args[0] == "[interrupted]"

    @pytest.mark.asyncio
    async def test_barge_in_truncates_item_at_played_audio(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        sent = asyncio.Event()
        count = 0

        async def send(_message):
            nonlocal count
            count += 1
            if count == 2:
                sent.set()

        await self.start_audio(srv, queue, conn)
        pacer = asyncio.create_task(srv._pace_audio_output(SimpleNamespace(send_text=send), queue))
        try:
            await asyncio.wait_for(sent.wait(), 1)
            await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")
        finally:
            pacer.cancel()
            await pacer
        # Each frame is 20 ms of 24 kHz PCM16 (960 bytes).
        conn.conversation.item.truncate.assert_awaited_once_with(item_id="item-old", content_index=0, audio_end_ms=40)

    @pytest.mark.asyncio
    async def test_barge_in_before_any_frame_truncates_at_zero(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        await self.start_audio(srv, queue, conn)
        await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")
        conn.conversation.item.truncate.assert_awaited_once_with(item_id="item-old", content_index=0, audio_end_ms=0)

    @pytest.mark.asyncio
    async def test_no_truncate_when_playback_already_finished(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        await self.start_audio(srv, queue, conn, frames=1)
        socket = SimpleNamespace(send_text=AsyncMock())
        pacer = asyncio.create_task(srv._pace_audio_output(socket, queue))
        try:
            await asyncio.wait_for(queue.join(), 1)
        finally:
            pacer.cancel()
            await pacer
        await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")
        conn.conversation.item.truncate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_truncate_failure_is_not_fatal(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        conn.conversation.item.truncate.side_effect = RuntimeError("unsupported")
        await self.start_audio(srv, queue, conn)
        await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_disconnect_logs_held_output_as_interrupted(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        await self.start_audio(srv, queue, conn)
        await self.finish(srv, queue, conn, "old", "Goodbye.")
        srv._flush_deferred_outputs_on_disconnect()
        srv.audit_log.append_assistant_output.assert_called_once()
        assert srv.audit_log.append_assistant_output.call_args.args[0] == "[interrupted]"

    @pytest.mark.asyncio
    async def test_barge_in_logs_only_the_estimated_heard_prefix(self):
        srv, queue, conn = _bare_server(), asyncio.Queue(), _connection()
        await self.start_audio(srv, queue, conn)
        await self.finish(srv, queue, conn, "old", "Your confirmation code is ABC123 and you are all set.")
        playback = srv._audio_playbacks["old"]
        assert playback.queued_pcm_bytes > 0
        playback.sent_pcm_bytes = playback.queued_pcm_bytes // 2  # caller heard about half

        await self.dispatch(srv, queue, conn, "input_audio_buffer.speech_started", item_id="user-1")

        # 52 chars * 0.5 = 26 -> cut inside "ABC123"; the partial word is dropped.
        assert srv.audit_log.append_assistant_output.call_args.args[0] == "Your confirmation code is [interrupted]"


class TestEstimateHeardText:
    @staticmethod
    def _playback(queued: int, sent: int) -> _AudioPlayback:
        return _AudioPlayback(queued_pcm_bytes=queued, sent_pcm_bytes=sent)

    def test_full_text_when_all_audio_was_sent(self):
        text = "Hello there friend."
        assert AprielRealtimeAssistantServer._estimate_heard_text(self._playback(100, 100), text) == text

    def test_full_text_without_audio_information(self):
        assert AprielRealtimeAssistantServer._estimate_heard_text(None, "Hello there.") == "Hello there."
        assert (
            AprielRealtimeAssistantServer._estimate_heard_text(self._playback(0, 0), "Hello there.") == "Hello there."
        )

    def test_nothing_heard_when_no_audio_was_sent(self):
        assert AprielRealtimeAssistantServer._estimate_heard_text(self._playback(100, 0), "Hello there.") == ""

    def test_cut_at_a_word_boundary(self):
        # 0.5 of 24 chars = 12 -> inside "everyone"; the partial word is dropped.
        assert (
            AprielRealtimeAssistantServer._estimate_heard_text(self._playback(100, 50), "Hello and welcome everyone!")
            == "Hello and"
        )

    def test_cut_exactly_on_a_space_keeps_the_whole_words(self):
        # 6 of 12 chars lands on the space after "Hello".
        assert AprielRealtimeAssistantServer._estimate_heard_text(self._playback(100, 50), "Hello world!") == "Hello"

    def test_single_long_word_cut_gives_empty_text(self):
        assert AprielRealtimeAssistantServer._estimate_heard_text(self._playback(100, 50), "Supercalifragilistic") == ""


class TestAprielNames:
    def test_apriel_session_controls_are_forwarded(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params["apriel"] = {"voice_history_mode": "transcript"}
        session = srv._build_session_config()
        assert session["apriel"] == {"voice_history_mode": "transcript"}

    def test_explicit_empty_session_settings_are_preserved(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params["apriel"] = {}
        assert srv._build_session_config()["apriel"] == {}

    def test_proxy_key_is_used_without_mutating_settings(self):
        srv = _bare_server()
        params = {
            "api_key": "ingress-test",
            "apriel_api_key": "application-test",
            "websocket_base_url": "wss://test_job.example/v1",
        }
        srv.pipeline_config.s2s_params = dict(params)
        options = srv._websocket_options()
        assert options["subprotocols"] == ["realtime", "openai-insecure-api-key.application-test"]
        assert options["server_hostname"] == "test-job.example"
        assert srv.pipeline_config.s2s_params == params
        assert srv._create_client().api_key == "ingress-test"

    def test_direct_auth_does_not_add_a_proxy_subprotocol(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {"api_key": "application-test"}
        assert "subprotocols" not in srv._websocket_options()
        assert srv._create_client().api_key == "application-test"


class TestWebsocketOptionsHostAndSubprotocols:
    def test_sni_override_uses_base_url_when_websocket_base_url_is_unset(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {"api_key": "k", "base_url": "https://test_job.example/v1"}
        assert srv._websocket_options()["server_hostname"] == "test-job.example"

    def test_websocket_base_url_takes_precedence_over_base_url(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {
            "api_key": "k",
            "base_url": "https://other_host.example/v1",
            "websocket_base_url": "wss://test_job.example/v1",
        }
        assert srv._websocket_options()["server_hostname"] == "test-job.example"

    def test_proxy_subprotocols_are_merged_with_user_supplied_ones(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {
            "apriel_api_key": "application-test",
            "websocket_connection_options": {"subprotocols": ["custom-proto", "realtime"]},
        }
        options = srv._websocket_options()
        assert options["subprotocols"] == [
            "realtime",
            "openai-insecure-api-key.application-test",
            "custom-proto",
        ]
        assert srv.pipeline_config.s2s_params["websocket_connection_options"]["subprotocols"] == [
            "custom-proto",
            "realtime",
        ]


class TestStreamedFunctionCallFlag:
    @pytest.mark.asyncio
    async def test_streamed_function_call_flag_skips_mixed_response_without_audio(self):
        srv = _bare_server()
        srv._log_assistant_output = MagicMock()
        # The function call was seen while streaming, but response.done carries no function_call item.
        srv._on_function_call_done(_arguments_done_event("resp_1", "call_1", "{}", name="lookup"))
        assert srv._assistant_state.has_function_calls
        srv._assistant_state.transcript_done_text = "Let me check that."

        await srv._record_response_done(_response_done_event("resp_1", []))

        srv._log_assistant_output.assert_not_called()


class TestContinuationWhileUserSpeaking:
    @pytest.mark.asyncio
    async def test_outputs_are_sent_but_response_create_is_skipped_while_user_speaks(self):
        srv = _bare_server()
        srv.execute_tool = AsyncMock(return_value={"status": "success"})
        srv._user_speaking = True
        conn = _connection()
        item = _function_item("call_1", "lookup", "{}")
        srv._on_response_output_item(_output_item_event("resp_1", item))
        srv._on_function_call_done(_arguments_done_event("resp_1", "call_1", "{}"))

        await srv._on_response_done(_response_done_event("resp_1", [item]), conn)

        srv.execute_tool.assert_awaited_once_with("lookup", {})
        conn.conversation.item.create.assert_awaited_once()
        conn.response.create.assert_not_awaited()

    def test_session_config_sets_create_response_explicitly(self):
        srv = _bare_server()
        assert srv._build_session_config()["audio"]["input"]["turn_detection"]["create_response"] is True


class TestSessionConfigBooleans:
    @pytest.mark.parametrize("field", ["enable_thinking", "pre_tool_speech"])
    @pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
    def test_non_boolean_values_are_rejected(self, field, value):
        srv = _bare_server()
        srv.pipeline_config.s2s_params[field] = value
        with pytest.raises(ValueError, match=field):
            srv._build_session_config()


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_client_is_closed_when_the_session_ends(self):
        srv = _bare_server()
        srv.conversation_id = "conv_1"
        client = MagicMock()
        client.close = AsyncMock()
        client.realtime.connect.side_effect = RuntimeError("connect failed")
        srv._create_client = MagicMock(return_value=client)
        websocket = MagicMock()

        await srv._handle_session(websocket)

        client.close.assert_awaited_once_with()


PLACEHOLDER = "[user speech - transcription unavailable]"


def _user_entries(srv) -> list[str]:
    return [call.args[0] for call in srv.audit_log.append_user_input.call_args_list]


async def _answer_the_turn(srv, response_id: str = "resp_1") -> None:
    """Record a spoken assistant response, which marks the current user turn answered."""
    srv._assistant_state.transcript_done_text = "Sure, one moment."
    srv._assistant_state.first_audio_wall_ms = "1"
    await srv._record_response_done(_response_done_event(response_id, []))


class TestMissingUserTranscript:
    @pytest.mark.asyncio
    async def test_answered_turn_without_transcript_gets_a_placeholder_at_the_next_speech_start(self):
        srv = _bare_server()
        await srv._on_speech_started(SimpleNamespace(item_id="user-1"))
        await _answer_the_turn(srv)

        await srv._on_speech_started(SimpleNamespace(item_id="user-2"))

        assert _user_entries(srv) == [PLACEHOLDER]
        assert srv._user_turn.item_id == "user-2"
        assert srv._user_turn.flushed is False

    @pytest.mark.asyncio
    async def test_unanswered_turn_gets_no_placeholder(self):
        srv = _bare_server()
        await srv._on_speech_started(SimpleNamespace(item_id="user-1"))

        await srv._on_speech_started(SimpleNamespace(item_id="user-1"))

        srv.audit_log.append_user_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_turn_with_a_transcript_gets_no_placeholder(self):
        srv = _bare_server()
        await srv._on_speech_started(SimpleNamespace(item_id="user-1"))
        await srv._on_transcription_completed(SimpleNamespace(item_id="user-1", transcript="I need a flight"))
        await _answer_the_turn(srv)

        await srv._on_speech_started(SimpleNamespace(item_id="user-2"))

        assert _user_entries(srv) == ["I need a flight"]

    @pytest.mark.asyncio
    async def test_tool_call_only_response_does_not_mark_the_turn_answered(self):
        srv = _bare_server()
        await srv._on_speech_started(SimpleNamespace(item_id="user-1"))
        item = _function_item("call_1", "lookup", "{}")
        await srv._record_response_done(_response_done_event("resp_1", [item]))

        assert srv._user_turn.answered is False

    def test_unflushed_turn_gets_a_placeholder_at_session_end(self):
        srv = _bare_server()
        srv._user_turn = _UserTurnRecord(item_id="user-1", speech_started_wall_ms="100")

        srv._flush_user_turn_without_transcript()
        srv._flush_user_turn_without_transcript()  # idempotent

        assert _user_entries(srv) == [PLACEHOLDER]

    def test_buffered_transcript_is_left_alone(self):
        srv = _bare_server()
        srv._user_turn = _UserTurnRecord(item_id="user-1", speech_started_wall_ms="100", transcript="buffered")

        srv._flush_user_turn_without_transcript()

        srv.audit_log.append_user_input.assert_not_called()


class TestStartFailure:
    @pytest.mark.asyncio
    async def test_start_raises_instead_of_hanging_when_the_server_fails_to_start(self, tmp_path):
        srv = _bare_server()
        srv._running = False
        srv.output_dir = tmp_path
        srv.port = 8765

        class _FailingServer:
            started = False

            def __init__(self, config):
                pass

            async def serve(self):
                return  # e.g. the port is already in use

        with patch("eva.assistant.apriel_realtime_server.uvicorn.Server", _FailingServer):
            with pytest.raises(RuntimeError, match="failed to start"):
                await asyncio.wait_for(srv.start(), timeout=2)

        assert srv._running is False
