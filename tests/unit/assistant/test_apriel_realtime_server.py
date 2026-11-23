"""Apriel-only session, tool continuation and playback regression tests."""

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI

from eva.assistant.apriel_realtime_server import AprielRealtimeAssistantServer, _AssistantResponseState


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
    conn.response.create = AsyncMock()
    return conn


async def _wait_for_finalizers(srv: AprielRealtimeAssistantServer) -> None:
    tasks = list(srv._tool_finalizer_tasks)
    assert tasks
    await asyncio.gather(*tasks)


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
    def test_default_voice_is_marin(self):
        srv = _bare_server()
        assert srv._default_voice() == "marin"


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

    def test_voice_falls_back_to_default(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {"api_key": "sk-test", "model": "gpt-realtime-mini"}
        cfg = srv._build_session_config()
        assert cfg["audio"]["output"]["voice"] == "marin"

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
    async def test_fast_tool_waits_for_response_done_before_result_and_continuation(self):
        srv = _bare_server()
        srv.execute_tool = AsyncMock(return_value={"status": "success", "value": "sunny"})
        conn = _connection()
        item = _function_item("call_1", "get_weather", '{"city":"Paris"}')

        srv._on_response_output_item(_output_item_event("resp_1", item), start_tool=False)
        await srv._on_function_call_done(_arguments_done_event("resp_1", "call_1", '{"city":"Paris"}'))
        await asyncio.sleep(0)

        srv.execute_tool.assert_awaited_once_with("get_weather", {"city": "Paris"})
        conn.conversation.item.create.assert_not_awaited()
        conn.response.create.assert_not_awaited()

        await srv._on_response_done(_response_done_event("resp_1", [item]), conn)
        await _wait_for_finalizers(srv)

        output_item = conn.conversation.item.create.await_args.kwargs["item"]
        assert output_item["call_id"] == "call_1"
        assert '"value": "sunny"' in output_item["output"]
        conn.response.create.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_tool_execution_overlaps_response_tail(self):
        srv = _bare_server()
        tool_started = asyncio.Event()
        release_tool = asyncio.Event()

        async def execute_tool(name: str, arguments: dict) -> dict:
            tool_started.set()
            await release_tool.wait()
            return {"status": "success", "name": name, "arguments": arguments}

        srv.execute_tool = AsyncMock(side_effect=execute_tool)
        conn = _connection()
        item = _function_item("call_1", "lookup", '{"id":"42"}')

        srv._on_response_output_item(_output_item_event("resp_1", item), start_tool=False)
        await srv._on_function_call_done(_arguments_done_event("resp_1", "call_1", '{"id":"42"}'))
        await tool_started.wait()
        await srv._on_response_done(_response_done_event("resp_1", [item]), conn)
        finalizers = list(srv._tool_finalizer_tasks)

        assert len(finalizers) == 1
        conn.conversation.item.create.assert_not_awaited()
        conn.response.create.assert_not_awaited()

        release_tool.set()
        await asyncio.gather(*finalizers)
        conn.conversation.item.create.assert_awaited_once()
        conn.response.create.assert_awaited_once()

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
            srv._on_response_output_item(_output_item_event("resp_1", item), start_tool=False)
            await srv._on_function_call_done(_arguments_done_event("resp_1", item.call_id, item.arguments))

        await srv._on_response_done(_response_done_event("resp_1", [second, first]), conn)
        await _wait_for_finalizers(srv)

        sent_call_ids = [entry.kwargs["item"]["call_id"] for entry in conn.conversation.item.create.await_args_list]
        assert sent_call_ids == ["call_2", "call_1"]
        assert conn.conversation.item.create.await_count == 2
        conn.response.create.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_cancelled_response_does_not_send_tool_output_or_continue(self):
        srv = _bare_server()
        tool_started = asyncio.Event()

        async def execute_tool(name: str, arguments: dict) -> dict:
            tool_started.set()
            await asyncio.Event().wait()
            return {"status": "success"}

        srv.execute_tool = AsyncMock(side_effect=execute_tool)
        conn = _connection()
        item = _function_item("call_1", "lookup", "{}")

        srv._on_response_output_item(_output_item_event("resp_1", item), start_tool=False)
        await srv._on_function_call_done(_arguments_done_event("resp_1", "call_1", "{}"))
        await tool_started.wait()
        await srv._on_response_done(
            _response_done_event("resp_1", [item], status="cancelled"),
            conn,
        )
        await asyncio.sleep(0)

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
        await _wait_for_finalizers(srv)
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
