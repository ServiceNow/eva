"""Tests for OpenAIRealtimeAssistantServer extension hooks and tool lifecycle.

These tests verify behavior of the hooks that GrokVoiceAssistantServer overrides.
They guard against regressions when refactoring shared logic.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI

from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer, _AssistantResponseState


def _bare_server() -> OpenAIRealtimeAssistantServer:
    """Construct an instance without running __init__ (skips PromptManager + tool building)."""
    srv = object.__new__(OpenAIRealtimeAssistantServer)
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
    srv._reset_tool_tracking()
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


async def _wait_for_finalizers(srv: OpenAIRealtimeAssistantServer) -> None:
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
            "api_key": "aur_sk_test",
            "model": "aurion-endpoint",
            "base_url": "http://aurion.example/v1",
            "websocket_base_url": "ws://aurion.example/v1",
        }

        client = srv._create_client()

        assert str(client.base_url) == "http://aurion.example/v1/"
        assert client.websocket_base_url == "ws://aurion.example/v1"

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
            "api_key": "aur_sk_test",
            "model": "aurion-endpoint",
            "transcription_model": None,
        }

        cfg = srv._build_session_config()

        assert cfg["audio"]["input"]["transcription"] is None

    def test_aurion_session_controls_pass_through(self):
        srv = _bare_server()
        aurion = {"tts_input_streaming": "token"}
        srv.pipeline_config.s2s_params = {
            "api_key": "aur_sk_test",
            "model": "aurion-endpoint",
            "voice": "auto",
            "enable_thinking": True,
            "pre_tool_speech": False,
            "aurion": aurion,
        }

        cfg = srv._build_session_config()

        assert cfg["audio"]["output"]["voice"] == "auto"
        assert cfg["enable_thinking"] is True
        assert cfg["pre_tool_speech"] is False
        assert cfg["aurion"] == aurion

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
