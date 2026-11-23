"""Tests for OpenAIRealtimeAssistantServer extension hooks.

These tests verify behavior of the hooks that GrokVoiceAssistantServer overrides.
They guard against regressions when refactoring shared logic.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI

from eva.assistant.openai_realtime_server import (
    OpenAIRealtimeAssistantServer,
    _AssistantResponseState,
    _UserTurnRecord,
)


def _bare_server() -> OpenAIRealtimeAssistantServer:
    """Construct an instance without running __init__ (skips PromptManager + tool building)."""
    srv = object.__new__(OpenAIRealtimeAssistantServer)
    srv.pipeline_config = MagicMock()
    srv.pipeline_config.s2s_params = {"api_key": "sk-test", "model": "gpt-realtime-mini"}
    srv._model = "gpt-realtime-mini"
    srv._system_prompt = "you are a helpful assistant"
    srv._realtime_tools = []
    return srv


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


def _batching_server() -> OpenAIRealtimeAssistantServer:
    """Bare server with the state the tool-batching handlers touch."""
    srv = _bare_server()
    srv._assistant_state = _AssistantResponseState()
    srv._pending_tool_calls = []
    srv._user_turn = None
    srv._metrics_log = None
    srv._fw_log = None
    srv.audit_log = MagicMock()
    srv.user_audio_buffer = bytearray()
    srv.assistant_audio_buffer = bytearray()
    srv._bot_speaking = False
    return srv


def _arguments_done(response_id: str, call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(response_id=response_id, call_id=call_id, name=name, arguments=arguments)


def _response_done(response_id: str, status: str = "completed") -> SimpleNamespace:
    return SimpleNamespace(response=SimpleNamespace(id=response_id, status=status, output=[], usage=None))


def _connection() -> MagicMock:
    conn = MagicMock()
    conn.conversation.item.create = AsyncMock()
    conn.response.create = AsyncMock()
    return conn


class TestToolBatching:
    @pytest.mark.asyncio
    async def test_calls_are_recorded_while_streaming_and_run_at_response_done(self):
        srv = _batching_server()
        srv.execute_tool = AsyncMock(return_value={"status": "success"})
        conn = _connection()

        srv._on_function_call_done(_arguments_done("resp_1", "call_1", "lookup", '{"id":"42"}'))

        srv.execute_tool.assert_not_awaited()
        conn.conversation.item.create.assert_not_awaited()
        conn.response.create.assert_not_awaited()

        await srv._on_response_done(_response_done("resp_1"), conn)

        srv.execute_tool.assert_awaited_once_with("lookup", {"id": "42"})
        conn.conversation.item.create.assert_awaited_once()
        conn.response.create.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_parallel_calls_run_in_order_with_all_outputs_and_one_response_create(self):
        srv = _batching_server()
        order: list[str] = []

        async def execute_tool(name: str, arguments: dict) -> dict:
            order.append(name)
            return {"status": "success", "tool": name}

        srv.execute_tool = AsyncMock(side_effect=execute_tool)
        conn = _connection()
        srv._on_function_call_done(_arguments_done("resp_1", "call_1", "first_tool", "{}"))
        srv._on_function_call_done(_arguments_done("resp_1", "call_2", "second_tool", "{}"))

        await srv._on_response_done(_response_done("resp_1"), conn)

        assert order == ["first_tool", "second_tool"]
        sent = [entry.kwargs["item"]["call_id"] for entry in conn.conversation.item.create.await_args_list]
        assert sent == ["call_1", "call_2"]
        conn.response.create.assert_awaited_once_with()
        assert srv._pending_tool_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["cancelled", "incomplete", "failed"])
    async def test_unfinished_response_runs_no_tools_and_sends_nothing(self, status):
        srv = _batching_server()
        srv.execute_tool = AsyncMock(return_value={"status": "success"})
        conn = _connection()
        srv._on_function_call_done(_arguments_done("resp_1", "call_1", "lookup", "{}"))

        await srv._on_response_done(_response_done("resp_1", status=status), conn)

        srv.execute_tool.assert_not_awaited()
        conn.conversation.item.create.assert_not_awaited()
        conn.response.create.assert_not_awaited()
        assert srv._pending_tool_calls == []

    @pytest.mark.asyncio
    async def test_only_the_finished_responses_calls_run(self):
        srv = _batching_server()
        srv.execute_tool = AsyncMock(return_value={"status": "success"})
        conn = _connection()
        srv._on_function_call_done(_arguments_done("resp_1", "call_1", "lookup", "{}"))
        srv._on_function_call_done(_arguments_done("resp_2", "call_2", "other", "{}"))

        await srv._on_response_done(_response_done("resp_1"), conn)

        srv.execute_tool.assert_awaited_once_with("lookup", {})
        assert [call.call_id for call in srv._pending_tool_calls] == ["call_2"]

    @pytest.mark.asyncio
    async def test_grok_override_forwards_the_connection(self):
        from eva.assistant.grok_voice_server import GrokVoiceAssistantServer

        srv = object.__new__(GrokVoiceAssistantServer)
        srv._flush_pending_user_transcript = MagicMock()
        srv._pending_tool_calls = []
        srv._user_turn = None
        srv._assistant_state = _AssistantResponseState()
        srv._metrics_log = None
        srv._fw_log = None
        srv.audit_log = MagicMock()
        srv.user_audio_buffer = bytearray()
        srv.assistant_audio_buffer = bytearray()
        srv._bot_speaking = False
        srv.execute_tool = AsyncMock(return_value={"status": "success"})
        conn = _connection()
        srv._on_function_call_done(_arguments_done("resp_1", "call_1", "lookup", "{}"))

        await srv._on_response_done(_response_done("resp_1"), conn)

        conn.response.create.assert_awaited_once_with()


class TestWebsocketOptions:
    def test_sni_override_uses_base_url_when_websocket_base_url_is_unset(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {"api_key": "k", "base_url": "https://test_job.example/v1"}
        assert srv._websocket_options()["server_hostname"] == "test-job.example"

    def test_websocket_base_url_takes_precedence(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {
            "api_key": "k",
            "base_url": "https://other_host.example/v1",
            "websocket_base_url": "wss://test_job.example/v1",
        }
        assert srv._websocket_options()["server_hostname"] == "test-job.example"

    def test_no_override_without_underscore_and_user_options_are_kept(self):
        srv = _bare_server()
        srv.pipeline_config.s2s_params = {
            "api_key": "k",
            "base_url": "https://api.openai.com/v1",
            "websocket_connection_options": {"subprotocols": ["x"]},
        }
        assert srv._websocket_options() == {"subprotocols": ["x"]}


class TestStartFailure:
    @pytest.mark.asyncio
    async def test_start_raises_instead_of_hanging_when_the_server_fails_to_start(self, tmp_path):
        srv = _bare_server()
        srv._running = False
        srv._service_name = "OpenAI Realtime"
        srv.output_dir = tmp_path
        srv.port = 8765

        class _FailingServer:
            started = False

            def __init__(self, config):
                pass

            async def serve(self):
                return  # e.g. the port is already in use

        with patch("eva.assistant.openai_realtime_server.uvicorn.Server", _FailingServer):
            with pytest.raises(RuntimeError, match="failed to start"):
                await asyncio.wait_for(srv.start(), timeout=2)

        assert srv._running is False


PLACEHOLDER = "[user speech - transcription unavailable]"


def _speech_server() -> OpenAIRealtimeAssistantServer:
    srv = _batching_server()
    srv._user_turn = None
    srv._user_speaking = False
    srv._audio_interface_speech_start_ts = None
    srv._stream_sid = "s"
    return srv


def _user_entries(srv) -> list[str]:
    return [call.args[0] for call in srv.audit_log.append_user_input.call_args_list]


class TestMissingUserTranscript:
    @pytest.mark.asyncio
    async def test_answered_turn_without_transcript_gets_a_placeholder_at_the_next_speech_start(self):
        srv = _speech_server()
        await srv._on_speech_started(SimpleNamespace())
        await srv._on_speech_stopped(SimpleNamespace())
        srv._assistant_state.transcript_done_text = "Sure, one moment."
        srv._assistant_state.first_audio_wall_ms = "1"
        srv._record_response_done(_response_done("resp_1"))

        await srv._on_speech_started(SimpleNamespace())

        assert _user_entries(srv) == [PLACEHOLDER]
        assert srv._user_turn.flushed is False  # a fresh turn for the new speech

    @pytest.mark.asyncio
    async def test_unanswered_turn_keeps_merging_vad_fragments(self):
        srv = _speech_server()
        await srv._on_speech_started(SimpleNamespace())
        await srv._on_speech_stopped(SimpleNamespace())
        first_turn = srv._user_turn

        await srv._on_speech_started(SimpleNamespace())

        assert srv._user_turn is first_turn
        srv.audit_log.append_user_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_turn_with_a_transcript_gets_no_placeholder(self):
        srv = _speech_server()
        await srv._on_speech_started(SimpleNamespace())
        await srv._on_transcription_completed(SimpleNamespace(transcript="I need a flight"))
        srv._assistant_state.transcript_done_text = "Sure."
        srv._assistant_state.first_audio_wall_ms = "1"
        srv._record_response_done(_response_done("resp_1"))

        await srv._on_speech_started(SimpleNamespace())

        assert _user_entries(srv) == ["I need a flight"]

    def test_unanswered_last_turn_gets_a_placeholder_at_session_end(self):
        srv = _speech_server()
        srv._user_turn = _UserTurnRecord(speech_started_wall_ms="100")

        srv._flush_user_turn_without_transcript()
        srv._flush_user_turn_without_transcript()  # idempotent

        assert _user_entries(srv) == [PLACEHOLDER]

    def test_buffered_transcript_is_left_for_the_subclass_to_write(self):
        srv = _speech_server()
        srv._user_turn = _UserTurnRecord(speech_started_wall_ms="100", transcript="buffered")

        srv._flush_user_turn_without_transcript()

        srv.audit_log.append_user_input.assert_not_called()

    def test_tool_call_only_response_does_not_mark_the_turn_answered(self):
        srv = _speech_server()
        srv._user_turn = _UserTurnRecord(speech_started_wall_ms="100")
        srv._on_function_call_done(_arguments_done("resp_1", "call_1", "lookup", "{}"))
        event = _response_done("resp_1")
        event.response.output = [SimpleNamespace(type="function_call")]

        srv._record_response_done(event)

        assert srv._user_turn.answered is False
