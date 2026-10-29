"""Tests for the OpenAI GPT-Live assistant server and its helpers."""

import asyncio
import base64
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from eva.assistant.openai_live_server import (
    ASSISTANT_QUIET_FLUSH_MS,
    INTERRUPT_TAIL_MS,
    LIVE_SAMPLE_RATE,
    USER_QUIET_FLUSH_MS,
    OpenAILiveAssistantServer,
    TurnEvent,
    TurnTracker,
    build_function_tools,
    build_session_config,
    load_live_prompts,
)
from eva.models.agents import AgentConfig, AgentTool, AgentToolParameter
from eva.models.config import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
AIRLINE_YAML = REPO_ROOT / "configs" / "agents" / "airline_agent.yaml"


def _write_agent_dir(tmp_path: Path, gpt_live: dict | None) -> Path:
    agent_yaml = tmp_path / "airline_agent.yaml"
    agent_yaml.write_text("id: agent_airline\n", encoding="utf-8")
    if gpt_live is not None:
        live_dir = tmp_path / "gpt-live"
        live_dir.mkdir()
        (live_dir / "airline_agent.yaml").write_text(yaml.safe_dump(gpt_live), encoding="utf-8")
    return agent_yaml


class TestLoadLivePrompts:
    def test_reads_sibling_gpt_live_file(self, tmp_path):
        agent_yaml = _write_agent_dir(
            tmp_path, {"frontend_instructions": "voice prompt", "backend_instructions": "backend prompt"}
        )
        assert load_live_prompts(agent_yaml) == ("voice prompt", "backend prompt")

    def test_accepts_string_path(self, tmp_path):
        agent_yaml = _write_agent_dir(tmp_path, {"frontend_instructions": "f", "backend_instructions": "b"})
        assert load_live_prompts(str(agent_yaml)) == ("f", "b")

    def test_missing_file_raises_value_error(self, tmp_path):
        agent_yaml = _write_agent_dir(tmp_path, None)
        with pytest.raises(ValueError, match="gpt-live"):
            load_live_prompts(agent_yaml)

    @pytest.mark.parametrize("missing", ["frontend_instructions", "backend_instructions"])
    def test_missing_key_raises_value_error(self, tmp_path, missing):
        prompts = {"frontend_instructions": "f", "backend_instructions": "b"}
        del prompts[missing]
        agent_yaml = _write_agent_dir(tmp_path, prompts)
        with pytest.raises(ValueError, match=missing):
            load_live_prompts(agent_yaml)

    def test_blank_key_raises_value_error(self, tmp_path):
        agent_yaml = _write_agent_dir(tmp_path, {"frontend_instructions": "  ", "backend_instructions": "b"})
        with pytest.raises(ValueError, match="frontend_instructions"):
            load_live_prompts(agent_yaml)


class TestBuildFunctionTools:
    def test_tools_are_flat_function_definitions(self):
        agent = AgentConfig.from_yaml(AIRLINE_YAML)
        tools = build_function_tools(agent)
        assert len(tools) == len(agent.tools) > 0
        for tool in tools:
            assert set(tool) == {"type", "name", "description", "parameters"}
            assert tool["type"] == "function"
            assert tool["parameters"]["type"] == "object"
            assert set(tool["parameters"]) == {"type", "properties", "required"}
        assert tools[0]["name"] == agent.tools[0].function_name

    def test_agent_without_tools_gives_empty_list(self):
        agent = AgentConfig.from_yaml(AIRLINE_YAML).model_copy(update={"tools": []})
        assert build_function_tools(agent) == []


_BASE = {
    "model": "gpt-live-1",
    "frontend_instructions": "FRONT",
    "backend_instructions": "BACK",
    "delegation_model": "gpt-6-luna",
    "tools": [{"type": "function", "name": "t", "description": "d", "parameters": {"type": "object"}}],
}


class TestBuildSessionConfig:
    def test_minimal_payload(self):
        cfg = build_session_config(**_BASE)
        assert LIVE_SAMPLE_RATE == 24000
        assert cfg["model"] == "gpt-live-1"
        assert cfg["instructions"] == "FRONT"
        assert cfg["audio"] == {
            "format": {"type": "audio/pcm", "rate": 24000},
            "output": {"voice": "marin"},
        }
        assert cfg["delegation"]["type"] == "responses"
        responses = cfg["delegation"]["responses"]
        assert responses["model"] == "gpt-6-luna"
        assert responses["instructions"] == "BACK"
        assert responses["tools"] == _BASE["tools"]
        assert responses["tool_choice"] == "auto"

    def test_optional_fields_omitted_by_default(self):
        cfg = build_session_config(**_BASE)
        assert "parallel_tool_calls" not in cfg["delegation"]["responses"]
        assert "reasoning" not in cfg["delegation"]["responses"]
        assert "acknowledgement_delay_ms" not in cfg["delegation"]

    @pytest.mark.parametrize("value", [True, False])
    def test_parallel_tool_calls_passed_through(self, value):
        cfg = build_session_config(**_BASE, parallel_tool_calls=value)
        assert cfg["delegation"]["responses"]["parallel_tool_calls"] is value

    def test_optional_fields_land_in_spec_locations(self):
        cfg = build_session_config(**_BASE, voice="cedar", reasoning_effort="low", acknowledgement_delay_ms=250)
        assert cfg["audio"]["output"]["voice"] == "cedar"
        assert cfg["delegation"]["responses"]["reasoning"] == {"effort": "low"}
        assert cfg["delegation"]["acknowledgement_delay_ms"] == 250


QUIET = b"\x00" * 960
LOUD = (3000).to_bytes(2, "little", signed=True) * 480


class TestTurnTracker:
    def test_user_then_assistant_turn_flush_in_order(self):
        t = TurnTracker()
        t.on_user_speech_start(1000)
        t.on_input_transcript("I need", 1500)
        t.on_input_transcript(" a flight", 1700)
        t.on_user_speech_stop(2000)
        # The assistant answering does not flush the user turn: input transcripts can still be trailing.
        assert t.on_output_transcript("Sure.", 3000) == []
        assert t.tick(3000) == [TurnEvent("user", "I need a flight", "1000")]
        assert t.finalize(5000) == [TurnEvent("assistant", "Sure.", "3000")]

    def test_late_user_transcript_tail_stays_in_its_turn(self):
        t = TurnTracker()
        t.on_user_speech_start(1000)
        t.on_input_transcript("I need", 1500)
        t.on_user_speech_stop(2000)
        assert t.on_output_transcript("Okay,", 2100) == []
        t.on_input_transcript(" a flight", 2600)  # trails the user's stop by 0.6 s
        assert t.on_output_transcript(" let me see", 3500) == []  # keeps the assistant turn active
        assert t.tick(2600 + USER_QUIET_FLUSH_MS - 1) == []
        assert t.tick(2600 + USER_QUIET_FLUSH_MS) == [TurnEvent("user", "I need a flight", "1000")]

    def test_user_turn_not_flushed_by_quiet_while_user_still_speaking(self):
        t = TurnTracker()
        t.on_user_speech_start(1000)
        t.on_input_transcript("I need", 1500)
        assert t.tick(1500 + USER_QUIET_FLUSH_MS + 5000) == []

    def test_transcriptless_blip_does_not_leave_stale_start_time(self):
        t = TurnTracker()
        t.on_user_speech_start(1000)
        t.on_user_speech_stop(1100)
        assert t.on_user_speech_start(10_000) == []
        t.on_input_transcript("Real question", 10_500)
        t.on_user_speech_stop(11_000)
        assert t.finalize(13_000) == [TurnEvent("user", "Real question", "10000")]

    def test_back_to_back_user_turns_are_separate(self):
        t = TurnTracker()
        t.on_user_speech_start(1000)
        t.on_input_transcript("First question", 1400)
        t.on_user_speech_stop(2000)
        assert t.on_user_speech_start(4000) == [TurnEvent("user", "First question", "1000")]
        t.on_input_transcript("Second question", 4500)
        t.on_user_speech_stop(5000)
        assert t.finalize(6000) == [TurnEvent("user", "Second question", "4000")]

    def test_repeated_speech_start_without_text_keeps_original_timestamp(self):
        t = TurnTracker()
        t.on_user_speech_start(1000)
        t.on_user_speech_stop(1100)
        assert t.on_user_speech_start(1200) == []
        t.on_input_transcript("One utterance", 1500)
        assert t.finalize(3000) == [TurnEvent("user", "One utterance", "1000")]

    def test_interruption_flushes_spoken_text_with_marker(self):
        t = TurnTracker()
        assert t.on_output_transcript("one... ", 1000) == []
        t.on_output_transcript("two... three...", 1500)
        assert t.assistant_speaking(1600)
        # The turn stays open briefly so transcript fragments still in flight can join it.
        assert t.on_user_speech_start(1800) == []
        assert t.tick(1800 + INTERRUPT_TAIL_MS - 1) == []
        assert t.tick(1800 + INTERRUPT_TAIL_MS) == [
            TurnEvent("assistant", "one... two... three... [interrupted]", "1000", interrupted=True)
        ]
        assert t.finalize(5000) == []

    def test_transcript_arriving_after_interruption_joins_the_interrupted_turn(self):
        t = TurnTracker()
        t.on_output_transcript("one...", 1000)
        t.on_output_transcript(" two...", 1500)
        assert t.on_user_speech_start(1800) == []
        assert t.on_output_transcript(" three...", 2000) == []  # arrived 200 ms after the interruption
        assert t.tick(1800 + INTERRUPT_TAIL_MS) == [
            TurnEvent("assistant", "one... two... three... [interrupted]", "1000", interrupted=True)
        ]

    def test_interruption_before_any_transcript_still_records_the_turn(self):
        t = TurnTracker()
        t.on_audio_chunk(LOUD, 1000)
        assert t.on_user_speech_start(1100) == []
        assert t.on_output_transcript("Sure thing", 1500) == []
        assert t.tick(1100 + INTERRUPT_TAIL_MS) == [
            TurnEvent("assistant", "Sure thing [interrupted]", "1000", interrupted=True)
        ]

    def test_transcript_after_the_interrupt_window_starts_a_new_turn(self):
        t = TurnTracker()
        t.on_output_transcript("one...", 1000)
        t.on_user_speech_start(1200)
        assert t.on_output_transcript("Okay.", 1200 + INTERRUPT_TAIL_MS + 500) == [
            TurnEvent("assistant", "one... [interrupted]", "1000", interrupted=True)
        ]
        assert t.finalize(9000) == [TurnEvent("assistant", "Okay.", str(1200 + INTERRUPT_TAIL_MS + 500))]

    def test_interrupting_audio_only_response_logs_nothing(self):
        t = TurnTracker()
        t.on_audio_chunk(LOUD, 1000)
        assert t.on_user_speech_start(1100) == []
        assert t.tick(1100 + INTERRUPT_TAIL_MS) == []
        assert t.finalize(5000) == []

    def test_assistant_timestamp_prefers_first_loud_audio(self):
        t = TurnTracker()
        t.on_audio_chunk(LOUD, 900)
        t.on_output_transcript("Hi.", 1700)
        assert t.finalize(5000) == [TurnEvent("assistant", "Hi.", "900")]

    def test_latency_measured_from_user_speech_stop_to_first_loud_chunk(self):
        t = TurnTracker()
        t.on_user_speech_start(1000)
        t.on_user_speech_stop(2000)
        assert t.on_audio_chunk(QUIET, 2100) is None
        assert t.on_audio_chunk(LOUD, 2800) == pytest.approx(0.8)
        assert t.on_audio_chunk(LOUD, 2900) is None

    def test_no_latency_without_user_speech_stop(self):
        t = TurnTracker()
        assert t.on_audio_chunk(LOUD, 100) is None

    def test_latency_over_thirty_seconds_is_dropped(self):
        t = TurnTracker()
        t.on_user_speech_stop(1000)
        assert t.on_audio_chunk(LOUD, 40_000) is None

    def test_silent_session_produces_nothing(self):
        t = TurnTracker()
        for now in range(0, 10_000, 100):
            assert t.on_audio_chunk(QUIET, now) is None
            assert t.tick(now) == []
        assert not t.assistant_speaking(10_000)
        assert t.finalize(10_000) == []

    def test_tool_call_splits_assistant_turn(self):
        t = TurnTracker()
        t.on_output_transcript("Let me check.", 1000)
        assert t.on_tool_call(1500) == [TurnEvent("assistant", "Let me check.", "1000")]
        t.on_output_transcript("Found it.", 3000)
        assert t.finalize(9000) == [TurnEvent("assistant", "Found it.", "3000")]

    def test_tick_flushes_after_quiet_period(self):
        t = TurnTracker()
        t.on_output_transcript("Hello.", 1000)
        assert t.tick(1000 + ASSISTANT_QUIET_FLUSH_MS - 1) == []
        assert t.tick(1000 + ASSISTANT_QUIET_FLUSH_MS) == [TurnEvent("assistant", "Hello.", "1000")]

    def test_tool_call_mid_sentence_defers_the_split_to_the_sentence_end(self):
        t = TurnTracker()
        t.on_output_transcript("Okay, let's find your", 1000)
        assert t.on_tool_call(1100) == []  # the acknowledgement is still being spoken
        assert t.on_output_transcript(" booking", 1200) == []
        assert t.on_output_transcript(" number.", 1500) == [
            TurnEvent("assistant", "Okay, let's find your booking number.", "1000")
        ]
        t.on_output_transcript("I see your flight.", 2500)
        assert t.finalize(9000) == [TurnEvent("assistant", "I see your flight.", "2500")]

    def test_tool_call_without_prior_speech_does_not_split_the_answer(self):
        t = TurnTracker()
        assert t.on_tool_call(1000) == []
        t.on_output_transcript("Found it.", 3000)
        t.on_output_transcript(" Your flight is confirmed.", 3500)
        assert t.finalize(9000) == [TurnEvent("assistant", "Found it. Your flight is confirmed.", "3000")]


def _bare_server(execute_tool=None) -> OpenAILiveAssistantServer:
    """Instance without running __init__ (skips agent/tool-executor setup)."""
    srv = object.__new__(OpenAILiveAssistantServer)
    srv._pending_calls = {}
    srv._delegation_started = {}
    srv._fw_log = MagicMock()
    srv.execute_tool = execute_tool or AsyncMock(return_value={"ok": True})
    return srv


def _response_event(inner: dict, delegation_id: str | None = "d1") -> SimpleNamespace:
    return SimpleNamespace(type="response.event", delegation_id=delegation_id, event=inner)


def _call_done(call_id: str, name: str, arguments: str, delegation_id: str | None = "d1") -> SimpleNamespace:
    item = {"type": "function_call", "name": name, "arguments": arguments, "call_id": call_id}
    return _response_event({"type": "response.output_item.done", "item": item}, delegation_id)


def _completed(delegation_id: str | None = "d1") -> SimpleNamespace:
    return _response_event({"type": "response.completed"}, delegation_id)


def _recording_conn() -> tuple[MagicMock, list]:
    """A fake Live connection that records the order of response.item.create / response.create."""
    calls: list = []
    conn = MagicMock()
    conn.response.item.create = AsyncMock(
        side_effect=lambda **kw: calls.append(("item", kw["item"]["call_id"], kw["item"]["output"]))
    )
    conn.response.create = AsyncMock(side_effect=lambda **kw: calls.append(("create",)))
    return conn, calls


@pytest.mark.asyncio
class TestDelegatedToolCalls:
    async def test_single_call_round_trip(self):
        execute = AsyncMock(return_value={"name": "Zoë", "found": True})
        srv = _bare_server(execute)
        conn, calls = _recording_conn()

        await srv._on_response_event(_call_done("call_1", "lookup_booking", '{"confirmation_number": "ABC123"}'), conn)
        assert calls == []  # nothing is sent until the backend response completes
        await srv._on_response_event(_completed(), conn)

        execute.assert_awaited_once_with("lookup_booking", {"confirmation_number": "ABC123"})
        output = json.dumps({"name": "Zoë", "found": True}, ensure_ascii=False)
        assert calls == [("item", "call_1", output), ("create",)]
        srv._fw_log.write.assert_called_once_with(
            "tool_call",
            {
                "frame": "tool_call",
                "tool_name": "lookup_booking",
                "arguments": {"confirmation_number": "ABC123"},
                "result": {"name": "Zoë", "found": True},
            },
        )

    async def test_two_parallel_calls_send_outputs_in_call_order_then_one_response_create(self):
        async def execute(name, arguments):
            if name == "slow_tool":
                await asyncio.sleep(0.05)
            return {"tool": name}

        srv = _bare_server(AsyncMock(side_effect=execute))
        conn, calls = _recording_conn()

        await srv._on_response_event(_call_done("call_a", "slow_tool", "{}"), conn)
        await srv._on_response_event(_call_done("call_b", "fast_tool", "{}"), conn)
        await srv._on_response_event(_completed(), conn)

        assert calls == [
            ("item", "call_a", json.dumps({"tool": "slow_tool"})),
            ("item", "call_b", json.dumps({"tool": "fast_tool"})),
            ("create",),
        ]

    async def test_completed_without_pending_calls_sends_nothing(self):
        srv = _bare_server()
        conn, calls = _recording_conn()
        await srv._on_response_event(_completed(), conn)
        assert calls == []
        srv.execute_tool.assert_not_awaited()

    async def test_invalid_arguments_json_becomes_empty_dict(self):
        srv = _bare_server()
        conn, _ = _recording_conn()
        await srv._on_response_event(_call_done("call_1", "t", "{not json"), conn)
        await srv._on_response_event(_completed(), conn)
        srv.execute_tool.assert_awaited_once_with("t", {})

    async def test_pending_calls_are_cleared_after_completion(self):
        srv = _bare_server()
        conn, calls = _recording_conn()
        await srv._on_response_event(_call_done("call_1", "t", "{}"), conn)
        await srv._on_response_event(_completed(), conn)
        assert srv._pending_calls == {}
        n_calls = len(calls)
        await srv._on_response_event(_completed(), conn)
        assert len(calls) == n_calls

    async def test_completion_only_flushes_its_own_delegation(self):
        srv = _bare_server()
        conn, calls = _recording_conn()
        await srv._on_response_event(_call_done("call_1", "t", "{}", delegation_id="d1"), conn)
        await srv._on_response_event(_completed(delegation_id="d2"), conn)
        assert calls == []
        await srv._on_response_event(_completed(delegation_id="d1"), conn)
        assert [c[0] for c in calls] == ["item", "create"]

    async def test_non_function_call_items_are_ignored(self):
        srv = _bare_server()
        conn, calls = _recording_conn()
        message = {"type": "response.output_item.done", "item": {"type": "message"}}
        await srv._on_response_event(_response_event(message), conn)
        await srv._on_response_event(_completed(), conn)
        assert calls == []
        srv.execute_tool.assert_not_awaited()

    async def test_failing_tool_sends_error_output_and_still_continues(self):
        srv = _bare_server(AsyncMock(side_effect=RuntimeError("boom")))
        conn, calls = _recording_conn()
        await srv._on_response_event(_call_done("call_1", "t", "{}"), conn)
        await srv._on_response_event(_completed(), conn)
        assert [c[0] for c in calls] == ["item", "create"]
        assert "boom" in json.loads(calls[0][2])["error"]

    async def test_one_failing_call_does_not_block_the_other_outputs(self):
        async def execute(name, arguments):
            if name == "bad":
                raise RuntimeError("boom")
            return {"ok": name}

        srv = _bare_server(AsyncMock(side_effect=execute))
        conn, calls = _recording_conn()
        await srv._on_response_event(_call_done("call_a", "bad", "{}"), conn)
        await srv._on_response_event(_call_done("call_b", "good", "{}"), conn)
        await srv._on_response_event(_completed(), conn)
        assert [c[:2] for c in calls] == [("item", "call_a"), ("item", "call_b"), ("create",)]
        assert json.loads(calls[1][2]) == {"ok": "good"}

    @pytest.mark.parametrize("inner_type", ["response.failed", "response.incomplete"])
    async def test_failed_backend_response_drops_pending_calls(self, inner_type):
        srv = _bare_server()
        conn, calls = _recording_conn()
        await srv._on_response_event(_call_done("call_1", "t", "{}"), conn)
        await srv._on_response_event(_response_event({"type": inner_type}), conn)
        assert calls == []
        assert srv._pending_calls == {}


S2S_PARAMS = {"model": "gpt-live-1", "api_key": "sk-test", "delegation_model": "gpt-6-luna"}


def _agent() -> AgentConfig:
    return AgentConfig(
        id="a1",
        name="Test Agent",
        description="desc",
        role="role",
        instructions="be helpful",
        tool_module_path="eva.assistant.tools.airline_tools",
        tools=[
            AgentTool(
                id="t1",
                name="Lookup Booking",
                description="Look up a booking",
                required_parameters=[AgentToolParameter(name="booking_id", type="str", description="The booking id")],
            )
        ],
    )


def _make_server(tmp_path, s2s_params=None, prompts=True, parallel_tool_calls=None) -> OpenAILiveAssistantServer:
    agent_yaml = _write_agent_dir(
        tmp_path, {"frontend_instructions": "FRONT", "backend_instructions": "BACK"} if prompts else None
    )
    model_config = ModelConfig(
        s2s="gpt-live-1",
        s2s_params=S2S_PARAMS if s2s_params is None else s2s_params,
        parallel_tool_calls=parallel_tool_calls,
    )
    with patch("eva.assistant.base_server.ToolExecutor"):
        return OpenAILiveAssistantServer(
            current_date_time="2026-10-28 12:00",
            pipeline_config=model_config,
            agent=_agent(),
            agent_config_path=str(agent_yaml),
            scenario_db_path=str(tmp_path / "db.json"),
            output_dir=tmp_path / "out",
            port=0,
            conversation_id="conv",
        )


class TestServerInit:
    @pytest.mark.parametrize("missing", ["model", "api_key", "delegation_model"])
    def test_missing_required_s2s_param_raises_value_error(self, tmp_path, missing):
        params = {k: v for k, v in S2S_PARAMS.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            _make_server(tmp_path, params)

    def test_missing_gpt_live_prompt_file_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="gpt-live"):
            _make_server(tmp_path, prompts=False)

    def test_session_config_uses_s2s_params_and_pipeline_flag(self, tmp_path):
        params = {**S2S_PARAMS, "voice": "cedar", "delegation_reasoning_effort": "low", "acknowledgement_delay_ms": 100}
        srv = _make_server(tmp_path, params, parallel_tool_calls=True)
        cfg = srv._build_session_config()
        assert cfg["model"] == "gpt-live-1"
        assert cfg["instructions"] == "FRONT"
        assert cfg["audio"]["output"]["voice"] == "cedar"
        responses = cfg["delegation"]["responses"]
        assert responses["model"] == "gpt-6-luna"
        assert responses["instructions"] == "BACK"
        assert responses["reasoning"] == {"effort": "low"}
        assert responses["parallel_tool_calls"] is True
        assert [t["name"] for t in responses["tools"]] == [t.function_name for t in _agent().tools]
        assert cfg["delegation"]["acknowledgement_delay_ms"] == 100

    def test_recording_sample_rate_is_live_rate(self, tmp_path):
        assert _make_server(tmp_path)._audio_sample_rate == 24000


class TestEmit:
    def test_user_and_assistant_turns_written_to_audit_and_framework_logs(self, tmp_path):
        srv = _make_server(tmp_path)
        srv.audit_log = MagicMock()
        srv._fw_log = MagicMock()
        srv._emit(
            [
                TurnEvent("user", "I need a flight", "1000"),
                TurnEvent("assistant", "one... two [interrupted]", "2000", interrupted=True),
            ]
        )
        srv.audit_log.append_user_input.assert_called_once_with("I need a flight", timestamp_ms="1000")
        srv.audit_log.append_assistant_output.assert_called_once_with("one... two [interrupted]", timestamp_ms="2000")
        srv._fw_log.turn_start.assert_called_once_with(timestamp_ms=1000)
        srv._fw_log.llm_response.assert_called_once_with("one... two [interrupted]")
        srv._fw_log.s2s_transcript.assert_called_once_with("one... two [interrupted]")
        srv._fw_log.turn_end.assert_called_once_with(was_interrupted=True)

    def test_uninterrupted_assistant_turn_has_no_s2s_transcript(self, tmp_path):
        srv = _make_server(tmp_path)
        srv.audit_log = MagicMock()
        srv._fw_log = MagicMock()
        srv._emit([TurnEvent("assistant", "Hello.", "1000")])
        srv._fw_log.s2s_transcript.assert_not_called()
        srv._fw_log.turn_end.assert_called_once_with(was_interrupted=False)

    def test_empty_text_event_writes_nothing(self, tmp_path):
        srv = _make_server(tmp_path)
        srv.audit_log = MagicMock()
        srv._fw_log = MagicMock()
        srv._emit([TurnEvent("user", "", "1000"), TurnEvent("assistant", "", "2000")])
        srv.audit_log.append_user_input.assert_not_called()
        srv.audit_log.append_assistant_output.assert_not_called()
        srv._fw_log.turn_end.assert_not_called()


@pytest.mark.asyncio
class TestLiveEvents:
    async def test_output_audio_delta_updates_buffer_queue_and_latency(self, tmp_path):
        srv = _make_server(tmp_path)
        srv._metrics_log = MagicMock()
        srv._tracker.on_user_speech_stop(int(time.time() * 1000) - 500)
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        event = SimpleNamespace(type="session.output_audio.delta", delta=base64.b64encode(LOUD).decode("ascii"))

        await srv._handle_live_event(event, MagicMock(), queue)

        assert len(srv.assistant_audio_buffer) == len(LOUD)
        assert queue.qsize() == 1 and len(queue.get_nowait()) == 160
        srv._metrics_log.write_latency.assert_called_once()
        stage, seconds, model = srv._metrics_log.write_latency.call_args.args
        assert stage == "model_response" and model == "gpt-live-1" and 0.4 < seconds < 2.0

    async def test_silent_audio_delta_writes_no_latency(self, tmp_path):
        srv = _make_server(tmp_path)
        srv._metrics_log = MagicMock()
        srv._tracker.on_user_speech_stop(int(time.time() * 1000) - 500)
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        event = SimpleNamespace(type="session.output_audio.delta", delta=base64.b64encode(QUIET).decode("ascii"))
        await srv._handle_live_event(event, MagicMock(), queue)
        srv._metrics_log.write_latency.assert_not_called()

    async def test_idle_silence_is_recorded_but_not_forwarded_to_the_simulator(self, tmp_path):
        srv = _make_server(tmp_path)
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        event = SimpleNamespace(type="session.output_audio.delta", delta=base64.b64encode(QUIET).decode("ascii"))
        await srv._handle_live_event(event, MagicMock(), queue)
        assert len(srv.assistant_audio_buffer) == len(QUIET)
        assert queue.empty()

    async def test_quiet_chunk_inside_speaking_hold_is_forwarded(self, tmp_path):
        srv = _make_server(tmp_path)
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        loud = SimpleNamespace(type="session.output_audio.delta", delta=base64.b64encode(LOUD).decode("ascii"))
        quiet = SimpleNamespace(type="session.output_audio.delta", delta=base64.b64encode(QUIET).decode("ascii"))
        await srv._handle_live_event(loud, MagicMock(), queue)
        await srv._handle_live_event(quiet, MagicMock(), queue)
        assert queue.qsize() == 2  # a pause inside speech keeps the span continuous

    async def test_user_speech_stop_is_stamped_with_receive_time(self, tmp_path):
        srv = _make_server(tmp_path)
        # The bridge sends loop.time() (monotonic) here, not epoch ms.
        events = [
            json.dumps({"event": "user_speech_stop", "timestamp_ms": "12345"}),
            json.dumps({"event": "stop"}),
        ]
        websocket = MagicMock()
        websocket.receive_text = AsyncMock(side_effect=events)
        before = int(time.time() * 1000)
        await srv._forward_user_audio(websocket, MagicMock())
        assert before <= srv._tracker._user_stop_ms <= int(time.time() * 1000)

    async def test_session_started_sets_started_event(self, tmp_path):
        srv = _make_server(tmp_path)
        assert not srv._started.is_set()
        await srv._handle_live_event(SimpleNamespace(type="session.started"), MagicMock(), asyncio.Queue())
        assert srv._started.is_set()

    async def test_unknown_event_type_is_ignored(self, tmp_path):
        srv = _make_server(tmp_path)
        await srv._handle_live_event(SimpleNamespace(type="session.something.new"), MagicMock(), asyncio.Queue())

    async def test_transcript_events_produce_audit_entries(self, tmp_path):
        srv = _make_server(tmp_path)
        srv.audit_log = MagicMock()
        srv._fw_log = MagicMock()
        conn, queue = MagicMock(), asyncio.Queue()
        srv._tracker.on_user_speech_start(1000)
        srv._tracker.on_user_speech_stop(2000)
        await srv._handle_live_event(
            SimpleNamespace(type="session.input_transcript.delta", delta="I need a flight"), conn, queue
        )
        await srv._handle_live_event(
            SimpleNamespace(type="session.output_transcript.delta", delta="Sure."), conn, queue
        )
        srv.audit_log.append_user_input.assert_not_called()  # input transcripts may still be trailing
        srv._emit(srv._tracker.tick(int(time.time() * 1000) + USER_QUIET_FLUSH_MS + 10))
        srv.audit_log.append_user_input.assert_called_once()
        assert srv.audit_log.append_user_input.call_args.args == ("I need a flight",)

    async def test_function_call_splits_pre_tool_speech_into_its_own_turn(self, tmp_path):
        srv = _make_server(tmp_path)
        srv.audit_log = MagicMock()
        srv._fw_log = MagicMock()
        srv.execute_tool = AsyncMock(return_value={})
        conn, queue = MagicMock(), asyncio.Queue()
        await srv._handle_live_event(
            SimpleNamespace(type="session.output_transcript.delta", delta="Let me check."), conn, queue
        )
        await srv._handle_live_event(_call_done("call_1", "lookup_booking", "{}"), conn, queue)
        srv.audit_log.append_assistant_output.assert_called_once()
        assert srv.audit_log.append_assistant_output.call_args.args == ("Let me check.",)


@pytest.mark.asyncio
class TestSessionSetup:
    async def test_open_session_sends_session_start_payload(self, tmp_path):
        srv = _make_server(tmp_path)
        conn = MagicMock()
        conn.session.start = AsyncMock()
        await srv._open_session(conn)
        conn.session.start.assert_awaited_once_with(session=srv._build_session_config())

    async def test_greeting_waits_for_session_started_then_uses_commentary(self, tmp_path):
        srv = _make_server(tmp_path)
        conn = MagicMock()
        conn.session.commentary.append = AsyncMock()

        greeting = asyncio.create_task(srv._greet_when_started(conn))
        await asyncio.sleep(0.05)
        conn.session.commentary.append.assert_not_awaited()

        srv._started.set()
        await asyncio.wait_for(greeting, timeout=1)
        conn.session.commentary.append.assert_awaited_once()
        kwargs = conn.session.commentary.append.call_args.kwargs
        assert kwargs["delegation_id"] is None
        assert srv.initial_message in kwargs["content"]

    async def test_greeting_raises_when_session_never_starts(self, tmp_path, monkeypatch):
        monkeypatch.setattr("eva.assistant.openai_live_server.SESSION_START_TIMEOUT_S", 0.05)
        srv = _make_server(tmp_path)
        conn = MagicMock()
        conn.session.commentary.append = AsyncMock()
        with pytest.raises(RuntimeError, match="session.started"):
            await srv._greet_when_started(conn)
        conn.session.commentary.append.assert_not_awaited()


class _FakeLiveConn:
    """Stand-in for an AsyncLiveConnection: records sends, yields pushed server events."""

    def __init__(self):
        self.session = SimpleNamespace(
            start=AsyncMock(),
            commentary=SimpleNamespace(append=AsyncMock()),
            input_audio=SimpleNamespace(append=AsyncMock()),
        )
        self.response = SimpleNamespace(item=SimpleNamespace(create=AsyncMock()), create=AsyncMock())
        self._events: asyncio.Queue = asyncio.Queue()

    def push(self, event) -> None:
        self._events.put_nowait(event)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._events.get()


class _FakeConnectCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


def _fake_websocket() -> MagicMock:
    async def hang():
        await asyncio.sleep(3600)

    ws = MagicMock()
    ws.receive_text = AsyncMock(side_effect=hang)
    ws.send_text = AsyncMock()
    return ws


@pytest.mark.asyncio
class TestHandleSession:
    async def test_greeting_is_sent_once_session_started_arrives(self, tmp_path):
        srv = _make_server(tmp_path)
        conn = _FakeLiveConn()
        fake_client = MagicMock()
        fake_client.live.connect.return_value = _FakeConnectCM(conn)

        with patch("eva.assistant.openai_live_server.AsyncOpenAI", return_value=fake_client):
            session = asyncio.create_task(srv._handle_session(_fake_websocket()))
            await asyncio.sleep(0.1)
            conn.session.start.assert_awaited_once()
            conn.session.commentary.append.assert_not_awaited()  # not before session.started

            conn.push(SimpleNamespace(type="session.started"))
            await asyncio.sleep(0.2)
            conn.session.commentary.append.assert_awaited_once()
            kwargs = conn.session.commentary.append.call_args.kwargs
            assert kwargs["delegation_id"] is None and srv.initial_message in kwargs["content"]

            session.cancel()
            with pytest.raises(asyncio.CancelledError):
                await session


@pytest.mark.asyncio
class TestSilenceFeed:
    """Live's output is clocked by the input audio it receives, so idle gaps must be filled with silence."""

    async def _run_feed(self, srv, conn, seconds, during=None):
        task = asyncio.create_task(srv._feed_silence_to_live(conn))
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if during:
                during()
            await asyncio.sleep(0.02)
        task.cancel()
        await task

    async def test_feeds_20ms_silence_chunks_when_no_user_audio_arrives(self, tmp_path):
        srv = _make_server(tmp_path)
        srv._started.set()
        conn = _FakeLiveConn()
        await self._run_feed(srv, conn, 0.3)
        calls = conn.session.input_audio.append.await_args_list
        assert len(calls) >= 8
        for call in calls:
            assert base64.b64decode(call.kwargs["audio"]) == b"\x00" * (LIVE_SAMPLE_RATE // 50 * 2)

    async def test_sends_nothing_before_session_started(self, tmp_path):
        srv = _make_server(tmp_path)
        conn = _FakeLiveConn()
        await self._run_feed(srv, conn, 0.2)
        conn.session.input_audio.append.assert_not_awaited()

    async def test_sends_nothing_while_user_audio_is_flowing(self, tmp_path):
        srv = _make_server(tmp_path)
        srv._started.set()
        conn = _FakeLiveConn()

        def user_frame():
            srv._last_user_audio_mono = time.monotonic()

        await self._run_feed(srv, conn, 0.3, during=user_frame)
        conn.session.input_audio.append.assert_not_awaited()


def _info_messages(logger_mock) -> list[str]:
    return [call.args[0] for call in logger_mock.info.call_args_list]


@pytest.mark.asyncio
class TestDelegationLogging:
    async def test_delegation_created_records_its_start_time(self, tmp_path):
        srv = _make_server(tmp_path)
        event = SimpleNamespace(type="session.delegation.created", delegation=SimpleNamespace(id="d1"))
        await srv._handle_live_event(event, MagicMock(), asyncio.Queue())
        assert time.monotonic() - srv._delegation_started["d1"] < 1.0

    async def test_backend_answer_is_logged_with_text_and_latency(self):
        srv = _bare_server()
        srv._delegation_started["d1"] = time.monotonic() - 2.0
        done = _response_event({"type": "response.output_text.done", "text": "I found your reservation."})
        with patch("eva.assistant.openai_live_server.logger") as log:
            await srv._on_response_event(done, MagicMock())
        [msg] = [m for m in _info_messages(log) if "Backend answer" in m]
        assert "d1" in msg and "I found your reservation." in msg and "2.0s" in msg
        assert "d1" not in srv._delegation_started  # one answer per delegation: the start time is consumed

    async def test_backend_answer_without_a_recorded_start_is_still_logged(self):
        srv = _bare_server()
        done = _response_event({"type": "response.output_text.done", "text": "Done."})
        with patch("eva.assistant.openai_live_server.logger") as log:
            await srv._on_response_event(done, MagicMock())
        [msg] = [m for m in _info_messages(log) if "Backend answer" in m]
        assert "Done." in msg and "since" not in msg

    async def test_long_backend_answer_is_truncated(self):
        srv = _bare_server()
        done = _response_event({"type": "response.output_text.done", "text": "x" * 1000})
        with patch("eva.assistant.openai_live_server.logger") as log:
            await srv._on_response_event(done, MagicMock())
        [msg] = [m for m in _info_messages(log) if "Backend answer" in m]
        assert "x" * 300 in msg and "x" * 301 not in msg

    async def test_response_completed_logs_status_and_token_usage(self):
        srv = _bare_server()
        inner = {
            "type": "response.completed",
            "response": {"status": "completed", "usage": {"input_tokens": 1231, "output_tokens": 57}},
        }
        with patch("eva.assistant.openai_live_server.logger") as log:
            await srv._on_response_event(_response_event(inner), MagicMock())
        [msg] = [m for m in _info_messages(log) if "Backend response" in m]
        assert "d1" in msg and "completed" in msg and "1231" in msg and "57" in msg


@pytest.mark.asyncio
class TestDelegationFrameworkLog:
    """The delegation_rate metric reads these records, so their shape is a contract."""

    async def test_session_started_writes_the_delegation_session_marker(self, tmp_path):
        srv = _make_server(tmp_path)
        srv._fw_log = MagicMock()
        await srv._handle_live_event(SimpleNamespace(type="session.started"), MagicMock(), asyncio.Queue())
        srv._fw_log.write.assert_called_once_with("delegation_session", {"frame": "delegation_session"})

    async def test_delegation_created_writes_a_delegation_record(self, tmp_path):
        srv = _make_server(tmp_path)
        srv._fw_log = MagicMock()
        delegation = SimpleNamespace(id="item_1", response_id="resp_1")
        event = SimpleNamespace(type="session.delegation.created", delegation=delegation)
        await srv._handle_live_event(event, MagicMock(), asyncio.Queue())
        srv._fw_log.write.assert_called_once_with(
            "delegation", {"frame": "delegation", "delegation_id": "item_1", "response_id": "resp_1"}
        )

    async def test_delegation_events_without_a_framework_log_are_fine(self, tmp_path):
        srv = _make_server(tmp_path)
        srv._fw_log = None
        queue = asyncio.Queue()
        await srv._handle_live_event(SimpleNamespace(type="session.started"), MagicMock(), queue)
        event = SimpleNamespace(type="session.delegation.created", delegation=SimpleNamespace(id="d", response_id="r"))
        await srv._handle_live_event(event, MagicMock(), queue)
