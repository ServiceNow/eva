"""Unit tests for CASCADE latency optimizations."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.agentic.system import GENERIC_ERROR, AgenticSystem
from eva.models.agents import AgentConfig, AgentTool, AgentToolParameter

_STATS = {
    "model": "test-model",
    "prompt_tokens": 1,
    "completion_tokens": 1,
    "reasoning_tokens": 0,
    "finish_reason": "stop",
    "cost": 0.0,
    "cost_source": "litellm",
    "latency": 0.1,
    "reasoning": None,
    "reasoning_content": None,
    "thinking_blocks": None,
    "responses_output_items": None,
}


def _agent(tools=None):
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="A test agent",
        role="You are a test agent.",
        instructions="Help the user.",
        tools=tools or [],
        tool_module_path="eva.assistant.tools.test_tools",
    )


def _tool(name="get_reservation"):
    return AgentTool(
        id="t1",
        name=name,
        description=f"Tool: {name}",
        required_parameters=[AgentToolParameter(name="confirmation_number", type="string", description="code")],
    )


def _msg(content, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls, model="test-model")


def _tc(call_id, name, arguments):
    tc = SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))
    tc.model_dump = lambda exclude_none=False: {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    return tc


class _StreamLLM:
    use_responses_api = False

    def __init__(self, turns):
        self._turns = list(turns)
        self._i = 0

    async def complete_stream(self, messages, tools=None):
        deltas, final = self._turns[self._i]
        self._i += 1
        for d in deltas:
            yield ("delta", d)
        yield ("final", (final, dict(_STATS)))


class _CancellingStreamLLM:
    use_responses_api = False

    def __init__(self):
        self.calls = 0

    async def complete_stream(self, messages, tools=None):
        self.calls += 1
        yield ("delta", "Let me check. ")
        yield ("delta", "I")
        raise asyncio.CancelledError()


class _FailingStreamLLM:
    use_responses_api = False

    def __init__(self):
        self.calls = 0

    async def complete_stream(self, messages, tools=None):
        self.calls += 1
        yield ("delta", "Let me check. ")
        yield ("delta", "I")
        raise RuntimeError("stream dropped")


async def _collect(system, query):
    out = []
    async for r in system.process_query(query):
        out.append(r)
    return out


class TestPreToolSpeechDirective:
    @pytest.mark.parametrize("mode,present", [("off", False), ("auto", True)])
    def test_directive_injection(self, mode, present):
        system = AgenticSystem(
            current_date_time="2026-02-25 10:00:00",
            agent=_agent(),
            tool_handler=MagicMock(),
            audit_log=AuditLog(),
            llm_client=MagicMock(),
            pre_tool_speech=mode,
        )
        assert ("Responsiveness" in system.system_prompt) is present


class TestSilentToolCallNoFabricatedSpeech:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["off", "auto"])
    async def test_silent_tool_call_speaks_only_final(self, mode):
        llm = MagicMock()
        llm.use_responses_api = False
        llm.complete = AsyncMock(
            side_effect=[
                (_msg("", tool_calls=[_tc("c1", "get_reservation", "{}")]), dict(_STATS)),
                (_msg("All set."), dict(_STATS)),
            ]
        )
        tool_handler = MagicMock()
        tool_handler.execute = AsyncMock(return_value={"status": "success"})
        system = AgenticSystem("x", _agent([_tool()]), tool_handler, AuditLog(), llm, pre_tool_speech=mode)
        assert await _collect(system, "look up A") == ["All set."]

    @pytest.mark.asyncio
    async def test_model_lead_in_is_spoken(self):
        llm = MagicMock()
        llm.use_responses_api = False
        llm.complete = AsyncMock(
            side_effect=[
                (_msg("Sure, one sec.", tool_calls=[_tc("c1", "get_reservation", "{}")]), dict(_STATS)),
                (_msg("All set."), dict(_STATS)),
            ]
        )
        tool_handler = MagicMock()
        tool_handler.execute = AsyncMock(return_value={"status": "success"})
        system = AgenticSystem("x", _agent([_tool()]), tool_handler, AuditLog(), llm, pre_tool_speech="auto")
        assert await _collect(system, "look up A") == ["Sure, one sec.", "All set."]


class TestLLMStreaming:
    @pytest.mark.asyncio
    async def test_streams_sentences_without_double_speak(self):
        turns = [
            (["Sure, Mr. ", "Smith. The fee ", "is 3.5 dollars."], _msg("Sure, Mr. Smith. The fee is 3.5 dollars."))
        ]
        system = AgenticSystem("x", _agent(), MagicMock(), AuditLog(), _StreamLLM(turns), llm_streaming=True)
        assert await _collect(system, "hi") == ["Sure, Mr. Smith.", "The fee is 3.5 dollars."]

    @pytest.mark.asyncio
    async def test_streaming_silent_tool_call_no_fabricated_speech(self):
        turns = [
            ([], _msg("", tool_calls=[_tc("c1", "get_reservation", "{}")])),
            (["All set."], _msg("All set.")),
        ]
        tool_handler = MagicMock()
        tool_handler.execute = AsyncMock(return_value={"status": "success"})
        system = AgenticSystem(
            "x",
            _agent([_tool()]),
            tool_handler,
            AuditLog(),
            _StreamLLM(turns),
            pre_tool_speech="auto",
            llm_streaming=True,
        )
        assert await _collect(system, "look up A") == ["All set."]

    @pytest.mark.asyncio
    async def test_streaming_cancellation_records_spoken_prefix(self):
        audit_log = AuditLog()
        llm = _CancellingStreamLLM()
        system = AgenticSystem("x", _agent(), MagicMock(), audit_log, llm, llm_streaming=True)

        assert await _collect(system, "hi") == ["Let me check."]
        assert llm.calls == 1
        assert [m.model_dump(exclude_none=True) for m in audit_log.get_conversation_messages()] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Let me check."},
        ]

    @pytest.mark.asyncio
    async def test_streaming_failure_records_prefix_without_generic_error(self):
        audit_log = AuditLog()
        llm = _FailingStreamLLM()
        system = AgenticSystem("x", _agent(), MagicMock(), audit_log, llm, llm_streaming=True)

        assert await _collect(system, "hi") == ["Let me check."]
        assert llm.calls == 1
        assert [m.model_dump(exclude_none=True) for m in audit_log.get_conversation_messages()] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Let me check."},
        ]
        assert all(entry.get("value") != GENERIC_ERROR for entry in audit_log.transcript)

    @pytest.mark.asyncio
    async def test_responses_api_streaming_falls_back_with_one_warning(self, caplog):
        llm = MagicMock()
        llm.use_responses_api = True
        llm.complete = AsyncMock(
            side_effect=[
                (_msg("First response."), dict(_STATS)),
                (_msg("Second response."), dict(_STATS)),
            ]
        )
        llm.complete_stream = MagicMock()
        system = AgenticSystem("x", _agent(), MagicMock(), AuditLog(), llm, llm_streaming=True)

        with caplog.at_level("WARNING", logger="eva.assistant.agentic.system"):
            assert await _collect(system, "hi") == ["First response."]
            assert await _collect(system, "again") == ["Second response."]

        assert llm.complete.await_count == 2
        llm.complete_stream.assert_not_called()
        assert caplog.text.count("llm_streaming is not supported for Responses API deployments") == 1


class TestCompleteStream:
    @pytest.mark.asyncio
    async def test_yields_deltas_then_assembled_final(self, monkeypatch):
        from eva.assistant.services import llm as llm_mod

        def _chunk(content):
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=None)]
            )

        async def _acompletion(**kwargs):
            async def gen():
                for c in ["Hello ", "there."]:
                    yield _chunk(c)

            return gen()

        mock_router = MagicMock()
        mock_router.model_list = []
        mock_router.acompletion = _acompletion
        monkeypatch.setattr(llm_mod.router, "get", lambda: mock_router)

        assembled = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hello there.", tool_calls=[], reasoning_content=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
            model="test-model",
            _hidden_params={},
        )
        monkeypatch.setattr(llm_mod.litellm, "stream_chunk_builder", lambda chunks, messages=None: assembled)

        client = llm_mod.LiteLLMClient(model="test-model")
        events = [ev async for ev in client.complete_stream([{"role": "user", "content": "hi"}])]

        deltas = [p for k, p in events if k == "delta"]
        finals = [p for k, p in events if k == "final"]
        assert deltas == ["Hello ", "there."]
        assert len(finals) == 1
        message, stats = finals[0]
        assert message.content == "Hello there."
        assert stats["model"] == "test-model" and "latency" in stats
