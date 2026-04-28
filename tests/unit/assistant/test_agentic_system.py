"""Unit tests for AgenticSystem.process_query and ToolExecutor."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.agentic.system import GENERIC_ERROR, AgenticSystem
from eva.models.agents import AgentConfig, AgentTool, AgentToolParameter


def _make_agent(tools: list[AgentTool] | None = None) -> AgentConfig:
    """Create a minimal AgentConfig for testing."""
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="A test agent",
        role="You are a test agent.",
        instructions="Help the user.",
        tools=tools or [],
        tool_module_path="eva.assistant.tools.test_tools",
    )


def _make_tool(name: str = "get_reservation", tool_id: str = "t1") -> AgentTool:
    """Create a minimal AgentTool for testing."""
    return AgentTool(
        id=tool_id,
        name=name,
        description=f"Tool: {name}",
        required_parameters=[
            AgentToolParameter(name="confirmation_number", type="string", description="Confirmation number"),
        ],
    )


def _make_llm_response(content: str, tool_calls: list | None = None):
    """Create a mock LLM response object (mimics LiteLLM response.choices[0].message)."""
    response = SimpleNamespace(content=content, tool_calls=tool_calls, model="test-model")
    return response


def _conv_to_dicts(messages) -> list[dict]:
    """Convert ConversationMessage objects to plain dicts for assertion."""
    return [m.model_dump(exclude_none=True) for m in messages]


def _make_tool_call(call_id: str, name: str, arguments: str):
    """Create a mock tool call object (mimics LiteLLM ChatCompletionMessageToolCall)."""
    tc = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    tc.model_dump = lambda exclude_none=False: {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    return tc


class TestProcessQueryNoTools:
    """Tests for process_query when the LLM responds with plain text (no tool calls)."""

    @pytest.mark.asyncio
    async def test_simple_response(self):
        """LLM returns a text response with no tool calls."""
        agent = _make_agent()
        audit_log = AuditLog()
        llm_client = MagicMock()
        llm_client.complete = AsyncMock(
            return_value=(
                _make_llm_response("Hello, how can I help you?"),
                {"prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"},
            )
        )

        system = AgenticSystem(
            current_date_time="2026-02-25 10:00:00",
            agent=agent,
            tool_handler=MagicMock(),
            audit_log=audit_log,
            llm_client=llm_client,
        )

        responses = []
        async for msg in system.process_query("Hi there"):
            responses.append(msg)

        assert responses == ["Hello, how can I help you?"]
        llm_client.complete.assert_awaited_once()

        # Verify transcript
        transcript = audit_log.transcript
        message_types = [e["message_type"] for e in transcript]
        assert message_types == ["user", "llm_call", "assistant"]
        assert transcript[0]["value"] == "Hi there"
        assert transcript[1]["value"] == {"agent": "Test Agent", "response": "Hello, how can I help you?"}
        assert transcript[2]["value"] == "Hello, how can I help you?"

        # Verify conversation messages
        assert _conv_to_dicts(audit_log.get_conversation_messages()) == [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello, how can I help you?"},
        ]


class TestProcessQueryWithToolCall:
    @pytest.mark.asyncio
    async def test_single_tool_call_then_response(self):
        """LLM calls a tool, gets the result, then responds with text."""
        tool = _make_tool("get_reservation")
        agent = _make_agent(tools=[tool])
        audit_log = AuditLog()

        tool_call = _make_tool_call("call_1", "get_reservation", '{"confirmation_number": "ABC123"}')

        # First LLM call returns a tool call, second returns final text
        llm_client = MagicMock()
        llm_client.complete = AsyncMock(
            side_effect=[
                (
                    _make_llm_response("What if there is text here", tool_calls=[tool_call]),
                    {"prompt_tokens": 20, "completion_tokens": 10, "finish_reason": "tool_calls"},
                ),
                (
                    _make_llm_response("Your reservation ABC123 is confirmed."),
                    {"prompt_tokens": 30, "completion_tokens": 15, "finish_reason": "stop"},
                ),
            ]
        )

        tool_handler = MagicMock()
        tool_handler.execute = AsyncMock(
            return_value={
                "status": "success",
                "reservation": {"confirmation_number": "ABC123", "status": "confirmed"},
            }
        )

        system = AgenticSystem(
            current_date_time="2026-02-25 10:00:00",
            agent=agent,
            tool_handler=tool_handler,
            audit_log=audit_log,
            llm_client=llm_client,
        )

        responses = []
        async for msg in system.process_query("Check reservation ABC123"):
            responses.append(msg)

        assert responses == ["What if there is text here", "Your reservation ABC123 is confirmed."]

        # Verify tool was executed with correct params
        tool_handler.execute.assert_awaited_once_with("get_reservation", {"confirmation_number": "ABC123"})

        # Verify LLM was called twice (tool call + final response)
        assert llm_client.complete.await_count == 2

        # Verify transcript — content alongside tool calls now appears as an assistant entry
        transcript = audit_log.transcript
        message_types = [e["message_type"] for e in transcript]
        assert message_types == [
            "user",
            "llm_call",
            "assistant",
            "tool_call",
            "tool_response",
            "llm_call",
            "assistant",
        ]
        assert transcript[0]["value"] == "Check reservation ABC123"
        assert transcript[2]["value"] == "What if there is text here"
        assert transcript[3]["value"]["tool"] == "get_reservation"
        assert transcript[4]["value"]["response"]["status"] == "success"
        assert transcript[6]["value"] == "Your reservation ABC123 is confirmed."

        # Verify conversation messages — content is preserved even with tool calls
        assert _conv_to_dicts(audit_log.get_conversation_messages()) == [
            {"role": "user", "content": "Check reservation ABC123"},
            {
                "role": "assistant",
                "content": "What if there is text here",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_reservation", "arguments": '{"confirmation_number": "ABC123"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "status": "success",
                        "reservation": {"confirmation_number": "ABC123", "status": "confirmed"},
                    }
                ),
                "tool_call_id": "call_1",
            },
            {"role": "assistant", "content": "Your reservation ABC123 is confirmed."},
        ]

    @pytest.mark.asyncio
    async def test_tool_call_with_error_result(self):
        """LLM calls a tool that returns an error, then responds."""
        tool = _make_tool("get_reservation")
        agent = _make_agent(tools=[tool])
        audit_log = AuditLog()

        tool_call = _make_tool_call("call_1", "get_reservation", '{"confirmation_number": "INVALID"}')

        llm_client = MagicMock()
        llm_client.complete = AsyncMock(
            side_effect=[
                (
                    _make_llm_response("", tool_calls=[tool_call]),
                    {"prompt_tokens": 20, "completion_tokens": 10, "finish_reason": "tool_calls"},
                ),
                (
                    _make_llm_response("I couldn't find that reservation."),
                    {"prompt_tokens": 30, "completion_tokens": 10, "finish_reason": "stop"},
                ),
            ]
        )

        tool_handler = MagicMock()
        tool_handler.execute = AsyncMock(
            return_value={
                "status": "error",
                "error_type": "not_found",
                "message": "No reservation found",
            }
        )

        system = AgenticSystem(
            current_date_time="2026-02-25 10:00:00",
            agent=agent,
            tool_handler=tool_handler,
            audit_log=audit_log,
            llm_client=llm_client,
        )

        responses = []
        async for msg in system.process_query("Check reservation INVALID"):
            responses.append(msg)

        assert responses == ["I couldn't find that reservation."]

        # Verify transcript
        transcript = audit_log.transcript
        message_types = [e["message_type"] for e in transcript]
        assert message_types == ["user", "llm_call", "tool_call", "tool_response", "llm_call", "assistant"]
        assert transcript[2]["value"]["tool"] == "get_reservation"
        assert transcript[3]["value"]["response"]["status"] == "error"
        assert transcript[5]["value"] == "I couldn't find that reservation."

        # Verify conversation messages
        assert _conv_to_dicts(audit_log.get_conversation_messages()) == [
            {"role": "user", "content": "Check reservation INVALID"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_reservation", "arguments": '{"confirmation_number": "INVALID"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "status": "error",
                        "error_type": "not_found",
                        "message": "No reservation found",
                    }
                ),
                "tool_call_id": "call_1",
            },
            {"role": "assistant", "content": "I couldn't find that reservation."},
        ]


class TestProcessQueryTransfer:
    @pytest.mark.asyncio
    async def test_transfer_to_agent(self):
        """LLM calls transfer_to_agent, conversation ends with transfer message."""
        transfer_tool = AgentTool(
            id="t_transfer",
            name="transfer_to_agent",
            description="Transfer to a live agent",
        )
        agent = _make_agent(tools=[transfer_tool])
        audit_log = AuditLog()

        tool_call = _make_tool_call("call_t", "transfer_to_agent", "{}")

        llm_client = MagicMock()
        llm_client.complete = AsyncMock(
            return_value=(
                _make_llm_response("", tool_calls=[tool_call]),
                {"prompt_tokens": 20, "completion_tokens": 5, "finish_reason": "tool_calls"},
            )
        )

        tool_handler = MagicMock()
        tool_handler.execute = AsyncMock()

        system = AgenticSystem(
            current_date_time="2026-02-25 10:00:00",
            agent=agent,
            tool_handler=tool_handler,
            audit_log=audit_log,
            llm_client=llm_client,
        )

        responses = []
        async for msg in system.process_query("I need to talk to a human"):
            responses.append(msg)

        assert responses == ["Transferring you to a live agent. Please wait."]

        # Tool handler should NOT have been called (transfer is special)
        tool_handler.execute.assert_not_awaited()

        # LLM should only be called once (no continuation after transfer)
        llm_client.complete.assert_awaited_once()

        # Verify transcript
        transcript = audit_log.transcript
        message_types = [e["message_type"] for e in transcript]
        assert message_types == ["user", "llm_call", "tool_call", "tool_response", "assistant"]
        assert transcript[2]["value"]["tool"] == "transfer_to_agent"
        assert transcript[3]["value"]["response"] == {"status": "transfer_initiated"}
        assert transcript[4]["value"] == "Transferring you to a live agent. Please wait."

        # Verify conversation messages
        assert _conv_to_dicts(audit_log.get_conversation_messages()) == [
            {"role": "user", "content": "I need to talk to a human"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_t", "type": "function", "function": {"name": "transfer_to_agent", "arguments": "{}"}}
                ],
            },
            {"role": "assistant", "content": "Transferring you to a live agent. Please wait."},
        ]


class TestResponsesOutputItemsThreading:
    @pytest.mark.asyncio
    async def test_responses_output_items_passed_to_next_llm_call(self):
        """When stats includes responses_output_items (OpenAI Responses API encrypted reasoning).

        They are attached to the assistant message and forwarded to the next LLM call.
        """
        tool = _make_tool("get_info")
        agent = _make_agent(tools=[tool])
        audit_log = AuditLog()

        tool_call = _make_tool_call("call_1", "get_info", '{"id": "42"}')
        output_items = [
            {"type": "reasoning", "encrypted_content": "enc_abc"},
            {"type": "function_call", "call_id": "call_1", "name": "get_info", "arguments": '{"id": "42"}'},
        ]

        llm_client = MagicMock()
        llm_client.complete = AsyncMock(
            side_effect=[
                (
                    _make_llm_response("", tool_calls=[tool_call]),
                    {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "finish_reason": "tool_calls",
                        "responses_output_items": output_items,
                        "reasoning_content": "enc_abc",
                    },
                ),
                (
                    _make_llm_response("Here is the result."),
                    {"prompt_tokens": 30, "completion_tokens": 10, "finish_reason": "stop"},
                ),
            ]
        )

        tool_handler = MagicMock()
        tool_handler.execute = AsyncMock(return_value={"status": "success", "data": "some data"})

        system = AgenticSystem(
            current_date_time="2026-02-25 10:00:00",
            agent=agent,
            tool_handler=tool_handler,
            audit_log=audit_log,
            llm_client=llm_client,
        )

        responses = []
        async for msg in system.process_query("Get info for 42"):
            responses.append(msg)

        assert responses == ["Here is the result."]

        # Verify the second LLM call received the assistant message with responses_output_items
        second_call_messages = llm_client.complete.call_args_list[1][0][0]
        assistant_msgs = [m for m in second_call_messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["responses_output_items"] == output_items


class TestProcessQueryLLMError:
    @pytest.mark.asyncio
    async def test_llm_error_yields_generic_error(self):
        """LLM raises an exception, system yields the generic error message."""
        agent = _make_agent()
        audit_log = AuditLog()

        llm_client = MagicMock()
        llm_client.complete = AsyncMock(side_effect=Exception("API rate limit exceeded"))

        system = AgenticSystem(
            current_date_time="2026-02-25 10:00:00",
            agent=agent,
            tool_handler=MagicMock(),
            audit_log=audit_log,
            llm_client=llm_client,
        )

        responses = []
        async for msg in system.process_query("Hello"):
            responses.append(msg)

        assert responses == [GENERIC_ERROR]

        # Verify transcript - user input + failed llm_call
        transcript = audit_log.transcript
        message_types = [e["message_type"] for e in transcript]
        assert message_types == ["user", "llm_call"]
        assert transcript[0]["value"] == "Hello"

        # Verify conversation messages - only user input (no assistant output on error)
        assert _conv_to_dicts(audit_log.get_conversation_messages()) == [
            {"role": "user", "content": "Hello"},
        ]

    @pytest.mark.asyncio
    async def test_cancellation_error_yields_nothing(self):
        """asyncio.CancelledError should exit gracefully with no output."""
        import asyncio

        agent = _make_agent()
        audit_log = AuditLog()

        llm_client = MagicMock()
        llm_client.complete = AsyncMock(side_effect=asyncio.CancelledError())

        system = AgenticSystem(
            current_date_time="2026-02-25 10:00:00",
            agent=agent,
            tool_handler=MagicMock(),
            audit_log=audit_log,
            llm_client=llm_client,
        )

        responses = []
        async for msg in system.process_query("Hello"):
            responses.append(msg)

        assert responses == []

        # Verify transcript - only user input (graceful exit, no llm_call logged)
        transcript = audit_log.transcript
        assert len(transcript) == 1
        assert transcript[0]["message_type"] == "user"
        assert transcript[0]["value"] == "Hello"

        # Verify conversation messages
        assert _conv_to_dicts(audit_log.get_conversation_messages()) == [
            {"role": "user", "content": "Hello"},
        ]
