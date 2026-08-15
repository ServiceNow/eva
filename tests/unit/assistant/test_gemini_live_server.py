"""Tests for Gemini Live blocking tool-result batching."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eva.assistant.gemini_live_server import GeminiLiveAssistantServer


def _bare_server() -> GeminiLiveAssistantServer:
    return object.__new__(GeminiLiveAssistantServer)


@pytest.mark.asyncio
async def test_multiple_blocking_tool_results_are_sent_as_one_batch():
    srv = _bare_server()
    srv.execute_tool = AsyncMock(
        side_effect=[
            {"status": "success", "value": "one"},
            {"status": "success", "value": "two"},
        ]
    )
    session = MagicMock()
    session.send_tool_response = AsyncMock()
    tool_call = SimpleNamespace(
        function_calls=[
            SimpleNamespace(id="call_1", name="first_tool", args={"n": 1}),
            SimpleNamespace(id="call_2", name="second_tool", args={"n": 2}),
        ]
    )

    await srv._handle_tool_call(tool_call, session)

    assert srv.execute_tool.await_count == 2
    session.send_tool_response.assert_awaited_once()
    responses = session.send_tool_response.await_args.kwargs["function_responses"]
    assert [response.id for response in responses] == ["call_1", "call_2"]
    assert [response.name for response in responses] == ["first_tool", "second_tool"]
    assert [response.response["value"] for response in responses] == ["one", "two"]
    assert all(response.scheduling is None for response in responses)


@pytest.mark.asyncio
async def test_empty_tool_batch_sends_nothing():
    srv = _bare_server()
    srv.execute_tool = AsyncMock()
    session = MagicMock()
    session.send_tool_response = AsyncMock()

    await srv._handle_tool_call(SimpleNamespace(function_calls=[]), session)

    srv.execute_tool.assert_not_awaited()
    session.send_tool_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_tool_does_not_drop_other_results_from_batch():
    srv = _bare_server()
    srv.execute_tool = AsyncMock(
        side_effect=[
            ValueError("malformed result"),
            {"status": "success", "value": "two"},
        ]
    )
    srv.audit_log = MagicMock()
    session = MagicMock()
    session.send_tool_response = AsyncMock()
    tool_call = SimpleNamespace(
        function_calls=[
            SimpleNamespace(id="call_1", name="first_tool", args={}),
            SimpleNamespace(id="call_2", name="second_tool", args={}),
        ]
    )

    await srv._handle_tool_call(tool_call, session)

    responses = session.send_tool_response.await_args.kwargs["function_responses"]
    assert len(responses) == 2
    assert responses[0].response["status"] == "error"
    assert responses[1].response["status"] == "success"
    srv.audit_log.append_tool_response.assert_called_once()
