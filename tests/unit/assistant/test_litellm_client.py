"""Unit tests for LiteLLMClient — Responses API helpers.

Covers:
- _convert_tools_for_responses_api: pure schema conversion
- _convert_messages_for_responses_api: pure message conversion with all three assistant cases
- _complete_via_responses_api: reasoning extraction (encrypted fallback, human-readable summary)
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eva.assistant.services.llm import LiteLLMClient
from eva.utils import router

# ---------------------------------------------------------------------------
# _convert_tools_for_responses_api
# ---------------------------------------------------------------------------


class TestConvertToolsForResponsesApi:
    def test_strips_function_wrapper(self):
        """Chat-completions tool format is flattened to Responses API format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_reservation",
                    "description": "Look up a reservation",
                    "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
                },
            }
        ]
        result = LiteLLMClient._convert_tools_for_responses_api(tools)
        assert result == [
            {
                "type": "function",
                "name": "get_reservation",
                "description": "Look up a reservation",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
            }
        ]

    def test_multiple_tools(self):
        """All tools in the list are converted."""
        tools = [
            {"type": "function", "function": {"name": "tool_a", "description": "A", "parameters": {}}},
            {"type": "function", "function": {"name": "tool_b", "description": "B", "parameters": {}}},
        ]
        result = LiteLLMClient._convert_tools_for_responses_api(tools)
        assert [t["name"] for t in result] == ["tool_a", "tool_b"]


# ---------------------------------------------------------------------------
# _convert_messages_for_responses_api
# ---------------------------------------------------------------------------


class TestConvertMessagesForResponsesApi:
    def test_system_extracted_as_instructions(self):
        """System message becomes instructions; user message becomes input item."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        instructions, items = LiteLLMClient._convert_messages_for_responses_api(messages)
        assert instructions == "You are helpful"
        assert items == [{"role": "user", "content": "Hello"}]

    def test_assistant_tool_calls_reconstructed_and_tool_messages_consumed(self):
        """Assistant message with tool_calls is converted to function_call items.

        The immediately following tool messages are consumed as function_call_output.
        """
        messages = [
            {"role": "user", "content": "Check reservation"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "function": {"name": "get_reservation", "arguments": '{"id": "1"}'}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "ok"}'},
            {"role": "user", "content": "Thanks"},
        ]
        _, items = LiteLLMClient._convert_messages_for_responses_api(messages)
        assert items == [
            {"role": "user", "content": "Check reservation"},
            {"type": "function_call", "call_id": "call_1", "name": "get_reservation", "arguments": '{"id": "1"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": '{"status": "ok"}'},
            {"role": "user", "content": "Thanks"},
        ]

    def test_assistant_tool_calls_with_text_content_preserved(self):
        """When an assistant message has both content and tool_calls, content is emitted before tool items."""
        messages = [
            {"role": "user", "content": "Go"},
            {
                "role": "assistant",
                "content": "Let me check that.",
                "tool_calls": [{"id": "c1", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "found"},
        ]
        _, items = LiteLLMClient._convert_messages_for_responses_api(messages)
        assert items == [
            {"role": "user", "content": "Go"},
            {"role": "assistant", "content": "Let me check that."},
            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "found"},
        ]

    def test_responses_output_items_injected_directly(self):
        """When an assistant message has responses_output_items (current-turn loop).

        Those raw items are injected directly into input_items without reconstruction.
        """
        raw_items = [
            {"type": "reasoning", "encrypted_content": "enc_abc"},
            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
        ]
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "", "responses_output_items": raw_items},
        ]
        _, items = LiteLLMClient._convert_messages_for_responses_api(messages)
        assert items == [{"role": "user", "content": "Hello"}] + raw_items


# ---------------------------------------------------------------------------
# _complete_via_responses_api — reasoning extraction
# ---------------------------------------------------------------------------


def _make_mock_router_with_no_deployments():
    """Return a mock router whose model_list is empty (no credential lookup needed)."""
    mock_router = MagicMock()
    mock_router.model_list = []
    return mock_router


def _make_reasoning_item(summary_texts: list[str], encrypted_content: str) -> MagicMock:
    item = MagicMock()
    item.type = "reasoning"
    item.summary = [SimpleNamespace(text=t) for t in summary_texts]
    item.encrypted_content = encrypted_content
    item.model_dump.return_value = {"type": "reasoning", "encrypted_content": encrypted_content}
    return item


def _make_message_item(text: str) -> MagicMock:
    item = MagicMock()
    item.type = "message"
    item.content = [SimpleNamespace(text=text)]
    item.model_dump.return_value = {"type": "message"}
    return item


def _make_mock_response(output_items: list, reasoning_tokens: int = 0) -> MagicMock:
    response = MagicMock()
    response.output = output_items
    response.usage = MagicMock(
        input_tokens=100,
        output_tokens=50,
        output_tokens_details=MagicMock(reasoning_tokens=reasoning_tokens),
    )
    response.model = "gpt-5.2"
    response.status = "completed"
    return response


class TestCompleteViaResponsesApiReasoning:
    @pytest.fixture(autouse=True, scope="class")
    def _init_router(self):
        router.init(
            model_list=[
                {
                    "model_name": "gpt-5.2",
                    "litellm_params": {
                        "model": "openai/gpt-5.2",
                        "api_key": "test-key",
                    },
                }
            ]
        )
        yield
        router.reset()

    @pytest.mark.asyncio
    async def test_encrypted_content_used_when_no_human_readable_summary(self):
        """When the reasoning item has no summary, encrypted_content becomes the reasoning value.

        This is the gpt-5.2 case: no human-readable text is returned, but the encrypted
        blob must flow through the system as the 'reasoning text' equivalent.
        """
        client = LiteLLMClient(model="gpt-5.2")

        reasoning_item = _make_reasoning_item(summary_texts=[], encrypted_content="enc_blob_xyz")
        message_item = _make_message_item("Here is your answer.")
        mock_response = _make_mock_response([reasoning_item, message_item], reasoning_tokens=42)

        with (
            patch("eva.assistant.services.llm.router.get", return_value=_make_mock_router_with_no_deployments()),
            patch("litellm.aresponses", new_callable=AsyncMock, return_value=mock_response),
        ):
            result, stats = await client.complete(messages=[{"role": "user", "content": "Hello"}])

        assert result == "Here is your answer."
        assert stats["reasoning"] == "enc_blob_xyz"
        assert stats["reasoning_content"] == "enc_blob_xyz"
        assert stats["reasoning_tokens"] == 42
        assert stats["responses_output_items"] is not None

    @pytest.mark.asyncio
    async def test_human_readable_summary_preferred_over_encrypted_content(self):
        """When the reasoning item has a human-readable summary, it takes priority."""
        client = LiteLLMClient(model="gpt-5.2")

        reasoning_item = _make_reasoning_item(
            summary_texts=["I thought about this carefully."],
            encrypted_content="enc_blob_xyz",
        )
        message_item = _make_message_item("Done.")
        mock_response = _make_mock_response([reasoning_item, message_item], reasoning_tokens=10)

        with (
            patch("eva.assistant.services.llm.router.get", return_value=_make_mock_router_with_no_deployments()),
            patch("litellm.aresponses", new_callable=AsyncMock, return_value=mock_response),
        ):
            _, stats = await client.complete(messages=[{"role": "user", "content": "Hi"}])

        assert stats["reasoning"] == "I thought about this carefully."

    @pytest.mark.asyncio
    async def test_tool_call_response_includes_output_items(self):
        """Tool call responses return a SimpleNamespace with tool_calls.

        Responses_output_items in stats contains all output items for the next turn.
        """
        client = LiteLLMClient(model="gpt-5.2")

        reasoning_item = _make_reasoning_item([], "enc_abc")

        fn_call_item = MagicMock()
        fn_call_item.type = "function_call"
        fn_call_item.call_id = "call_1"
        fn_call_item.name = "get_info"
        fn_call_item.arguments = '{"id": "42"}'
        fn_call_item.model_dump.return_value = {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_info",
            "arguments": '{"id": "42"}',
        }

        mock_response = _make_mock_response([reasoning_item, fn_call_item], reasoning_tokens=20)

        with (
            patch("eva.assistant.services.llm.router.get", return_value=_make_mock_router_with_no_deployments()),
            patch("litellm.aresponses", new_callable=AsyncMock, return_value=mock_response),
        ):
            result, stats = await client.complete(
                messages=[{"role": "user", "content": "Get info"}],
                tools=[{"type": "function", "function": {"name": "get_info", "description": "d", "parameters": {}}}],
            )

        # Should return a message object (not a string) when there are tool calls
        assert hasattr(result, "tool_calls")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_info"

        # responses_output_items must be present for next-turn threading
        output_items = stats["responses_output_items"]
        assert output_items is not None
        assert any(it["type"] == "reasoning" for it in output_items)
        assert any(it["type"] == "function_call" for it in output_items)


def _standard_chat_response():
    msg = SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None, thinking_blocks=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        model="gpt-5.2",
        _hidden_params={},
    )


_A_TOOL = [
    {
        "type": "function",
        "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}},
    }
]


async def _empty_stream():
    for chunk in ():
        yield chunk


class TestParallelToolCallsForwarding:
    @pytest.fixture(autouse=True, scope="class")
    def _init_router(self):
        router.init(
            model_list=[
                {
                    "model_name": "gpt-5.2",
                    "litellm_params": {"model": "openai/gpt-5.2", "api_key": "test-key"},
                    "use_chat_completions": True,
                }
            ]
        )
        yield
        router.reset()

    def _mock_router(self):
        r = MagicMock()
        r.model_list = []
        r.acompletion = AsyncMock(return_value=_standard_chat_response())
        return r

    @pytest.mark.asyncio
    async def test_forwarded_when_set_with_tools(self):
        client = LiteLLMClient(model="gpt-5.2", parallel_tool_calls=False)
        r = self._mock_router()
        with patch("eva.assistant.services.llm.router.get", return_value=r):
            await client.complete([{"role": "user", "content": "hi"}], tools=_A_TOOL)
        assert r.acompletion.call_args.kwargs["parallel_tool_calls"] is False

    @pytest.mark.asyncio
    async def test_absent_when_none(self):
        client = LiteLLMClient(model="gpt-5.2")
        r = self._mock_router()
        with patch("eva.assistant.services.llm.router.get", return_value=r):
            await client.complete([{"role": "user", "content": "hi"}], tools=_A_TOOL)
        assert "parallel_tool_calls" not in r.acompletion.call_args.kwargs

    @pytest.mark.asyncio
    async def test_absent_when_no_tools(self):
        client = LiteLLMClient(model="gpt-5.2", parallel_tool_calls=True)
        r = self._mock_router()
        with patch("eva.assistant.services.llm.router.get", return_value=r):
            await client.complete([{"role": "user", "content": "hi"}])
        assert "parallel_tool_calls" not in r.acompletion.call_args.kwargs

    @pytest.mark.asyncio
    async def test_streaming_forwards_when_set(self):
        client = LiteLLMClient(model="gpt-5.2", parallel_tool_calls=False)

        r = MagicMock()
        r.model_list = []
        r.acompletion = AsyncMock(return_value=_empty_stream())
        with (
            patch("eva.assistant.services.llm.router.get", return_value=r),
            patch("litellm.stream_chunk_builder", return_value=_standard_chat_response()),
        ):
            async for _kind, _payload in client.complete_stream([{"role": "user", "content": "hi"}], tools=_A_TOOL):
                pass
        assert r.acompletion.call_args.kwargs["parallel_tool_calls"] is False

    @pytest.mark.asyncio
    async def test_streaming_absent_when_no_tools(self):
        client = LiteLLMClient(model="gpt-5.2", parallel_tool_calls=True)

        r = MagicMock()
        r.model_list = []
        r.acompletion = AsyncMock(return_value=_empty_stream())
        with (
            patch("eva.assistant.services.llm.router.get", return_value=r),
            patch("litellm.stream_chunk_builder", return_value=_standard_chat_response()),
        ):
            async for _kind, _payload in client.complete_stream([{"role": "user", "content": "hi"}]):
                pass
        assert "parallel_tool_calls" not in r.acompletion.call_args.kwargs


class TestCompleteStreamViaResponsesApi:
    @pytest.fixture(autouse=True, scope="class")
    def _init_router(self):
        router.init(
            model_list=[
                {
                    "model_name": "gpt-5.2",
                    "litellm_params": {"model": "openai/gpt-5.2", "api_key": "test-key"},
                }
            ]
        )
        yield
        router.reset()

    @staticmethod
    def _stream_of(events):
        async def _aresponses(**kwargs):
            assert kwargs["stream"] is True

            async def gen():
                for e in events:
                    yield e

            return gen()

        return _aresponses

    @pytest.mark.asyncio
    async def test_yields_deltas_then_final_from_completed_event(self):
        client = LiteLLMClient(model="gpt-5.2")
        response = _make_mock_response([_make_message_item("Hello there.")])
        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta="Hello "),
            SimpleNamespace(type="response.output_text.delta", delta="there."),
            SimpleNamespace(type="response.completed", response=response),
        ]

        with (
            patch("eva.assistant.services.llm.router.get", return_value=_make_mock_router_with_no_deployments()),
            patch("litellm.aresponses", self._stream_of(events)),
        ):
            out = [item async for item in client.complete_stream([{"role": "user", "content": "hi"}])]

        assert out[:2] == [("delta", "Hello "), ("delta", "there.")]
        kind, (message, stats) = out[2]
        assert kind == "final"
        assert message == "Hello there."
        assert stats["responses_output_items"] is not None

    @pytest.mark.asyncio
    async def test_function_call_from_completed_event(self):
        client = LiteLLMClient(model="gpt-5.2")
        item = MagicMock()
        item.type = "function_call"
        item.call_id = "call_1"
        item.name = "get_weather"
        item.arguments = '{"location":"Paris, France"}'
        item.model_dump.return_value = {"type": "function_call", "call_id": "call_1"}
        response = _make_mock_response([_make_message_item("Checking. "), item])
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="Checking. "),
            SimpleNamespace(type="response.function_call_arguments.delta", delta="{"),
            SimpleNamespace(type="response.completed", response=response),
        ]

        with (
            patch("eva.assistant.services.llm.router.get", return_value=_make_mock_router_with_no_deployments()),
            patch("litellm.aresponses", self._stream_of(events)),
        ):
            out = [item async for item in client.complete_stream([{"role": "user", "content": "hi"}])]

        assert out[0] == ("delta", "Checking. ")
        _, (message, stats) = out[-1]
        assert message.content == "Checking. "
        assert message.tool_calls[0].id == "call_1"
        assert message.tool_calls[0].function.name == "get_weather"
        assert len(stats["responses_output_items"]) == 2

    @pytest.mark.asyncio
    async def test_missing_completed_event_raises(self):
        client = LiteLLMClient(model="gpt-5.2")
        events = [SimpleNamespace(type="response.output_text.delta", delta="Hi")]

        with (
            patch("eva.assistant.services.llm.router.get", return_value=_make_mock_router_with_no_deployments()),
            patch("litellm.aresponses", self._stream_of(events)),
        ):
            with pytest.raises(RuntimeError, match="without a completed response"):
                [item async for item in client.complete_stream([{"role": "user", "content": "hi"}], max_retries=0)]


class TestUseResponsesApiResolution:
    @staticmethod
    def _resolve(deployment: dict) -> bool:
        router.init(model_list=[{"model_name": "m", **deployment}])
        try:
            return LiteLLMClient(model="m").use_responses_api
        finally:
            router.reset()

    def test_native_openai_defaults_to_responses_api(self):
        assert self._resolve({"litellm_params": {"model": "openai/gpt-5.5", "api_key": "k"}}) is True

    def test_use_chat_completions_opts_out(self):
        assert (
            self._resolve({"litellm_params": {"model": "openai/gpt-5.5", "api_key": "k"}, "use_chat_completions": True})
            is False
        )

    @pytest.mark.parametrize("key", ["base_url", "api_base"])
    def test_openai_prefixed_custom_endpoint_defaults_to_responses_api(self, key):
        params = {"model": "openai/phonellm", "api_key": "k", key: "https://example.test/v1"}
        assert self._resolve({"litellm_params": params}) is True

    def test_custom_endpoint_opts_out_with_use_chat_completions(self):
        params = {"model": "openai/phonellm", "api_key": "k", "base_url": "https://example.test/v1"}
        assert self._resolve({"litellm_params": params, "use_chat_completions": True}) is False

    def test_non_openai_provider_stays_on_chat_completions(self):
        assert self._resolve({"litellm_params": {"model": "vertex_ai/gemini-3.1-pro-preview"}}) is False

    def test_use_responses_api_forces_on_for_non_openai_provider(self):
        params = {"model": "vertex_ai/gemini-3.1-pro-preview"}
        assert self._resolve({"litellm_params": params, "use_responses_api": True}) is True

    def test_use_chat_completions_wins_over_use_responses_api(self):
        params = {"model": "openai/gpt-5.5", "api_key": "k"}
        deployment = {"litellm_params": params, "use_responses_api": True, "use_chat_completions": True}
        assert self._resolve(deployment) is False


class TestResponsesApiBaseUrl:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", ["base_url", "api_base"])
    async def test_custom_endpoint_forwarded_to_aresponses(self, key):
        router.init(
            model_list=[
                {
                    "model_name": "gemma",
                    "litellm_params": {"model": "openai/gemma", "api_key": "k", key: "https://example.test/v1"},
                    "use_responses_api": True,
                }
            ]
        )
        try:
            client = LiteLLMClient(model="gemma")
            mock_response = _make_mock_response([_make_message_item("ok")])
            with patch("litellm.aresponses", new_callable=AsyncMock, return_value=mock_response) as m:
                await client.complete([{"role": "user", "content": "hi"}])
            assert m.call_args.kwargs["api_base"] == "https://example.test/v1"
        finally:
            router.reset()
