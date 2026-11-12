"""Tests for stream-chunk reassembly in alm_base.py.

Gemini's raw OpenAI-compatible endpoint (hit directly by ALMGeminiClient, bypassing
litellm's own Gemini transport) carries the thought signature required for multi-turn
function calling under an `extra_content` key on each tool_call delta. litellm's
stream_chunk_builder only preserves id/type/function/provider_specific_fields when
merging streamed tool_call deltas, so that key is silently dropped unless we re-merge it
ourselves -- see _merge_streamed_tool_call_extras.
"""

from unittest.mock import AsyncMock

import pytest
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from eva.assistant.pipeline.alm_base import _assemble_stream_chunks, _merge_streamed_tool_call_extras
from eva.assistant.pipeline.alm_vllm import ALMvLLMClient


def _make_chunk(delta: dict, finish_reason: str | None = None, usage: dict | None = None) -> ChatCompletionChunk:
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gemini-3.6-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage:
        payload["usage"] = usage
    return ChatCompletionChunk.model_validate(payload)


def test_assemble_stream_chunks_preserves_gemini_thought_signature():
    """extra_content.google.thought_signature must survive stream reassembly."""
    chunks = [
        _make_chunk(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_flight_status", "arguments": ""},
                        "extra_content": {"google": {"thought_signature": "sig123"}},
                    }
                ],
            }
        ),
        _make_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"flight": "AA1"}'}}]}),
        _make_chunk(
            {},
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
    ]

    message, _usage, finish_reason = _assemble_stream_chunks(chunks, messages=[{"role": "user", "content": "hi"}])

    assert finish_reason == "tool_calls"
    tc_dict = message.tool_calls[0].model_dump(exclude_none=True)
    assert tc_dict["function"]["name"] == "get_flight_status"
    assert tc_dict["function"]["arguments"] == '{"flight": "AA1"}'
    assert tc_dict["extra_content"]["google"]["thought_signature"] == "sig123"


def test_assemble_stream_chunks_unaffected_when_no_extra_fields():
    """Providers with no unrecognized delta keys (e.g. vLLM) get litellm's own object untouched."""
    chunks = [
        _make_chunk(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ],
            }
        ),
        _make_chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
        _make_chunk(
            {},
            finish_reason="tool_calls",
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        ),
    ]

    message, _usage, _finish_reason = _assemble_stream_chunks(chunks, messages=[{"role": "user", "content": "hi"}])

    tc_dict = message.tool_calls[0].model_dump(exclude_none=True)
    assert tc_dict == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{}"},
    }


def test_merge_streamed_tool_call_extras_returns_none_without_extra_keys():
    dict_chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }
    ]
    assert _merge_streamed_tool_call_extras(dict_chunks) is None


def test_merge_streamed_tool_call_extras_handles_multiple_tool_calls():
    """Two parallel tool calls, each streamed across multiple chunks, each with its own signature."""
    dict_chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_flight_status", "arguments": ""},
                                "extra_content": {"google": {"thought_signature": "sig_a"}},
                            },
                            {
                                "index": 1,
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                                "extra_content": {"google": {"thought_signature": "sig_b"}},
                            },
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"flight": "AA1"}'}},
                            {"index": 1, "function": {"arguments": '{"city": "SF"}'}},
                        ]
                    }
                }
            ]
        },
    ]

    tool_calls = _merge_streamed_tool_call_extras(dict_chunks)

    assert tool_calls is not None
    assert len(tool_calls) == 2
    dumped = [tc.model_dump(exclude_none=True) for tc in tool_calls]
    assert dumped[0]["id"] == "call_1"
    assert dumped[0]["function"]["arguments"] == '{"flight": "AA1"}'
    assert dumped[0]["extra_content"]["google"]["thought_signature"] == "sig_a"
    assert dumped[1]["id"] == "call_2"
    assert dumped[1]["function"]["arguments"] == '{"city": "SF"}'
    assert dumped[1]["extra_content"]["google"]["thought_signature"] == "sig_b"


@pytest.mark.asyncio
async def test_vllm_complete_stream_emits_tool_call_start_once_before_final():
    """The lead-in text is followed by a single tool_call_start, emitted when tool-call tokens begin."""
    tool_delta = {
        "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_x", "arguments": ""}}]
    }
    args_delta = {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}
    chunks = [
        _make_chunk({"role": "assistant", "content": "Let me pull that up."}),
        _make_chunk(tool_delta),
        _make_chunk(args_delta),
        _make_chunk({}, finish_reason="tool_calls"),
    ]

    async def _stream():
        for chunk in chunks:
            yield chunk

    client = ALMvLLMClient(base_url="http://localhost:1")
    client._client.chat.completions.create = AsyncMock(return_value=_stream())

    events = [
        ev async for ev in client.complete_stream([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    ]

    assert [kind for kind, _ in events] == ["delta", "tool_call_start", "final"]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
async def test_vllm_complete_stream_captures_reasoning_under_either_field(field):
    """Streamed reasoning is surfaced whether vLLM names the delta field `reasoning` or `reasoning_content`."""
    chunks = [
        _make_chunk({"role": "assistant", field: "Let me "}),
        _make_chunk({field: "think."}),
        _make_chunk({"content": "Hello."}),
        _make_chunk({}, finish_reason="stop"),
    ]

    async def _stream():
        for chunk in chunks:
            yield chunk

    client = ALMvLLMClient(base_url="http://localhost:1")
    client._client.chat.completions.create = AsyncMock(return_value=_stream())

    events = [ev async for ev in client.complete_stream([{"role": "user", "content": "hi"}])]

    _, (_, stats) = events[-1]
    assert stats["reasoning"] == "Let me think."
    assert stats["reasoning_content"] == "Let me think."


def test_vllm_reasoning_effort_in_extra_body():
    assert "reasoning_effort" not in ALMvLLMClient(base_url="http://localhost:1")._build_extra_body()
    client = ALMvLLMClient(base_url="http://localhost:1", reasoning_effort="low")
    assert client._build_extra_body()["reasoning_effort"] == "low"
