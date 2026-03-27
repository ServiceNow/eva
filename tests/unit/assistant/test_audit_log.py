"""Tests for AuditLog."""

import json

from eva.assistant.agentic.audit_log import (
    AuditLog,
    ConversationMessage,
    LLMCall,
    MessageRole,
    current_timestamp_ms,
)


class TestCurrentTimestampMs:
    def test_returns_string(self):
        result = current_timestamp_ms()
        assert isinstance(result, str)

    def test_returns_millisecond_epoch(self):
        result = current_timestamp_ms()
        ts = int(result)
        # Should be 13 digits (ms since epoch)
        assert ts > 1_000_000_000_000


class TestConversationMessage:
    def test_to_dict_excludes_none_fields(self):
        msg = ConversationMessage(role=MessageRole.USER, content="hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hello"}
        assert "tool_calls" not in d
        assert "tool_call_id" not in d

    def test_to_dict_includes_tool_calls(self):
        msg = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[{"id": "1", "function": {"name": "f"}}],
        )
        d = msg.to_dict()
        assert "tool_calls" in d
        assert len(d["tool_calls"]) == 1

    def test_tool_message(self):
        msg = ConversationMessage(
            role=MessageRole.TOOL,
            content='{"result": "ok"}',
            tool_call_id="call_1",
            name="my_tool",
        )
        d = msg.to_dict()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call_1"
        assert d["name"] == "my_tool"


class TestAuditLog:
    def setup_method(self):
        self.log = AuditLog()

    def test_append_user_input_adds_transcript_entry(self):
        self.log.append_user_input("Hi there")
        assert len(self.log.transcript) == 1
        entry = self.log.transcript[0]
        assert entry["value"] == "Hi there"
        assert entry["message_type"] == "user"
        assert entry["isBotMessage"] is False

    def test_append_user_input_adds_conversation_message(self):
        self.log.append_user_input("Hi")
        assert len(self.log.conversation_messages) == 1
        msg = self.log.conversation_messages[0]
        assert msg.role == MessageRole.USER
        assert msg.content == "Hi"

    def test_append_user_input_custom_timestamp(self):
        self.log.append_user_input("Hi", timestamp_ms="1234567890000")
        assert self.log.transcript[0]["timestamp"] == "1234567890000"

    def test_append_assistant_output_text(self):
        self.log.append_assistant_output("Hello!")
        assert len(self.log.transcript) == 1
        entry = self.log.transcript[0]
        assert entry["value"] == "Hello!"
        assert entry["message_type"] == "assistant"
        assert entry["isBotMessage"] is True

    def test_append_assistant_output_with_tool_calls_skips_transcript(self):
        tool_calls = [{"id": "1", "function": {"name": "f", "arguments": "{}"}}]
        self.log.append_assistant_output("", tool_calls=tool_calls)
        # Should NOT add to transcript when tool_calls present
        assert len(self.log.transcript) == 0
        # But should add to conversation_messages
        assert len(self.log.conversation_messages) == 1
        msg = self.log.conversation_messages[0]
        assert msg.content == ""
        assert msg.tool_calls == tool_calls

    def test_append_assistant_output_custom_timestamp(self):
        self.log.append_assistant_output("Hi", timestamp_ms="9999999999999")
        assert self.log.transcript[0]["timestamp"] == "9999999999999"

    def test_append_tool_message(self):
        self.log.append_tool_message("call_1", '{"result": "ok"}')
        assert len(self.log.conversation_messages) == 1
        msg = self.log.conversation_messages[0]
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "call_1"

    def test_append_llm_call(self):
        response_msg = ConversationMessage(role=MessageRole.ASSISTANT, content="Sure!")
        llm_call = LLMCall(
            messages=[{"role": "user", "content": "Hi"}],
            response=response_msg,
            duration_seconds=1.5,
            start_time="100",
            end_time="200",
            model="gpt-4o",
            latency_ms=1500.0,
        )
        self.log.append_llm_call(llm_call, agent_name="TestAgent")

        assert len(self.log.llm_prompts) == 1
        assert self.log.llm_prompts[0]["response"] == "Sure!"
        assert self.log.llm_prompts[0]["model"] == "gpt-4o"
        assert self.log.llm_prompts[0]["latency_ms"] == 1500.0
        # Also adds to transcript
        assert len(self.log.transcript) == 1
        assert self.log.transcript[0]["message_type"] == "llm_call"

    def test_append_llm_call_no_response(self):
        llm_call = LLMCall(
            messages=[{"role": "user", "content": "Hi"}],
            response=None,
            status="error",
        )
        self.log.append_llm_call(llm_call)
        assert self.log.llm_prompts[0]["response"] == ""
        assert self.log.llm_prompts[0]["response_message"] is None

    def test_append_llm_call_with_reasoning(self):
        response_msg = ConversationMessage(
            role=MessageRole.ASSISTANT, content="Sure!", reasoning="I thought about this carefully..."
        )
        llm_call = LLMCall(
            messages=[{"role": "user", "content": "Hi"}],
            response=response_msg,
            duration_seconds=1.5,
            start_time="100",
            end_time="200",
            model="o1-preview",
            latency_ms=1500.0,
        )
        self.log.append_llm_call(llm_call, agent_name="TestAgent")

        # Check that reasoning is added to transcript entry
        assert len(self.log.transcript) == 1
        assert "reasoning" in self.log.transcript[0]["value"]
        assert self.log.transcript[0]["value"]["reasoning"] == "I thought about this carefully..."
        assert self.log.transcript[0]["value"]["response"] == "Sure!"

    def test_append_tool_call_without_response(self):
        self.log.append_tool_call("search", {"query": "test"})
        assert len(self.log.transcript) == 1
        assert self.log.transcript[0]["type"] == "tool_call"
        assert self.log._tool_calls_count == 1
        assert self.log._tools_called == ["search"]

    def test_append_tool_call_with_response(self):
        self.log.append_tool_call("search", {"query": "test"}, response={"results": []})
        # Should have tool_call + tool_response entries
        assert len(self.log.transcript) == 2
        assert self.log.transcript[0]["type"] == "tool_call"
        assert self.log.transcript[1]["type"] == "tool_response"

    def test_tool_call_deduplicates_tools_called(self):
        self.log.append_tool_call("search", {"q": "a"})
        self.log.append_tool_call("search", {"q": "b"})
        assert self.log._tools_called == ["search"]
        assert self.log._tool_calls_count == 2

    def test_last_tool_call_tracked(self):
        self.log.append_tool_call("search", {})
        self.log.append_tool_call("book", {})
        assert self.log._last_tool_call == "book"

    def test_append_realtime_tool_call(self):
        self.log.append_realtime_tool_call("get_flight", {"id": "123"})
        assert len(self.log.transcript) == 1
        assert self.log._tool_calls_count == 1
        assert self.log._tools_called == ["get_flight"]

    def test_get_conversation_messages_empty(self):
        result = self.log.get_conversation_messages()
        assert result == []

    def test_get_conversation_messages_returns_copy(self):
        self.log.append_user_input("Hi")
        msgs = self.log.get_conversation_messages()
        assert len(msgs) == 1
        # Modifying returned list should not affect original
        msgs.clear()
        assert len(self.log.conversation_messages) == 1

    def test_get_conversation_messages_with_max_messages(self):
        # Add 5 user/assistant pairs
        for i in range(5):
            self.log.append_user_input(f"User msg {i}")
            self.log.append_assistant_output(f"Bot msg {i}")

        # Limit to last 2 user messages
        msgs = self.log.get_conversation_messages(max_messages=2)
        user_msgs = [m for m in msgs if m.role == MessageRole.USER]
        assert len(user_msgs) == 2
        assert user_msgs[0].content == "User msg 3"
        assert user_msgs[1].content == "User msg 4"

    def test_get_conversation_messages_max_messages_larger_than_total(self):
        self.log.append_user_input("Only one")
        msgs = self.log.get_conversation_messages(max_messages=10)
        assert len(msgs) == 1

    def test_get_stats_empty(self):
        stats = self.log.get_stats()
        assert stats["num_turns"] == 0
        assert stats["num_tool_calls"] == 0
        assert stats["tools_called"] == []
        assert stats["total_transcript_entries"] == 0
        assert stats["total_llm_calls"] == 0

    def test_get_stats_populated(self):
        self.log.append_user_input("Hi")
        self.log.append_assistant_output("Hello")
        self.log.append_tool_call("search", {"q": "test"})
        self.log.append_user_input("Thanks")

        stats = self.log.get_stats()
        assert stats["num_turns"] == 2
        assert stats["num_tool_calls"] == 1
        assert stats["tools_called"] == ["search"]

    def test_get_stats_returns_copy_of_tools_called(self):
        self.log.append_tool_call("search", {})
        stats = self.log.get_stats()
        stats["tools_called"].append("hacked")
        assert "hacked" not in self.log._tools_called

    def test_save(self, tmp_path):
        self.log.append_user_input("Hi", timestamp_ms="2000")
        self.log.append_assistant_output("Hello", timestamp_ms="1000")

        path = tmp_path / "audit_log.json"
        self.log.save(path)

        data = json.loads(path.read_text())
        assert "transcript" in data
        assert "llm_prompts" in data
        assert "conversation_messages" in data
        # Should be sorted by timestamp
        assert data["transcript"][0]["timestamp"] == "1000"
        assert data["transcript"][1]["timestamp"] == "2000"

    def test_save_transcript_jsonl(self, tmp_path):
        self.log.append_user_input("Hi", timestamp_ms="1000")
        self.log.append_assistant_output("Hello", timestamp_ms="2000")
        # Tool calls should be excluded from jsonl
        self.log.append_tool_call("search", {"q": "test"})

        path = tmp_path / "transcript.jsonl"
        self.log.save_transcript_jsonl(path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2  # Only user + assistant
        record0 = json.loads(lines[0])
        assert record0["type"] == "user"
        assert record0["content"] == "Hi"

    def test_reset_clears_everything(self):
        self.log.append_user_input("Hi")
        self.log.append_assistant_output("Hello")
        self.log.append_tool_call("search", {})

        self.log.reset()

        assert self.log.transcript == []
        assert self.log.llm_prompts == []
        assert self.log.conversation_messages == []
        assert self.log._tool_calls_count == 0
        assert self.log._tools_called == []
        assert self.log._last_tool_call is None
