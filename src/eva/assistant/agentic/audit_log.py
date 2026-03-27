"""Audit log for tracking conversation history."""

import json
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from eva.utils.logging import get_logger

logger = get_logger(__name__)


def current_timestamp_ms() -> str:
    """Return current POSIX timestamp in milliseconds as string."""
    return str(int(round(time.time() * 1000)))


class MessageRole(StrEnum):
    """Message roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationMessage(BaseModel):
    """A message in the conversation."""

    role: MessageRole
    content: str
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool messages
    turn_id: Optional[int] = None  # For associating transcription updates
    reasoning: Optional[str] = None  # For model reasoning (e.g., from OpenAI o1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict, excluding None fields and internal tracking fields."""
        return self.model_dump(exclude_none=True, exclude={"turn_id"})


class LLMCall(BaseModel):
    """Record of an LLM call."""

    messages: list[dict]
    tools: Optional[list[dict]] = None
    response: Optional[ConversationMessage] = None
    duration_seconds: float = 0.0
    start_time: str = ""
    end_time: str = ""
    status: str = "success"
    model: Optional[str] = None
    # New fields for enhanced tracking (optional for backward compatibility)
    latency_ms: Optional[float] = None
    error_type: Optional[str] = None
    error_source: Optional[str] = None
    retry_attempt: int = 0


class AuditLog:
    """Tracks conversation history for auditing and metrics.

    Structure:
    - transcript: array of conversation entries (user, assistant, tool_call, tool_response)
    - llm_prompts: array of LLM call entries
    """

    def __init__(self):
        self.transcript: list[dict[str, Any]] = []
        self.llm_prompts: list[dict[str, Any]] = []
        self.conversation_messages: list[ConversationMessage] = []  # Full message sequence for LLM context
        self._tool_calls_count = 0
        self._tools_called: list[str] = []
        self._last_tool_call: Optional[str] = None  # Track last tool called for matching responses

    def append_user_input(
        self,
        content: str,
        timestamp_ms: Optional[str] = None,
        turn_id: Optional[int] = None,
    ) -> None:
        """Record user input.

        Args:
            content: User message text.
            timestamp_ms: Epoch-millisecond string to use as the entry
                timestamp.  When *None* (default), ``current_timestamp_ms()``
                is used.  The realtime instrumented LLM service passes the
                wall-clock captured at ``speech_started`` so the audit log
                reflects when the user actually spoke.
            turn_id: Optional turn identifier for associating transcription
                updates with the correct entry when transcription runs in
                parallel with processing.
        """
        entry = {
            "value": content,
            "displayName": "User",
            "type": "text",
            "isBotMessage": False,
            "timestamp": timestamp_ms or current_timestamp_ms(),
            "message_type": "user",
        }
        if turn_id is not None:
            entry["turn_id"] = turn_id
        self.transcript.append(entry)

        # Also add to conversation messages (with turn_id for matching)
        self.conversation_messages.append(
            ConversationMessage(
                role=MessageRole.USER,
                content=content,
                turn_id=turn_id,
            )
        )

        logger.debug(f"Audit: user input (turn_id={turn_id}) - {content[:50]}...")

    def update_last_user_input(self, content: str) -> bool:
        """Update the most recent user input with real transcription.

        This is used when transcription runs in parallel with processing and
        completes after the placeholder was written.

        Args:
            content: The real transcription text to replace the placeholder.

        Returns:
            True if an entry was updated, False if no user entry was found.
        """
        # Update the most recent user entry in transcript
        for entry in reversed(self.transcript):
            if entry.get("message_type") == "user":
                entry["value"] = content
                break
        else:
            logger.warning("No user entry found in transcript to update")
            return False

        # Update the most recent USER message in conversation_messages
        for msg in reversed(self.conversation_messages):
            if msg.role == MessageRole.USER:
                msg.content = content
                break

        logger.debug(f"Audit: updated last user input to - {content[:50]}...")
        return True

    def update_user_input_by_turn_id(self, turn_id: int, content: str) -> bool:
        """Update a specific user input entry by turn_id.

        This is used when transcription runs in parallel with processing and
        completes after the placeholder was written. Using turn_id ensures the
        correct entry is updated even if a new turn has started.

        Args:
            turn_id: The turn identifier to match.
            content: The real transcription text to replace the placeholder.

        Returns:
            True if an entry was updated, False if no matching entry was found.
        """
        found = False

        # Update the matching user entry in transcript
        for entry in self.transcript:
            if entry.get("message_type") == "user" and entry.get("turn_id") == turn_id:
                entry["value"] = content
                found = True
                break

        if not found:
            logger.warning(f"No user entry found with turn_id={turn_id} to update")
            return False

        # Update the matching USER message in conversation_messages
        for msg in self.conversation_messages:
            if msg.role == MessageRole.USER and msg.turn_id == turn_id:
                msg.content = content
                break

        logger.debug(f"Audit: updated user input (turn_id={turn_id}) to - {content[:50]}...")
        return True

    def append_assistant_output(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        timestamp_ms: Optional[str] = None,
    ) -> None:
        """Record assistant output.

        Args:
            content: Assistant message text.
            tool_calls: Optional list of tool call dicts (for conversation_messages).
            timestamp_ms: Epoch-millisecond string to use as the entry
                timestamp.  When *None* (default), ``current_timestamp_ms()``
                is used.  The realtime instrumented LLM service passes the
                wall-clock captured at the first ``audio_delta`` so the audit
                log reflects when the assistant actually started responding.
        """
        if content and not tool_calls:
            entry = {
                "value": content,
                "displayName": "Bot",
                "type": "text",
                "isBotMessage": True,
                "timestamp": timestamp_ms or current_timestamp_ms(),
                "message_type": "assistant",
            }
            self.transcript.append(entry)

        # With tool calls, we save an empty content regardless because it is never returned to the client.
        # TODO Implement returning the content to the client while tool calls are in progress
        self.conversation_messages.append(
            ConversationMessage(
                role=MessageRole.ASSISTANT,
                content="" if tool_calls else content,
                tool_calls=tool_calls,
            )
        )

        logger.debug(
            f"Audit: assistant output - {content[:50]}...{f'with {len(tool_calls)} tool_calls' if tool_calls else ''}"
        )

    def append_tool_message(self, tool_call_id: str, content: str) -> None:
        """Record tool response message to conversation messages."""
        self.conversation_messages.append(
            ConversationMessage(
                role=MessageRole.TOOL,
                tool_call_id=tool_call_id,
                content=content,
            )
        )
        logger.debug(f"Audit: tool message for call_id {tool_call_id}")

    def append_llm_call(self, llm_call: LLMCall, agent_name: Optional[str] = None) -> None:
        """Record an LLM call."""
        response_content = llm_call.response.content if llm_call.response else ""
        response_dict = llm_call.response.to_dict() if llm_call.response else None

        # Add to llm_prompts (full LLM call details)
        llm_entry = {
            "response": response_content,
            "response_message": response_dict,  # Full response message with tool_calls
            "prompt": llm_call.messages,
            "status": llm_call.status,
            "start_time": llm_call.start_time,
            "end_time": llm_call.end_time,
            "model": llm_call.model,
            "latency_ms": llm_call.latency_ms,
            "error_type": llm_call.error_type,
            "error_source": llm_call.error_source,
            "retry_attempt": llm_call.retry_attempt,
            "message_type": "llm_call",
        }
        self.llm_prompts.append(llm_entry)

        # Also add to transcript (user-facing view)
        transcript_entry = {
            "value": {"agent": agent_name or "Assistant", "response": response_content},
            "displayName": "LLM",
            "type": "text",
            "isBotMessage": True,
            "timestamp": current_timestamp_ms(),
            "message_type": "llm_call",
        }
        # Add reasoning if present
        if llm_call.response and llm_call.response.reasoning:
            transcript_entry["value"]["reasoning"] = llm_call.response.reasoning
        self.transcript.append(transcript_entry)

    def append_tool_call(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        response: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a tool call and its response."""
        # Record tool call in transcript
        tool_call_entry = {
            "value": {"tool": tool_name, "parameters": parameters},
            "displayName": "Tool",
            "type": "tool_call",
            "isBotMessage": True,
            "timestamp": current_timestamp_ms(),
            "message_type": "tool_call",
        }
        self.transcript.append(tool_call_entry)

        self._tool_calls_count += 1
        if tool_name not in self._tools_called:
            self._tools_called.append(tool_name)
        self._last_tool_call = tool_name

        if response is not None:
            self.append_tool_response(tool_name, response)

        logger.debug(f"Audit: tool_name call - {tool_name}")

    def append_tool_response(self, tool_name: str, response: dict[str, Any]) -> None:
        """Record a tool response."""
        tool_response_entry = {
            "value": {"tool": tool_name, "response": response},
            "displayName": "Tool Response",
            "type": "tool_response",
            "isBotMessage": True,
            "timestamp": current_timestamp_ms(),
            "message_type": "tool_response",
        }
        self.transcript.append(tool_response_entry)
        logger.debug(f"Audit: tool response for {tool_name}")

    def append_realtime_tool_call(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> None:
        """Record a tool call from the realtime pipeline (no AgentConfig/AgentTool required).

        Note: the S2S model processes raw audio and can call tools *before* transcription.completed fires,
        so this may be appended before the corresponding user entry.  Correct chronological order is guaranteed
        by ``save()`` sorting the transcript by timestamp — the user entry carries the ``speech_started`` wall-clock
        which is always earlier than the tool call's ``current_timestamp_ms()``.
        """
        tool_call_entry = {
            "value": {"tool": tool_name, "parameters": parameters},
            "displayName": "Tool",
            "type": "tool_call",
            "isBotMessage": True,
            "timestamp": current_timestamp_ms(),
            "message_type": "tool_call",
        }
        self.transcript.append(tool_call_entry)
        self._tool_calls_count += 1
        if tool_name not in self._tools_called:
            self._tools_called.append(tool_name)
        self._last_tool_call = tool_name
        logger.debug(f"Audit: realtime tool call - {tool_name}")

    def get_conversation_messages(
        self,
        max_messages: Optional[int] = None,
    ) -> list[ConversationMessage]:
        """Get conversation messages for LLM context.

        Args:
            max_messages: Maximum number of messages to return (applied to user messages)

        Returns:
            List of conversation messages in OpenAI format
        """
        if not self.conversation_messages:
            logger.warning("conversation_messages is empty.")
            return []

        messages: list[ConversationMessage] = list(self.conversation_messages)

        # Apply max_messages limit if specified
        if max_messages:
            # Count user messages to apply the limit
            user_message_count = sum(1 for m in messages if m.role == MessageRole.USER)
            if user_message_count > max_messages:
                # Find the index to start from
                user_count = 0
                start_idx = 0
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].role == MessageRole.USER:
                        user_count += 1
                        if user_count == max_messages:
                            start_idx = i
                            break
                messages = messages[start_idx:]
        return messages

    def get_stats(self) -> dict[str, Any]:
        """Get conversation statistics."""
        num_turns = sum(1 for e in self.transcript if e.get("message_type") == "user")
        return {
            "num_turns": num_turns,
            "num_tool_calls": self._tool_calls_count,
            "tools_called": self._tools_called.copy(),
            "total_transcript_entries": len(self.transcript),
            "total_llm_calls": len(self.llm_prompts),
        }

    def save(self, path: Path) -> None:
        """Save audit log to JSON file.

        The transcript list is sorted by timestamp as a safety net to guarantee
        chronological order (important for the realtime pipeline where events
        may arrive slightly out of order).
        """
        # Sort transcript by timestamp for correct chronological order
        self.transcript.sort(key=lambda e: int(e.get("timestamp", "0")))

        data = {
            "transcript": self.transcript,
            "llm_prompts": self.llm_prompts,
            "conversation_messages": [m.to_dict() for m in self.conversation_messages],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Audit log saved to {path}")

    def save_transcript_jsonl(self, path: Path) -> None:
        """Save transcript in JSONL format for metrics processing."""
        with open(path, "w") as f:
            for entry in self.transcript:
                # Filter to user and assistant messages only for simplified transcript
                if entry.get("message_type") in ["user", "assistant"]:
                    record = {"timestamp": entry["timestamp"], "type": entry["message_type"], "content": entry["value"]}
                    f.write(json.dumps(record) + "\n")

    def reset(self) -> None:
        """Reset the audit log."""
        self.transcript.clear()
        self.llm_prompts.clear()
        self.conversation_messages.clear()
        self._tool_calls_count = 0
        self._tools_called.clear()
        self._last_tool_call = None
