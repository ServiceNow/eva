"""Agentic system - orchestrates interaction between users and a single agent."""

import asyncio
import csv
import json
import time
import warnings
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from eva.assistant.agentic.audit_log import (
    AuditLog,
    ConversationMessage,
    LLMCall,
    MessageRole,
)
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.models.agents import AgentConfig
from eva.utils.error_handler import categorize_error
from eva.utils.log_processing import truncate_data_uris
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

# Suppress LiteLLM's Pydantic serialization warnings (harmless internal warnings)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Pydantic serializer warnings.*")

# Response messages
GENERIC_ERROR = "I'm sorry, I encountered an error processing your request."


def _clean_tool_name(name: str) -> str:
    """Strip Harmony special tokens that leak into tool names due to a known vLLM bug.

    vLLM's openai tool-call parser does not always correctly handle the Harmony
    chat template's <|channel|> delimiters, causing tokens like
    '<|channel|>commentary' to be appended to the tool name intermittently.
    See: https://github.com/vllm-project/vllm/issues/32587
    """
    if "<|channel|>" in name:
        cleaned = name.split("<|channel|>")[0].strip()
        logger.warning(f"Harmony token leak in tool name: {name!r} → {cleaned!r}")
        return cleaned
    return name


class AgenticSystem:
    """Orchestrates the interaction between users and a single agent.

    Single-agent mode: directly executes the configured agent without matching.

    The system handles:
    - Agent execution (running the agent with tool calls)
    - Conversation state management
    """

    def __init__(
        self,
        current_date_time: str,
        agent: AgentConfig,
        tool_handler: ToolExecutor,
        audit_log: AuditLog,
        llm_client: Any,  # LLM client for model calls
        output_dir: Path | None = None,  # Output directory for performance stats
    ):
        """Initialize the agentic system.

        Args:
            current_date_time: Current date and time string for prompt
            agent: Single agent configuration to use for all interactions
            tool_handler: Handler for tool calls (ToolExecutor)
            audit_log: Audit log for conversation tracking
            llm_client: Client for LLM calls
            output_dir: Optional output directory for saving performance stats
        """
        self.agent = agent
        self.tool_handler = tool_handler
        self.audit_log = audit_log
        self.llm_client = llm_client
        self.output_dir = output_dir
        self.current_date_time = current_date_time

        self.prompt_manager = PromptManager()

        # Track agent performance stats
        self.agent_perf_stats: list[dict[str, Any]] = []

        # Build the agent prompt
        self.system_prompt = self.prompt_manager.get_prompt(
            "agent.system_prompt",
            agent_personality=agent.description,
            agent_instructions=agent.instructions,
            datetime=self.current_date_time,
        )
        # Build tools for the LLM
        self.tools = agent.build_tools_for_agent()

    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        """Process a user query and yield response messages.

        This is the main entry point for handling user input.
        Directly executes the configured agent without matching.

        Args:
            query: User's input text

        Yields:
            Text responses to be sent to TTS
        """
        logger.info(f"Processing query: {query}")

        # Record user input
        self.audit_log.append_user_input(query)

        # Execute agent interaction
        async for response in self._execute_agent(self.agent):
            yield response

    async def _execute_agent(
        self,
        agent: AgentConfig,
    ) -> AsyncGenerator[str, None]:
        """Execute an agent interaction with tool calling loop.

        Args:
            agent: Agent to execute

        Yields:
            Response messages
        """
        # Initial messages - start with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        # Add conversation history (includes current query since we already called append_user_input)
        conversation_history = self.audit_log.get_conversation_messages(max_messages=30)
        messages.extend(msg.to_dict() for msg in conversation_history)

        async for response in self._run_tool_loop(messages, agent):
            yield response

    async def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        agent: AgentConfig,
    ) -> AsyncGenerator[str, None]:
        """Core tool-calling loop. Calls the LLM, executes tool calls, and repeats.

        Separated from _execute_agent() so subclasses can build custom message lists
        (e.g. with audio content) and reuse the same tool-calling logic.

        Args:
            messages: Initial messages list (will be mutated with tool results).
            agent: Agent configuration with tools.

        Yields:
            Response messages.
        """
        # Tool calling loop (no max iterations)
        while True:
            start_time = str(int(time.time() * 1000))
            try:
                # Truncate data URIs for logging only (full audio stays in `messages`)
                messages_for_log = truncate_data_uris(messages)
                prompt_str = json.dumps(messages_for_log, indent=2, ensure_ascii=False)

                response, llm_stats = await self.llm_client.complete(
                    messages,
                    tools=self.tools,
                )
                end_time = str(int(time.time() * 1000))

                # Convert tool calls to dicts if present and extract content as string
                response_tool_calls = getattr(response, "tool_calls", []) or []
                tool_calls_dicts = [
                    {
                        "id": str(tc.id),
                        "type": "function",
                        "function": {
                            "name": _clean_tool_name(str(tc.function.name)),
                            "arguments": str(tc.function.arguments),
                        },
                    }
                    for tc in response_tool_calls
                ]

                response_content = getattr(response, "content", "") or (response if isinstance(response, str) else "")
                if response_content:
                    response_content = response_content.strip()

                response_tool_calls_for_stats = (
                    [
                        {"name": tool["function"]["name"], "arguments": tool["function"]["arguments"]}
                        for tool in tool_calls_dicts
                    ]
                    if tool_calls_dicts
                    else None
                )

                # Store performance stats
                perf_stat = {
                    "prompt": prompt_str,
                    "response": response_content,
                    "prompt_tokens": llm_stats.get("prompt_tokens", 0),
                    "output_tokens": llm_stats.get("completion_tokens", 0),
                    "cost": llm_stats.get("cost", 0.0),
                    "cost_source": llm_stats.get("cost_source", "unknown"),
                    "stop_reason": llm_stats.get("finish_reason", "unknown"),
                    "latency": llm_stats.get("latency", 0.0),
                    "parameters": json.dumps(llm_stats.get("parameters", {})),
                    "tool_calls": json.dumps(response_tool_calls_for_stats) if response_tool_calls_for_stats else "",
                    "reasoning": f'"{llm_stats.get("reasoning_content", "")}"',
                }
                self.agent_perf_stats.append(perf_stat)
                logger.debug(
                    f"Collected agent perf stat: tokens={perf_stat['prompt_tokens']}/{perf_stat['output_tokens']}, stop_reason={perf_stat['stop_reason']}"
                )

                llm_call_response = ConversationMessage(
                    role=MessageRole.ASSISTANT,
                    content=response_content,
                    tool_calls=tool_calls_dicts or None,
                    reasoning=llm_stats.get("reasoning"),
                )

                llm_call = LLMCall(
                    messages=messages_for_log,
                    tools=self.tools,
                    response=llm_call_response,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=(int(end_time) - int(start_time)) / 1000.0,
                    status="success",
                    model=getattr(response, "model", None),
                    latency_ms=float(int(end_time) - int(start_time)),
                )

                self.audit_log.append_llm_call(llm_call, agent_name=agent.name)

            except asyncio.CancelledError:
                # Pipeline is shutting down - log at debug and exit gracefully
                logger.debug("LLM call cancelled during pipeline shutdown")
                return  # Don't yield error message, just exit
            except Exception as e:
                # Check if this is a real error or a cancellation-related error
                error_msg = str(e).strip()
                # Cancellation errors often show as "APIError - " with no details
                if error_msg.endswith("APIError -") or error_msg.endswith("APIError"):
                    # Likely a cancellation-related error (no error details)
                    logger.debug(f"LLM call failed during shutdown: {e}")
                else:
                    # Real API error with details - log and yield error
                    logger.error(f"LLM call failed in agent execution: {e}")

                    # Categorize error using centralized error handler
                    error_info = categorize_error(e)
                    error_type = error_info.error_type
                    error_source = error_info.error_source

                    # Create failed LLM call record
                    end_time_failed = str(int(time.time() * 1000))
                    failed_llm_call = LLMCall(
                        messages=messages_for_log,
                        tools=self.tools,
                        response=None,
                        start_time=start_time,
                        end_time=end_time_failed,
                        duration_seconds=(int(end_time_failed) - int(start_time)) / 1000.0,
                        status="error",
                        model=None,
                        latency_ms=float(int(end_time_failed) - int(start_time)),
                        error_type=error_type,
                        error_source=error_source,
                        retry_attempt=0,
                    )
                    self.audit_log.append_llm_call(failed_llm_call, agent_name=agent.name)
                    yield GENERIC_ERROR
                return

            if response_content:
                logger.info(f"💬 Assistant LLM response: {response_content}")
                yield response_content

            self.audit_log.append_assistant_output(content=response_content, tool_calls=tool_calls_dicts or None)

            if not tool_calls_dicts:
                # No tool calls, this is the final response
                return

            messages.append(
                {
                    "role": "assistant",
                    "content": response_content,
                    "tool_calls": tool_calls_dicts,
                }
            )

            # Execute each tool call
            for tool_call in response_tool_calls:
                tool_name = _clean_tool_name(tool_call.function.name)
                try:
                    # TODO Consider this a model error instead of handling this gracefully
                    params = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    params = {}

                # Log tool call
                logger.info(f"🔧 Tool call: {tool_name}")
                logger.info(f"   Parameters: {json.dumps(params, indent=2)}")

                # Special handling for transfer to live agent
                if tool_name == "transfer_to_agent":
                    transfer_message = "Transferring you to a live agent. Please wait."
                    self.audit_log.append_tool_call(
                        tool_name=tool_name,
                        parameters=params,
                        response={"status": "transfer_initiated"},
                    )

                    logger.info(f"🔀 Transfer initiated: {transfer_message}")
                    yield transfer_message
                    self.audit_log.append_assistant_output(transfer_message)
                    return

                result = await self.tool_handler.execute(tool_name, params)

                if result.get("status") == "error":
                    logger.warning(f"❌ Tool error: {tool_name} - {result.get('message', 'Unknown error')}")
                else:
                    logger.info(f"✅ Tool response: {tool_name}")
                    logger.info(f"   Result: {json.dumps(result, indent=2)}")

                self.audit_log.append_tool_call(
                    tool_name=tool_name,
                    parameters=params,
                    response=result,
                )

                # Add tool response to messages
                tool_content = json.dumps(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    }
                )

                self.audit_log.append_tool_message(tool_call_id=tool_call.id, content=tool_content)

    def get_stats(self) -> dict[str, Any]:
        """Get conversation statistics."""
        stats = self.audit_log.get_stats()
        stats["agent"] = self.agent.name
        return stats

    def save_agent_perf_stats(self) -> None:
        """Save agent performance stats to CSV file."""
        logger.info(
            f"save_agent_perf_stats called: output_dir={self.output_dir}, stats_count={len(self.agent_perf_stats)}"
        )

        if not self.output_dir:
            logger.warning("No output_dir set, skipping agent perf stats save")
            return

        if not self.agent_perf_stats:
            logger.warning("No agent perf stats collected, skipping save")
            return

        csv_path = Path(self.output_dir) / "agent_perf_stats.csv"

        try:
            # Write to CSV
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "prompt",
                    "response",
                    "prompt_tokens",
                    "output_tokens",
                    "cost",
                    "cost_source",
                    "stop_reason",
                    "parameters",
                    "tool_calls",
                    "latency",
                    "reasoning",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.agent_perf_stats)

            logger.info(f"Saved {len(self.agent_perf_stats)} agent performance stats to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save agent performance stats: {e}", exc_info=True)
