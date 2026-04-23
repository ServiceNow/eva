"""LLM client factory and implementations using LiteLLM."""

import asyncio
import os
import time
from types import SimpleNamespace
from typing import Any

import litellm
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageToolCall

from eva.utils import router
from eva.utils.error_handler import is_retryable_error
from eva.utils.logging import get_logger

load_dotenv()

logger = get_logger(__name__)


class LiteLLMClient:
    """Universal LLM client using LiteLLM.

    Provider routing is handled by the LiteLLM Router based on
    ``litellm_params.model`` in the ``EVA_MODEL_LIST`` deployment config.
    """

    def __init__(self, model: str, use_responses_api: bool | None = None):
        """Initialize LiteLLM client.

        Args:
            model: Model name matching a model_name in EVA_MODEL_LIST (e.g., 'gpt-5.2', 'gemini-3-pro')
            use_responses_api: Whether to route calls through the OpenAI Responses API instead of
                chat completions. Required for multi-turn encrypted reasoning on OpenAI o-series /
                gpt-5.x models. When None (default), the value is read from the ``use_responses_api``
                top-level field in the matching EVA_MODEL_LIST deployment. Pass True/False explicitly
                to override.
        """
        self.model = model
        self.use_responses_api = (
            self._lookup_use_responses_api_from_router() if use_responses_api is None else use_responses_api
        )

        logger.info(f"Initialized LiteLLM client with model: {self.model}, use_responses_api={self.use_responses_api}")
        litellm.drop_params = True

    def _lookup_use_responses_api_from_router(self) -> bool:
        """Read use_responses_api for self.model from the EVA_MODEL_LIST deployment config.

        The field lives at the top level of the deployment object (not inside litellm_params),
        since it is an EVA routing decision rather than a LiteLLM parameter.

        Returns False if the router is not yet initialized or the deployment is not found.
        """
        try:
            r = router.get()
        except RuntimeError:
            return False
        for deployment in getattr(r, "model_list", []):
            if deployment.get("model_name") == self.model:
                return bool(deployment.get("use_responses_api", False))
        return False

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        reasoning_effort: str | None = None,
        max_retries: int = 5,
        initial_delay: float = 1.0,
    ) -> tuple[Any, dict[str, Any]]:
        """Generate a completion using LiteLLM with exponential backoff retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tools in OpenAI format
            reasoning_effort: Optional reasoning effort level ("low", "medium", "high").
                            If provided, enables extended thinking for models that support it.
                            Can also be set in model config via litellm_params.reasoning_effort.
            max_retries: Maximum number of retry attempts for rate limits
            initial_delay: Initial delay in seconds before first retry

        Returns:
            Tuple of (message, stats) where:
            - message: LLM response message (content string or message object with tool calls)
            - stats: Dict with usage info (prompt_tokens, completion_tokens, finish_reason, model, parameters)
        """
        # OpenAI Responses API path: stateless multi-turn encrypted reasoning
        if self.use_responses_api:
            return await self._complete_via_responses_api(messages, tools, reasoning_effort, max_retries, initial_delay)

        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Enable extended thinking if reasoning_effort is provided
        # Note: reasoning_effort can also be set in model config (litellm_params.reasoning_effort)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = await router.get().acompletion(**kwargs)
                elapsed_time = time.time() - start_time

                message = response.choices[0].message
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                # Extract reasoning tokens if present (OpenAI o1 models include this in usage)
                reasoning_tokens = 0
                if usage and hasattr(usage, "completion_tokens_details"):
                    details = usage.completion_tokens_details
                    if details and hasattr(details, "reasoning_tokens"):
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0)

                finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
                model = getattr(response, "model", self.model)
                hidden_params = getattr(response, "_hidden_params", {}) or {}
                response_cost = hidden_params.get("response_cost")
                cost_source = "litellm"

                # Extract reasoning content — LiteLLM provides a unified interface:
                # reasoning_content: concatenated thinking text (all providers including Anthropic)
                # thinking_blocks: raw thinking block objects (Anthropic only, needed for multi-turn threading)
                reasoning_content = getattr(message, "reasoning_content", None)
                thinking_blocks = (
                    getattr(message, "thinking_blocks", None) if hasattr(message, "thinking_blocks") else None
                )

                if reasoning_content:
                    logger.info(f"💭 Reasoning content from {model} ({len(reasoning_content)} chars)")
                    logger.debug(f"Reasoning content preview: {reasoning_content[:200]}...")

                # Gemini thought signatures are handled automatically by LiteLLM
                # They are stored in provider_specific_fields and preserved across turns
                # The reasoning_content field will contain any reasoning output from Gemini

                stats = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "finish_reason": finish_reason,
                    "model": model,
                    "cost": response_cost,
                    "cost_source": cost_source,
                    "latency": round(elapsed_time, 3),
                    "reasoning": reasoning_content,
                    "reasoning_content": reasoning_content,  # Keep for backward compatibility
                    "thinking_blocks": thinking_blocks,  # Anthropic-specific thinking blocks
                }

                if hasattr(message, "tool_calls") and message.tool_calls:
                    return message, stats
                else:
                    return message.content or "", stats

            except Exception as e:
                last_exception = e

                # Use centralized retry logic
                if is_retryable_error(e) and attempt < max_retries:
                    delay = initial_delay * (2**attempt)
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.exception(f"LiteLLM completion failed: {e}")
                    raise

        logger.error(f"LiteLLM completion failed after {max_retries} retries")
        raise last_exception

    # ------------------------------------------------------------------
    # OpenAI Responses API helpers (stateless multi-turn reasoning)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools_for_responses_api(tools: list[dict]) -> list[dict]:
        """Convert chat completions tool format to Responses API format.

        Chat completions: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Responses API:    {"type": "function", "name": ..., "description": ..., "parameters": ...}
        """
        result = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                fn = tool["function"]
                result.append(
                    {
                        "type": "function",
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    }
                )
        return result

    @staticmethod
    def _convert_messages_for_responses_api(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert chat completions messages to Responses API (instructions, input_items).

        Handles three cases for assistant messages:
        - Has ``responses_output_items``: inject raw items from the previous Responses API call
          (preserves encrypted reasoning within a tool-calling loop).
        - Has ``tool_calls`` but no output items (history from prior user turns): reconstruct
          function_call items and consume the following tool messages.
        - Plain assistant message: convert as-is.

        Tool messages (role "tool") are converted to function_call_output items and are normally
        consumed during the assistant look-ahead above; orphaned ones are handled as a fallback.
        """
        instructions: str | None = None
        input_items: list[dict[str, Any]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            if role == "system":
                instructions = msg.get("content")
            elif role == "user":
                input_items.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                if msg.get("responses_output_items"):
                    # Current-turn tool loop: inject raw Responses API output items directly
                    # (includes the reasoning item with encrypted_content)
                    input_items.extend(msg["responses_output_items"])
                elif msg.get("tool_calls"):
                    # Prior turn in conversation history: reconstruct function_call items.
                    # Emit assistant content first (narration preceded tool execution).
                    if content := msg.get("content"):
                        input_items.append({"role": "assistant", "content": content})
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": tc["id"],
                                "name": fn["name"],
                                "arguments": fn.get("arguments", "{}"),
                            }
                        )
                    # Consume the immediately following tool-result messages
                    j = i + 1
                    while j < len(messages) and messages[j].get("role") == "tool":
                        tool_msg = messages[j]
                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": tool_msg["tool_call_id"],
                                "output": tool_msg["content"],
                            }
                        )
                        j += 1
                    i = j - 1  # will be incremented at end of loop
                else:
                    input_items.append({"role": "assistant", "content": msg.get("content", "")})
            elif role == "tool":
                # Orphaned tool message (normally consumed in the assistant look-ahead above)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg["tool_call_id"],
                        "output": msg["content"],
                    }
                )
            i += 1
        return instructions, input_items

    def _get_router_litellm_params(self) -> dict[str, Any]:
        """Look up the litellm_params for self.model from the router config.

        Assumes the router is initialized (only called from _complete_via_responses_api,
        which requires use_responses_api=True read from the router at init time).
        """
        r = router.get()
        for deployment in getattr(r, "model_list", []):
            if deployment.get("model_name") == self.model:
                return deployment.get("litellm_params", {})
        return {}

    async def _complete_via_responses_api(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None,
        reasoning_effort: str | None,
        max_retries: int,
        initial_delay: float,
    ) -> tuple[Any, dict[str, Any]]:
        """Call the OpenAI Responses API for stateless multi-turn encrypted reasoning.

        Uses ``include=["reasoning.encrypted_content"]`` + ``store=False`` so encrypted
        reasoning is threaded through tool-call loops without managing conversation IDs.
        The caller (system.py) stores ``responses_output_items`` from stats on the
        assistant message so the next iteration injects them back into the input.
        """
        # Resolve model credentials from router config (handles Azure, custom api_base, etc.)
        litellm_params = self._get_router_litellm_params()
        litellm_model = litellm_params.get("model", self.model)

        # Resolve api_key (may be "os.environ/VAR_NAME")
        raw_key = litellm_params.get("api_key", "")
        if raw_key.startswith("os.environ/"):
            raw_key = os.environ.get(raw_key.split("/", 1)[1], "")
        api_key = raw_key or None

        api_base = litellm_params.get("api_base") or None

        # reasoning_effort: caller takes priority, then fall back to litellm_params
        effective_reasoning_effort = reasoning_effort or litellm_params.get("reasoning_effort")

        instructions, input_items = self._convert_messages_for_responses_api(messages)

        kwargs: dict[str, Any] = {
            "model": litellm_model,
            "input": input_items,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if effective_reasoning_effort:
            kwargs["reasoning"] = {"effort": effective_reasoning_effort}
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = self._convert_tools_for_responses_api(tools)
            kwargs["tool_choice"] = "auto"
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = await litellm.aresponses(**kwargs)
                elapsed = time.time() - start_time

                text_content = ""
                reasoning_summary: str | None = None
                tool_calls: list[ChatCompletionMessageToolCall] = []
                output_items_for_next_turn: list[dict[str, Any]] = []

                for item in response.output:
                    if item.type == "reasoning":
                        # Prefer human-readable summary; fall back to encrypted_content as the
                        # reasoning-text equivalent (e.g. gpt-5.2 returns no readable summary)
                        summary_parts = getattr(item, "summary", []) or []
                        if summary_parts:
                            reasoning_summary = " ".join(s.text for s in summary_parts if hasattr(s, "text"))
                        if not reasoning_summary:
                            reasoning_summary = getattr(item, "encrypted_content", None)
                        output_items_for_next_turn.append(item.model_dump(exclude_none=True))
                    elif item.type == "message":
                        for c in item.content:
                            if hasattr(c, "text") and c.text:
                                text_content += c.text
                        output_items_for_next_turn.append(item.model_dump(exclude_none=True))
                    elif item.type == "function_call":
                        tool_calls.append(
                            ChatCompletionMessageToolCall(
                                id=item.call_id,
                                type="function",
                                function={"name": item.name, "arguments": item.arguments},
                            )
                        )
                        output_items_for_next_turn.append(item.model_dump(exclude_none=True))

                usage = getattr(response, "usage", None)
                reasoning_tokens = 0
                if usage and hasattr(usage, "output_tokens_details"):
                    details = usage.output_tokens_details
                    if details:
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

                if reasoning_summary:
                    logger.info(f"💭 Reasoning summary from Responses API ({len(reasoning_summary)} chars)")

                stats = {
                    "prompt_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                    "reasoning_tokens": reasoning_tokens,
                    "finish_reason": getattr(response, "status", "completed"),
                    "model": getattr(response, "model", self.model),
                    "cost": None,
                    "cost_source": "litellm",
                    "latency": round(elapsed, 3),
                    "reasoning": reasoning_summary,
                    "reasoning_content": reasoning_summary,
                    "thinking_blocks": None,
                    "responses_output_items": output_items_for_next_turn or None,
                }

                if tool_calls:
                    message = SimpleNamespace(
                        content=text_content,
                        tool_calls=tool_calls,
                        reasoning_content=reasoning_summary,
                    )
                    return message, stats
                else:
                    return text_content, stats

            except Exception as e:
                last_exception = e
                if is_retryable_error(e) and attempt < max_retries:
                    delay = initial_delay * (2**attempt)
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.exception(f"Responses API call failed: {e}")
                    raise

        raise last_exception
