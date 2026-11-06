"""LLM client factory and implementations using LiteLLM."""

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import litellm
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageToolCall

from eva.utils import router
from eva.utils.error_handler import is_retryable_error
from eva.utils.llm_utils import approximate_reasoning_tokens
from eva.utils.logging import get_logger

load_dotenv()

logger = get_logger(__name__)

# Responses API stream error codes that succeed on retry (rate_limit_exceeded is handled separately).
_TRANSIENT_STREAM_ERROR_CODES = frozenset({"server_error", "vector_store_timeout"})


class LiteLLMClient:
    """Universal LLM client using LiteLLM.

    Provider routing is handled by the LiteLLM Router based on
    ``litellm_params.model`` in the ``EVA_MODEL_LIST`` deployment config.
    """

    def __init__(self, model: str, parallel_tool_calls: bool | None = None):
        """Initialize LiteLLM client.

        Args:
            model: Model name matching a model_name in EVA_MODEL_LIST (e.g., 'gpt-5.2', 'gemini-3-pro')
            parallel_tool_calls: Optional provider pass-through for tool calls.
        """
        self.model = model
        self.parallel_tool_calls = parallel_tool_calls
        self.use_responses_api = self._lookup_use_responses_api_from_router()
        self._reasoning_token_fallback_warned = False

        logger.info(f"Initialized LiteLLM client with model: {self.model}, use_responses_api={self.use_responses_api}")
        litellm.drop_params = True

    def _lookup_use_responses_api_from_router(self) -> bool:
        """Decide whether self.model uses the OpenAI Responses API, from its EVA_MODEL_LIST deployment.

        Precedence (flags live at the top level of the deployment object, not inside litellm_params,
        since they are EVA routing decisions rather than LiteLLM parameters):
        1. ``use_chat_completions: true`` -> chat completions.
        2. ``use_responses_api: true`` -> Responses API, even for providers that don't use an ``openai/``
           model prefix.
        3. Default: Responses API when ``litellm_params.model`` starts with ``openai/`` (this includes
           OpenAI-compatible servers such as vLLM reached through ``api_base``/``base_url``, so set
           ``use_chat_completions: true`` on those if the server has no /v1/responses endpoint);
           chat completions for everything else (Vertex, Azure, ...).
        """
        r = router.get()
        for deployment in getattr(r, "model_list", []):
            if deployment.get("model_name") == self.model:
                if deployment.get("use_chat_completions", False):
                    return False
                if deployment.get("use_responses_api", False):
                    return True
                params = deployment.get("litellm_params", {})
                is_native_openai = str(params.get("model", "")).startswith("openai/")
                return is_native_openai
        return False

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        max_retries: int = 5,
        initial_delay: float = 1.0,
    ) -> tuple[Any, dict[str, Any]]:
        """Generate a completion using LiteLLM with exponential backoff retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tools in OpenAI format
            max_retries: Maximum number of retry attempts for rate limits
            initial_delay: Initial delay in seconds before first retry

        Returns:
            Tuple of (message, stats) where:
            - message: LLM response message (content string or message object with tool calls)
            - stats: Dict with usage info (prompt_tokens, completion_tokens, finish_reason, model, parameters)
        """
        # OpenAI Responses API path: stateless multi-turn encrypted reasoning
        if self.use_responses_api:
            return await self._complete_via_responses_api(messages, tools, max_retries, initial_delay)

        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            if self.parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = self.parallel_tool_calls

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
                    if reasoning_tokens == 0:
                        reasoning_tokens = approximate_reasoning_tokens(reasoning_content, self.model, self, logger)

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

        Assumes the router is initialized (only called from the Responses API paths,
        which require use_responses_api resolved from the router at init time).
        """
        r = router.get()
        for deployment in getattr(r, "model_list", []):
            if deployment.get("model_name") == self.model:
                return deployment.get("litellm_params", {})
        return {}

    async def complete_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        max_retries: int = 5,
        initial_delay: float = 1.0,
    ):
        """Yield text deltas, then the assembled final message and stats."""
        if self.use_responses_api:
            async for item in self._complete_stream_via_responses_api(messages, tools, max_retries, initial_delay):
                yield item
            return

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            if self.parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = self.parallel_tool_calls

        last_exception = None
        for attempt in range(max_retries + 1):
            chunks: list[Any] = []
            first_token = False
            tool_call_started = False
            try:
                start_time = time.time()
                stream = await router.get().acompletion(**kwargs)
                async for chunk in stream:
                    chunks.append(chunk)
                    choices = getattr(chunk, "choices", None) or []
                    if choices:
                        delta = getattr(choices[0], "delta", None)
                        text = getattr(delta, "content", None) if delta else None
                        if text:
                            first_token = True
                            yield ("delta", text)
                        if not tool_call_started and delta and getattr(delta, "tool_calls", None):
                            # Text before this point is complete; let the consumer release it now
                            # instead of waiting for the tool arguments to finish streaming.
                            tool_call_started = True
                            yield ("tool_call_start", None)
                elapsed_time = time.time() - start_time

                full = litellm.stream_chunk_builder(chunks, messages=messages)
                message = full.choices[0].message
                usage = getattr(full, "usage", None)
                reasoning_content = getattr(message, "reasoning_content", None)
                thinking_blocks = (
                    getattr(message, "thinking_blocks", None) if hasattr(message, "thinking_blocks") else None
                )
                hidden_params = getattr(full, "_hidden_params", {}) or {}
                # Reasoning tokens arrive in the final usage chunk.
                reasoning_tokens = 0
                if usage and hasattr(usage, "completion_tokens_details"):
                    details = usage.completion_tokens_details
                    if details and hasattr(details, "reasoning_tokens"):
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                stats = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "reasoning_tokens": reasoning_tokens,
                    "finish_reason": getattr(full.choices[0], "finish_reason", "unknown"),
                    "model": getattr(full, "model", self.model),
                    "cost": hidden_params.get("response_cost"),
                    "cost_source": "litellm",
                    "latency": round(elapsed_time, 3),
                    "reasoning": reasoning_content,
                    "reasoning_content": reasoning_content,
                    "thinking_blocks": thinking_blocks,
                    "responses_output_items": None,
                }
                yield ("final", (message, stats))
                return
            except Exception as e:
                last_exception = e
                if is_retryable_error(e) and attempt < max_retries and not first_token:
                    delay = initial_delay * (2**attempt)
                    logger.warning(
                        f"Retryable streaming error on attempt {attempt + 1}/{max_retries + 1}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.exception(f"LiteLLM streaming completion failed: {e}")
                raise
        raise last_exception

    def _build_responses_kwargs(self, messages: list[dict[str, Any]], tools: list[dict] | None) -> dict[str, Any]:
        """Build ``router.aresponses`` kwargs.

        The Router resolves credentials and forwards the deployment's ``litellm_params`` (api_key,
        api_base, reasoning_effort, extra_body, timeout, ...), so only per-call arguments are set here.
        Uses ``include=["reasoning.encrypted_content"]`` + ``store=False`` so encrypted
        reasoning is threaded through tool-call loops without managing conversation IDs.
        """
        litellm_params = self._get_router_litellm_params()
        instructions, input_items = self._convert_messages_for_responses_api(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = self._convert_tools_for_responses_api(tools)
            kwargs["tool_choice"] = "auto"
            if self.parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = self.parallel_tool_calls

        # The Router's Responses path ignores ``base_url`` (chat completions accepts it as an alias
        # for ``api_base``) and has no ``max_tokens`` (Responses calls it ``max_output_tokens``).
        if not litellm_params.get("api_base") and litellm_params.get("base_url"):
            kwargs["api_base"] = litellm_params["base_url"]
        if litellm_params.get("max_tokens") and not litellm_params.get("max_output_tokens"):
            kwargs["max_output_tokens"] = litellm_params["max_tokens"]
        return kwargs

    @staticmethod
    def _response_cost(response: Any) -> float | None:
        """Cost of a Responses API response, or None when LiteLLM has no pricing for the model."""
        hidden = getattr(response, "_hidden_params", None)
        cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
        if cost is None:
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = None
        return cost

    def _stream_failure_exception(self, event: Any) -> Exception:
        """Turn a ``response.failed`` / ``error`` stream event into an exception.

        Transient codes become LiteLLM exceptions so ``is_retryable_error`` retries them like the
        HTTP-level errors of chat completions. Everything else (``invalid_prompt``, image errors, ...)
        would fail identically on retry and stays a plain ``RuntimeError``.
        """
        # response.failed nests the error under response.error (a dict or object); an error event carries it directly.
        error = getattr(getattr(event, "response", None), "error", None) or event
        if isinstance(error, dict):
            code, message = error.get("code"), error.get("message")
        else:
            code, message = getattr(error, "code", None), getattr(error, "message", None)
        text = f"Responses API stream failed ({code}): {message}"
        if code == "rate_limit_exceeded":
            return litellm.RateLimitError(message=text, llm_provider="openai", model=self.model)
        if code in _TRANSIENT_STREAM_ERROR_CODES:
            return litellm.InternalServerError(message=text, llm_provider="openai", model=self.model)
        return RuntimeError(text)

    def _raise_if_responses_api_unsupported(self, error: Exception) -> None:
        """Replace a missing-endpoint error with one that names the ``use_chat_completions`` opt-out."""
        if getattr(error, "status_code", None) in (404, 405, 501):
            raise RuntimeError(
                f"Model '{self.model}' is using the OpenAI Responses API, but its endpoint does not support it "
                f'({type(error).__name__}: {error}). Set "use_chat_completions": true on the deployment in '
                "EVA_MODEL_LIST to use Chat Completions instead."
            ) from error

    def _parse_responses_output(self, response: Any, elapsed: float) -> tuple[Any, dict[str, Any]]:
        """Convert a completed Responses API response into (message, stats).

        Shared by the non-streaming and streaming paths. ``stats["responses_output_items"]``
        holds the raw output items that the caller threads into the next tool-loop iteration.
        """
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
            "cost": self._response_cost(response),
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
        return text_content, stats

    async def _complete_via_responses_api(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None,
        max_retries: int,
        initial_delay: float,
    ) -> tuple[Any, dict[str, Any]]:
        """Call the OpenAI Responses API (non-streaming) for stateless multi-turn encrypted reasoning.

        The caller (system.py) stores ``responses_output_items`` from stats on the
        assistant message so the next iteration injects them back into the input.
        """
        kwargs = self._build_responses_kwargs(messages, tools)

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = await router.get().aresponses(**kwargs)
                return self._parse_responses_output(response, time.time() - start_time)
            except Exception as e:
                self._raise_if_responses_api_unsupported(e)
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

    async def _complete_stream_via_responses_api(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None,
        max_retries: int,
        initial_delay: float,
    ):
        """Streaming variant of the Responses API call.

        Yields ("delta", text) for each ``response.output_text.delta`` event, then
        ("final", (message, stats)) built from the terminal ``response.completed`` /
        ``response.incomplete`` event's full response object (same shape as the
        non-streaming path, including ``responses_output_items``). Function calls are read from
        that object's ``output``; the argument delta events are not consumed.
        """
        kwargs = self._build_responses_kwargs(messages, tools)
        kwargs["stream"] = True

        last_exception = None
        for attempt in range(max_retries + 1):
            first_token = False
            tool_call_started = False
            try:
                start_time = time.time()
                stream = await router.get().aresponses(**kwargs)
                final_response = None
                async for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "response.output_text.delta":
                        text = getattr(event, "delta", None)
                        if text:
                            first_token = True
                            yield ("delta", text)
                    elif event_type == "response.output_item.added" and (
                        getattr(getattr(event, "item", None), "type", None) == "function_call"
                    ):
                        if not tool_call_started:
                            tool_call_started = True
                            yield ("tool_call_start", None)
                    elif event_type in ("response.completed", "response.incomplete"):
                        final_response = event.response
                    elif event_type in ("response.failed", "error"):
                        raise self._stream_failure_exception(event)

                if final_response is None:
                    raise RuntimeError("Responses API stream ended without a completed response")

                yield ("final", self._parse_responses_output(final_response, time.time() - start_time))
                return
            except Exception as e:
                self._raise_if_responses_api_unsupported(e)
                last_exception = e
                if is_retryable_error(e) and attempt < max_retries and not first_token:
                    delay = initial_delay * (2**attempt)
                    logger.warning(
                        f"Retryable streaming error on attempt {attempt + 1}/{max_retries + 1}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.exception(f"Responses API streaming call failed: {e}")
                raise
        raise last_exception
