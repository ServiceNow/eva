"""LLM client factory and implementations using LiteLLM."""

import asyncio
import time
from typing import Any, Optional

import litellm
from dotenv import load_dotenv

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

    def __init__(self, model: str):
        """Initialize LiteLLM client.

        Args:
            model: Model name matching a model_name in EVA_MODEL_LIST (e.g., 'gpt-5.2', 'gemini-3-pro')
        """
        self.model = model

        logger.info(f"Initialized LiteLLM client with model: {self.model}")
        litellm.drop_params = True

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict]] = None,
        reasoning_effort: Optional[str] = None,
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

                # Extract reasoning content - LiteLLM provides unified interface
                # reasoning_content: string containing reasoning (all providers)
                # thinking_blocks: list of thinking blocks (Anthropic only)
                reasoning_content = getattr(message, "reasoning_content", None)
                thinking_blocks = getattr(message, "thinking_blocks", None) if hasattr(message, "thinking_blocks") else None

                # For Anthropic models, extract human-readable reasoning from thinking blocks
                if thinking_blocks and isinstance(thinking_blocks, list):
                    # Thinking blocks are list of dicts with 'type' and 'thinking' (and optional 'signature')
                    thinking_texts = []
                    for block in thinking_blocks:
                        if isinstance(block, dict) and block.get("type") == "thinking" and "thinking" in block:
                            thinking_texts.append(block["thinking"])
                    if thinking_texts:
                        # Override reasoning_content with extracted thinking text
                        reasoning_content = "\n\n".join(thinking_texts)
                        total_chars = sum(len(t) for t in thinking_texts)
                        logger.info(f"💭 Extracted {len(thinking_texts)} thinking block(s) from Anthropic response ({total_chars} chars)")
                        logger.debug(f"Thinking content preview: {reasoning_content[:200]}...")
                elif reasoning_content:
                    # Non-Anthropic model with reasoning_content (e.g., Gemini)
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
