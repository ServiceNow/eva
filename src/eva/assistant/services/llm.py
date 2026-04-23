"""LLM client factory and implementations using LiteLLM."""

import asyncio
import time
from typing import Any

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
        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

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
                finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
                model = getattr(response, "model", self.model)
                hidden_params = getattr(response, "_hidden_params", {}) or {}
                response_cost = hidden_params.get("response_cost")
                cost_source = "litellm"

                # Extract reasoning if present (OpenAI o1 and compatible models)
                reasoning = getattr(message, "reasoning_content", None)

                stats = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                    "model": model,
                    "cost": response_cost,
                    "cost_source": cost_source,
                    "latency": round(elapsed_time, 3),
                    "reasoning": reasoning,
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
