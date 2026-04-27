"""Unified LLM client using LiteLLM for all model calls."""

import asyncio
import itertools
import random
from typing import ClassVar

from dotenv import load_dotenv

from eva.utils import router
from eva.utils.error_handler import is_retryable_error
from eva.utils.logging import get_logger

logger = get_logger(__name__)
load_dotenv()


class LLMClient:
    """Unified LLM client using LiteLLM.

    Concurrency is managed by the LiteLLM Router per-deployment `max_parallel_requests` and `rpm`/`tpm` limits configured in `EVA_MODEL_LIST`.
    """

    _call_counter: ClassVar[itertools.count] = itertools.count(1)  # Used as a unique identifier for logs.

    def __init__(
        self,
        model: str,
        timeout: int = 480,
        params: dict | None = None,
        max_retries: int = 5,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 60.0,
        retry_multiplier: float = 2.0,
    ):
        """Initialize the LLM client.

        Args:
            model: Model name matching a model_name in EVA_MODEL_LIST (e.g., 'gpt-5.2', 'gemini-3-pro')
            timeout: Timeout in seconds for requests
            params: Dictionary of model parameters (temperature, max_tokens, top_p, etc.)
                    passed directly to litellm acompletion
            max_retries: Maximum number of retry attempts (default: 5)
            retry_min_wait: Minimum wait time in seconds between retries (default: 1.0)
            retry_max_wait: Maximum wait time in seconds between retries (default: 60.0)
            retry_multiplier: Exponential backoff multiplier (default: 2.0)
        """
        self.model = model
        self.params = params or {}
        self.timeout = timeout

        # Retry configuration
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.retry_multiplier = retry_multiplier

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable using centralized error handling.

        Args:
            error: The exception that occurred

        Returns:
            True if the error is retryable, False otherwise
        """
        return is_retryable_error(error)

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter.

        Args:
            attempt: The current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Calculate exponential backoff: min_wait * (multiplier ^ attempt)
        delay = self.retry_min_wait * (self.retry_multiplier**attempt)

        # Cap at max_wait
        delay = min(delay, self.retry_max_wait)

        # Add jitter (randomize ±20%)
        jitter = delay * 0.2 * (2 * random.random() - 1)
        delay = delay + jitter

        # Ensure delay is positive
        return max(0, delay)

    async def generate_text(self, messages: list[dict], response_format: dict | None = None) -> tuple[str, dict | None]:
        """Generate text completion with automatic retries.

        Args:
            messages: List of message dicts with role and content
            response_format: Optional response format specification

        Returns:
            Tuple of (generated text, usage dict with prompt_tokens/completion_tokens or None)

        Raises:
            Exception: If the LLM call fails after all retries
        """
        last_error = None
        call_id = next(self._call_counter)

        for attempt in range(self.max_retries + 1):
            try:
                call_retry_id = f"LLM call {call_id} (attempt {attempt + 1}/{self.max_retries + 1})"

                # Build kwargs for acompletion
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "timeout": self.timeout,
                }
                # Merge model params (temperature, max_tokens, top_p, etc.)
                kwargs.update(self.params)

                # Add response format if specified
                if response_format:
                    kwargs["response_format"] = response_format

                # Log details about audio content if present (only on first attempt)
                if attempt == 0:
                    for msg in messages:
                        if isinstance(msg.get("content"), list):
                            for content_item in msg["content"]:
                                if content_item.get("type") in ["audio_url", "image_url"]:
                                    url = content_item.get("audio_url", content_item.get("image_url", {})).get(
                                        "url", ""
                                    )
                                    if url.startswith("data:audio"):
                                        # Extract size info from base64
                                        base64_data = url.split(",")[1] if "," in url else ""
                                        logger.info(
                                            f"{call_retry_id} sending audio to {self.model}: "
                                            f"type={content_item['type']}, "
                                            f"base64_length={len(base64_data)}"
                                        )
                                    break

                # Make the API call
                logger.debug(f"{call_retry_id} started for {self.model}")
                response = await router.get().acompletion(**kwargs)

                (logger.info if attempt > 0 else logger.debug)(f"{call_retry_id} succeeded for {self.model}")

                text = response.choices[0].message.content
                usage = None
                if hasattr(response, "usage") and response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "model_name": getattr(response, "model", None),
                    }
                return text, usage

            except Exception as e:
                last_error = e

                # Check if this is the last attempt
                is_last_attempt = attempt == self.max_retries

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    logger.error(
                        f"{call_retry_id} failed with non-retryable error for {self.model}: {type(e).__name__}: {e}"
                    )
                    raise

                if is_last_attempt:
                    logger.error(f"{call_retry_id} failed last retry for {self.model}: {type(e).__name__}: {e}")
                    raise

                # Calculate backoff delay
                delay = self._calculate_backoff_delay(attempt)

                # Extract short error message, stripping repeated litellm prefixes
                err_msg = str(e).split("\n")[0]
                for prefix in ("litellm.RateLimitError: ", "litellm."):
                    while err_msg.startswith(prefix):
                        err_msg = err_msg[len(prefix) :]

                logger.warning(
                    f"{call_retry_id} failed for {self.model} (retrying in {delay:.3f} s): {type(e).__name__}: {err_msg}"
                )

                # Wait before retrying
                await asyncio.sleep(delay)

        # This should never be reached, but just in case
        if last_error:
            raise last_error
        raise Exception("LLM call failed for unknown reason")
