"""Audio language model vLLM client for chat completions and transcription.

Talks to a self-hosted audio language model served via vLLM's OpenAI-compatible HTTP API.
Provides chat completions with audio content support and audio transcription.
"""

import asyncio
import time
from typing import Any

from openai import AsyncOpenAI

from eva.assistant.pipeline.alm_base import (
    DEFAULT_NUM_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLE_WIDTH,
    DEFAULT_TRANSCRIPTION_PROMPT,
    BaseALMClient,
)
from eva.utils.logging import get_logger

logger = get_logger(__name__)


class ALMvLLMClient(BaseALMClient):
    """Client for self-hosted audio language model via vLLM's OpenAI-compatible HTTP API."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = "ultravox-v07",
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            initial_delay=initial_delay,
            sample_rate=sample_rate,
            num_channels=num_channels,
            sample_width=sample_width,
        )
        # Normalize base_url: ensure it ends with /v1 for the OpenAI client
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"

        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=120.0,
        )

        logger.info(
            f"Initialized ALMvLLMClient: base_url={self.base_url}, model={self.model}, "
            f"sample_rate={self.sample_rate}, num_channels={self.num_channels}, "
            f"sample_width={self.sample_width}"
        )

    def _audio_content_part(self, audio_b64: str) -> dict[str, Any]:
        return {
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
        }

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Chat completion with audio and tool support.

        Same return signature as LiteLLMClient.complete():
        Returns (message_or_content, stats_dict).

        When tool_calls are present, returns the full message object.
        Otherwise returns the content string.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                response = await self._client.chat.completions.create(**kwargs)
                elapsed = time.time() - start_time

                message = response.choices[0].message
                usage = response.usage

                # Extract reasoning content if present (OpenAI o1 and compatible models)
                reasoning_content = getattr(message, "reasoning_content", None)

                # Extract reasoning tokens if present
                reasoning_tokens = 0
                if usage and hasattr(usage, "completion_tokens_details"):
                    details = usage.completion_tokens_details
                    if details and hasattr(details, "reasoning_tokens"):
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0)

                stats = {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "reasoning_tokens": reasoning_tokens,
                    "finish_reason": response.choices[0].finish_reason or "unknown",
                    "model": response.model or self.model,
                    "cost": 0.0,  # Self-hosted, no API cost
                    "cost_source": "self_hosted",
                    "latency": round(elapsed, 3),
                    "reasoning": reasoning_content,
                    "reasoning_content": reasoning_content,  # Keep for backward compatibility
                }

                if hasattr(message, "tool_calls") and message.tool_calls:
                    return message, stats
                else:
                    return message.content or "", stats

            except Exception as e:
                last_exception = e
                if self._is_retryable(e) and attempt < self.max_retries:
                    delay = self.initial_delay * (2**attempt)
                    logger.warning(
                        f"Retryable error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"UltravoxVLLM completion failed: {e}")
                    raise

        raise last_exception  # type: ignore[misc]

    async def transcribe(
        self,
        audio_bytes: bytes,
        source_sample_rate: int,
        system_prompt: str | None = None,
    ) -> str | None:
        """Transcribe a chunk of PCM16 audio via vLLM chat completions."""
        if not audio_bytes:
            return None

        prompt = system_prompt or DEFAULT_TRANSCRIPTION_PROMPT
        user_msg = self.build_audio_user_message(audio_bytes, source_sample_rate)
        messages = [{"role": "system", "content": prompt}, user_msg]

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = response.choices[0].message.content if response.choices else None
            return text.strip() if text else None
        except Exception as e:
            logger.error(f"ALMvLLMClient transcription failed: {e}")
            return None
