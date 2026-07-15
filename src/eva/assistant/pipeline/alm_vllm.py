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
    BaseALMClient,
    _StreamedMessage,
    _assemble_stream_chunks,
)
from eva.utils.llm_utils import approximate_reasoning_tokens
from eva.utils.logging import get_logger

logger = get_logger(__name__)

# vLLM / OpenAI-compatible decoding params that are safe to forward via extra_body.
# Unknown keys are still forwarded (vLLM evolves) but warned about, to catch typos.
_KNOWN_SAMPLING_PARAMS = frozenset(
    {
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
        "length_penalty",
        "seed",
        "stop",
        "stop_token_ids",
        "min_tokens",
        "ignore_eos",
        "skip_special_tokens",
        "spaces_between_special_tokens",
    }
)

# Keys the client controls itself — must not be set via sampling_params, or they would
# silently override the managed request (e.g. temperature/max_tokens land in extra_body
# and clobber the top-level args; model/messages/tools break the call entirely).
_RESERVED_SAMPLING_PARAMS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "tools",
        "tool_choice",
        "stream",
        "n",
        "extra_body",
        "chat_template_kwargs",
    }
)


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
        language: str | None = None,
        enable_thinking: bool = False,
        sampling_params: dict[str, Any] | None = None,
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
            language=language,
        )
        self._reasoning_token_fallback_warned = False
        self.enable_thinking = enable_thinking
        # vLLM-specific decoding params (repetition_penalty, top_p, top_k, min_p,
        # frequency_penalty, presence_penalty, ...). Validated here, then merged into
        # extra_body of every complete()/complete_stream() call; not applied to
        # transcribe(), which is kept deterministic.
        self.sampling_params: dict[str, Any] = self._validate_sampling_params(sampling_params)
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
            f"sample_width={self.sample_width}, "
            f"sampling_params={self.sampling_params}, enable_thinking={self.enable_thinking}"
        )

    @staticmethod
    def _validate_sampling_params(sampling_params: dict[str, Any] | None) -> dict[str, Any]:
        """Validate user-supplied vLLM sampling params before they reach extra_body.

        Rejects keys the client manages itself (these would silently clobber the
        request — e.g. a ``temperature`` here overrides the top-level arg via
        extra_body). Warns on keys outside the known vLLM set so typos are visible,
        but still forwards them since vLLM's parameter surface evolves.
        """
        if sampling_params is None:
            return {}
        if not isinstance(sampling_params, dict):
            raise ValueError(f"sampling_params must be a dict, got {type(sampling_params).__name__}")

        reserved = _RESERVED_SAMPLING_PARAMS & sampling_params.keys()
        if reserved:
            raise ValueError(
                f"sampling_params may not override client-managed keys {sorted(reserved)}. "
                "Set temperature/max_tokens via their dedicated params; model/messages/tools "
                "are controlled by the client."
            )

        non_string_keys = [k for k in sampling_params if not isinstance(k, str)]
        if non_string_keys:
            raise ValueError(f"sampling_params keys must be strings, got non-string keys: {non_string_keys}")

        unknown = sampling_params.keys() - _KNOWN_SAMPLING_PARAMS
        if unknown:
            logger.warning(
                f"sampling_params contains keys not in the known vLLM set {sorted(unknown)}; "
                "forwarding to vLLM as-is — check for typos."
            )
        return dict(sampling_params)

    def _audio_content_part(self, audio_b64: str) -> dict[str, Any]:
        return {
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
        }

    def _build_extra_body(self) -> dict[str, Any]:
        """Build the extra_body payload: chat_template_kwargs + vLLM sampling params.

        The validated sampling params ride alongside chat_template_kwargs; vLLM
        accepts non-OpenAI keys at the top level of extra_body.
        """
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        extra_body.update(self.sampling_params)
        return extra_body

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
            "extra_body": self._build_extra_body(),
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
                # vLLM versions use different field names: "reasoning_content" vs "reasoning"
                reasoning_content = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)

                # Extract reasoning tokens if present
                reasoning_tokens = 0
                if usage and hasattr(usage, "completion_tokens_details"):
                    details = usage.completion_tokens_details
                    if details and hasattr(details, "reasoning_tokens"):
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0)

                if reasoning_content and reasoning_tokens == 0:
                    reasoning_tokens = approximate_reasoning_tokens(reasoning_content, self.model, self, logger)

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
                    logger.error(f"VLLM completion failed: {e}")
                    raise

        raise last_exception  # type: ignore[misc]

    async def complete_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ):
        """Stream chat completion from vLLM, yielding text deltas then the final message/stats."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "extra_body": self._build_extra_body(),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            chunks: list[Any] = []
            first_token = False
            try:
                start_time = time.time()
                stream = await self._client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    chunks.append(chunk)
                    choices = getattr(chunk, "choices", None) or []
                    if choices:
                        delta = getattr(choices[0], "delta", None)
                        text = getattr(delta, "content", None) if delta else None
                        if text:
                            first_token = True
                            yield ("delta", text)
                elapsed = time.time() - start_time

                full_content, reasoning_content, finish_reason, usage, assembled_tool_calls = _assemble_stream_chunks(
                    chunks
                )
                reasoning_content = reasoning_content or None

                # Reasoning tokens from usage if the server reported them; else approximate from
                # the streamed reasoning text (mirrors the non-streaming complete() path).
                reasoning_tokens = 0
                if usage and hasattr(usage, "completion_tokens_details"):
                    details = usage.completion_tokens_details
                    if details and hasattr(details, "reasoning_tokens"):
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                if reasoning_content and reasoning_tokens == 0:
                    reasoning_tokens = approximate_reasoning_tokens(reasoning_content, self.model, self, logger)

                stats = {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "reasoning_tokens": reasoning_tokens,
                    "finish_reason": finish_reason,
                    "model": self.model,
                    "cost": 0.0,
                    "cost_source": "self_hosted",
                    "latency": round(elapsed, 3),
                    "reasoning": reasoning_content,
                    "reasoning_content": reasoning_content,
                }
                yield ("final", (_StreamedMessage(full_content, assembled_tool_calls), stats))
                return

            except Exception as e:
                last_exception = e
                if self._is_retryable(e) and attempt < self.max_retries and not first_token:
                    delay = self.initial_delay * (2**attempt)
                    logger.warning(
                        f"Retryable streaming error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"ALMvLLMClient streaming completion failed: {e}")
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

        prompt = system_prompt or self.default_transcription_prompt
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
