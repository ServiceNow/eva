"""Audio language model vLLM client for chat completions and transcription.

Talks to a self-hosted audio language model served via vLLM's OpenAI-compatible HTTP API.
Provides chat completions with audio content support and audio transcription.
"""

import asyncio
import base64
import io
import struct
import time
import wave
from typing import Any, Optional

from openai import AsyncOpenAI

from eva.utils.logging import get_logger

logger = get_logger(__name__)

# Default audio parameters (Ultravox: 16kHz PCM16 mono)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NUM_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit PCM

VALID_SAMPLE_RATES = {8000, 16000, 24000, 44100, 48000}


def pcm16_to_wav_bytes(
    pcm_data: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_channels: int = DEFAULT_NUM_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def resample_pcm16(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM16 mono audio via linear interpolation.

    Args:
        pcm_data: Raw PCM16 audio bytes.
        from_rate: Source sample rate in Hz.
        to_rate: Target sample rate in Hz.

    Returns:
        Resampled PCM16 bytes at the target rate.
    """
    if from_rate == to_rate:
        return pcm_data
    num_samples = len(pcm_data) // 2
    if num_samples == 0:
        return pcm_data
    samples = struct.unpack(f"<{num_samples}h", pcm_data)
    ratio = to_rate / from_rate
    out_count = int(num_samples * ratio)
    out_samples = []
    for i in range(out_count):
        src_idx = i / ratio
        idx0 = int(src_idx)
        idx1 = min(idx0 + 1, num_samples - 1)
        frac = src_idx - idx0
        val = int(samples[idx0] * (1 - frac) + samples[idx1] * frac)
        val = max(-32768, min(32767, val))
        out_samples.append(val)
    return struct.pack(f"<{len(out_samples)}h", *out_samples)


class ALMvLLMClient:
    """Client for self-hosted audio language model via vLLM's OpenAI-compatible HTTP API.

    Provides:
    - complete(): Chat completions with audio content support + tool calling
    - transcribe(): Audio transcription via chat completion prompt
    - build_audio_user_message(): Build OpenAI-format message with audio content
    """

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
        # Normalize base_url: ensure it ends with /v1 for the OpenAI client
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.initial_delay = initial_delay

        if sample_rate not in VALID_SAMPLE_RATES:
            raise ValueError(f"Invalid sample_rate={sample_rate}. Must be one of {sorted(VALID_SAMPLE_RATES)}")
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.sample_width = sample_width

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

    def build_audio_user_message(
        self,
        audio_bytes: bytes,
        source_sample_rate: int,
        text_hint: str = "",
    ) -> dict[str, Any]:
        """Build an OpenAI-format user message with audio content.

        Args:
            audio_bytes: Raw PCM16 audio bytes at source_sample_rate.
            source_sample_rate: Sample rate of the input audio.
            text_hint: Optional text to include alongside audio.

        Returns:
            Message dict with audio_url content for vLLM.
        """
        resampled = resample_pcm16(audio_bytes, source_sample_rate, self.sample_rate)
        wav_bytes = pcm16_to_wav_bytes(
            resampled,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            sample_width=self.sample_width,
        )
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        content: list[dict[str, Any]] = []
        if text_hint:
            content.append({"type": "text", "text": text_hint})
        content.append(
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
            }
        )

        return {"role": "user", "content": content}

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict]] = None,
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

        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                response = await self._client.chat.completions.create(**kwargs)
                elapsed = time.time() - start_time

                message = response.choices[0].message
                usage = response.usage

                # Extract reasoning if present (OpenAI o1 and compatible models)
                reasoning = getattr(message, "reasoning_content", None)

                stats = {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "finish_reason": response.choices[0].finish_reason or "unknown",
                    "model": response.model or self.model,
                    "cost": 0.0,  # Self-hosted, no API cost
                    "cost_source": "self_hosted",
                    "latency": round(elapsed, 3),
                    "reasoning": reasoning,
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

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Check if an error is retryable (connection, timeout, server errors)."""
        error_str = str(error).lower()
        retryable_patterns = [
            "connection",
            "timeout",
            "502",
            "503",
            "504",
            "rate limit",
            "too many requests",
            "server error",
        ]
        return any(pattern in error_str for pattern in retryable_patterns)
