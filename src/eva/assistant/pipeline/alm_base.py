"""Abstract base class for audio language model clients.

Audio-LLM clients accept audio input + text context and return text output.
Concrete implementations (vLLM-hosted models, Gemini, etc.) differ in auth,
endpoint shape, and provider-specific request quirks, but share:

- A common interface (build_audio_user_message, complete, transcribe)
- Audio utilities (PCM resampling, WAV encoding)
- Common numeric configuration (sample rate, retry policy, etc.)

The methods used by the audio-LLM pipeline:

- build_audio_user_message: serialize a chunk of PCM audio into a chat message
- complete: chat completion with audio + text + tool support (used by AudioLLMAgenticSystem)
- transcribe: transcription-only call (used by AudioTranscriptionProcessor for
  logging and conversation history, since a regular multimodal LLM does not
  surface a separate transcription stream)
"""

import base64
import io
import struct
import wave
from abc import ABC, abstractmethod
from typing import Any

# Default audio parameters (Ultravox: 16kHz PCM16 mono)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NUM_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit PCM

VALID_SAMPLE_RATES = {8000, 16000, 24000, 44100, 48000}

# Default system prompt for audio-LLM transcription calls. Used by client
# transcribe() implementations when no override is supplied, and by
# AudioTranscriptionProcessor when no system_prompt is configured.
DEFAULT_TRANSCRIPTION_PROMPT = """You are an audio transcriber. Your job is to transcribe the input audio to text exactly as it was said by the user.

Rules:
- Respond with an exact transcription of the audio input only.
- Do not include any text other than the transcription.
- Do not explain or add to your response.
- Transcribe the audio input simply and precisely.
- If the audio is not clear, respond with exactly: UNCLEAR"""


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
    """Resample PCM16 mono audio via linear interpolation."""
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


class BaseALMClient(ABC):
    """Common interface and shared behavior for audio-LLM clients."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
    ):
        if sample_rate not in VALID_SAMPLE_RATES:
            raise ValueError(f"Invalid sample_rate={sample_rate}. Must be one of {sorted(VALID_SAMPLE_RATES)}")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.sample_width = sample_width

    def _audio_to_b64_wav(self, audio_bytes: bytes, source_sample_rate: int) -> str:
        """Resample, WAV-wrap, and base64-encode raw PCM16 audio."""
        resampled = resample_pcm16(audio_bytes, source_sample_rate, self.sample_rate)
        wav_bytes = pcm16_to_wav_bytes(
            resampled,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            sample_width=self.sample_width,
        )
        return base64.b64encode(wav_bytes).decode("utf-8")

    def build_audio_user_message(
        self,
        audio_bytes: bytes,
        source_sample_rate: int,
        text_hint: str = "",
    ) -> dict[str, Any]:
        """Build a user message with audio content. Provider-specific shape comes
        from the subclass _audio_content_part hook."""
        audio_b64 = self._audio_to_b64_wav(audio_bytes, source_sample_rate)
        content: list[dict[str, Any]] = []
        if text_hint:
            content.append({"type": "text", "text": text_hint})
        content.append(self._audio_content_part(audio_b64))
        return {"role": "user", "content": content}

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

    @abstractmethod
    def _audio_content_part(self, audio_b64: str) -> dict[str, Any]:
        """Return the provider-specific content dict for a base64-WAV audio blob."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Chat completion with audio and tool support.

        Returns (message_or_content, stats_dict). When tool_calls are present
        on the response, returns the full message object; otherwise returns
        the content string.
        """

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        source_sample_rate: int,
        system_prompt: str | None = None,
    ) -> str | None:
        """Transcribe a chunk of PCM16 audio to text.

        Returns the transcript, or None on error / empty audio.
        """
