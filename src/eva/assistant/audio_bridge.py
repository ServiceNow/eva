"""Shared audio bridge utilities for framework-specific assistant servers.

All framework servers need to:
1. Accept Twilio-framed WebSocket connections from the user simulator
2. Convert audio between Twilio's mulaw 8kHz and the framework's native format
3. Write framework_logs.jsonl with timestamped events

This module provides the common infrastructure.
"""

import audioop
import base64
import json
import struct
import time
from pathlib import Path

import numpy as np
import soxr

from eva.utils.logging import get_logger

logger = get_logger(__name__)


# ── Audio format conversion ──────────────────────────────────────────


def mulaw_8k_to_pcm16_16k(mulaw_bytes: bytes) -> bytes:
    """Convert 8kHz mu-law audio to 16kHz 16-bit PCM."""
    # Decode mu-law to 16-bit PCM at 8kHz
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    # Upsample from 8kHz to 16kHz
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
    return pcm_16k


def mulaw_8k_to_pcm16_24k(mulaw_bytes: bytes) -> bytes:
    """Convert 8kHz mu-law audio to 24kHz 16-bit PCM."""
    # Decode mu-law to 16-bit PCM at 8kHz
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    # Upsample from 8kHz to 24kHz
    pcm_24k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 24000, None)
    # audioop.ratecv can produce ±2 samples; clamp to exact 3× input length
    # so that the inverse conversion recovers the original sample count.
    expected_bytes = len(pcm_8k) * 3
    if len(pcm_24k) < expected_bytes:
        pcm_24k = pcm_24k + b"\x00" * (expected_bytes - len(pcm_24k))
    elif len(pcm_24k) > expected_bytes:
        pcm_24k = pcm_24k[:expected_bytes]
    return pcm_24k


def pcm16_24k_to_mulaw_8k(pcm_bytes: bytes) -> bytes:
    """Convert 24kHz 16-bit PCM to 8kHz mu-law.

    Uses soxr VHQ resampling (same as Pipecat) for proper anti-aliasing during the 3:1 downsampling.
    audioop.ratecv produces muffled audio because it lacks an anti-aliasing filter.
    """
    # Downsample from 24kHz to 8kHz using high-quality resampler
    audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
    resampled = soxr.resample(audio_data, 24000, 8000, quality="VHQ")
    # Both audioop.ratecv (upstream) and soxr can produce ±1 sample due to filter rounding.
    # Use round() so that e.g. 2399 input samples → round(2399/3) = 800, not 799.
    expected_samples = round(len(audio_data) * 8000 / 24000)
    if len(resampled) < expected_samples:
        resampled = np.pad(resampled, (0, expected_samples - len(resampled)))
    elif len(resampled) > expected_samples:
        resampled = resampled[:expected_samples]
    pcm_8k = resampled.astype(np.int16).tobytes()
    # Encode to mu-law
    return audioop.lin2ulaw(pcm_8k, 2)


def sync_buffer_to_position(buffer: bytearray, target_position: int) -> None:
    """Pad *buffer* with silence bytes so it reaches *target_position*.

    Mirrors pipecat's ``AudioBufferProcessor._sync_buffer_to_position``.
    Call this **before** extending the *other* track so both tracks stay
    positionally aligned.
    """
    current_len = len(buffer)
    if current_len < target_position:
        buffer.extend(b"\x00" * (target_position - current_len))


def pcm16_mix(track_a: bytes, track_b: bytes) -> bytes:
    """Mix two 16-bit PCM tracks by sample-wise addition with clipping.

    Both tracks must be the same sample rate. If lengths differ,
    the shorter track is zero-padded.
    """
    len_a, len_b = len(track_a), len(track_b)
    max_len = max(len_a, len_b)

    # Zero-pad shorter track
    if len_a < max_len:
        track_a = track_a + b"\x00" * (max_len - len_a)
    if len_b < max_len:
        track_b = track_b + b"\x00" * (max_len - len_b)

    # Mix with clipping
    n_samples = max_len // 2
    fmt = f"<{n_samples}h"
    samples_a = struct.unpack(fmt, track_a)
    samples_b = struct.unpack(fmt, track_b)
    mixed = struct.pack(fmt, *(max(-32768, min(32767, a + b)) for a, b in zip(samples_a, samples_b)))
    return mixed


# ── Twilio WebSocket Protocol ────────────────────────────────────────


def parse_twilio_media_message(message: str) -> bytes | None:
    """Parse a Twilio media WebSocket message and extract raw audio bytes.

    Returns None if the message is not a media message.
    """
    try:
        data = json.loads(message)
        if data.get("event") == "media":
            payload = data["media"]["payload"]
            return base64.b64decode(payload)
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def create_twilio_media_message(stream_sid: str, audio_bytes: bytes) -> str:
    """Create a Twilio media WebSocket message with the given audio bytes."""
    payload = base64.b64encode(audio_bytes).decode("ascii")
    return json.dumps(
        {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": payload,
            },
        }
    )


# ── Framework Logs Writer ────────────────────────────────────────────


class FrameworkLogWriter:
    """Write framework_logs.jsonl (replacement for pipecat_logs.jsonl).

    Capture turn boundaries, TTS text, and LLM responses with accurate
    wall-clock timestamps.
    """

    def __init__(self, output_dir: Path):
        self.log_file = output_dir / "framework_logs.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, data: dict, timestamp_ms: int | None = None) -> None:
        """Write a single log entry.

        Args:
            event_type: One of 'turn_start', 'turn_end', 'tts_text', 'llm_response'
            data: Event data dict. Must contain a 'frame' key for tts_text/llm_response.
            timestamp_ms: Wall-clock timestamp in milliseconds. Defaults to now.
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        entry = {
            "timestamp": timestamp_ms,
            "type": event_type,
            "data": data,
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error writing framework log: {e}")

    def turn_start(self, timestamp_ms: int | None = None) -> None:
        """Log a turn start event."""
        self.write("turn_start", {"frame": "turn_start"}, timestamp_ms)

    def turn_end(self, was_interrupted: bool = False, timestamp_ms: int | None = None) -> None:
        """Log a turn end event."""
        self.write("turn_end", {"frame": "turn_end", "was_interrupted": was_interrupted}, timestamp_ms)

    def tts_text(self, text: str, timestamp_ms: int | None = None) -> None:
        """Log TTS text (what was actually spoken)."""
        self.write("tts_text", {"frame": text}, timestamp_ms)

    def llm_response(self, text: str, timestamp_ms: int | None = None) -> None:
        """Log LLM response text (full intended response)."""
        self.write("llm_response", {"frame": text}, timestamp_ms)

    def s2s_transcript(self, text: str, timestamp_ms: int | None = None) -> None:
        """Log S2S transcript (what was actually spoken)."""
        self.write("s2s_transcript", {"frame": text}, timestamp_ms)


# ── Metrics Log Writer ───────────────────────────────────────────────


class MetricsLogWriter:
    """Writes pipecat_metrics.jsonl for non-Pipecat frameworks.

    Pipecat writes its own metrics natively via MetricsFileObserver.  This
    writer covers OpenAI Realtime, Gemini Live, and any other framework that
    manages its own session loop.
    """

    def __init__(self, output_dir: Path):
        self.log_file = output_dir / "pipecat_metrics.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)

    def write_latency(self, stage: str, value_seconds: float, model: str = "") -> None:
        """Write a LatencyMetric entry.

        Args:
            stage: Semantic label for the stage being measured. Use ``"stt"``
                for STT processing time, ``"tts"`` for TTS time-to-first-byte,
                ``"model_response"`` for s2s/realtime time from user speech end
                to first model audio chunk.
            value_seconds: Latency in seconds.
            model: Model identifier (optional).
        """
        entry = {
            "timestamp": int(time.time() * 1000),
            "type": "LatencyMetric",
            "stage": stage,
            "model": model,
            "value": value_seconds,
        }
        self._append(entry)

    def write_token_usage(
        self,
        processor: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Write an LLMTokenUsageMetricsData entry."""
        entry = {
            "timestamp": int(time.time() * 1000),
            "type": "LLMTokenUsageMetricsData",
            "processor": processor,
            "model": model,
            "value": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        self._append(entry)

    def _append(self, entry: dict) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Error writing metrics log: {e}")
