"""Streaming caller STT backed by LiveKit Agents plugins, used without a room."""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any

from eva.user_simulator.cascade.constants import CALLER_SAMPLE_RATE
from eva.user_simulator.cascade.stt import TranscriptBuffer
from eva.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODELS = {"elevenlabs": "scribe_v2_realtime"}
API_KEY_ENV = {"elevenlabs": "ELEVENLABS_API_KEY"}


def build_livekit_stt(provider: str, params: dict[str, Any]) -> Any:
    """Construct the LiveKit plugin STT for a provider, without provider-side endpointing."""
    model = params.get("model") or DEFAULT_MODELS.get(provider)
    api_key = params.get("api_key") or os.environ.get(API_KEY_ENV.get(provider, ""), "")
    if provider == "elevenlabs":
        from livekit.plugins import elevenlabs

        # server_vad is deliberately not passed: the plugin then selects
        # commit_strategy=manual, leaving the turn boundary ours to decide.
        return elevenlabs.STT(model=model, api_key=api_key)
    raise ValueError(f"Unsupported caller STT provider: {provider!r}. Supported: {sorted(DEFAULT_MODELS)}")


class LiveKitStreamingSTT:
    """Transcribes assistant audio via a LiveKit STT plugin driven by our tick clock.

    Finalization is ours: `feed(..., commit=True)` maps to the plugin's `flush()`
    sentinel rather than waiting for provider endpointing, which keeps the tick
    scheduler the only thing deciding where a turn ends.
    """

    def __init__(self, provider: str, params: dict[str, Any], *, language: str = "en") -> None:
        self.provider = provider
        self.params = dict(params)
        self.model = params.get("model") or DEFAULT_MODELS.get(provider, "")
        self.language = language
        self.buffer = TranscriptBuffer()
        self._stt: Any = None
        self._stream: Any = None
        self._reader: asyncio.Task | None = None
        self._http: Any = None

    async def start(self) -> None:
        """Open the plugin's HTTP context and streaming recognizer."""
        from livekit.agents.utils import http_context

        # Plugins outside the agent worker have no ambient session; without this they
        # raise "http session outside of a job context" on first use.
        self._http = http_context.open()
        await self._http.__aenter__()
        self._stt = build_livekit_stt(self.provider, self.params)
        self._stream = self._stt.stream()
        self._reader = asyncio.create_task(self._receive_loop())

    async def feed(self, pcm: bytes, *, commit: bool = False) -> None:
        """Push one tick of assistant audio, optionally closing the utterance."""
        if self._stream is None:
            return
        from livekit import rtc

        try:
            self._stream.push_frame(
                rtc.AudioFrame(
                    data=pcm,
                    sample_rate=CALLER_SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=len(pcm) // 2,
                )
            )
            if commit:
                self._stream.flush()
        except Exception as exc:
            logger.warning(f"LiveKit STT feed failed: {exc}")

    async def stop(self) -> None:
        """Close the recognizer and HTTP context. Safe to call twice."""
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        if self._stream is not None:
            with contextlib.suppress(Exception):
                await self._stream.aclose()
            self._stream = None
        if self._http is not None:
            with contextlib.suppress(Exception):
                await self._http.__aexit__(None, None, None)
            self._http = None

    async def _receive_loop(self) -> None:
        """Fold interim and final transcripts into the buffer as the plugin emits them."""
        from livekit.agents import stt as lk_stt

        try:
            async for event in self._stream:
                alternatives = getattr(event, "alternatives", None) or []
                text = alternatives[0].text if alternatives else ""
                if not text:
                    continue
                if event.type == lk_stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    self.buffer.apply_partial(text)
                elif event.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT:
                    self.buffer.commit(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("LiveKit STT receive loop failed; caller may stop hearing the assistant")
