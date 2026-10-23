"""Streaming transcription of the assistant's audio for the cascade caller."""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Any

try:
    import audioop
except ImportError:  # Python 3.13+
    import audioop_lts as audioop

from elevenlabs.realtime import AudioFormat, CommitStrategy, RealtimeConnection, RealtimeEvents, ScribeRealtime

from eva.utils.logging import get_logger

logger = get_logger(__name__)

INPUT_SAMPLE_RATE = 8000
"""The bridge delivers the assistant's audio as 8kHz mu-law, which is what send() accepts."""

STT_AUDIO_FORMAT = AudioFormat.PCM_16000
STT_SAMPLE_RATE = 16000
"""What Scribe is sent. The same 8kHz audio decoded and upsampled to 16kHz PCM16 transcribed
~120ms faster (STT lag p50 422 vs 540ms) than sending Scribe the mu-law directly, with no
change in the information it carries (scripts/measure_barge_in_latency.py, STT_BACKENDS)."""

MIN_COMMIT_SECONDS = 0.3
"""Scribe ignores a commit with less uncommitted audio than this, and sends no final for it."""


class ScribeStreamingSTT:
    """Transcribes assistant audio continuously over ElevenLabs Scribe realtime.

    Partials are tracked as they arrive so a listener can act on what has been heard
    mid-utterance. Commits are manual: the caller decides where a segment ends rather
    than the provider's VAD, so turn boundaries have a single owner.
    """

    def __init__(
        self,
        model: str,
        *,
        language: str = "en",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.language = language.split("-")[0]
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self._connection: RealtimeConnection | None = None
        self._committed: list[str] = []
        self.partial = ""
        self.error: str | None = None
        self._closing = False
        self.committed_event = asyncio.Event()
        self._pending_commits = 0
        self._uncommitted_bytes = 0
        self._resample_state = None
        self.updated = asyncio.Event()
        """Set on every partial or committed transcript, so a listener can react to each update
        (e.g. an interruption decision) without polling. Clear it before waiting."""

    async def start(self) -> None:
        """Open the realtime session."""
        self._connection = await ScribeRealtime(api_key=self._api_key).connect(
            {
                "model_id": self.model,
                "audio_format": STT_AUDIO_FORMAT,
                "sample_rate": STT_SAMPLE_RATE,
                "commit_strategy": CommitStrategy.MANUAL,
                "language_code": self.language,
            }
        )
        self._connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, self._on_partial)
        self._connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, self._on_committed)
        self._connection.on(RealtimeEvents.ERROR, self._on_error)

    async def send(self, mulaw: bytes) -> None:
        """Stream one chunk of the assistant's 8kHz mu-law audio, as 16kHz PCM16."""
        if self.error is not None:
            raise RuntimeError(f"Caller STT failed: {self.error}")
        if self._connection is not None and mulaw:
            pcm, self._resample_state = audioop.ratecv(
                audioop.ulaw2lin(mulaw, 2), 2, 1, INPUT_SAMPLE_RATE, STT_SAMPLE_RATE, self._resample_state
            )
            self._uncommitted_bytes += len(pcm)
            await self._connection.send({"audio_base_64": base64.b64encode(pcm).decode("ascii")})

    def expect_commit(self) -> None:
        """Count a commit as pending from the moment it is requested, before ``commit`` sends it.

        ``committed_event`` is set once every expected commit has its final text.
        """
        self._pending_commits += 1
        self.committed_event.clear()

    async def commit(self) -> None:
        """Close the current segment, answering the oldest ``expect_commit``."""
        if self._connection is not None:
            # A pause commit can follow an earlier one by a few ms of audio; pad the tail with silence so
            # Scribe accepts it, rather than never answering a commit that was counted as pending.
            missing = int(MIN_COMMIT_SECONDS * STT_SAMPLE_RATE * 2) - self._uncommitted_bytes
            if missing > 0:
                await self._connection.send({"audio_base_64": base64.b64encode(bytes(missing)).decode("ascii")})
            self._uncommitted_bytes = 0
            await self._connection.commit()

    def heard_text(self) -> str:
        """Everything heard since the last ``take``, including the in-flight partial."""
        return " ".join([*self._committed, self.partial]).strip()

    def take(self) -> str:
        """Return and clear the committed text."""
        text = " ".join(self._committed).strip()
        self._committed.clear()
        return text

    async def stop(self) -> None:
        """Close the session. Safe to call twice."""
        connection, self._connection = self._connection, None
        # Our own close makes the SDK's receive loop raise (Scribe sends no close frame back),
        # which it reports as an ERROR event; anything after this point is teardown, not a failure.
        self._closing = True
        if connection is not None:
            try:
                await connection.close()
            except Exception as exc:
                logger.warning(f"Error closing caller STT: {exc}")

    def _on_partial(self, data: dict[str, Any]) -> None:
        self.partial = data.get("text", "")
        self.updated.set()

    def _on_committed(self, data: dict[str, Any]) -> None:
        text = data.get("text", "").strip()
        self.partial = ""
        self.updated.set()
        if not self._pending_commits:
            # Scribe also commits on its own (e.g. a long unbroken stretch of speech). That text is real
            # and owes nobody a commit, so keep it rather than lose it or fail the call.
            logger.warning(f"Final transcript with no commit outstanding, keeping it: {text[:120]!r}")
            if text:
                self._committed.append(text)
            return
        if text:
            self._committed.append(text)
        self._pending_commits -= 1
        if self._pending_commits == 0:
            self.committed_event.set()

    def _on_error(self, data: dict[str, Any]) -> None:
        if self._closing:
            logger.debug(f"Caller STT closed: {data}")
            return
        self.error = str(data.get("error") or data.get("message_type") or data)
        logger.error(f"Caller STT error: {self.error}")
