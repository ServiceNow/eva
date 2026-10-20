"""Streaming transcription of the assistant's audio for the cascade caller."""

from __future__ import annotations

import asyncio
import base64
import os
from collections.abc import Callable
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
        on_dropped_final: Callable[[str], None] | None = None,
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
        self._stale_commits = 0
        """Commits whose segment was already taken from the partial; their late finals are dropped."""
        self._on_dropped_final = on_dropped_final
        """Called with each dropped final, so the caller can record what it acted on without."""
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
            await self._connection.send({"audio_base_64": base64.b64encode(pcm).decode("ascii")})

    def expect_commit(self) -> None:
        """Count a commit as pending from the moment it is requested, before ``commit`` sends it.

        ``committed_event`` is set once every expected commit has its final text. Counting at request
        time means a partial fallback taken before the commit is sent still marks its final as stale.
        """
        self._pending_commits += 1
        self.committed_event.clear()

    async def commit(self) -> None:
        """Close the current segment, answering the oldest ``expect_commit``."""
        if self._connection is not None:
            await self._connection.commit()

    def heard_text(self) -> str:
        """Everything heard since the last ``take``, including the in-flight partial."""
        return " ".join([*self._committed, self.partial]).strip()

    def take(self, *, include_partial: bool = False) -> str:
        """Return and clear the committed text, optionally with the in-flight partial."""
        text = self.heard_text() if include_partial else " ".join(self._committed).strip()
        self._committed.clear()
        if include_partial:
            self.partial = ""
            # Finals still owed for this turn would otherwise land in the next one.
            self._stale_commits += self._pending_commits
            self._pending_commits = 0
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
        # Scribe answers commits in order, so the oldest outstanding one is the stale one.
        if self._stale_commits:
            self._stale_commits -= 1
            logger.warning(f"Dropping late final for an already-answered turn: {text[:120]!r}")
            if self._on_dropped_final is not None:
                self._on_dropped_final(text)
            return
        if not self._pending_commits:
            # Finals are matched to commits by order; one with no commit owed means that matching is off,
            # and later text could be dropped or heard twice. Fail like a provider error rather than guess.
            self.error = f"final transcript with no commit outstanding: {text[:120]!r}"
            logger.error(f"Caller STT error: {self.error}")
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
