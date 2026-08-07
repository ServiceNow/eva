"""Streaming speech-to-text client that transcribes the assistant's audio for the caller."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any

import websockets

from eva.user_simulator.cascade.constants import CALLER_SAMPLE_RATE
from eva.utils.logging import get_logger

logger = get_logger(__name__)

SCRIBE_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

INCOMPLETE_MARKER = "[CURRENTLY SPEAKING, INCOMPLETE]"


class TranscriptBuffer:
    """Accumulates committed transcript segments separately from the in-flight partial."""

    def __init__(self) -> None:
        self.committed = ""
        self.in_flight = ""

    def apply_partial(self, text: str) -> None:
        """Replace the in-flight partial with the latest partial transcript."""
        self.in_flight = text

    def commit(self, text: str) -> None:
        """Append a finalized segment to the committed text and clear the partial."""
        self.committed = f"{self.committed} {text}".strip() if self.committed else text
        self.in_flight = ""

    def current_text(self) -> str:
        """Render everything heard so far, marking an in-flight utterance as incomplete."""
        if not self.in_flight:
            return self.committed
        prefix = f"{self.committed} " if self.committed else ""
        return f"{prefix}{self.in_flight} {INCOMPLETE_MARKER}"

    def take_committed(self) -> str:
        """Return and clear the committed text."""
        text = self.committed
        self.committed = ""
        return text


class ScribeStreamingSTT:
    """Streams PCM16 to Scribe and folds results into a TranscriptBuffer.

    Uses commit_strategy=manual so the caller's own turn-end decision drives
    commits; Scribe's built-in VAD would be a second, independent detector on its
    own timing, which is the problem the tick design exists to remove.
    """

    def __init__(self, params: dict[str, Any], *, language: str = "en") -> None:
        self._model = params.get("model", "scribe_v2_realtime")
        self._api_key = params.get("api_key") or os.environ.get("ELEVENLABS_API_KEY", "")
        self._language = language
        self.buffer = TranscriptBuffer()
        self._ws: Any = None
        self._receive_task: asyncio.Task | None = None
        self._error: Exception | None = None
        self._closed = False

    async def start(self) -> None:
        """Open the transcription socket and begin consuming results."""
        self._ws = await websockets.connect(
            f"{SCRIBE_URL}?model_id={self._model}"
            f"&audio_format=pcm_{CALLER_SAMPLE_RATE}"
            f"&language_code={self._language}"
            "&commit_strategy=manual",
            additional_headers={"xi-api-key": self._api_key},
        )
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def feed(self, pcm: bytes, *, commit: bool = False) -> None:
        """Send one tick of assistant audio, optionally closing the utterance.

        Raises once the session has closed, instead of repeating a swallowed warning every
        tick while the caller goes silently deaf for the rest of the call.
        """
        if self._ws is None:
            return
        if self._closed:
            raise RuntimeError("Scribe session is closed; caller can no longer hear the assistant")
        message: dict[str, Any] = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm).decode(),
        }
        if commit:
            message["commit"] = True
        try:
            await self._ws.send(json.dumps(message))
        except Exception as exc:
            self._closed = True
            logger.error(f"Scribe session closed unexpectedly; caller is now deaf: {exc}")
            raise RuntimeError("Scribe session closed unexpectedly") from exc

    async def stop(self) -> None:
        """Close the socket and stop consuming. Safe to call twice."""
        if self._receive_task is not None:
            self._receive_task.cancel()
            self._receive_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None

    async def _receive_loop(self) -> None:
        """Fold partial and committed transcripts into the buffer as they arrive."""
        while True:
            try:
                raw = await self._ws.recv()
            except Exception as exc:
                self._error = exc
                logger.exception("Scribe receive loop failed")
                return
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = message.get("message_type", "")
            text = message.get("text", "")
            if not text:
                continue
            if kind == "partial_transcript":
                self.buffer.apply_partial(text)
            elif kind.startswith("committed_transcript") or kind.startswith("final_transcript"):
                self.buffer.commit(text)
