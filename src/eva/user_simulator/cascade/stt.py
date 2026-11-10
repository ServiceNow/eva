"""Streaming transcription of the assistant's audio for the cascade caller."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from contextlib import suppress
from typing import Any

try:
    import audioop
except ImportError:  # Python 3.13+
    import audioop_lts as audioop

import websockets
from elevenlabs.realtime import AudioFormat, CommitStrategy, RealtimeConnection, RealtimeEvents, ScribeRealtime

from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

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
        self._stale_commits = 0
        """Commits whose segment was already taken from the partial; their late finals are dropped."""
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

    def take(self, *, include_partial: bool = False) -> str:
        """Return and clear the committed text, optionally with the in-flight partial.

        With the partial, call ``expect_commit`` first so its segment gets closed: that final, and
        those of commits already pending, repeat what is returned here, so they are dropped.
        """
        text = self.heard_text() if include_partial else " ".join(self._committed).strip()
        self._committed.clear()
        if include_partial:
            self.partial = ""
            self._stale_commits += self._pending_commits
            self._pending_commits = 0
            self.committed_event.set()
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
        logger.debug(f"Partial transcript ({self._pending_commits} commits pending): {self.partial[:120]!r}")
        self.updated.set()

    def _on_committed(self, data: dict[str, Any]) -> None:
        text = data.get("text", "").strip()
        self.partial = ""
        self.updated.set()
        # Scribe answers commits in order, so the oldest outstanding one is the stale one.
        if self._stale_commits:
            self._stale_commits -= 1
            logger.debug(f"Dropping late final for an already-taken segment: {text[:120]!r}")
            return
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


OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
OPENAI_SAMPLE_RATE = 24000
"""What gpt-live-transcribe is sent: the bridge's 8kHz mu-law, decoded and upsampled to 24kHz PCM16."""
OPENAI_MIN_COMMIT_SECONDS = 0.1
"""OpenAI rejects a commit with less buffered audio than this (input_audio_buffer_commit_empty)."""
OPENAI_EMPTY_COMMIT_CODE = "input_audio_buffer_commit_empty"
"""The one error event that leaves the session usable; every other error fails the STT."""
OPENAI_SESSION_TIMEOUT_SECONDS = 10.0


class OpenAIStreamingSTT:
    """Transcribes assistant audio continuously over an OpenAI realtime transcription session.

    Same contract as ``ScribeStreamingSTT``: partials are tracked as deltas arrive, and commits are manual
    (no server VAD). ``prompt`` and ``keywords`` describe the call to the model: at the minimal delay they
    cut wrong words about as much as a higher delay does, without its lag.
    """

    def __init__(
        self,
        model: str,
        *,
        language: str = "en",
        delay: str = "minimal",
        prompt: str | None = None,
        keywords: list[str] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.language = language.split("-")[0]
        self.delay = delay
        self.prompt = prompt
        self.keywords = keywords
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._ws: websockets.ClientConnection | None = None
        self._receiver: asyncio.Task | None = None
        self._committed: list[str] = []
        self._partials: dict[str, str] = {}
        """Deltas so far of each item not yet completed, in arrival order."""
        self._stale_items: set[str] = set()
        """Items already taken with the partial; their late deltas are dropped."""
        self.error: str | None = None
        self._closing = False
        self._session_ready = asyncio.Event()
        self.committed_event = asyncio.Event()
        self._unacked_commits = 0
        """Expected commits OpenAI has not yet acknowledged with an item id."""
        self._stale_unacked = 0
        """Unacknowledged commits whose segment was already taken from the partial."""
        self._pending_items: set[str] = set()
        """Acknowledged commits still owed a final. Finals for different items may arrive out of order,
        so they are matched by item id rather than counted."""
        self._uncommitted_bytes = 0
        self._resample_state = None
        self.updated = asyncio.Event()
        """Set on every delta or completed transcript, so a listener can react to each update
        (e.g. an interruption decision) without polling. Clear it before waiting."""

    @property
    def partial(self) -> str:
        return "".join(self._partials.values()).strip()

    @property
    def _pending_commits(self) -> int:
        return self._unacked_commits + len(self._pending_items)

    async def start(self) -> None:
        """Open the session and wait until it is configured; a rejected configuration sets ``error``."""
        self._ws = await websockets.connect(
            OPENAI_REALTIME_URL, additional_headers={"Authorization": f"Bearer {self._api_key}"}
        )
        self._receiver = asyncio.create_task(self._receive())
        await self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE},
                            "transcription": self._transcription_config(),
                            "turn_detection": None,
                        }
                    },
                },
            }
        )
        try:
            await asyncio.wait_for(self._session_ready.wait(), timeout=OPENAI_SESSION_TIMEOUT_SECONDS)
        except TimeoutError:
            self.error = self.error or "No session.updated from OpenAI transcription"
        if self.error is not None:
            logger.error(f"Caller STT error: {self.error}")

    def _transcription_config(self) -> dict[str, Any]:
        # English too: codes, names and acronyms are often spoken in English whatever the call's language.
        config: dict[str, Any] = {
            "model": self.model,
            "languages": list(dict.fromkeys([self.language, "en"])),
            "delay": self.delay,
        }
        if self.prompt:
            config["prompt"] = self.prompt
        if self.keywords:
            config["keywords"] = self.keywords
        logger.info(f"Caller STT transcription config: {config}")
        return config

    async def send(self, mulaw: bytes) -> None:
        """Stream one chunk of the assistant's 8kHz mu-law audio, as 24kHz PCM16."""
        if self.error is not None:
            raise RuntimeError(f"Caller STT failed: {self.error}")
        if self._ws is not None and mulaw:
            pcm, self._resample_state = audioop.ratecv(
                audioop.ulaw2lin(mulaw, 2), 2, 1, INPUT_SAMPLE_RATE, OPENAI_SAMPLE_RATE, self._resample_state
            )
            self._uncommitted_bytes += len(pcm)
            await self._append(pcm)

    def expect_commit(self) -> None:
        """Count a commit as pending from the moment it is requested, before ``commit`` sends it.

        ``committed_event`` is set once every expected commit has its final text.
        """
        self._unacked_commits += 1
        self.committed_event.clear()

    async def commit(self) -> None:
        """Close the current segment, answering the oldest ``expect_commit``."""
        if self._ws is not None:
            # Pad a commit too short for OpenAI, rather than never answering a commit counted as pending.
            missing = int(OPENAI_MIN_COMMIT_SECONDS * OPENAI_SAMPLE_RATE * 2) - self._uncommitted_bytes
            if missing > 0:
                await self._append(bytes(missing))
            self._uncommitted_bytes = 0
            await self._send({"type": "input_audio_buffer.commit"})

    def heard_text(self) -> str:
        """Everything heard since the last ``take``, including the in-flight partial."""
        return " ".join([*self._committed, self.partial]).strip()

    def take(self, *, include_partial: bool = False) -> str:
        """Return and clear the committed text, optionally with the in-flight partial.

        With the partial, call ``expect_commit`` first so its segment gets closed: that final, and
        those of commits already pending, repeat what is returned here, so they are dropped.
        """
        text = self.heard_text() if include_partial else " ".join(self._committed).strip()
        self._committed.clear()
        if include_partial:
            self._stale_items.update(self._partials)
            self._stale_items.update(self._pending_items)
            self._partials.clear()
            self._pending_items.clear()
            self._stale_unacked += self._unacked_commits
            self._unacked_commits = 0
            self.committed_event.set()
        return text

    async def stop(self) -> None:
        """Close the session. Safe to call twice."""
        ws, self._ws = self._ws, None
        self._closing = True
        if ws is not None:
            try:
                await ws.close()
            except Exception as exc:
                logger.warning(f"Error closing caller STT: {exc}")
        if self._receiver is not None:
            self._receiver.cancel()
            with suppress(asyncio.CancelledError):
                await self._receiver

    async def _append(self, pcm: bytes) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")})

    async def _send(self, event: dict[str, Any]) -> None:
        if self._ws is not None:
            await self._ws.send(json.dumps(event))

    async def _receive(self) -> None:
        try:
            async for message in self._ws:
                self._on_event(json.loads(message))
        except Exception as exc:
            self._on_error({"error": f"connection lost: {exc}"})
        else:
            self._on_error({"error": "connection closed"})

    def _on_event(self, data: dict[str, Any]) -> None:
        kind = data.get("type", "")
        if kind == "session.updated":
            self._session_ready.set()
        elif kind == "input_audio_buffer.committed":
            self._on_committed(data.get("item_id", ""))
        elif kind == "conversation.item.input_audio_transcription.delta":
            self._on_delta(data)
        elif kind == "conversation.item.input_audio_transcription.completed":
            self._on_completed(data.get("item_id", ""), data.get("transcript", ""))
        elif kind == "conversation.item.input_audio_transcription.failed":
            logger.warning(f"Caller STT could not transcribe a segment: {data.get('error')}")
            self._on_completed(data.get("item_id", ""), "")
        elif kind == "error":
            error = data.get("error") or data
            if isinstance(error, dict) and error.get("code") == OPENAI_EMPTY_COMMIT_CODE:
                # The session survives a rejected commit; answer it as an empty segment rather than leave
                # it pending (and the caller waiting on it) or fail the call.
                logger.warning(f"Caller STT rejected a commit as empty: {error.get('message')}")
                self._on_rejected_commit()
            else:
                self._on_error(error)

    def _on_delta(self, data: dict[str, Any]) -> None:
        item, delta = data.get("item_id", ""), data.get("delta", "")
        if item in self._stale_items or not delta:
            return
        self._partials[item] = self._partials.get(item, "") + delta
        logger.debug(f"Partial transcript ({self._pending_commits} commits pending): {self.partial[:120]!r}")
        self.updated.set()

    def _on_committed(self, item: str) -> None:
        # Commits are acknowledged in order, so the oldest unacknowledged one is the one acknowledged.
        if self._stale_unacked:
            self._stale_unacked -= 1
            self._stale_items.add(item)
        elif self._unacked_commits:
            self._unacked_commits -= 1
            self._pending_items.add(item)

    def _on_rejected_commit(self) -> None:
        # Rejections, like acknowledgements, answer the oldest unacknowledged commit.
        if self._stale_unacked:
            self._stale_unacked -= 1
        elif self._unacked_commits:
            self._unacked_commits -= 1
        if self._pending_commits == 0:
            self.committed_event.set()

    def _on_completed(self, item: str, transcript: str) -> None:
        text = transcript.strip()
        self._partials.pop(item, None)
        self.updated.set()
        if item in self._stale_items:
            self._stale_items.discard(item)
            logger.debug(f"Dropping late final for an already-taken segment: {text[:120]!r}")
            return
        if item not in self._pending_items:
            logger.warning(f"Final transcript with no commit outstanding, keeping it: {text[:120]!r}")
            if text:
                self._committed.append(text)
            return
        if text:
            self._committed.append(text)
        self._pending_items.discard(item)
        if self._pending_commits == 0:
            self.committed_event.set()

    def _on_error(self, data: Any) -> None:
        if self._closing:
            logger.debug(f"Caller STT closed: {data}")
            return
        self.error = str(data.get("message") or data) if isinstance(data, dict) else str(data)
        self._session_ready.set()  # unblock start()
        logger.error(f"Caller STT error: {self.error}")


CallerSTT = ScribeStreamingSTT | OpenAIStreamingSTT


def create_caller_stt(
    provider: str,
    model: str,
    *,
    language: str = "en",
    domain: str | None = None,
) -> CallerSTT:
    """The caller's streaming STT for a ``CascadeSimulatorConfig.stt`` provider.

    ``domain`` picks the call description from ``user_simulator.stt_prompt_{domain}`` and
    ``stt_keywords_{domain}`` (comma-separated) in the simulation prompts; a domain without them gets none.
    """
    if provider == "openai":
        prompt, keywords = None, None
        if domain:
            prompts = PromptManager()
            with suppress(KeyError):
                prompt = prompts.get_template(f"user_simulator.stt_prompt_{domain}").strip()
            with suppress(KeyError):
                keywords = [k.strip() for k in prompts.get_template(f"user_simulator.stt_keywords_{domain}").split(",")]
        return OpenAIStreamingSTT(model, language=language, prompt=prompt, keywords=keywords)
    return ScribeStreamingSTT(model, language=language)
