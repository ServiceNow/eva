"""Pre-rendered audio for the caller's fixed phrase vocabularies."""

from __future__ import annotations

import asyncio
import random
from typing import ClassVar, Protocol

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class SpeechSynthesizer(Protocol):
    """Minimal interface the cache needs from a TTS client."""

    async def synthesize(self, text: str, *, voice_id: str) -> bytes:
        """Render text to PCM16 audio."""
        ...


class PhraseCache:
    """Renders a fixed vocabulary once so it can be voiced at zero latency.

    Backchannels and barge-in openers are short, fixed, and voice-stable, which
    makes them cacheable — and caching is the only way a 300ms "mm-hmm" reliably
    lands on the tick the decision chose.

    The audio is held **per process, keyed by (voice_id, phrase)**, not per
    simulator. The vocabulary depends only on the voice, so a run of N records
    against two voices needs two renders of each phrase rather than 2N: every
    conversation after the first finds the audio already there and starts without
    waiting on TTS at all. Only the RNG is per-instance, so each conversation
    still chooses its phrases independently and reproducibly.
    """

    _audio: ClassVar[dict[tuple[str, str], bytes]] = {}
    _render_lock: ClassVar[asyncio.Lock | None] = None

    def __init__(self, tts: SpeechSynthesizer, *, voice_id: str, seed: int = 0) -> None:
        self._tts = tts
        self._voice_id = voice_id
        self._rng = random.Random(seed)

    @classmethod
    def _lock(cls) -> asyncio.Lock:
        """Lazily create the shared render lock, on whichever loop is running."""
        if cls._render_lock is None:
            cls._render_lock = asyncio.Lock()
        return cls._render_lock

    @classmethod
    def clear(cls) -> None:
        """Drop all cached audio. For tests, and for a voice set changing mid-process."""
        cls._audio.clear()

    async def prerender(self, phrases: list[str]) -> None:
        """Synthesize any phrase this voice has not rendered yet.

        Serialized across conversations: concurrent records share one voice, so
        without the lock each would discover the same empty cache and render the
        same phrases. The lock is held only for genuine misses.
        """
        async with self._lock():
            missing = [p for p in phrases if (self._voice_id, p) not in self._audio]
            if not missing:
                logger.debug(f"All {len(phrases)} caller phrases already rendered for {self._voice_id}")
                return
            rendered = await asyncio.gather(*(self._tts.synthesize(p, voice_id=self._voice_id) for p in missing))
            self._audio.update(
                {(self._voice_id, phrase): audio for phrase, audio in zip(missing, rendered, strict=True)}
            )
            logger.info(f"Pre-rendered {len(missing)} caller phrases for voice {self._voice_id}")

    def get(self, phrase: str) -> bytes:
        """Return cached audio for a phrase in this cache's voice."""
        key = (self._voice_id, phrase)
        if key not in self._audio:
            raise KeyError(f"Phrase not pre-rendered for voice {self._voice_id}: {phrase!r}")
        return self._audio[key]

    def choose(self, phrases: list[str]) -> str:
        """Pick a phrase using the cache's seeded RNG, so runs stay reproducible."""
        return self._rng.choice(phrases)
