"""Pre-rendered audio for the caller's fixed phrase vocabularies."""

from __future__ import annotations

import asyncio
import random
from typing import Protocol

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class SpeechSynthesizer(Protocol):
    """Minimal interface the cache needs from a TTS client."""

    async def synthesize(self, text: str, *, voice_id: str) -> bytes:
        """Render text to PCM16 audio."""
        ...


class PhraseCache:
    """Renders a fixed vocabulary once at init so it can be voiced at zero latency.

    Backchannels and barge-in openers are short, fixed, and voice-stable, which
    makes them cacheable — and caching is the only way a 300ms "mm-hmm" reliably
    lands on the tick the decision chose.
    """

    def __init__(self, tts: SpeechSynthesizer, *, voice_id: str, seed: int = 0) -> None:
        self._tts = tts
        self._voice_id = voice_id
        self._rng = random.Random(seed)
        self._audio: dict[str, bytes] = {}

    async def prerender(self, phrases: list[str]) -> None:
        """Synthesize every phrase concurrently and hold the audio in memory."""
        rendered = await asyncio.gather(*(self._tts.synthesize(p, voice_id=self._voice_id) for p in phrases))
        self._audio.update(dict(zip(phrases, rendered, strict=True)))
        logger.info(f"Pre-rendered {len(phrases)} caller phrases")

    def get(self, phrase: str) -> bytes:
        """Return cached audio for a phrase."""
        if phrase not in self._audio:
            raise KeyError(f"Phrase not pre-rendered: {phrase!r}")
        return self._audio[phrase]

    def choose(self, phrases: list[str]) -> str:
        """Pick a phrase using the cache's seeded RNG, so runs stay reproducible."""
        return self._rng.choice(phrases)
