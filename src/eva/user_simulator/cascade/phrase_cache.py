"""Short caller phrases pre-rendered at startup, so playing one costs no TTS latency."""

from __future__ import annotations

import asyncio
import random

from eva.user_simulator.cascade.barge_in import trim_silence
from eva.user_simulator.cascade.tts import CartesiaStreamingTTS


class PhraseCache:
    """Phrases rendered once as PCM; ``pick`` returns a random one to play."""

    def __init__(self, phrases: list[str]) -> None:
        self.phrases = phrases
        self.entries: list[tuple[str, bytes]] = []

    def __bool__(self) -> bool:
        return bool(self.entries)

    async def render(self, tts: CartesiaStreamingTTS, voice_id: str, *, speed: float | None = None) -> None:
        async def render_one(phrase: str) -> tuple[str, bytes]:
            # Trimmed, so every millisecond of playout is speech and the gap is not padded.
            pcm = b"".join([chunk async for chunk in tts.stream(phrase, voice_id=voice_id, speed=speed)])
            return phrase, trim_silence(pcm)

        self.entries = list(await asyncio.gather(*(render_one(phrase) for phrase in self.phrases)))

    def pick(self) -> tuple[str, bytes]:
        return random.choice(self.entries)
