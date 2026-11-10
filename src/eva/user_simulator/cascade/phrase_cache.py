"""Short caller sounds pre-rendered at startup, so playing one costs no TTS latency."""

from __future__ import annotations

import asyncio
import json
import random
import wave
from pathlib import Path

try:
    import audioop
except ImportError:  # Python 3.13+
    import audioop_lts as audioop

from eva.user_simulator.cascade.barge_in import trim_silence
from eva.user_simulator.cascade.tts import TTS_SAMPLE_RATE, CartesiaStreamingTTS

SOUND_BANK_DIR = Path(__file__).resolve().parents[4] / "data" / "sound_bank"
"""``<sound_name>/phrases.json`` (phrases by language code, rendered with the caller's own voice) and/or
``<sound_name>/male/*.wav`` and ``<sound_name>/female/*.wav`` (recordings, played as they are)."""

PHRASES_FILE = "phrases.json"
DEFAULT_LANGUAGE = "en"
"""Whose phrases a sound uses for a language its phrases.json does not have."""
OPENER_SOUND = "barge_in_opener"
"""The sound-bank entry holding barge-in openers. It plays only as the start of a barge-in and is never
rolled for, so it is not a name for the ``back_channel`` list."""


def load_wav_pcm(path: Path) -> bytes:
    """A WAV file as 16kHz mono PCM16, the format the bridge plays."""
    with wave.open(str(path), "rb") as wav:
        channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    if width != 2:
        pcm = audioop.lin2lin(pcm, width, 2)
    if channels > 1:
        pcm = audioop.tomono(pcm, 2, 1 / channels, 1 / channels) if channels == 2 else pcm[: len(pcm) // channels]
    if rate != TTS_SAMPLE_RATE:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, TTS_SAMPLE_RATE, None)
    return trim_silence(pcm)


class PhraseCache:
    """Sounds rendered once as PCM; ``pick`` returns a random one to play.

    Entries come from text phrases (synthesized in the caller's voice) and from recorded WAV clips.
    """

    def __init__(self, phrases: list[str], clips: list[Path] | None = None) -> None:
        self.phrases = phrases
        self.clips = clips or []
        self.entries: list[tuple[str, bytes]] = []

    @classmethod
    def from_bank(cls, sound: str, language: str, gender: str, bank_dir: Path = SOUND_BANK_DIR) -> PhraseCache:
        """The cache for ``sound``: its phrases for ``language`` plus the WAVs for ``gender`` ("M"/"F")."""
        folder = bank_dir / sound
        phrases_file = folder / PHRASES_FILE
        table = json.loads(phrases_file.read_text(encoding="utf-8")) if phrases_file.is_file() else {}
        # pt-BR tries pt-BR, then pt, then the default language.
        phrases = table.get(language) or table.get(language.split("-")[0].lower()) or table.get(DEFAULT_LANGUAGE, [])
        clips = sorted((folder / ("male" if gender == "M" else "female")).glob("*.wav"))
        if not phrases and not clips:
            raise ValueError(f"Sound {sound!r} has no {PHRASES_FILE} phrases or WAV clips for this caller in {folder}")
        return cls(phrases, clips)

    def __bool__(self) -> bool:
        return bool(self.entries)

    async def render(self, tts: CartesiaStreamingTTS, voice_id: str, *, speed: float | None = None) -> None:
        async def render_one(phrase: str) -> tuple[str, bytes]:
            # Trimmed, so every millisecond of playout is speech and the gap is not padded.
            pcm = b"".join([chunk async for chunk in tts.stream(phrase, voice_id=voice_id, speed=speed)])
            return phrase, trim_silence(pcm)

        rendered = await asyncio.gather(*(render_one(phrase) for phrase in self.phrases))
        self.entries = [*rendered, *((clip.stem, load_wav_pcm(clip)) for clip in self.clips)]

    def pick(self) -> tuple[str, bytes]:
        return random.choice(self.entries)
