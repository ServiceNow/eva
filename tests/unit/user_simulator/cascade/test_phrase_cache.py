import pytest

from eva.user_simulator.cascade.phrase_cache import PhraseCache


class FakeTTS:
    """Counts synthesis calls and returns deterministic audio per phrase."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(self, text: str, *, voice_id: str) -> bytes:
        self.calls.append(text)
        return text.encode()


async def test_prerender_synthesizes_every_phrase_once():
    tts = FakeTTS()
    cache = PhraseCache(tts, voice_id="voice-f")

    await cache.prerender(["uh-huh", "mm-hmm"])

    assert sorted(tts.calls) == ["mm-hmm", "uh-huh"]


async def test_cached_audio_is_returned_without_further_synthesis():
    tts = FakeTTS()
    cache = PhraseCache(tts, voice_id="voice-f")
    await cache.prerender(["uh-huh"])

    audio = cache.get("uh-huh")

    assert audio == b"uh-huh"
    assert len(tts.calls) == 1


async def test_choose_returns_a_phrase_from_the_cache_deterministically():
    tts = FakeTTS()
    cache = PhraseCache(tts, voice_id="voice-f", seed=7)
    await cache.prerender(["uh-huh", "mm-hmm"])

    first = cache.choose(["uh-huh", "mm-hmm"])
    replay = PhraseCache(FakeTTS(), voice_id="voice-f", seed=7)
    await replay.prerender(["uh-huh", "mm-hmm"])

    assert first == replay.choose(["uh-huh", "mm-hmm"])


async def test_requesting_an_unrendered_phrase_raises():
    cache = PhraseCache(FakeTTS(), voice_id="voice-f")

    with pytest.raises(KeyError, match="not pre-rendered"):
        cache.get("never-rendered")
