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


async def test_a_second_conversation_on_the_same_voice_renders_nothing():
    # This is the point of the global cache: a run of N records against one voice
    # renders each phrase once, not N times.
    tts = FakeTTS()
    first = PhraseCache(tts, voice_id="voice-f")
    await first.prerender(["uh-huh", "mm-hmm"])

    second = PhraseCache(tts, voice_id="voice-f")
    await second.prerender(["uh-huh", "mm-hmm"])

    assert len(tts.calls) == 2
    assert second.get("uh-huh") == b"uh-huh"


async def test_only_the_phrases_a_voice_is_missing_are_rendered():
    tts = FakeTTS()
    await PhraseCache(tts, voice_id="voice-f").prerender(["uh-huh"])
    tts.calls.clear()

    await PhraseCache(tts, voice_id="voice-f").prerender(["uh-huh", "mm-hmm"])

    assert tts.calls == ["mm-hmm"]


async def test_each_voice_keeps_its_own_audio():
    # Two genders means two voices; one must never be served the other's audio.
    tts = FakeTTS()
    female = PhraseCache(tts, voice_id="voice-f")
    male = PhraseCache(tts, voice_id="voice-m")
    await female.prerender(["uh-huh"])
    await male.prerender(["uh-huh"])

    assert len(tts.calls) == 2
    assert female.get("uh-huh") == b"uh-huh"
    assert male.get("uh-huh") == b"uh-huh"


async def test_concurrent_conversations_do_not_double_render():
    import asyncio

    tts = FakeTTS()
    caches = [PhraseCache(tts, voice_id="voice-f") for _ in range(4)]

    await asyncio.gather(*(c.prerender(["uh-huh", "mm-hmm"]) for c in caches))

    assert sorted(tts.calls) == ["mm-hmm", "uh-huh"]
