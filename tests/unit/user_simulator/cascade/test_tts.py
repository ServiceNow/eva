import pytest

from eva.user_simulator.cascade.tts import CartesiaTTS


def test_voice_id_selected_for_female_persona():
    tts = CartesiaTTS({"model": "sonic-3.5", "female_voice": "voice-f", "male_voice": "voice-m"})

    assert tts.voice_for_persona({"user_persona_id": 1}) == "voice-f"


def test_voice_id_selected_for_male_persona():
    tts = CartesiaTTS({"model": "sonic-3.5", "female_voice": "voice-f", "male_voice": "voice-m"})

    assert tts.voice_for_persona({"user_persona_id": 2}) == "voice-m"


def test_unknown_persona_falls_back_to_female_voice():
    tts = CartesiaTTS({"model": "sonic-3.5", "female_voice": "voice-f", "male_voice": "voice-m"})

    assert tts.voice_for_persona({}) == "voice-f"


async def test_empty_text_synthesizes_to_no_audio():
    tts = CartesiaTTS({"model": "sonic-3.5", "api_key": "k"})

    assert await tts.synthesize("", voice_id="voice-f") == b""


async def test_missing_api_key_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    tts = CartesiaTTS({"model": "sonic-3.5"})

    with pytest.raises(ValueError, match="Cartesia API key"):
        await tts.synthesize("hello", voice_id="voice-f")
