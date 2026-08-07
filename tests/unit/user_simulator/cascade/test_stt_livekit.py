import pytest

from eva.user_simulator.cascade.stt import TranscriptBuffer
from eva.user_simulator.cascade.stt_livekit import LiveKitStreamingSTT, build_livekit_stt


def test_unknown_provider_is_rejected_with_a_clear_message():
    with pytest.raises(ValueError, match="Unsupported caller STT provider"):
        build_livekit_stt("nope", {})


def test_elevenlabs_provider_defaults_to_the_realtime_scribe_model():
    #  Only scribe_v2_realtime streams interim transcripts and honours flush().
    stt = LiveKitStreamingSTT("elevenlabs", {"api_key": "k"})

    assert stt.model == "scribe_v2_realtime"


def test_explicit_model_overrides_the_default():
    stt = LiveKitStreamingSTT("elevenlabs", {"api_key": "k", "model": "scribe_v2"})

    assert stt.model == "scribe_v2"


def test_buffer_starts_empty_and_is_a_transcript_buffer():
    stt = LiveKitStreamingSTT("elevenlabs", {"api_key": "k"})

    assert isinstance(stt.buffer, TranscriptBuffer)
    assert stt.buffer.committed == ""
    assert stt.buffer.in_flight == ""


async def test_feed_before_start_is_a_noop_rather_than_an_error():
    stt = LiveKitStreamingSTT("elevenlabs", {"api_key": "k"})

    await stt.feed(b"\x00" * 320)


async def test_stop_is_safe_before_start_and_twice():
    stt = LiveKitStreamingSTT("elevenlabs", {"api_key": "k"})

    await stt.stop()
    await stt.stop()
