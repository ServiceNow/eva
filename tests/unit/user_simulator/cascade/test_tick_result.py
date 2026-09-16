import pytest

from eva.user_simulator.cascade.tick_result import TickResult, split_tick_audio


@pytest.mark.parametrize("input_length", [0, 1, 7, 8, 9, 17])
def test_split_chunk_is_always_bytes_per_tick(input_length):
    chunk, overflow = split_tick_audio(b"\x01" * input_length, bytes_per_tick=8)

    assert len(chunk) == 8
    assert len(overflow) == max(0, input_length - 8)


def test_split_pads_short_audio_with_silence():
    chunk, overflow = split_tick_audio(b"\x01\x02", bytes_per_tick=8)

    assert chunk == b"\x01\x02" + b"\x00" * 6
    assert overflow == b""


def test_split_carries_overflow_to_next_tick():
    chunk, overflow = split_tick_audio(b"\x01" * 12, bytes_per_tick=8)

    assert chunk == b"\x01" * 8
    assert overflow == b"\x01" * 4


def test_split_of_empty_audio_is_all_silence():
    chunk, overflow = split_tick_audio(b"", bytes_per_tick=8)

    assert chunk == b"\x00" * 8
    assert overflow == b""


def test_has_assistant_speech_is_false_for_padded_silence():
    result = TickResult(tick_number=3, assistant_audio=b"\x00" * 8, assistant_audio_raw_bytes=0, wall_clock_ms=1)

    assert result.has_assistant_speech is False


def test_has_assistant_speech_is_true_when_real_audio_arrived():
    result = TickResult(tick_number=3, assistant_audio=b"\x01" * 8, assistant_audio_raw_bytes=8, wall_clock_ms=1)

    assert result.has_assistant_speech is True
