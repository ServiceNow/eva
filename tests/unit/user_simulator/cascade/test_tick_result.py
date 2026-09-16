import pytest

from eva.user_simulator.cascade.tick_result import TickResult, played_audio_ms, split_tick_audio


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


def test_played_position_reflects_released_ticks_not_received_bytes():
    # 12 ticks released at 200ms each, regardless of how much arrived early.
    assert played_audio_ms(ticks_released=12) == 2400


def test_played_position_is_zero_before_anything_is_released():
    assert played_audio_ms(ticks_released=0) == 0


def test_truncation_defaults_are_inert():
    result = TickResult(tick_number=1, assistant_audio=b"\x00" * 8, assistant_audio_raw_bytes=0, wall_clock_ms=0)

    assert result.skip_item_id is None
    assert result.interruption_audio_start_ms is None


def test_truncation_fields_carry_the_played_position():
    result = TickResult(
        tick_number=1,
        assistant_audio=b"\x00" * 8,
        assistant_audio_raw_bytes=0,
        wall_clock_ms=0,
        skip_item_id="item_42",
        interruption_audio_start_ms=2400,
    )

    assert result.skip_item_id == "item_42"
    assert result.interruption_audio_start_ms == 2400
