from eva.user_simulator.cascade.constants import (
    BYTES_PER_TICK,
    TICK_DURATION_MS,
    WAIT_TO_RESPOND_OTHER_MS,
    WAIT_TO_RESPOND_SELF_MS,
    ms_to_ticks,
)


def test_tick_duration_matches_tau_voice():
    assert TICK_DURATION_MS == 200


def test_bytes_per_tick_is_one_tick_of_pcm16_at_16khz():
    # 16000 samples/s * 0.2s * 2 bytes/sample
    assert BYTES_PER_TICK == 6400


def test_ms_to_ticks_floors_to_whole_ticks():
    assert ms_to_ticks(WAIT_TO_RESPOND_OTHER_MS) == 5
    assert ms_to_ticks(WAIT_TO_RESPOND_SELF_MS) == 25
    assert ms_to_ticks(150) == 0
