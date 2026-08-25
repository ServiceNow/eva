from eva.user_simulator.cascade.constants import (
    BYTES_PER_TICK,
    TICK_DURATION_MS,
    WAIT_TO_RESPOND_OTHER_MS,
    WAIT_TO_RESPOND_SELF_MS,
    ms_to_ticks,
)

THRESHOLD_MS_CONSTANTS = [
    WAIT_TO_RESPOND_OTHER_MS,
    WAIT_TO_RESPOND_SELF_MS,
]


def test_bytes_per_tick_is_one_tick_of_pcm16_at_16khz():
    # 16000 samples/s * 0.2s * 2 bytes/sample
    assert BYTES_PER_TICK == 6400


def test_threshold_constants_are_exact_multiples_of_tick_duration():
    for threshold_ms in THRESHOLD_MS_CONSTANTS:
        assert threshold_ms % TICK_DURATION_MS == 0


def test_ms_to_ticks_converts_and_floors_sub_tick_remainder():
    assert ms_to_ticks(WAIT_TO_RESPOND_OTHER_MS) == 5
    assert ms_to_ticks(150) == 0


def test_listener_check_interval_is_two_seconds_in_ticks():
    from eva.user_simulator.cascade.constants import LISTENER_CHECK_INTERVAL_MS, ms_to_ticks

    assert LISTENER_CHECK_INTERVAL_MS == 2000
    assert ms_to_ticks(LISTENER_CHECK_INTERVAL_MS) == 10


def test_the_vocabularies_are_no_longer_constants():
    # They moved to configs/caller_phrases.yaml because they are language data, not
    # timing. Timing constants staying here is the whole distinction.
    from eva.user_simulator.cascade import constants

    assert not hasattr(constants, "BACKCHANNEL_PHRASES")
    assert not hasattr(constants, "BARGE_IN_OPENERS")
