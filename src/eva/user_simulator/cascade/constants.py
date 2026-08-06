"""Timing constants for the tick-based cascade user simulator.

Values mirror tau-voice (tau2-bench/src/tau2/config.py:104-113). These are
module constants rather than config fields on purpose: there is no run-level
reason to vary them, and exposing them would let runs drift apart in ways that
make their metrics incomparable.
"""

TICK_DURATION_MS = 200
"""One simulation tick: 200ms of PCM16 audio at CALLER_SAMPLE_RATE, converted by
the adapter into ten 20ms mulaw frames when written to the wire."""

WAIT_TO_RESPOND_OTHER_MS = 1000
"""Silence required from the assistant before the caller starts a turn."""

WAIT_TO_RESPOND_SELF_MS = 5000
"""Silence required from the caller itself before it starts another turn."""

YIELD_WHEN_INTERRUPTED_MS = 1000
"""How long the caller keeps talking after the assistant barges in."""

YIELD_WHEN_INTERRUPTING_MS = 5000
"""How long the caller holds the floor after barging in itself."""

CALLER_SAMPLE_RATE = 16000
"""PCM16 sample rate for the caller's own audio track."""

_BYTES_PER_SAMPLE = 2

BYTES_PER_TICK = CALLER_SAMPLE_RATE * TICK_DURATION_MS // 1000 * _BYTES_PER_SAMPLE
"""PCM16 bytes carried per tick at CALLER_SAMPLE_RATE."""

SILENCE_BYTE = b"\x00"
"""PCM16 silence, used to pad partial ticks."""


def ms_to_ticks(milliseconds: int) -> int:
    """Convert milliseconds to whole ticks, flooring."""
    return milliseconds // TICK_DURATION_MS
