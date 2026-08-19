"""Timing constants for the tick-based cascade caller.

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

TRANSCRIPT_WAIT_MS = 1500
"""How long the caller waits for the assistant's transcript to finalize before falling back
to the in-flight partial, sized above the slowest measured finalization (ink-2, 1.2s)."""

INACTIVITY_TIMEOUT_MS = 120000
"""Assistant silence that ends the conversation, matching ElevenLabsUserSimulator's 12
keep-alives so both providers record the same inactivity_timeout terminal state."""

CALLER_SAMPLE_RATE = 16000
"""PCM16 sample rate for the caller's own audio track."""

_BYTES_PER_SAMPLE = 2

BYTES_PER_TICK = CALLER_SAMPLE_RATE * TICK_DURATION_MS // 1000 * _BYTES_PER_SAMPLE
"""PCM16 bytes carried per tick at CALLER_SAMPLE_RATE."""

SILENCE_BYTE = b"\x00"
"""PCM16 silence, used to pad partial ticks."""

LISTENER_CHECK_INTERVAL_MS = 2000
"""How often the interrupt and backchannel checks run while the assistant speaks."""

MAX_INTERRUPT_SLIP_MS = 1500
"""Drop a reactive barge-in whose audio arrived this far past its intended tick."""

SELF_CORRECTION_DELAY_MS = 1200
"""How long after the assistant starts replying to play a pre-authored correction."""

SELF_CORRECTION_RATE = 0.15
"""Fraction of caller turns generated with a self-correction attached."""

INTERRUPT_RATE = 0.15
"""Fraction of assistant turns eligible for one barge-in.

Rolled once when the assistant starts speaking, not once per check: the interrupt prompt's
YES criteria ("has the user heard enough to have a response ready?") are satisfied at the end
of every assistant turn, so ungated every caller utterance became an interruption and normal
turn-taking was bypassed entirely.
"""

BACKCHANNEL_PHRASES = ["uh-huh", "mm-hmm"]
"""Fixed continuer vocabulary (tau: voice_config.py:126). Pre-rendered at init."""

BARGE_IN_OPENERS = ["Wait—", "Sorry—", "Hold on—", "Actually—"]
"""Fixed barge-in openers. Pre-rendered so a decision can be voiced at zero latency."""


def ms_to_ticks(milliseconds: int) -> int:
    """Convert milliseconds to whole ticks, flooring."""
    return milliseconds // TICK_DURATION_MS
