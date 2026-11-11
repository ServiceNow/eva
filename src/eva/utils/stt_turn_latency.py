"""STT latency for pipelines that report no STT TTFB (Cartesia ink-2).

ink-2 runs have no VAD (``Using VAD analyzer: none``), so pipecat never starts its STT TTFB timer
and reports only a startup ``0.0``. Latency is reconstructed after the run by pairing the
committed-turn times the assistant server recorded with the real end of the user's speech from
``user_simulator_events.jsonl``.
"""

import json
from pathlib import Path

# Entry type the assistant server writes to pipecat_metrics.jsonl for each committed ink-2 turn.
INK2_TURN_END_METRIC = "Ink2TurnEnd"

# A committing turn may be stamped marginally before the simulator's audio_end.
_END_TOLERANCE_S = 0.1
# Longest wait after audio_end that is attributed to an utterance (matches the worker's 30s filter).
_MAX_WINDOW_S = 30.0


def _read_user_utterances(events_path: Path) -> tuple[list[tuple[float, float]], list[float]]:
    """Return ((start, end) per simulated-user utterance, assistant audio_start times) in unix seconds.

    Events are read in file order. An audio_start with no audio_end before the next audio_start is
    dropped, as is an audio_end with no open audio_start, so one missing event does not shift the
    pairing of every later utterance.
    """
    utterances: list[tuple[float, float]] = []
    assistant_starts: list[float] = []
    open_start: float | None = None
    with open(events_path) as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = event.get("audio_timestamp")
            if not isinstance(ts, (int, float)):
                continue
            kind, who = event.get("event_type"), event.get("user")
            if who == "simulated_user" and kind == "audio_start":
                open_start = ts
            elif who == "simulated_user" and kind == "audio_end":
                if open_start is not None:
                    utterances.append((open_start, ts))
                open_start = None
            elif who == "assistant" and kind == "audio_start":
                assistant_starts.append(ts)
    return utterances, assistant_starts


def compute_turn_commit_latencies(turn_end_times_s: list[float], events_path: Path) -> list[float]:
    """Seconds from the real end of each user utterance to the turn that committed it.

    The committing turn is the last turn end between the utterance start and the earliest of the
    first assistant audio after it ends, the next utterance's start, and a 30s cap (ink-2 can split
    one utterance into several turns). An utterance with no turn end at or after its real end is
    skipped. Includes any wait the simulator imposes after the user stops, since the assistant
    cannot respond before the turn commits.

    Args:
        turn_end_times_s: Wall-clock unix seconds of each committed ink-2 turn.
        events_path: user_simulator_events.jsonl for the same conversation.
    """
    utterances, assistant_starts = _read_user_utterances(events_path)
    latencies = []
    for i, (start, end) in enumerate(utterances):
        # The window ends at the first of: the assistant's reply, the next utterance, or the max wait.
        # Bounding by the next utterance keeps one turn end from being counted for two utterances.
        bounds = [end + _MAX_WINDOW_S]
        bounds.extend(a for a in assistant_starts if a > end)
        if i + 1 < len(utterances):
            bounds.append(utterances[i + 1][0])
        window_end = min(bounds)
        in_window = [t for t in turn_end_times_s if start <= t < window_end]
        if not in_window or in_window[-1] < end - _END_TOLERANCE_S:
            continue
        latencies.append(in_window[-1] - end)
    return latencies
