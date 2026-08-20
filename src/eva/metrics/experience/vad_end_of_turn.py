"""VAD-analyzer turn diagnostics for the turn_taking metric.

Surfaces Krisp's "no silence-duration fallback" hang and Smart Turn's silence-fallback
(stop_secs) usage directly in turn_taking.sub_metrics, for any pipecat run with local
turn-analyzer VAD (Krisp or Smart Turn). Null for runs with no local VAD (other
frameworks, or external turn strategies like self-endpointing STT/FLUX).

No pipecat modifications: all signals come from files eva already writes
(pipecat_logs.jsonl, pipecat_metrics.jsonl, audit_log.json) or pipecat's own log
output (logs.log).
"""

import json
import re
import statistics
from pathlib import Path
from typing import Any

from eva.metrics.base import MetricContext
from eva.metrics.processor import is_agent_timeout_on_user_turn
from eva.models.results import MetricScore
from eva.utils.conversation_correctly_finished.final_turn import final_turn_input_flags
from eva.utils.json_utils import load_jsonl
from eva.utils.log_processing import load_audit_log

_STOP_SECS_RE = re.compile(r"End of Turn complete due to stop_secs\. Silence in ms: ([\d.]+)")
_FINAL_TURN_FLAG_KEYS = ("short", "acknowledgement", "spelled_entity")

# Widest gap tolerated between a vad_stop and the dispatched user turn it closed. Two orders of
# magnitude above the observed +3..+8ms dispatch lag, and well under half the closest observed
# spacing between consecutive vad_stops (2244ms) - see ``_transcript_for_vad_stop``.
_DISPATCH_VS_VAD_STOP_TOLERANCE_MS = 1000


def _find_run_config(output_dir: str) -> dict[str, Any] | None:
    """Walk up from output_dir to the run root and load config.json, or None if absent/unreadable."""
    if not output_dir:
        return None
    path = Path(output_dir)
    while True:
        candidate = path / "config.json"
        if candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                return None
        if path == path.parent:
            return None
        path = path.parent


def vad_turn_metrics_applicable(output_dir: str) -> bool:
    """True when this record's run used local turn-analyzer VAD (Krisp or Smart Turn).

    False for non-pipecat frameworks (openai_realtime, elevenlabs, grok_voice,
    smallest_hydra) and for pipecat runs with turn_stop_strategy == "external"
    (self-endpointing STT / external turn detection, e.g. Deepgram FLUX). Also False
    when config.json can't be found at all — a safe default rather than assuming a
    pipeline shape we can't confirm.
    """
    config = _find_run_config(output_dir)
    if config is None:
        return False
    if config.get("framework", "pipecat") != "pipecat":
        return False
    model_cfg = config.get("model") or {}
    return model_cfg.get("turn_stop_strategy", "turn_analyzer") != "external"


def _load_turn_metrics_events(output_dir: str) -> list[dict[str, Any]]:
    """Return TurnMetricsData entries from pipecat_metrics.jsonl, sorted by timestamp.

    Covers both KrispVivaTurn (always is_complete=True when present) and BaseSmartTurn
    (is_complete True or False, one entry per VAD-stop evaluation) — same shape for both.
    """
    entries = load_jsonl(Path(output_dir) / "pipecat_metrics.jsonl")
    events = [
        {
            "timestamp": e["timestamp"],
            "is_complete": e["value"]["is_complete"],
            "e2e_processing_time_ms": e["value"].get("e2e_processing_time_ms"),
        }
        for e in entries
        if e.get("type") == "TurnMetricsData"
    ]
    return sorted(events, key=lambda e: e["timestamp"])


def _load_vad_events(output_dir: str) -> tuple[list[int], list[int]]:
    """Return (sorted user_started_speaking timestamps, sorted user_stopped_speaking timestamps).

    Both epoch-ms, from the same WallClock instance that produces pipecat_metrics.jsonl's
    timestamps (confirmed shared at pipecat_server.py:483-488) — directly comparable.
    """
    entries = load_jsonl(Path(output_dir) / "pipecat_logs.jsonl")
    starts = sorted(e["timestamp"] for e in entries if e.get("type") == "user_started_speaking")
    stops = sorted(e["timestamp"] for e in entries if e.get("type") == "user_stopped_speaking")
    return starts, stops


def _load_dispatched_user_turns(output_dir: str) -> list[tuple[int, str]]:
    """Return (timestamp, text) for each LLM-dispatched user turn in audit_log.json, sorted.

    Deliberately *not* pipecat_logs.jsonl's "transcript" events. Those are individual STT
    fragments, stamped when the fragment finalized, and neither property survives contact with
    real data: a single user turn routinely spans several fragments (confirmed in the fixtures -
    one turn arrived as "My employee ID is e n" + "p zero six six six six six." + "The last four
    of my phone are four four two five."), and a fragment can finalize *before* its own turn's
    user_started_speaking fires, so slicing fragments by VAD-start boundaries attributes them to
    the previous turn.

    audit_log.json's transcript entries instead carry the aggregated text actually handed to the
    LLM, one entry per real turn, stamped in the same epoch-ms clock as the VAD events.
    """
    audit = load_audit_log(Path(output_dir) / "audit_log.json")
    if not audit:
        return []
    turns = []
    for entry in audit.get("transcript", []):
        if entry.get("message_type") != "user" or not entry.get("value"):
            continue
        try:
            timestamp = int(entry["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        turns.append((timestamp, entry["value"]))
    return sorted(turns, key=lambda t: t[0])


def _transcript_for_vad_stop(dispatched_user_turns: list[tuple[int, str]], vad_stop: int) -> str | None:
    """Return the text of the user turn that ``vad_stop`` closed, or None if none matches.

    Each dispatched turn lands a handful of milliseconds after the user_stopped_speaking that
    ended it (measured at +3..+8ms across all 30 turns in the four fixtures, pairing 1:1 with
    the vad_stops), while consecutive vad_stops are seconds apart (2244ms at the closest observed).
    ``_DISPATCH_VS_VAD_STOP_TOLERANCE_MS`` sits between those two scales, so the nearest turn
    within tolerance is unambiguous - and a window whose utterance was never dispatched at all
    matches nothing rather than borrowing a neighbour's text.
    """
    candidates = [
        (abs(ts - vad_stop), text)
        for ts, text in dispatched_user_turns
        if abs(ts - vad_stop) <= _DISPATCH_VS_VAD_STOP_TOLERANCE_MS
    ]
    return min(candidates, key=lambda c: c[0])[1] if candidates else None


def _load_turn_start_timestamps(output_dir: str) -> set[int]:
    """Return the set of timestamps where pipecat's own TurnTrackingObserver started a turn.

    Sourced from pipecat_logs.jsonl's "turn_start" entries (observers.py:_start_turn), on the
    same shared epoch-ms clock as vad_starts/vad_stops. This is pipecat's own judgment of
    whether a given user_started_speaking frame began a genuinely new turn - independent of
    (and a check on) the turn-analyzer's is_complete/stop_secs signals used elsewhere in this
    module. Empty when the run's pipeline doesn't emit turn_start at all, which callers must
    treat as "signal unavailable" rather than "no turn ever started".
    """
    entries = load_jsonl(Path(output_dir) / "pipecat_logs.jsonl")
    return {e["timestamp"] for e in entries if e.get("type") == "turn_start"}


def _load_stop_secs_silence_ms(output_dir: str) -> list[float]:
    """Return the 'Silence in ms' value from each pipecat stop_secs forced-completion line, in file order.

    logs.log's local-time text timestamps are not safe to convert and compare against
    pipecat_logs.jsonl/pipecat_metrics.jsonl's shared epoch (confirmed ~1-3s of slop,
    machine/timezone-dependent) — only the count and self-contained silence value are used.
    """
    log_path = Path(output_dir) / "logs.log"
    if not log_path.exists():
        return []
    return [float(m) for m in _STOP_SECS_RE.findall(log_path.read_text(errors="ignore"))]


def _build_turn_windows(vad_starts: list[int]) -> list[tuple[int, int | None]]:
    """Pair consecutive VAD-start timestamps into (window_start, window_end); last window is open-ended."""
    return [(start, vad_starts[i + 1] if i + 1 < len(vad_starts) else None) for i, start in enumerate(vad_starts)]


def _classify_turns(
    vad_starts: list[int],
    vad_stops: list[int],
    turn_metrics_events: list[dict[str, Any]],
    stop_secs_silence_ms: list[float],
    dispatched_user_turns: list[tuple[int, str]] | None = None,
    turn_start_timestamps: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Classify each VAD-analyzer turn window as natural/forced/premature/stuck.

    natural: one or more TurnMetricsData(is_complete=True) entries fall inside the window.
        A raw VAD-start window can legitimately contain more than one - confirmed against real
        Krisp data where a single window held two genuine completions for two separate real
        user turns (Krisp's own "user_started_speaking" fired late/not-at-all for the second
        one, so both landed in the first turn's window). Each is emitted as its own result
        entry rather than keeping only the first and silently dropping the rest: the window is
        split at each natural completion in turn order, with each sub-turn's open_duration_ms
        measured from the previous split point (window_start for the first one).
    forced: no natural completion, AND this window has at least one user_stopped_speaking
        event in its range (Smart Turn's stop_secs mechanism only ever fires after VAD has
        registered a stop, so a window with no vad_stop cannot legitimately be the source of
        a forced completion). Such a window consumes the next unused stop_secs silence value
        in file order (Smart Turn only ever has one open turn waiting on stop_secs at a time,
        so ordinal matching is safe among windows that actually have a vad_stop — see spec's
        "Matching stop_secs lines to turn windows"). Close time = last vad_stop in the
        window; the consumed silence value is only used to confirm the fallback fired, not
        added to close_time — last vad_stop is emitted only after the turn analyzer has
        already waited out that full silence duration internally (confirmed against real
        data: it trails the LLM-dispatched turn by only 4-6ms), so adding it again would
        double-count that wait. Also tagged with ``final_turn_flags`` (short /
        acknowledgement / spelled_entity, via ``final_turn_input_flags``) computed on the
        dispatched user turn that this window's closing vad_stop ended (see
        ``_transcript_for_vad_stop``) — a candidate explanation for *why* the turn analyzer
        needed the stop_secs fallback instead of closing naturally. Both the text and the flags
        are omitted when no dispatched turn matches, so a flag is never computed on some other
        turn's words.
    premature: a natural or forced completion (see above) whose window is immediately
        followed by another VAD-start window that (a) itself resolves to a natural or forced
        completion - not "stuck" - and (b) opened with no "turn_start" logged by pipecat's own
        TurnTrackingObserver (see ``_load_turn_start_timestamps``). Both conditions matter:
        (b) alone is not enough, since a missing turn_start also shows up ahead of an
        ordinary Krisp VAD false-start/blip window that resolves "stuck" - confirmed against
        real Krisp data, where that shape is benign noise already covered by stuck_rate, not
        a wrongly-cut-short utterance. Condition (a) is what tells the two apart: only when
        the *next* window turns out to hold a genuine completion of its own does the pair
        read as one real utterance that got artificially split into two. Confirmed against
        real data (example/nonkrisp_12_0): the turn analyzer reported is_complete=True on
        "Sure. My employee ID is emp zero four eight two seven one." with 0.98 confidence,
        but the user was still mid-utterance - the rest ("Last four of my phone number are
        seven two nine four.") arrived in the very next VAD-start window, which produced no
        turn_start of its own and went on to complete naturally there. Only reclassifies the
        natural/forced result immediately bordering that missing turn_start (for a
        multi-natural window, that's the last split segment) - earlier segments in the same
        window, or the run's final window with no following window at all, are untouched.
        Only checked when ``turn_start_timestamps`` is non-empty; an empty set means this
        run's pipeline never emits turn_start at all, which must read as "signal
        unavailable", not "every transition is premature".
    stuck: neither signal found for this window (including: no natural completion and no
        vad_stop in range, in which case no stop_secs value is consumed — it's left for a
        later window that actually earned it). This is common, not a rare edge
        case (confirmed against real Krisp data: VAD false-starts and silently-swallowed
        utterances both produce it) - a middle window's open_duration_ms uses its own
        window_end (the next VAD start bounds it), never the conversation's global last
        timestamp. Only the true last, open-ended window (window_end is None) falls back
        to last_epoch as a conversation-end proxy.

    ``turn_index`` on each result is a running count over the *output* list, not the raw
    window index - windows that split into multiple natural completions push every later
    entry's index forward accordingly.
    """
    windows = _build_turn_windows(vad_starts)
    last_epoch = max([*vad_starts, *vad_stops, *(e["timestamp"] for e in turn_metrics_events)], default=0)
    stop_secs_iter = iter(stop_secs_silence_ms)
    dispatched_user_turns = dispatched_user_turns or []
    # One entry list per raw VAD-start window (before turn_index is assigned), so the
    # premature pass below can look at "the next window's own entries" without having to
    # re-derive window boundaries from the flattened, already-turn_indexed output.
    per_window: list[list[dict[str, Any]]] = []

    for window_start, window_end in windows:
        in_window = [
            e
            for e in turn_metrics_events
            if e["timestamp"] >= window_start and (window_end is None or e["timestamp"] < window_end)
        ]
        naturals = [e for e in in_window if e["is_complete"]]
        if naturals:
            entries = []
            segment_start = window_start
            for natural in naturals:
                entries.append(
                    {
                        "open_duration_ms": natural["timestamp"] - segment_start,
                        "time_to_complete_ms": natural["e2e_processing_time_ms"],
                        "completion": "natural",
                    }
                )
                segment_start = natural["timestamp"]
            per_window.append(entries)
            continue

        stops_in_window = [s for s in vad_stops if s >= window_start and (window_end is None or s < window_end)]
        if stops_in_window:
            silence_ms = next(stop_secs_iter, None)
            if silence_ms is not None:
                # silence_ms is only consumed here to confirm this window legitimately earned
                # the stop_secs fallback (ordinal matching) - it is NOT added to close_time.
                # last_stop (user_stopped_speaking) is emitted by
                # TurnAnalyzerUserTurnStopStrategy only once the analyzer's append_audio has
                # already waited out the full silence_ms internally (confirmed against real
                # data: last_stop trails the audit-log dispatch by only 4-6ms). Adding
                # silence_ms again here double-counted that wait, inflating open_duration_ms
                # by roughly one extra stop_secs timeout per forced completion.
                last_stop = max(stops_in_window)
                close_time = last_stop
                transcript_text = _transcript_for_vad_stop(dispatched_user_turns, last_stop)
                result = {
                    "open_duration_ms": close_time - window_start,
                    "time_to_complete_ms": None,
                    "completion": "forced",
                }
                if transcript_text is not None:
                    # Was stop_secs forced because the user's actual utterance is a shape the
                    # turn analyzer struggles to close on its own (short / ack / spelled-out)?
                    result["transcript_text"] = transcript_text
                    result["final_turn_flags"] = final_turn_input_flags(transcript_text)
                per_window.append([result])
                continue

        close_time = window_end if window_end is not None else last_epoch
        per_window.append(
            [
                {
                    "open_duration_ms": close_time - window_start,
                    "time_to_complete_ms": None,
                    "completion": "stuck",
                }
            ]
        )

    # Premature pass: a window's last entry (the one bordering the next window) gets
    # reclassified when the next window itself resolved to a real completion (not "stuck")
    # but opened with no turn_start - see "early" in this function's docstring for why
    # both conditions are required.
    if turn_start_timestamps:
        for i, (_, window_end) in enumerate(windows[:-1]):
            last_entry = per_window[i][-1]
            next_window_resolved = per_window[i + 1][0]["completion"] in ("natural", "forced")
            if (
                last_entry["completion"] in ("natural", "forced")
                and next_window_resolved
                and window_end not in turn_start_timestamps
            ):
                last_entry["completion"] = "early"

    results: list[dict[str, Any]] = []
    for entries in per_window:
        for entry in entries:
            results.append({"turn_index": len(results), **entry})
    return results


# Tolerance for matching the real, ground-truth last user turn (from
# context.audio_timestamps_user_turns, recorded by the user-simulator process) against the
# turn-analyzer's own vad_starts (recorded by the pipecat server process). These two are on
# different clocks with no fixed offset - confirmed against real Krisp/non-Krisp examples,
# where legitimate skew for a turn VAD *did* register ranged ~0.4-6.8s, while turns VAD never
# registered at all sat 10.8-24.4s away from the nearest vad_start. 8s sits between those two
# clusters: generous enough to absorb real skew, tight enough to catch a genuine miss.
_VAD_MISS_TOLERANCE_MS = 8000


def _final_user_turn_never_reached_vad(
    vad_starts: list[int], audio_timestamps_user_turns: dict[int, list[tuple[float, float]]] | None
) -> bool:
    """True if the turn analyzer's VAD never registered a start anywhere near the last real user turn.

    Catches the case where the turn-analyzer's own "user_started_speaking" never fired at all
    for the final utterance (confirmed against real Krisp examples: the conversation ends via
    inactivity_timeout with the user having spoken last, but that last utterance produced no
    VAD-start window whatsoever - so it never shows up in ``per_turn`` and
    ``per_turn[-1]["completion"] == "stuck"`` can't catch it, since ``per_turn[-1]`` is really
    describing an earlier, already-resolved turn).
    """
    if not audio_timestamps_user_turns or not vad_starts:
        return False
    last_key = max(audio_timestamps_user_turns)
    segments = audio_timestamps_user_turns[last_key]
    if not segments:
        return False
    last_start_ms = min(seg[0] for seg in segments) * 1000
    return all(abs(vs - last_start_ms) > _VAD_MISS_TOLERANCE_MS for vs in vad_starts)


def _turn_stop_strategy(output_dir: str) -> str | None:
    """Return this run's configured model.turn_stop_strategy, or None if config.json wasn't found."""
    config = _find_run_config(output_dir)
    if config is None:
        return None
    model_cfg = config.get("model") or {}
    return model_cfg.get("turn_stop_strategy", "turn_analyzer")


def compute_vad_turn_sub_metrics(context: MetricContext) -> tuple[dict[str, MetricScore], list[dict[str, Any]]] | None:
    """Compute the VAD-analyzer sub-metrics plus the per-turn diagnostic list.

    Returns None when the run used no local turn-analyzer VAD (vad_turn_metrics_applicable
    is False) or recorded no VAD-analyzer turns at all.
    """
    if not vad_turn_metrics_applicable(context.output_dir):
        return None

    vad_starts, vad_stops = _load_vad_events(context.output_dir)
    if not vad_starts:
        return None

    turn_metrics_events = _load_turn_metrics_events(context.output_dir)
    stop_secs_silence_ms = _load_stop_secs_silence_ms(context.output_dir)
    dispatched_user_turns = _load_dispatched_user_turns(context.output_dir)
    turn_start_timestamps = _load_turn_start_timestamps(context.output_dir)
    per_turn = _classify_turns(
        vad_starts,
        vad_stops,
        turn_metrics_events,
        stop_secs_silence_ms,
        dispatched_user_turns,
        turn_start_timestamps,
    )

    def _wrap(key: str, value: float, normalized: bool) -> MetricScore:
        return MetricScore(name=f"turn_taking.{key}", score=value, normalized_score=value if normalized else None)

    # A user turn that never even got a VAD-start window (see
    # _final_user_turn_never_reached_vad) is invisible to _classify_turns entirely - fold it
    # in here as an explicit stuck entry so it's counted by stuck_rate below, rather than
    # silently vanishing. Only added when it's also the reason the conversation died (matches
    # the narrower signal this used to be gated on): Krisp has no silence-duration fallback,
    # so a VAD-analyzer miss is what actually kills the call via inactivity_timeout.
    if is_agent_timeout_on_user_turn(
        context.conversation_ended_reason,
        context.audio_timestamps_user_turns,
        context.audio_timestamps_assistant_turns,
    ) and _final_user_turn_never_reached_vad(vad_starts, context.audio_timestamps_user_turns):
        per_turn = [
            *per_turn,
            {
                "turn_index": len(per_turn),
                "open_duration_ms": None,
                "time_to_complete_ms": None,
                "completion": "stuck",
                "vad_never_started": True,
            },
        ]

    sub: dict[str, MetricScore] = {}

    stuck_count = sum(1 for t in per_turn if t["completion"] == "stuck")
    sub["stuck_rate"] = _wrap("stuck_rate", round(stuck_count / len(per_turn), 4), True)

    natural_count = sum(1 for t in per_turn if t["completion"] == "natural")
    forced_count = sum(1 for t in per_turn if t["completion"] == "forced")
    premature_count = sum(1 for t in per_turn if t["completion"] == "early")
    strategy = _turn_stop_strategy(context.output_dir)
    if strategy != "krisp_viva_turn" and (natural_count + forced_count) > 0:
        sub["forced_completion_rate"] = _wrap(
            "forced_completion_rate", round(forced_count / (natural_count + forced_count), 4), True
        )

    # Rate of turn-analyzer completions (natural or forced) that turned out to be premature -
    # the turn analyzer reported the user done, but the next VAD-start window was never
    # promoted to a real pipecat turn (see "early" in _classify_turns' docstring), meaning
    # the user's utterance actually continued and got wrongly cut short.
    resolved_count = natural_count + forced_count + premature_count
    if resolved_count > 0:
        sub["premature_detection_rate"] = _wrap(
            "premature_detection_rate", round(premature_count / resolved_count, 4), True
        )

    # Among forced completions specifically: does the final utterance's shape (short /
    # acknowledgement / spelled-out entity) explain why the turn analyzer needed the stop_secs
    # fallback instead of closing naturally? Only emitted when there's at least one forced
    # completion with a matched transcript, so cross-record aggregates reflect real evidence.
    forced_flags = [t["final_turn_flags"] for t in per_turn if t["completion"] == "forced" and "final_turn_flags" in t]
    if forced_flags:
        for key in _FINAL_TURN_FLAG_KEYS:
            rate = sum(1 for flags in forced_flags if flags[key]) / len(forced_flags)
            sub[f"forced_completion_final_turn_{key}_rate"] = _wrap(
                f"forced_completion_final_turn_{key}_rate", round(rate, 4), True
            )

    natural_times = [
        t["time_to_complete_ms"]
        for t in per_turn
        if t["completion"] == "natural" and t["time_to_complete_ms"] is not None
    ]
    if natural_times:
        sorted_times = sorted(natural_times)
        n = len(sorted_times)

        def _pct(p: float) -> float:
            return sorted_times[min(n - 1, int(p * n))]

        sub["mean_time_to_complete_ms"] = _wrap(
            "mean_time_to_complete_ms", round(statistics.mean(natural_times), 3), False
        )
        sub["p50_time_to_complete_ms"] = _wrap("p50_time_to_complete_ms", round(_pct(0.50), 3), False)
        sub["p90_time_to_complete_ms"] = _wrap("p90_time_to_complete_ms", round(_pct(0.90), 3), False)

    return sub, per_turn
