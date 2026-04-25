# Turn Taking

> **Experience Metric**: poor timing — interrupting the user or leaving awkward silences — makes the conversation feel unnatural even if the content is correct.

## Overview

Code-based metric (no LLM) that scores each user→assistant transition on a continuous `[0, 1]` scale derived from the ElevenLabs audio timestamps already stored on `MetricContext`. The main score is the plain mean of the per-turn scores. A flat set of sub-metrics surfaces supporting headline numbers (latency percentiles, interruption rates, recovery/yield rates) that show up as their own columns in analysis tools.

## Scope

- **Greeting (turn 0) is excluded.**
- A turn is **evaluable** only when both `audio_timestamps_user_turns[t]` and `audio_timestamps_assistant_turns[t]` are non-empty. Turns without both sides are silently excluded from the set that feeds the score and sub-metrics.
- If no evaluable turns exist, the metric returns `normalized_score = None` with an error `"No turns with both user and assistant audio timestamps"`.

## Inputs (from `MetricContext`)

- `audio_timestamps_user_turns` / `audio_timestamps_assistant_turns` — per-turn audio segment lists `[(start_s, end_s), ...]`. Drive the evaluable set, overlap, and yield computations.
- `latency_assistant_turns` — per-turn latency (`first_asst_start - last_user_end`) in seconds. Drives the latency curve.
- `assistant_interrupted_turns` / `user_interrupted_turns` — turn-level interruption flags set by the processor.
- `conversation_trace` — used to detect which turns contain a tool call (`type == "tool_call"`). Tool-call turns get a more lenient upper end of the latency curve and a higher `late` threshold, since tool execution adds inherent latency.

## Per-turn Score

For each evaluable turn, one of four signal combinations is used depending on the turn's flags:

| Turn flag state | Signal → Score |
|---|---|
| `turn ∈ assistant_interrupted_turns` only | `min(overlap_score, count_score, post_interrupt_latency_score)` — each bounded by `AGENT_INTERRUPT_MAX_SCORE = 0.5` except the post-interrupt latency (unbounded [0, 1]), so the final turn score is capped at 0.5. |
| `turn ∈ user_interrupted_turns` only | **Yield score** |
| `turn` in **both** sets | `min(agent_interrupt_score, yield_score)` |
| neither flag | **Latency score** |

Each signal has its own continuous curve (below). Agent-interrupt sub-scores (`overlap_score`, `count_score`) are computed natively in `[0, AGENT_INTERRUPT_MAX_SCORE]` — they are never in [0, 1] internally; the "interrupting is never free" semantics are baked into the curves directly.

### Latency curve (piecewise linear)

Ramp up from `LATENCY_HARD_EARLY_MS` (-500ms) to `LATENCY_SWEET_SPOT_LOW_MS` (500ms), plateau at 1.0 through `LATENCY_SWEET_SPOT_HIGH_MS`, ramp down to `LATENCY_HARD_LATE_MS`. Outside the outer bounds the score is clamped at 0. The lower edges (`LATENCY_HARD_EARLY_MS`, `LATENCY_SWEET_SPOT_LOW_MS`) are shared across all turns; the upper edges are different for tool-call turns since tool execution adds inherent latency.

| | Non-tool turn | Tool-call turn |
|---|---|---|
| Sweet-spot upper edge | 2000 ms | 4000 ms (`LATENCY_SWEET_SPOT_HIGH_MS_TOOL`) |
| Hard-late edge | 5000 ms | 7000 ms (`LATENCY_HARD_LATE_MS_TOOL`) |

| Latency (ms) | Non-tool score | Tool-call score |
|---|---|---|
| ≤ -500 (hard early) | 0.00 | 0.00 |
| 0 | 0.50 | 0.50 |
| 200 | 0.70 | 0.70 |
| 500 | 1.00 | 1.00 |
| 2000 | 1.00 | 1.00 |
| 3500 | 0.50 | 1.00 |
| 4000 | 0.33 | 1.00 |
| 5000 | 0.00 | 0.67 |
| 6500 | 0.00 | 0.17 |
| ≥ 7000 | 0.00 | 0.00 |

A turn is considered a "tool-call turn" when `conversation_trace` has at least one entry with `type == "tool_call"` at that `turn_id`. The `has_tool_call` flag is echoed into `per_turn_evidence` for transparency.

### Overlap curve (agent-interrupt turns)

`overlap_ms` is the **total** simultaneous-speech duration between user and assistant in the turn, computed as the sum of pairwise segment intersections (streamed turns with interleaved silence would wildly over-count under a naive full-range intersection).

```
score = max(0, AGENT_INTERRUPT_MAX_SCORE * (1 - overlap_ms / OVERLAP_HARD_MS))
```

Output lives in `[0, AGENT_INTERRUPT_MAX_SCORE]` (default `[0, 0.5]`). The cap is baked into the formula — even zero-overlap barge-ins score at most 0.5, because interrupting the user is never fully "free".

| Overlap (ms) | Score |
|---|---|
| 0 | 0.50 (cap) |
| 500 | 0.375 |
| 1000 | 0.25 |
| ≥ 2000 | 0.00 |

### Count curve (agent-interrupt turns)

`n_interrupt_segments` is the number of **distinct** agent audio segments that overlap the user's speech in the same user turn. A single brief barge-in (n=1) is less disruptive than three separate barge-ins in a long user utterance; this sub-score penalizes the latter.

```
score = max(0, AGENT_INTERRUPT_MAX_SCORE * (1 - (n - 1) / (INTERRUPT_COUNT_HARD - 1)))
```

Only defined for `n ≥ 1` (a turn with zero overlap segments isn't really an agent-interrupt turn even if the flag is set — the count sub-score is simply omitted in that edge case). Output shares the same `[0, 0.5]` range as the overlap curve, so both agent-interrupt sub-scores are directly comparable.

| Distinct segments | Score |
|---|---|
| 1 | 0.50 (cap) |
| 2 | 0.25 |
| ≥ 3 (`INTERRUPT_COUNT_HARD`) | 0.00 |

### Post-interrupt latency (agent-interrupt turns)

After folding in `overlap_score` and `count_score`, the agent-interrupt turn score is additionally min'd with the **post-interrupt latency score**: the silent gap between the user's last audio end and the agent's first *settled* segment (the first agent segment starting strictly after the user finished) passed through the same latency curve as above (tool-call-aware). This catches the "brief barge-in, then 10s wait" failure mode that the overlap curve alone would not penalize.

Post-interrupt latency is omitted when the conversation ended during the overlap (no settled segment follows) or when an agent segment spans the user's end (continuous speech — there is no silent gap to measure). The post-interrupt latency score is uncapped (lives in `[0, 1]`), but since it is min'd with `overlap_score` and `count_score` (both ≤ 0.5), the final turn score is still ≤ 0.5.

### Yield curve (user-interrupt turns)

`yield_ms` is how long the agent kept speaking after the user barged in: `assistant_audio_end[t-1] - user_audio_start[t]`.

`score = max(0, 1 - yield_ms / YIELD_HARD_MS)`

User interruptions are not the agent's fault, so the score is uncapped — a fast-yielding agent gets the full `1.0`.

| Yield (ms) | Score |
|---|---|
| 0 | 1.00 |
| 600 | 0.70 |
| 1000 | 0.50 |
| ≥ 2000 | 0.00 |

## Main Score

`turn_taking.score = turn_taking.normalized_score = mean(per_turn_score)` over evaluable turns. No weighting.

## Per-turn Evidence

Every evaluated turn contributes one entry to `details.per_turn_evidence[turn_id]`. Always present:

- `has_tool_call` — whether `conversation_trace` has a tool_call entry on this turn. Drives the tool-aware branch of the latency curve.

The remaining fields depend on which signal fired:

**Latency turns**
- `latency_ms`
- `latency_score`

**Agent-interrupt turns** (either agent-only or dual)
- `overlap_ms`
- `overlap_score` — in `[0, AGENT_INTERRUPT_MAX_SCORE]`.
- `n_interrupt_segments` — how many distinct agent audio segments overlap the user's speech in this turn.
- `interrupt_count_score` — in `[0, AGENT_INTERRUPT_MAX_SCORE]`. Present only when `n_interrupt_segments ≥ 1`.
- `post_interrupt_latency_ms` — gap between user's last audio end and the agent's first *settled* segment (the first one starting **after** the user finished). Present only when such a segment exists.
- `post_interrupt_latency_score` — `_latency_score(post_interrupt_latency_ms, has_tool_call=…)`. The agent's turn score folds everything together: `turn_score = min(overlap_score, interrupt_count_score, post_interrupt_latency_score)` (each term included only when defined).

**User-interrupt turns** (either user-only or dual)
- `yield_ms`
- `yield_score`

## Details Fields

`details` on the main `MetricScore` contains:

| Field | Description |
|---|---|
| `per_turn_score` | `{turn_id: float}` — the final 0–1 score per turn. |
| `per_turn_reason` | `{turn_id: "latency" / "agent_interrupt" / "user_interrupt" / "dual_interrupt"}` — which signal fired. |
| `per_turn_evidence` | `{turn_id: {...}}` — see previous section. |
| `num_turns` | Highest turn_id present in either user or assistant audio timestamps (greeting excluded). |
| `num_evaluated` | Number of turns actually scored (both timestamp sides present). |

## Sub-metrics (flat)

Emitted as `sub_metrics` on the main `MetricScore`, in this order. The runner aggregates each generically into `metrics_summary.json` as its own column, preserving insertion order.

**Latency (always present when at least one latency measurement exists)**

| Key | Normalized? | Meaning |
|---|---|---|
| `mean_latency_ms` | no | Arithmetic mean of per-turn latencies in ms. |
| `p50_latency_ms` | no | Median latency. |
| `p90_latency_ms` | no | 90th-percentile latency. |
| `on_time_rate` | yes | Fraction with `EARLY_THRESHOLD_MS ≤ latency < late_threshold` (where the late threshold is `LATE_THRESHOLD_MS_TOOL` on tool-call turns, `LATE_THRESHOLD_MS` otherwise). |
| `early_rate` | yes | Fraction with `latency < EARLY_THRESHOLD_MS` (default 200 ms). |
| `late_rate` | yes | Fraction with `latency ≥ late_threshold` — 4000 ms on non-tool turns, 6000 ms on tool-call turns. |

**Agent interruptions** (dotted prefix so tables group them visibly)

| Key | Normalized? | When present | Meaning |
| --- | --- | --- | --- |
| `agent_interruption.rate` | yes | always | Fraction of evaluable turns in `assistant_interrupted_turns`. |
| `agent_interruption.num_interruptions` | no (raw count) | always, but `score=None` when rate = 0 | Total distinct barge-in segments summed across all agent-interrupt turns. `None` on clean runs so cross-record aggregates exclude them rather than averaging in zeros. |
| `agent_interruption.mean_overlap_ms` | no | rate > 0 | Arithmetic mean of `overlap_ms` across agent-interrupt turns. |
| `agent_interruption.mean_overlap_score` | yes | rate > 0 | Mean of the per-turn overlap scores (in `[0, AGENT_INTERRUPT_MAX_SCORE]`). |
| `agent_interruption.mean_count_score` | yes | rate > 0 and ≥ 1 interrupt turn had `n_interrupt_segments ≥ 1` | Mean of the per-turn count scores (in `[0, AGENT_INTERRUPT_MAX_SCORE]`). |
| `agent_interruption.mean_post_interrupt_latency_ms` | no | ≥ 1 interrupt turn has a settled response | Mean `post_interrupt_latency_ms` across agent-interrupt turns that emit a settled segment after the user finishes. |
| `agent_interruption.mean_post_interrupt_latency_score` | yes | same as above | Mean of the post-interrupt latency scores that feed the main score (tool-aware). |

Sub-metric aggregation reads the raw per-turn values from `per_turn_evidence` rather than recomputing them from audio timestamps — aggregates are guaranteed to be consistent with the per-turn scores.

**User interruptions**

| Key | Normalized? | When present | Meaning |
| --- | --- | --- | --- |
| `user_interruption.rate` | yes | always | Fraction of evaluable turns in `user_interrupted_turns`. |
| `user_interruption.mean_yield_ms` | no | rate > 0 | Arithmetic mean of `yield_ms` across user-interrupt turns. |
| `user_interruption.mean_yield_score` | yes | rate > 0 | Mean of the per-turn yield scores that feed the main score. |

Rate sub-metrics are emitted as `normalized_score` (they already live on `[0, 1]`). Raw-ms sub-metrics have `normalized_score = None` so they don't corrupt cross-metric averages.

## Tunable Constants

All thresholds live as class-level attributes on `TurnTakingMetric`. Override by subclassing or editing in place.

| Constant | Default | Purpose |
|---|---|---|
 | `LATENCY_HARD_EARLY_MS` | -500 | Left edge of the latency ramp (score = 0 at or below). Shared across tool / non-tool turns. |
| `LATENCY_SWEET_SPOT_LOW_MS` | 500 | Left edge of the latency plateau (score reaches 1). Shared. |
| `LATENCY_SWEET_SPOT_HIGH_MS` | 2000 | Right edge of the plateau on **non-tool** turns. |
| `LATENCY_HARD_LATE_MS` | 5000 | Right edge of the ramp on **non-tool** turns (score = 0 at or above). |
| `LATENCY_SWEET_SPOT_HIGH_MS_TOOL` | 4000 | Right edge of the plateau on **tool-call** turns. |
| `LATENCY_HARD_LATE_MS_TOOL` | 7000 | Right edge of the ramp on **tool-call** turns. |
| `OVERLAP_HARD_MS` | 2000 | Overlap at which the agent-interrupt overlap score hits 0. |
| `AGENT_INTERRUPT_MAX_SCORE` | 0.5 | Cap on the overlap and count sub-scores — interrupting is never fully "free". |
| `INTERRUPT_COUNT_HARD` | 3 | Number of distinct barge-in segments at which the count sub-score hits 0. |
| `YIELD_HARD_MS` | 2000 | Yield time at which the user-interrupt score hits 0. |
| `EARLY_THRESHOLD_MS` | 200 | Latency classification cutoff — below ⇒ "early". Shared across tool / non-tool turns. |
| `LATE_THRESHOLD_MS` | 4000 | Latency classification cutoff on **non-tool** turns — at or above ⇒ "late". |
| `LATE_THRESHOLD_MS_TOOL` | 6000 | Latency classification cutoff on **tool-call** turns. |

Note: the latency *curve* and the latency *classification* use independent thresholds. The curve is continuous (hard-early / hard-late), while `EARLY_THRESHOLD_MS` / `LATE_THRESHOLD_MS{,_TOOL}` bucket turns for the `early_rate` / `on_time_rate` / `late_rate` sub-metrics only.

## Example Output

```json
{
  "name": "turn_taking",
  "score": 0.83,
  "normalized_score": 0.83,
  "details": {
    "per_turn_score": {"1": 1.0, "2": 1.0, "3": 0.45, "4": 0.95},
    "per_turn_reason": {"1": "latency", "2": "latency", "3": "agent_interrupt", "4": "user_interrupt"},
    "per_turn_evidence": {
      "1": {"has_tool_call": false, "latency_ms": 1000, "latency_score": 1.0},
      "2": {"has_tool_call": true,  "latency_ms": 3500, "latency_score": 1.0},
      "3": {"has_tool_call": false,
            "overlap_ms": 200, "overlap_score": 0.45,
            "n_interrupt_segments": 1, "interrupt_count_score": 0.5,
            "post_interrupt_latency_ms": 1000, "post_interrupt_latency_score": 1.0},
      "4": {"has_tool_call": false, "yield_ms": 100, "yield_score": 0.95}
    },
    "num_turns": 4,
    "num_evaluated": 4
  },
  "sub_metrics": {
    "mean_latency_ms":                                        {"score": 2200.0, "normalized_score": null},
    "p50_latency_ms":                                         {"score": 2000.0, "normalized_score": null},
    "p90_latency_ms":                                         {"score": 3500.0, "normalized_score": null},
    "on_time_rate":                                           {"score": 1.0,    "normalized_score": 1.0},
    "early_rate":                                             {"score": 0.0,    "normalized_score": 0.0},
    "late_rate":                                              {"score": 0.0,    "normalized_score": 0.0},
    "agent_interruption.rate":                                {"score": 0.25,   "normalized_score": 0.25},
    "agent_interruption.num_interruptions":                   {"score": 1.0,    "normalized_score": null},
    "agent_interruption.mean_overlap_ms":                     {"score": 200.0,  "normalized_score": null},
    "agent_interruption.mean_overlap_score":                  {"score": 0.45,   "normalized_score": 0.45},
    "agent_interruption.mean_count_score":                    {"score": 0.5,    "normalized_score": 0.5},
    "agent_interruption.mean_post_interrupt_latency_ms":      {"score": 1000.0, "normalized_score": null},
    "agent_interruption.mean_post_interrupt_latency_score":   {"score": 1.0,    "normalized_score": 1.0},
    "user_interruption.rate":                                 {"score": 0.25,   "normalized_score": 0.25},
    "user_interruption.mean_yield_ms":                        {"score": 100.0,  "normalized_score": null},
    "user_interruption.mean_yield_score":                     {"score": 0.95,   "normalized_score": 0.95}
  }
}
```

## Related Metrics

- [`response_speed`](response_speed.md) — raw per-turn latency values, no curve or bucketing.

## Implementation Details

- **File**: `src/eva/metrics/experience/turn_taking.py`
- **Class**: `TurnTakingMetric`
- **Base class**: `CodeMetric`
