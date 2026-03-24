# Turn Taking

> **Experience Metric**: Poor timing — interrupting the user or leaving awkward silences — makes the conversation feel unnatural even if the content is correct.

## Overview

LLM-based metric that evaluates timing accuracy of assistant responses after user utterances, using the transcript and the latency measurements. It assesses whether the assistant takes the conversational floor at the correct time after user completion, avoids interrupting before the user finishes, responds promptly without awkward delays, and recognizes proper turn-completion points (TRPs). Greeting turn (turn 0) is excluded from evaluation. Turns with missing audio timestamps are skipped.

### Capabilities Measured

- **VAD**: Accurate detection of when the user has started and finished speaking is essential for well-timed responses.
- **Pipeline**: Turn-taking timing is a function of the end-to-end latency across all models in the pipeline (STT → LLM → TTS in cascade, or the single audio-native model). It reflects overall system performance.

## How It Works

### Evaluation Method

- **Type**: Text Judge (LLM-as-judge with timestamps and transcript context)
- **Model**: GPT-5.2
- **Granularity**: Per-turn (each user-to-assistant transition evaluated)

### Input Data

Uses the following MetricContext fields:
- `audio_timestamps_user_turns` / `audio_timestamps_assistant_turns`: Timing segments per turn
- `intended_user_turns` / `intended_assistant_turns`: What each speaker intended to say
- `transcribed_user_turns` / `transcribed_assistant_turns`: What was actually heard
- `assistant_interrupted_turns` / `user_interrupted_turns`: Interruption flags

The judge receives per-turn context blocks with timestamps, expected text, heard text, segment transitions, and interruption annotations. It does **not** receive raw audio.

### Audio-Native vs Cascade

This metric works with both architectures since it relies on ElevenLabs audio timestamps (available in both). The judge sees both intended and transcribed text for context, but the core evaluation is based on **timing** (latency between user end and assistant start), which is architecture-agnostic.

### Evaluation Methodology

For each user→assistant turn transition, the judge receives a context block containing:

1. **Segment transitions** — computed latencies between speaker handoffs (e.g., "user_end→assistant_start: 1.4s"). Positive values are gaps, negative values are overlaps.
2. **Interruption flags** — per-turn flags indicating who interrupted whom, detected from audio overlap analysis.
3. **User transcript** — both the intended text (what the user meant to say) and the heard text (what the STT transcribed), , plus interruption tags (e.g., `[assistant interrupts]`).
4. **Assistant transcript** — both the intended text and the heard text (transcribed by the user-side system), plus interruption tags (e.g., `[user interrupts]`, `[likely cut off by user]`).

The judge does **not** receive raw audio — it works entirely from timestamps, transcripts, and metadata tags.

**Decision logic**: The judge determines whether the user still holds the floor (mid-sentence, interruption tags indicate ongoing speech) or has yielded it (syntactically and pragmatically complete). Interruption tags are the strongest signal and override timing values when they conflict.

### Scoring

- **Scale**: -1 to +1 per turn
  - -1: Early/Interrupting — agent begins before user completion (< 200ms latency)
  - 0: On-Time — agent begins 200ms–4000ms after user completion
  - +1: Late — agent begins >= 4000ms after user completion
  - null: Indeterminate — no clear completion or unclear audio
- **Normalization**: `1.0 - abs(aggregated_score)` — perfect timing (all 0s) = 1.0, maximum deviation (all +/-1) = 0.0
- **Aggregation**: Absolute mean (default) — takes absolute value of each rating, then averages

In addition to the judge evaluation, the metric computes latency-based timing labels as a reference:
- Latency < 200ms → "Early / Interrupting"
- 200ms <= Latency < 4000ms → "On-Time"
- Latency >= 4000ms → "Late"

Agreement between judge labels and latency-derived labels is tracked in the output as `agreement_with_latency_values`.

## Example Output

```json
{
  "name": "turn_taking",
  "score": 0.2,
  "normalized_score": 0.8,
  "details": {
    "aggregation": "abs_mean",
    "num_turns": 5,
    "num_evaluated": 5,
    "per_turn_judge_timing_ratings": {"1": "On-Time", "2": "On-Time", "3": "Early / Interrupting", "4": "On-Time", "5": "On-Time"},
    "per_turn_latency": {"1": 0.412, "2": 0.589, "3": -0.15, "4": 0.501, "5": 0.32},
    "agreement_with_latency_values": 1.0
  }
}
```

## Related Metrics

- [response_speed.md](response_speed.md) - Code-based latency measurement (no judge involved)

## Implementation Details

- **File**: `src/eva/metrics/experience/turn_taking.py`
- **Class**: `TurnTakingMetric`
- **Base Class**: `TextJudgeMetric`
- **Prompt location**: `configs/prompts/judge.yaml` under `judge.turn_taking`
- **Configuration**: `judge_model` (default: "gpt-5.2"), `aggregation` (default: "abs_mean"; options: "mean", "abs_mean", "median")
