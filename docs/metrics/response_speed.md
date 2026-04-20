# Response Speed

> **Diagnostic Metric**: Raw latency is useful for benchmarking but not directly scorable — acceptable thresholds depend on the system, so this is diagnostic rather than evaluative. Turn taking provides the qualitative timing assessment.

## Overview

Deterministic metric that measures the elapsed time between the end of a user utterance and the start of the assistant's response. It evaluates conversation responsiveness by measuring response latency for each turn transition, identifying slow responses that create awkward pauses, and providing mean and max latency for system benchmarking.

### Capabilities Measured

- **Pipeline**: Measures end-to-end wall-clock latency across all pipeline components (STT + LLM + TTS in cascade, or the single audio-native model). Not attributable to a single model capability.

## How It Works

### Evaluation Method

- **Type**: Deterministic (latency analysis from Pipecat's [UserBotLatencyObserver](https://docs.pipecat.ai/server/utilities/observers/user-bot-latency-observer))
- **Granularity**: Per-turn with conversation-level aggregation

### Input Data

Uses `latency_assistant_turns` from MetricContext — per-turn latencies (in seconds) computed from audio timestamps (user speech end to assistant speech start).

### Audio-Native vs Cascade

This metric is **architecture-agnostic**. It measures end-to-end latency regardless of pipeline type. However, the absolute values may differ significantly:
- **Cascade**: Latency includes STT + LLM + TTS processing time
- **Audio-native (S2S):** Latency includes only the model's processing time (no separate STT/TTS steps)

### Sanity Checks

Invalid values are filtered out: negative latencies and extreme values (>1000s).

### Scoring

- **Scale**: Seconds (typically 0.5-10.0)
- **Normalization**: None. Raw latency in seconds is not normalizable to a 0-1 scale in a meaningful way — interpretation depends on the system being benchmarked.
- **Aggregation**: Mean response speed across all valid turn transitions

## Example Output

```json
{
  "name": "response_speed",
  "score": 2.8,
  "normalized_score": null,
  "details": {
    "mean_speed_seconds": 2.8,
    "max_speed_seconds": 4.2,
    "num_turns": 7,
    "per_turn_speeds": [2.1, 2.5, 3.2, 2.9, 4.2, 2.3, 2.4]
  }
}
```

## Related Metrics

- [turn_taking.md](turn_taking.md) - Evaluates timing appropriateness (early/on-time/late), not raw latency

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/response_speed.py`
- **Class**: `ResponseSpeedMetric`
- **Base Class**: `CodeMetric`
- **Configuration**: None (deterministic computation)
