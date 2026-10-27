# Time to Completion

> **Diagnostic Metric**: How long did a successful conversation take? Reported only for conversations that completed the task, so it answers "when the agent succeeds, how long does it take?" — not directly used in final pass/fail scores.

## Overview

Deterministic metric that reports the total wall-clock duration (in seconds) of a conversation, but **only for conversations where the task was actually completed**. Mixing in the durations of failed conversations would make the value meaningless as an efficiency signal, so unsuccessful conversations are skipped.

Task completion is determined using the exact same criteria as [`task_completion`](task_completion.md): the session must be authenticated correctly and the final scenario database state must match the expected state (SHA-256 hash comparison).

### Capabilities Measured

- **Pipeline**: End-to-end wall-clock efficiency of the full system in reaching the correct outcome. Not attributable to a single model capability.

## How It Works

### Evaluation Method

- **Type**: Deterministic (reads `duration_seconds`)
- **Granularity**: Conversation-level

### Input Data

Uses the following MetricContext fields:
- `expected_scenario_db`, `final_scenario_db`, `final_scenario_db_hash`: used to determine whether the task was completed (same as `task_completion`).
- `duration_seconds`: total conversation duration.

### Scoring

- **Scale**: Seconds (lower is better)
- **Normalization**: None. Raw duration in seconds is not meaningfully normalizable to a 0-1 scale.
- **Skipped when**: the task was not completed, or no valid duration was recorded. Skipped records are excluded from the run-level efficiency aggregate.

## Example Output

```json
{
  "name": "time_to_completion",
  "score": 42.5,
  "normalized_score": null,
  "details": {
    "task_completed": true,
    "duration_seconds": 42.5,
    "reason": "Task completed — reporting total conversation duration"
  }
}
```

When the task was not completed:

```json
{
  "name": "time_to_completion",
  "score": null,
  "normalized_score": null,
  "skipped": true,
  "details": {
    "task_completed": false,
    "reason": "Final database state differs from expected state"
  }
}
```

## Summary Aggregation

The run-level `metrics_summary.json` includes an `overall_scores.efficiency.time_to_completion` block with the mean/min/max duration across successful conversations, alongside the per-metric aggregate under `per_metric.time_to_completion`.

## Related Metrics

- [turns_to_completion.md](turns_to_completion.md) - Number of turns (rather than seconds) to complete the task
- [task_completion.md](task_completion.md) - The binary completion check that gates this metric
- [response_speed.md](response_speed.md) - Per-turn response latency (not total conversation time)

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/time_to_completion.py`
- **Class**: `TimeToCompletionMetric`
- **Base Class**: `CodeMetric`
- **Configuration**: None (deterministic computation)
