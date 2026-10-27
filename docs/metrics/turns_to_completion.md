# Turns to Completion

> **Diagnostic Metric**: How many turns did a successful conversation take? Reported only for conversations that completed the task, so it answers "when the agent succeeds, how many turns does it take?" — not directly used in final pass/fail scores.

## Overview

Deterministic metric that reports the total number of conversation turns, but **only for conversations where the task was actually completed**. Mixing in the turn counts of failed conversations would make the value meaningless as an efficiency signal, so unsuccessful conversations are skipped.

Task completion is determined using the exact same criteria as [`task_completion`](task_completion.md): the session must be authenticated correctly and the final scenario database state must match the expected state (SHA-256 hash comparison).

### Capabilities Measured

- **Language Model**: How efficiently the agent drives the conversation to the correct outcome (fewer back-and-forth turns for the same result is better).

## How It Works

### Evaluation Method

- **Type**: Deterministic (reads `num_turns`)
- **Granularity**: Conversation-level

### Input Data

Uses the following MetricContext fields:
- `expected_scenario_db`, `final_scenario_db`, `final_scenario_db_hash`: used to determine whether the task was completed (same as `task_completion`).
- `num_turns`: total number of conversation turns.

### Scoring

- **Scale**: Turn count (lower is better)
- **Normalization**: None. Raw turn count is not meaningfully normalizable to a 0-1 scale.
- **Skipped when**: the task was not completed, or no valid turn count was recorded. Skipped records are excluded from the run-level efficiency aggregate.

## Example Output

```json
{
  "name": "turns_to_completion",
  "score": 8.0,
  "normalized_score": null,
  "details": {
    "task_completed": true,
    "num_turns": 8,
    "reason": "Task completed — reporting total conversation turns"
  }
}
```

When the task was not completed:

```json
{
  "name": "turns_to_completion",
  "score": null,
  "normalized_score": null,
  "skipped": true,
  "details": {
    "task_completed": false,
    "reason": "Authentication failed — session mismatch on keys: ['user_id']"
  }
}
```

## Summary Aggregation

The run-level `metrics_summary.json` includes an `overall_scores.efficiency.turns_to_completion` block with the mean/min/max turn count across successful conversations, alongside the per-metric aggregate under `per_metric.turns_to_completion`.

## Related Metrics

- [time_to_completion.md](time_to_completion.md) - Wall-clock time (rather than turns) to complete the task
- [task_completion.md](task_completion.md) - The binary completion check that gates this metric

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/turns_to_completion.py`
- **Class**: `TurnsToCompletionMetric`
- **Base Class**: `CodeMetric`
- **Configuration**: None (deterministic computation)
