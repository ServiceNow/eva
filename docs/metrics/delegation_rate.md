# Delegation Rate

> **Diagnostic Metric**: How often does the frontend voice model hand a user turn to the backend model? Reported only for frameworks with explicit delegation (currently `openai_live`) — not used in final pass/fail scores, and neither a high nor a low rate is good or bad by itself.

## Overview

Deterministic metric: the fraction of **user turns** that triggered at least one delegation. The unit is the user turn, not the assistant turn, because assistant turns include the greeting, backchannels ("Hmm.") and acknowledgements ("Okay, I'll check that for you."), which would dilute the rate.

It is a tuning signal for the frontend prompt's delegation policy: a rate near 1 means almost every request pays the backend latency; a low rate means the frontend answers most turns itself. It cannot show the frontend answering a factual question *without* delegating — read it next to the accuracy metrics.

### Capabilities Measured

- **Language Model**: How the frontend decides between answering directly and delegating.

## How It Works

### Evaluation Method

- **Type**: Deterministic (reads `framework_logs.jsonl`)
- **Granularity**: Conversation-level

### Input Data

Reads `framework_logs.jsonl` in the record directory:
- `delegation_session`: written once when the session starts. Its presence marks the run as delegation-capable; without it the metric is skipped, so frameworks without delegation are not scored.
- `turn_start`: the start of each user turn.
- `delegation`: one per delegation, stamped with the time it was created.

Records are ordered by timestamp (the server writes a user turn's `turn_start` after the fact, with its original start time). Each delegation belongs to the latest user turn that started at or before it, so a correction that spawns a second delegation in the same turn counts that turn once.

### Scoring

- **Scale**: 0-1 (delegated user turns / user turns)
- **Normalization**: None. There is no better or worse direction.
- **Skipped when**: the framework does not use delegation, there is no `framework_logs.jsonl`, or no user turns were recorded. A delegation-capable run with no delegations scores `0.0`.

## Example Output

```json
{
  "name": "delegation_rate",
  "score": 0.5,
  "normalized_score": null,
  "details": {
    "num_user_turns": 4,
    "num_delegated_turns": 2,
    "num_delegations": 3,
    "num_unattributed_delegations": 0
  }
}
```

`num_unattributed_delegations` counts delegations created before the first user turn started; they are not part of the rate.

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/delegation_rate.py`
- **Class**: `DelegationRateMetric`
- **Base Class**: `CodeMetric`
- **Written by**: `src/eva/assistant/openai_live_server.py` (`delegation_session` and `delegation` records)
- **Configuration**: None (deterministic computation)
