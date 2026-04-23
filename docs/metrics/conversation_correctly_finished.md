# Conversation Correctly Finished

> **Diagnostic Metric**: Flags records where the agent failed to respond to the user's final turn, helping isolate agent-side timeouts/no-response failures from other conversation outcomes. Not directly used in final evaluation scores; excluded from pass@k.

## Overview

Deterministic diagnostic metric that flags records where the conversation ended with `inactivity_timeout` and the user was the last speaker by audio timeline — i.e. the agent failed to respond to the user's final turn. Conversations that ended normally, errored out, or ended on inactivity but with the assistant as the last speaker all score 1.0.

### Capabilities Measured

- **Language Model / Pipeline**: Whether the agent produced a response to the user's final turn. A 0.0 score indicates the agent went silent after the user spoke (e.g. generation stalled, tool call hung, or the model chose not to respond) and the session was closed by the inactivity timer.

## How It Works

### Evaluation Method

- **Type**: Deterministic (metadata + audio timeline inspection)
- **Granularity**: Conversation-level

### Input Data

Uses the following MetricContext fields:
- `conversation_ended_reason`: Reason the conversation terminated (e.g. `"goodbye"`, `"inactivity_timeout"`, `"error"`).
- `audio_timestamps_user_turns`: Per-turn `(start, end)` audio intervals for the user.
- `audio_timestamps_assistant_turns`: Per-turn `(start, end)` audio intervals for the assistant.

### Evaluation Methodology

1. Compute `last_audio_speaker` as whichever side (`"user"` or `"assistant"`) has the latest audio end-timestamp across all turns. Returns `None` if neither side recorded audio.
2. Flag the record as a missed turn iff `conversation_ended_reason == "inactivity_timeout"` **and** `last_audio_speaker == "user"`.
3. Score 0.0 if flagged, else 1.0.

### Scoring

- **Scale**: Binary (0.0 or 1.0)
  - 1.0: Agent responded to every user turn (normal end, error, or inactivity with assistant as last speaker).
  - 0.0: Conversation ended with `inactivity_timeout` and the user was the last speaker.
- **Normalization**: Already 0-1 scale.
- **Aggregation**: Mean across records (excluded from pass@k).

## Example Output

```json
{
  "name": "conversation_correctly_finished",
  "score": 0.0,
  "normalized_score": 0.0,
  "details": {
    "conversation_ended_reason": "inactivity_timeout",
    "last_audio_speaker": "user",
    "reason": "conversation ended with inactivity_timeout and user was the last speaker"
  }
}
```

## Related Metrics

- [conversation_valid_end.md](conversation_valid_end.md) - Validates the conversation ended via the `end_call` tool (simulator-side quality gate).
- [response_speed.md](response_speed.md) - Measures per-turn latency between user-end and assistant-start.

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/conversation_correctly_finished.py`
- **Class**: `ConversationCorrectlyFinishedMetric`
- **Base Class**: `CodeMetric`
- **Category**: `diagnostic`
- **`exclude_from_pass_at_k`**: `True`
- **Configuration**: None (deterministic)
