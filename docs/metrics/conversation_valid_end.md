# Conversation Finished

> **Validation Metric**: Incomplete conversations produce unreliable metric results — this must pass before other metrics can be trusted.

## Overview

Deterministic validation metric that verifies the conversation properly ended with the `end_call` tool. This is a quality control metric. It ensures simulation quality by checking that the user simulator called `end_call` to terminate the conversation, validating that conversations completed rather than timing out or erroring, and identifying incomplete or abandoned conversations.

## How It Works

### Evaluation Method

- **Type**: Deterministic (log file inspection)
- **Granularity**: Conversation-level

### Input Data

Uses `output_dir` from MetricContext to read `elevenlabs_events.jsonl`.

### Evaluation Methodology

1. Read `elevenlabs_events.jsonl` from output directory
2. Parse last line as JSON
3. Check if `type == "tool_response"` and `data.agent_tool_response.tool_name == "end_call"`
4. Return 1.0 if both conditions met, else 0.0

### Scoring

- **Scale**: Binary (0.0 or 1.0)
  - 1.0: Last event is `tool_response` with `tool_name="end_call"`
  - 0.0: Missing `end_call` or file issues
- **Normalization**: Already 0-1 scale
- Conversations scoring 0 may have timed out, encountered errors, or had the user simulator fail to call `end_call`

## Example Output

```json
{
  "name": "conversation_valid_end",
  "score": 1.0,
  "normalized_score": 1.0,
  "details": {
    "ended_properly": true,
    "last_event_type": "tool_response",
    "tool_name": "end_call"
  }
}
```

## Related Metrics

- [user_behavioral_fidelity.md](user_behavioral_fidelity.md) - Validates user simulation quality

## Implementation Details

- **File**: `src/eva/metrics/validation_metrics/conversation_valid_end.py`
- **Class**: `ConversationFinishedMetric`
- **Base Class**: `CodeMetric`
- **Configuration**: None (deterministic validation)
