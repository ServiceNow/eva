# Authentication Success

> **Diagnostic Metric**: Helps isolate whether task failures stem from the agent never authenticating the user vs. downstream issues — not used in accuracy directly since authentication failure is already reflected in task completion.

## Overview

Deterministic metric that checks whether the agent successfully authenticated the user by calling `get_reservation` with a successful result. Domain-specific to airline (checks `get_reservation` only). It diagnoses authentication failures by detecting if the agent never attempted to look up the reservation, identifying cases where `get_reservation` was called but failed (wrong confirmation code, wrong last name, etc.), and providing counts of attempts and successes.

### Capabilities Measured

- **Speech Recognition**: STT errors (cascade) or mishearing (audio-native) can garble the confirmation code or last name, causing authentication to fail even when the model's reasoning is correct.
- **Language Model**: Did the model correctly identify the need to authenticate and call `get_reservation` with the right parameters?

## How It Works

### Evaluation Method

- **Type**: Deterministic (tool response analysis)
- **Granularity**: Conversation-level

### Input Data

Uses `tool_responses` from MetricContext — list of tool call responses with `tool_name` and `tool_response`.

### Scoring

- **Scale**: Binary (0.0 or 1.0)
  - 1.0: At least one `get_reservation` call returned `status: "success"`
  - 0.0: No `get_reservation` calls, or none returned success
- **Normalization**: Same as raw score
- Score 0.0 with `get_reservation_found: false` means the agent never attempted authentication
- Score 0.0 with `get_reservation_found: true` means authentication was attempted but failed
- Multiple failed attempts followed by a success still scores 1.0

## Example Output

```json
{
  "name": "authentication_success",
  "score": 1.0,
  "normalized_score": 1.0,
  "details": {
    "get_reservation_found": true,
    "get_reservation_call_count": 2,
    "get_reservation_success_count": 1,
    "reason": "get_reservation called successfully"
  }
}
```

## Related Metrics

- [tool_call_validity.md](tool_call_validity.md) - Checks if tool calls have correctly formatted parameters

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/authentication_success.py`
- **Class**: `AuthenticationSuccessMetric`
- **Base Class**: `CodeMetric`
- **Configuration**: None (deterministic computation)
