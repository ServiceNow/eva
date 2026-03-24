# Tool Call Validity

> **Diagnostic Metric**: Helps isolate whether task failures are caused by malformed tool calls (schema issues) vs. incorrect reasoning — not scored directly since format errors are already reflected in task completion.

## Overview

Deterministic metric that measures the fraction of tool calls with correctly formatted parameters. It identifies tool call format errors including wrong tool names, missing or malformed parameters, and invalid enum values or wrong types — while distinguishing these from business-logic errors (e.g., "reservation not found" is not a format error).

### Capabilities Measured

- **Language Model**: Does the model generate well-formed tool calls with valid parameter names, types, and enum values?

## How It Works

### Evaluation Method

- **Type**: Deterministic (tool response analysis)
- **Granularity**: Conversation-level

### Input Data

Uses the following MetricContext fields:
- `tool_params`: List of tool call parameters (parallel to `tool_responses`)
- `tool_responses`: List of tool call responses with `tool_name` and `tool_response`

### Evaluation Methodology

**Infrastructure errors** (from ToolExecutor):
- `tool_not_found`: Non-existent tool called
- `function_not_found`: Function not found in tool module
- `execution_error`: Runtime error during tool execution
- `invalid_parameter`: Generic parameter validation failure

**Validation errors** (from Pydantic parameter models):
- Field-specific validation errors (e.g., `invalid_confirmation_number_format`, `invalid_flight_number_format`)

**Not counted as errors** (business-logic):
- `not_found`: Reservation not found
- `verification_failed`: Last name doesn't match
- `no_seats_available`: No available seats

### Scoring

- **Scale**: 0.0-1.0 (fraction of valid calls)
  - 1.0: All tool calls have valid format
  - 0.0: All tool calls have format errors
- **Normalization**: Same as raw score (already 0-1)
- **Edge case**: No tool calls → returns 1.0

## Example Output

```json
{
  "name": "tool_call_validity",
  "score": 0.75,
  "normalized_score": 0.75,
  "details": {
    "total_tool_calls": 4,
    "valid_tool_calls": 3,
    "invalid_tool_calls": 1,
    "errors": [
      {
        "tool_name": "get_reservation",
        "error_type": "invalid_confirmation_number_format",
        "message": "Invalid confirmation_number 'X'",
        "parameters": {"confirmation_number": "X", "last_name": "Doe"}
      }
    ]
  }
}
```

## Related Metrics

- [authentication_success.md](authentication_success.md) - Checks if get_reservation succeeded

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/tool_call_validity.py`
- **Class**: `ToolCallValidity`
- **Base Class**: `CodeMetric`
- **Configuration**: None (deterministic computation)
