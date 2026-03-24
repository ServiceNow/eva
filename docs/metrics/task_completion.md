# Task Completion

> **Accuracy Metric**: The bottom-line accuracy check — did the agent actually accomplish what it was supposed to?

## Overview

Deterministic code-based metric that validates task correctness by comparing scenario database states before and after the conversation. It answers: "Did the agent make the correct changes to the database?" by comparing the actual final state against the expected final state using SHA-256 hash comparison, providing binary pass/fail without LLM subjectivity.

To make sure that this metric is meaningful, the user simulator is tightly constrained to produce deterministic outcomes — i.e., the same scenario can always lead to the same expected final database state, regardless of conversational variation, if the agent doesn't make mistakes.

### Capabilities Measured

- **Speech Recognition**: In cascade pipelines, STT errors can cause the model to receive incorrect user input (e.g., wrong confirmation code), leading to wrong tool parameters and a mismatched final state. In audio-native models, the equivalent is the model mishearing user speech.
- **Language Model**: Did the model make the correct tool calls with the right parameters to reach the expected database state?

## How It Works

### Evaluation Method

- **Type**: Deterministic (hash comparison)
- **Granularity**: Conversation-level

### Input Data

Uses the following MetricContext fields:
- `expected_scenario_db`: Expected final database state (from ground truth)
- `final_scenario_db_hash`: Hash of actual final database state (computed during execution)

### Evaluation Methodology

When hashes don't match, the metric computes a detailed diff showing:
- Tables added/removed/modified
- Records added/removed/modified within tables
- Field-level changes within records

This makes it easy to diagnose exactly what went wrong.

### Scoring

- **Scale**: Binary (0.0 or 1.0)
  - 1.0: Expected and actual database state hashes match
  - 0.0: Hashes don't match (wrong changes, missing changes, or extra changes)
- **Normalization**: Same as raw score

## Example Output

```json
{
  "name": "task_completion",
  "score": 1.0,
  "normalized_score": 1.0,
  "details": {
    "match": true,
    "expected_hash": "a1b2c3...",
    "actual_hash": "a1b2c3..."
  }
}
```

## Related Metrics

- [faithfulness.md](faithfulness.md) - Evaluates how the agent arrived at the result (process)
- [tool_call_validity.md](tool_call_validity.md) - Checks tool call format validity

## Implementation Details

- **File**: `src/eva/metrics/accuracy/task_completion.py`
- **Class**: `TaskCompletion`
- **Base Class**: `BaseMetric`
- **Configuration**: None (deterministic computation)
