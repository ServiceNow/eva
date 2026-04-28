# User Behavioral Fidelity

> **Validation Metric**: If the simulated user behaved incorrectly (e.g., ended the call early, made off-script requests), the agent is being evaluated against a corrupted scenario — results are not trustworthy.

## Overview

LLM-based validation metric that detects whether a simulated user's behavior **corrupted the agent evaluation**. Unlike a general realism check, this metric specifically identifies cases where the user's actions caused the agent to be evaluated against an incorrect database state — either by triggering wrong modifications, preventing correct ones, or ending the conversation before the agent could complete its work. It answers: "Did the user cause the database to end up in a different state than it should have?"

This is a data quality gate — conversations scored 0 (corrupted) should be excluded or re-run before drawing conclusions about agent performance. The metric only cares about **modification tools** (write operations). User deviations that only trigger read-only tools are not considered corruption, and minor deviations that don't affect database state are acceptable. Each corruption type is analyzed independently with its own reasoning, making it easy to diagnose what went wrong.

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2
- **Granularity**: Conversation-level

### Input Data

Uses the following MetricContext fields:

- `conversation_trace`: Full conversation transcript including tool calls and responses
- `intended_user_turns`: What the user simulator was instructed to say (ground truth)
- `user_goal`: User's high-level goal and required information
- `user_persona`: User's personality traits and behavior description
- `agent_tools` (filtered to `tool_type == "write"`): The modification tools whose invocations are relevant to corruption analysis — read-only tools are not a concern

### Audio-Native vs Cascade

This metric has **pipeline-specific prompt text** and provides the judge with **two views** of the conversation:

- **Cascade**: The judge sees the agent-side transcript (`conversation_trace`, where user turns are STT transcriptions) alongside the `intended_user_turns` (ground truth). The prompt explains that discrepancies between the two are transcription errors — the user should not be penalized for the agent mishearing.
- **Audio-native (AUDIO_LLM, S2S):** The judge sees the conversation trace (where user turns are already intended text) alongside the `intended_user_turns`. The prompt explains this is an audio-native system and that discrepancies in agent behavior may be due to audio perception errors, not user corruption.

In both cases, `intended_user_turns` serves as ground truth for what the user actually said.

### Evaluation Methodology

The metric checks for five specific corruption scenarios:

#### 1. Extra Modifications (User Invented Requests)
The user made requests **outside their assigned goal** that caused the agent to call a modification tool. Off-script behavior that only triggers read-only tools (searching, looking up information) is NOT corruption.

#### 2. Premature Ending
The user ended the conversation before the agent could complete necessary modification tools. This is NOT flagged if the agent encountered an error, said it could not help, or was stuck/unhelpful for multiple turns — in those cases the user is correct to end the call.

#### 3. Missing Information
The user failed to provide information from their goal that the agent explicitly asked for, preventing a necessary modification tool call. NOT flagged if the agent never asked for the information.

#### 4. Duplicate Modifications (User Looping)
The user repeatedly made the same request in a loop, causing the agent to call the same modification tool multiple times when it should have been called once. NOT flagged if the agent handled the looping correctly without extra modification calls.

#### 5. Decision Tree Violation
The user violated a specific instruction in their decision tree (negotiation behavior, edge cases, escalation behavior, resolution/failure conditions) AND this caused a modification tool to be called with incorrect parameters. Examples: accepting an option that didn't meet must-have criteria, ignoring an edge case instruction (e.g., accepting a standby flight when told to reject standby). NOT flagged if the agent only presented options that failed to meet criteria and the user had no correct option available.

### Scoring

- **Scale**: Binary (0 or 1)
  - 1 (Clean): The user's behavior did not corrupt the agent evaluation. None of the corruption types occurred.
  - 0 (Corrupted): One or more corruption types occurred — the user's behavior caused the agent to be evaluated against an incorrect database state.
- **Normalization**: Score equals the rating directly (0.0 or 1.0)

## Example Output

**Example 1: Clean (1)**
```json
{
  "name": "user_behavioral_fidelity",
  "score": 1.0,
  "normalized_score": 1.0,
  "details": {
    "rating": 1,
    "corrupted": false,
    "corruption_analysis": {
      "extra_modifications": {"analysis": "User stayed on-goal throughout. All modification tool calls were related to the rebooking request.", "detected": false},
      "premature_ending": {"analysis": "User ended the call after the agent confirmed the rebooking was complete.", "detected": false},
      "missing_information": {"analysis": "User provided confirmation number and last name when asked.", "detected": false},
      "duplicate_modifications": {"analysis": "No repeated requests that caused duplicate tool calls.", "detected": false},
      "decision_tree_violation": {"analysis": "User correctly rejected the standby option per their instructions and accepted the direct flight that met criteria.", "detected": false}
    }
  }
}
```

**Example 2: Corrupted — Premature Ending (0)**
```json
{
  "name": "user_behavioral_fidelity",
  "score": 0.0,
  "normalized_score": 0.0,
  "details": {
    "rating": 0,
    "corrupted": true,
    "corruption_analysis": {
      "extra_modifications": {"analysis": "No off-script modification requests.", "detected": false},
      "premature_ending": {"analysis": "The agent found a suitable flight and was about to call change_reservation, but the user said 'okay thanks bye' and ended the call before the modification could be completed.", "detected": true},
      "missing_information": {"analysis": "User provided all requested information.", "detected": false},
      "duplicate_modifications": {"analysis": "No looping behavior observed.", "detected": false},
      "decision_tree_violation": {"analysis": "No instruction violations detected.", "detected": false}
    }
  }
}
```

**Example 3: Corrupted — Decision Tree Violation (0)**
```json
{
  "name": "user_behavioral_fidelity",
  "score": 0.0,
  "normalized_score": 0.0,
  "details": {
    "rating": 0,
    "corrupted": true,
    "corruption_analysis": {
      "extra_modifications": {"analysis": "No off-script requests.", "detected": false},
      "premature_ending": {"analysis": "Conversation completed normally.", "detected": false},
      "missing_information": {"analysis": "All information provided.", "detected": false},
      "duplicate_modifications": {"analysis": "No duplicate calls.", "detected": false},
      "decision_tree_violation": {"analysis": "User's instructions specified to reject standby flights, but user accepted a standby option. This caused change_reservation to be called with a standby flight, which should not have happened. The user had a correct action available (rejecting the option and asking for alternatives).", "detected": true}
    }
  }
}
```

## Related Metrics

- [conversation_valid_end.md](conversation_valid_end.md) - Validates conversation completed

## Implementation Details

- **File**: `src/eva/metrics/validation_metrics/user_behavioral_fidelity.py`
- **Class**: `UserBehavioralFidelityMetric`
- **Base Class**: `ConversationTextJudgeMetric`
- **Prompt location**: `configs/prompts/judge.yaml` under `judge.user_behavioral_fidelity`
- **Configuration options**:
  - `judge_model`: LLM model to use (default: "gpt-5.2")
  - `temperature`: LLM temperature (default: 0.0)
  - `batch_size`: Number of records to process in parallel (default: 50)
