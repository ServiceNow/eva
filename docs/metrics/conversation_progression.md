# Conversation Progression

> **Experience Metric**: A conversation that loops or repeats itself wastes the user's time even if the final outcome is correct.

## Overview

LLM-based metric that evaluates whether the assistant effectively moved the conversation forward without redundancy. It assesses whether the assistant makes consistent progress toward the user's goal, avoids repetitive tool calls with identical parameters, avoids redundant statements or questions, retains context across turns, and asks clarifying questions only when needed.

### Capabilities Measured

- **Language Model**: Does the model avoid repeating itself, retain context across turns, and make consistent forward progress toward the user's goal?

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2
- **Granularity**: Conversation-level (single rating for the whole conversation)

### Input Data

Uses `conversation_trace` from MetricContext (via `format_transcript`), which includes user turns, assistant turns, tool calls, and tool responses.

### Audio-Native vs Cascade

Uses `conversation_trace`, where user turns are transcribed text in cascade but intended text in audio-native systems. This difference can **subtly affect scoring**: in cascade, the judge sees what the assistant actually received (which may include STT errors that justify repeated questions), while in audio-native systems, the judge sees the clean intended text (so repeated questions may look less justified). The prompt acknowledges that speech recognition errors may cause repeated requests, which is expected behavior in voice interfaces and not penalized — but this mitigation is more relevant for cascade where STT errors are visible in the trace.

### Evaluation Methodology

The judge evaluates four independent dimensions (each flagged as true/false):

1. **unnecessary_tool_calls** — Repeated/unjustified tool calls (same params, no new info)
2. **unnecessary_questions** — Asking for information already provided or obtainable from tools
3. **unnecessary_repetition** — Restating information already communicated
4. **information_loss_from_interruption** — Critical info lost due to interruption and never restated

This metric evaluates **conversation efficiency only**. It does NOT evaluate whether the assistant followed policies or acted faithfully — those are faithfulness concerns. Even if a policy violation affected flow, it is only flagged here if the assistant's conversational choices (questions, repetition, tool calls) were themselves inefficient.

### Scoring

- **Scale**: 1-3 (integer rating)
  - 3: No progression issues — consistent forward movement, no repeated calls/questions
  - 2: Minor issues — some unnecessary questions or repeats, but conversation still progresses
  - 1: Significant issues — fails to move forward, repeatedly asks for provided info, or critical context loss
- **Normalization**: `(rating - 1) / 2` → 3→1.0, 2→0.5, 1→0.0

## Example Output

```json
{
  "name": "conversation_progression",
  "score": 2.0,
  "normalized_score": 0.5,
  "details": {
    "rating": 2,
    "explanation": {
      "dimensions": {
        "unnecessary_tool_calls": {"evidence": "Agent called search_flights twice with identical parameters (origin=JFK, destination=LAX, date=2026-03-25) without any new information between calls.", "flagged": true},
        "unnecessary_questions": {"evidence": "Agent asked for the confirmation number once and the user provided it. No repeated questions.", "flagged": false},
        "unnecessary_repetition": {"evidence": "Agent restated the flight options summary after the user had already selected an option, but this was minor and did not block progress.", "flagged": false},
        "information_loss_from_interruption": {"evidence": "No interruptions caused loss of critical information.", "flagged": false}
      },
      "flags_count": 1
    },
    "num_turns": 12
  }
}
```

## Related Metrics

- [conciseness.md](conciseness.md) - Evaluates response brevity (per-turn)
- [speakability.md](speakability.md) - Checks voice-friendliness of text

## Implementation Details

- **File**: `src/eva/metrics/experience/conversation_progression.py`
- **Class**: `ConversationProgressionJudgeMetric`
- **Base Class**: `ConversationTextJudgeMetric`
- **Prompt location**: `configs/prompts/judge.yaml` under `judge.conversation_progression`
- **Configuration options**:
  - `judge_model`: LLM model to use (default: "gpt-5.2")
