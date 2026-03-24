# User Speech Fidelity

> **Validation Metric**: If the user simulator's TTS garbled the intended speech, the agent received bad input — evaluating its performance against that conversation would be unfair.

## Overview

Audio-based validation metric that evaluates whether the user simulator's **spoken audio** accurately represents the intended text, using Gemini for multimodal analysis. The user simulator always uses a TTS engine, so this measures TTS quality directly. It validates simulation quality by checking that all words from the intended text are present (no missing words), no extra words were added (no insertions), words are spoken correctly (no substitutions), and key entities are accurately conveyed (dates, names, numbers, codes).

## How It Works

### Evaluation Method

- **Type**: Audio Judge (multimodal LLM with audio input)
- **Model**: Gemini 3.1 Pro
- **Granularity**: Per-turn (each user turn evaluated independently)

### Input Data

Uses the following MetricContext fields:
- `audio_user_path`: Path to user-only audio file
- `intended_user_turns`: What the user simulator was instructed to say

### Scoring

This metric uses a 1-3 scale instead of binary 0-1 (like agent speech fidelity) because it is a validation metric — its purpose is to flag simulations that need to be re-run, not to score agent performance. A binary scale was too severe: minor TTS artifacts that don't affect agent comprehension would unnecessarily invalidate otherwise usable simulations. Only turns rated 1 (poor fidelity) indicate that the simulation should be re-run.

- **Scale**: 1-3 per turn
  - 3: High Fidelity — all words correct including key entities
  - 2: Acceptable — minor issues that don't affect understanding
  - 1: Poor Fidelity — missing, added, or wrong words affecting key information (re-run simulation)
- **Special Handling**:
  - Minor pronunciation variations acceptable
  - Filler words (um, uh) ignored if they don't affect content
  - Missing words at END of LAST turn only not penalized (audio cutoff)
- **Normalization**: `(rating - 1) / 2` → 3→1.0, 2→0.5, 1→0.0
- **Aggregation**: Mean across all user turns

## Example Output

```json
{
  "name": "user_speech_fidelity",
  "score": 2.75,
  "normalized_score": 0.875,
  "details": {
    "aggregation": "mean",
    "num_turns": 8,
    "num_evaluated": 8,
    "per_turn_ratings": {"1": 3, "2": 3, "3": 3, "4": 1, "5": 3, "6": 3, "7": 3, "8": 3},
    "per_turn_explanations": {
      "4": "Missing word: intended 'my flight SW100' but audio said 'my flight SW10'. Key entity error."
    }
  }
}
```

## Related Metrics

- [agent_speech_fidelity.md](agent_speech_fidelity.md) - Same evaluation for the assistant side
- [user_behavioral_fidelity.md](user_behavioral_fidelity.md) - Validates user simulation behavior

## Implementation Details

- **File**: `src/eva/metrics/validation_metrics/user_speech_fidelity.py`
- **Class**: `UserSpeechFidelityMetric`
- **Base Class**: `SpeechFidelityBaseMetric` → `AudioJudgeMetric`
- **Prompt location**: `configs/prompts/judge.yaml` under `judge.user_speech_fidelity`
  - Uses the same speech fidelity prompt structure as `agent_speech_fidelity` but with `evaluation_mode="user"` and user turns
- **Configuration options**:
  - `audio_judge_model`: LLM model (default: Gemini 3.1 Pro)
  - `aggregation`: Aggregation method (default: "mean")
