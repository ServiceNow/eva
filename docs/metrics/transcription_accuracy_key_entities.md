# Transcription Accuracy Key Entities

> **Diagnostic Metric**: Helps isolate whether task failures are caused by the STT mishearing critical entities (names, dates, codes) vs. agent reasoning errors — not scored directly since entity errors surface through faithfulness and task completion.

## Overview

LLM-based metric that evaluates STT transcription accuracy for key entities across user turns. It identifies whether critical information was correctly transcribed from user audio, focusing on entities that matter for task completion: names, dates and times, confirmation codes, flight numbers, amounts, prices, addresses, phone numbers, and email addresses.

### Capabilities Measured

- **Speech Recognition**: Measures whether the STT pipeline correctly captured critical entities (names, dates, confirmation codes, amounts) that directly affect task completion.

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2
- **Granularity**: Per-turn (each user turn evaluated for entity accuracy)

### Input Data

Uses the following MetricContext fields:
- `intended_user_turns`: What the user simulator intended to say (reference)
- `transcribed_user_turns`: What the assistant's STT transcribed (hypothesis)

The judge receives both texts side by side and identifies key entities to compare.

### Audio-Native vs Cascade

- **Cascade**: Fully applicable — measures whether the assistant's STT correctly captured key entities, which directly affects downstream tool calls and responses.
- **Audio-native (AUDIO_LLM / S2S):** **Skipped entirely** (`supported_pipeline_types = {CASCADE}`). Audio-native models receive raw audio, not STT output, so entity-level STT accuracy is not meaningful. Entity perception issues in audio-native systems are captured instead by `faithfulness` (which treats mishearing as a faithfulness violation).

### Evaluation Methodology

- Entity must be present (not missing)
- Value must match (minor formatting variations OK)
- Numeric equivalence recognized ("150" = "one hundred fifty")
- Date equivalence recognized ("December 15th" = "Dec 15")

### Scoring

- **Scale**: 0.0-1.0 per turn (ratio of correct entities)
  - Turns with no key entities are marked as not applicable and excluded from aggregation
- **Normalization**: Already 0-1 scale (correct entities / total entities)
- **Aggregation**: Mean across all turns that have key entities

## Example Output

```json
{
  "name": "transcription_accuracy_key_entities",
  "score": 0.875,
  "normalized_score": 0.875,
  "details": {
    "aggregation": "mean",
    "num_turns": 3,
    "num_evaluated": 3,
    "num_not_applicable": 0,
    "per_turn_ratings": {"1": 1.0, "2": 0.5, "3": 1.0},
    "per_turn_entity_details": {
      "2": {
        "entities": [
          {"type": "amount", "value": "$150", "transcribed_value": "$115", "correct": false},
          {"type": "flight_number", "value": "SW100", "transcribed_value": "SW100", "correct": true}
        ]
      }
    }
  }
}
```

## Related Metrics

- [stt_wer.md](stt_wer.md) - Word-level STT accuracy (complementary)
- [user_speech_fidelity.md](user_speech_fidelity.md) - Upstream: checks if user TTS audio matches intended text

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/transcription_accuracy_key_entities.py`
- **Class**: `TranscriptionAccuracyKeyEntitiesMetric`
- **Base Class**: `TextJudgeMetric`
- **Prompt location**: `configs/prompts/judge.yaml` under `judge.transcription_accuracy_key_entities`
- **Configuration**: `judge_model` (default: "gpt-5.2"), `aggregation` (default: "mean")
