# STT WER

> **Diagnostic Metric**: Provides a broad word-level view of STT quality to complement entity-level accuracy — useful for diagnosing systemic STT issues, but not scored directly since individual word errors may not affect the outcome.

## Overview

Deterministic metric that measures Speech-to-Text (STT) transcription accuracy using Word Error Rate (WER). It evaluates the quality of the assistant's STT transcription by comparing what the user simulator intended to say to what the assistant's STT transcribed, computing word-level errors (substitutions, deletions, insertions), and providing detailed per-turn error analysis.

### Capabilities Measured

- **Speech Recognition**: Directly measures the quality of the speech-to-text pipeline by comparing transcribed output against intended text at the word level.

## How It Works

### Evaluation Method

- **Type**: Deterministic (uses jiwer library)
- **Granularity**: Per-turn with conversation-level aggregation

### Input Data

Uses the following MetricContext fields:
- `intended_user_turns`: What the user simulator intended to say (reference text)
- `transcribed_user_turns`: What the assistant's STT transcribed (hypothesis text)

### Audio-Native vs Cascade

- **Cascade**: Fully applicable — measures the quality of the assistant's STT pipeline, which directly affects the LLM's input.
- **Audio-native (AUDIO_LLM / S2S):** **Skipped entirely** (`supported_pipeline_types = {CASCADE}`). Audio-native models receive raw audio, not STT transcripts, so measuring STT accuracy is not meaningful. The `transcribed_user_turns` field in audio-native systems comes from a secondary transcription service, not the model's actual input.

### Evaluation Methodology

Before comparison, both reference and hypothesis texts go through a normalization pipeline:
- Unicode character conversion
- Digits-to-words conversion
- Whisper-based text normalization
- Apostrophe normalization
- Single letter collapsing
- Number suffix handling

### Scoring

- **Scale**: 0.0-∞ (WER, unbounded but typically 0.0-1.0)
  - 0.0: Perfect transcription
  - 0.1: 10% of words have errors
  - >1.0: More errors than reference words (excessive insertions)
- **Normalization**: `1 - WER` (clamped to 0-1). Lower WER → higher normalized score.
- **Algorithm**: Levenshtein distance at word level via jiwer:
  `WER = (Substitutions + Deletions + Insertions) / Total Reference Words`

## Example Output

```json
{
  "name": "stt_wer",
  "score": 0.045,
  "normalized_score": 0.955,
  "details": {
    "wer": 0.045,
    "accuracy": 0.955,
    "language": "en",
    "num_turns": 8,
    "per_turn_wer": {"1": 0.0, "2": 0.1, "3": 0.0, "4": 0.05},
    "per_turn_errors": {
      "2": {
        "substitutions": [{"ref": "confirmation", "hyp": "conformation"}],
        "deletions": ["please"],
        "insertions": ["um"]
      }
    },
    "error_summary": {
      "top_substitutions": [{"error": "confirmation → conformation", "count": 2}],
      "top_deletions": [{"word": "please", "count": 3}],
      "top_insertions": [{"word": "um", "count": 5}]
    },
    "total_substitutions": 3,
    "total_deletions": 5,
    "total_insertions": 8
  }
}
```

## Related Metrics

- [transcription_accuracy_key_entities.md](transcription_accuracy_key_entities.md) - Entity-level accuracy (complementary)
- [user_speech_fidelity.md](user_speech_fidelity.md) - Upstream: checks if user TTS matches intended text

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/stt_wer.py`
- **Class**: `STTWERMetric`
- **Base Class**: `CodeMetric`
- **Configuration**: `language` (default: "en")
