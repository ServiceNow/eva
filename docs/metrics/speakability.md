# Speakability

> **Diagnostic Metric**: This is a diagnostic metric because flagged issues may not actually cause problems for the TTS engine — some TTS systems handle markdown, missing spaces, or formatting gracefully. It surfaces potential concerns for manual review rather than definitive failures.

## Overview

LLM-based metric that evaluates whether assistant text is free of formatting and structural issues that could cause problems for TTS systems in a cascade pipeline. Specifically, it flags text containing elements a TTS engine may not handle correctly: markdown or visual formatting (bold, italic, headers, tables), non-spoken formatting (JSON brackets, repeated punctuation/symbols), missing spaces between words that would cause TTS to fail (e.g., "eighttwentypm"), and emojis.

### Capabilities Measured

- **Language Model**: Does the model generate text that is free of formatting artifacts (markdown, JSON, emojis) and suitable for TTS consumption? This is a text generation quality concern.

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2
- **Granularity**: Per-turn (each assistant turn evaluated independently)

### Input Data

Uses `intended_assistant_turns` from MetricContext — the text sent to the TTS engine.

### Audio-Native vs Cascade

- **Cascade**: Fully applicable — evaluates whether the LLM's text output is appropriate for TTS.
- **S2S:** **Skipped** (`supported_pipeline_types = {CASCADE, AUDIO_LLM}`). S2S models generate audio directly without a separate TTS step, so there is no intermediate text whose speakability can be evaluated. AUDIO_LLM models do have a TTS step and are evaluated.

### Evaluation Methodology

Any of the following violations result in a score of 0:

1. **Markdown / visual formatting**: Bold/italic (`**text**`, `*text*`), headers (`## Title`), markdown tables, repeated punctuation/symbols (`-----`, `*****`)
2. **Non-spoken formatting**: JSON with brackets, or other structured data formats
3. **Missing spaces**: Words run together that would cause TTS failure (e.g., "eighttwentypm" instead of "eight twenty PM"). Common acronyms are fine.
4. **Emojis**

### Scoring

- **Scale**: 0-1 (binary per turn)
  - 1: Voice-friendly — natural when spoken, no issues
  - 0: Voice-unfriendly — contains elements problematic for TTS
- **Normalization**: Already 0-1 scale
- **Aggregation**: Mean across all assistant turns

## Example Output

```json
{
  "name": "speakability",
  "score": 0.8,
  "normalized_score": 0.8,
  "details": {
    "num_turns": 5,
    "per_turn_ratings": {"0": 1, "1": 1, "2": 0, "3": 1, "4": 1},
    "per_turn_explanations": {
      "2": "Contains 'ASAP' acronym which sounds awkward when spoken. Should be 'as soon as possible'."
    },
    "mean_rating": 0.8
  }
}
```

## Related Metrics

- [conciseness.md](conciseness.md) - Evaluates response brevity
- [agent_speech_fidelity.md](agent_speech_fidelity.md) - Checks if TTS audio matches intended text (downstream)

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/speakability.py`
- **Class**: `SpeakabilityJudgeMetric`
- **Base Class**: `SingleTurnTextJudgeMetric`
- **Prompt location**: `configs/prompts/judge.yaml` under `judge.speakability`
- **Configuration options**:
  - `judge_model`: LLM model to use (default: "gpt-5.2")
