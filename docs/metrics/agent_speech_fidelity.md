# Agent Speech Fidelity

> **Accuracy Metric**: If the agent's spoken audio garbles or misstates key information, the user receives incorrect information regardless of how good the text reasoning was.

## Overview

Audio-based metric that evaluates whether the assistant's **spoken audio** accurately represents the **key entities** (dates, names, numbers, codes, addresses, etc.), using an audio LLM for multimodal analysis.

To keep the EVA score **apples-to-apples across all pipeline setups**, the same entity-focused metric runs for every pipeline type — cascade, S2S, and audio-LLM. It does not require any intended text.

### Capabilities Measured

- **Speech Synthesis**: Measures whether the assistant's spoken audio accurately represents the key entities.

## How It Works

### Evaluation Method

- **Type**: Audio Judge (multimodal LLM with audio input)
- **Model**: Gemini 3 Flash
- **Granularity**: Per-turn (each assistant turn evaluated independently)

### Input Data

Uses the following MetricContext fields:
- `audio_assistant_path`: Path to assistant-only audio file
- `conversation_trace`: User utterances and tool responses are kept as-is (the sources of the entities to listen for); assistant turns are **redacted** to a placeholder so the judge evaluates articulation, not whether the agent gave the "right" answer.

### Evaluation Methodology

The judge receives the agent audio plus a redacted conversation trace and, for each assistant turn, checks whether the spoken audio accurately represents the key entities: Names, dates, times, codes, dollar amounts, flight numbers, etc. Turns with no entities to evaluate are flagged (`has_entities = false`) and **excluded** from scoring.

### Scoring

- **Scale**: 0-1 (binary per turn)
  - 1: Entities clearly articulated
  - 0: An entity is unclear, garbled, or wrongly articulated
- **Normalization**: Already 0-1 scale
- **Aggregation**: Mean across scored assistant turns (turns with no entities are skipped)

## Example Output

```json
{
  "name": "agent_speech_fidelity",
  "score": 0.875,
  "normalized_score": 0.875,
  "details": {
    "aggregation": "mean",
    "num_turns": 7,
    "num_evaluated": 4,
    "num_skipped_no_entities": 3,
    "per_turn_ratings": {"0": 1, "1": 1, "3": 0, "5": 1},
    "per_turn_has_entities": {"0": true, "1": true, "2": false, "3": true},
    "per_turn_explanations": {
      "3": "Confirmation code unclear: heard 'ZK three F F' but trace shows 'ZK3FFW'."
    }
  }
}
```

## Related Metrics

- [tts_fidelity.md](tts_fidelity.md) - Stricter, word-for-word diagnostic metric, only for pipelines that expose intended text
- [faithfulness.md](faithfulness.md) - Faithfulness for the text layer: evaluates whether the assistant's responses are grounded in instructions, policies, and tool results
- [speakability.md](speakability.md) - Checks if text is voice-friendly (upstream concern)

## Implementation Details

- **File**: `src/eva/metrics/accuracy/speech_fidelity.py`
- **Class**: `SpeechFidelityMetric`
- **Base Class**: `SpeechFidelityBaseMetric` → `AudioJudgeMetric`
- **Prompt**: `configs/prompts/judge.yaml` under `judge.agent_speech_fidelity`
- **Configuration**: `audio_judge_model` (default: Gemini 3 Flash), `aggregation` (default: "mean")
