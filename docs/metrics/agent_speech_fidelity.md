# Agent Speech Fidelity

> **Accuracy Metric**: If the agent's spoken audio doesn't match what it intended to say, the user receives incorrect information regardless of how good the text reasoning was.

## Overview

Audio-based metric that evaluates whether the assistant's **spoken audio** accurately represents the intended text, using Gemini for multimodal analysis. This metric evaluates the speech output regardless of how it was produced — whether by a separate TTS engine or generated directly by an audio-native model. Specifically, it checks that all words from the intended text are present (no missing words), no extra words were added (no insertions), words are spoken correctly (no substitutions), and key entities are accurately conveyed (dates, names, numbers, codes, addresses).

### Capabilities Measured

- **Speech Synthesis**: Measures whether the TTS engine (cascade) or the model's direct audio generation (audio-native) accurately produces the intended text as spoken audio.

## How It Works

### Evaluation Method

- **Type**: Audio Judge (multimodal LLM with audio input)
- **Model**: Gemini 3.1 Pro
- **Granularity**: Per-turn (each assistant turn evaluated independently)

### Input Data

Uses the following MetricContext fields:
- `audio_assistant_path`: Path to assistant-only audio file
- `intended_assistant_turns`: What the assistant intended to say

### Audio-Native vs Cascade

The evaluation is the same in both cases — compare `intended_assistant_turns` against the actual spoken audio. The only difference is where the intended text comes from:

- **Cascade**: The intended text is the input to the TTS engine (i.e., the LLM's text output).
- **Audio-native (S2S, S2T+TTS):** The intended text is the text output that the audio-native model returns alongside its generated speech.

### Evaluation Methodology

The judge compares intended text against spoken audio, focusing on:

- **TTS-critical entities**: Names, dates, times, codes, dollar amounts, flight numbers — these are the highest-priority items
- **Error types**: Missing words, added words, wrong words, entity errors

**Special handling:**
- Minor pronunciation variations that don't change meaning are acceptable
- Filler words (um, uh) that don't affect core content are ignored
- Interruption tags (e.g., `[likely cut off by user]`, `[assistant interrupts]`) are non-spoken metadata in the intended text — words in regions flagged by these tags as likely not spoken are not penalized
- Missing words at END of LAST turn only are not penalized (audio cutoff)

### Scoring

- **Scale**: 0-1 (binary per turn)
  - 1: High Fidelity — audio accurately says all words from intended text
  - 0: Low Fidelity — missing, added, or wrong words detected
- **Normalization**: Already 0-1 scale
- **Aggregation**: Mean across all assistant turns

## Example Output

```json
{
  "name": "agent_speech_fidelity",
  "score": 0.875,
  "normalized_score": 0.875,
  "details": {
    "aggregation": "mean",
    "num_turns": 7,
    "num_evaluated": 7,
    "per_turn_ratings": {"0": 1, "1": 1, "2": 1, "3": 0, "4": 1, "5": 1, "6": 1},
    "per_turn_explanations": {
      "3": "Missing word: intended 'flight SW102' but audio said 'flight SW12'. Key entity error."
    }
  }
}
```

## Related Metrics

- [user_speech_fidelity.md](user_speech_fidelity.md) - Same metric for the user simulator side
- [faithfulness.md](faithfulness.md) - Faithfulness for the text layer: evaluates whether the assistant's responses are grounded in instructions, policies, and tool results
- [speakability.md](speakability.md) - Checks if text is voice-friendly (upstream concern)

## Implementation Details

- **File**: `src/eva/metrics/accuracy/agent_speech_fidelity.py`
- **Class**: `AgentSpeechFidelityMetric`
- **Base Class**: `SpeechFidelityBaseMetric` → `AudioJudgeMetric`
- **Prompt**: `configs/prompts/judge.yaml` under `judge.agent_speech_fidelity`
- **Configuration**: `audio_judge_model` (default: Gemini 3.1 Pro), `aggregation` (default: "mean")
