# Conciseness

> **Experience Metric**: Verbose responses don't affect accuracy but degrade the voice experience — users can't skim spoken output the way they can scan text.

## Overview

LLM-based metric that evaluates whether assistant responses are appropriately concise for voice interaction. In voice, users must process information in real time — they cannot skim, re-read, or scroll back. Verbose or information-dense responses increase cognitive load, making it harder for users to retain key details and respond appropriately. This metric assesses whether responses are brief and to the point without filler or rambling, appropriately scoped for spoken delivery to minimize cognitive load (typically 2-4 sentences), free of overwhelming lists or excessive detail that a listener cannot absorb in real time, and conversationally appropriate (not cramming too much into one turn).

### Capabilities Measured

- **Language Model**: Is the model's output appropriately brief and scoped for voice interaction? This is a text generation quality concern.

## How It Works

### Evaluation Method

- **Type**: Judge (LLM-as-judge)
- **Model**: GPT-5.2
- **Granularity**: Per-turn (each assistant turn rated independently, within a single LLM call)

### Input Data

Uses `conversation_trace` from MetricContext (via `format_transcript`), which includes user turns, assistant turns, tool calls, and tool responses for full context.

### Audio-Native vs Cascade

Uses `conversation_trace`, where user turns are transcribed text in cascade but intended text in audio-native systems. Since this metric **only evaluates assistant turns** (user turns are context only), the difference in user text source has **negligible impact** on scoring.

### Evaluation Methodology

The judge identifies which failure modes are present when rating < 3:

1. **verbosity_or_filler** — Unnecessary wording, repetition within the same turn, hedging
2. **excess_information_density** — Too many distinct facts/options/numbers at once
3. **over_enumeration_or_list_exhaustion** — Reading out long lists instead of summarizing
4. **contextually_disproportionate_detail** — More explanation than the situation warrants

**Allowed exceptions** (not penalized):
- Phonetic confirmation of codes (NATO alphabet) when user misheard
- Delivery of essential reference codes (vouchers, booking references)
- Slightly longer end-of-call wrap-up with recap/confirmation
- Essential information (confirmation codes, voucher numbers) regardless of length
- Interrupted/truncated content caused by user interruptions

### Scoring

- **Scale**: 1-3 (integer rating per turn)
  - 3: Highly Concise — clear, appropriately scoped for voice, no cognitive overload
  - 2: Adequate — one minor failure mode present, but still processable
  - 1: Not Concise — significant failure modes that hinder comprehension in voice
- **Normalization**: `(rating - 1) / 2` → 3→1.0, 2→0.5, 1→0.0
- **Aggregation**: Mean across all evaluated assistant turns (turns with no assistant content are excluded)

## Example Output

```json
{
  "name": "conciseness",
  "score": 2.6,
  "normalized_score": 0.8,
  "details": {
    "aggregation": "mean",
    "num_turns": 5,
    "num_evaluated": 5,
    "per_turn_ratings": {"0": 3, "1": 3, "2": 2, "3": 3, "4": 2},
    "per_turn_explanations": {"0": "Clear greeting...", "1": "Concise confirmation..."},
    "per_turn_failure_modes": {"2": ["excess_information_density"]},
    "mean_rating": 2.6
  }
}
```

## Related Metrics

- [conversation_progression.md](conversation_progression.md) - Evaluates conversation flow efficiency
- [speakability.md](speakability.md) - Checks voice-friendliness of text

## Implementation Details

- **File**: `src/eva/metrics/experience/conciseness.py`
- **Class**: `ConcisenessJudgeMetric`
- **Base Class**: `ConversationTextJudgeMetric`
- **Prompt location**: `configs/prompts/judge.yaml` under `judge.conciseness`
- **Configuration options**:
  - `judge_model`: LLM model to use (default: "gpt-5.2")
  - `aggregation`: Aggregation method for per-turn scores (default: "mean")
