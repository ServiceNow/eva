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

#### WER calculation for new languages

For new languages, normalization rules are generated once per language via `add_culture_data.py` and stored in `src/eva/utils/wer_normalization/configs/`.

Languages within the **Romance** (French, Spanish, Italian, Portuguese, Romanian, …), **Indic** (Hindi, Bengali, Punjabi, …), and **Sino-CJK** (Mandarin, Cantonese, Japanese, Korean) families receive particularly thorough coverage because the normalization engine has explicit support for their structural patterns (vigesimal forms, reversed units, positional kanji arithmetic, etc.). Languages outside these families should still benefit from the generic pipeline.

A language config can also opt into three rules for patterns that only some languages need (see the [configs README](../../src/eva/utils/wer_normalization/configs/README.md)):
- `char_folds`: treat two spellings of a character as equal (Hindi nukta, German ü = ue).
- `join_hyphenated_compounds`: treat `Anmelde-Konto` and `Anmeldekonto` as the same word.
- `spoken_symbols`: the spoken word for a symbol in a spelled ID (French "trait d'union", German "Bindestrich") matches the written symbol.

#### Code-switching (Hindi)

Hindi speakers mix English words into Hindi ("मेरा flight कल है"). The intended text and the STT transcript often write those words differently: in a different form (`seven बजे` vs `7 बजे`, `I'm` vs `I am`) or in a different script (`मार्च` vs `march`). With Hindi rules alone, a correct transcript would be scored as wrong.

When a language config sets `code_switch_languages` (today only `hi`, with `["en"]`), the metric runs in two passes (`src/eva/utils/wer_normalization/code_switch.py`):

1. **Hindi alignment.** Both texts are normalized with the Hindi rules and aligned as usual. This gives `strict_wer`.
2. **English re-check of the errors.** Each error region (a run of mismatched words, up to 4 words per side) is compared again with an English view of the same text. In that view, Latin-script parts use the English rules ("twenty five" → 25, "I'm" → "i am") and Devanagari parts keep the Hindi rules. With `code_switch_phonetic: true` (on for `hi`), a single English word written in Devanagari on one side and in Latin on the other (`कन्फर्म` / `confirm`) also matches when the two words have the same consonants and a compatible first vowel. Native Hindi words that look like an English word (कमरा / camera) are excluded.

A region stops counting as an error only when the two sides match **and** the intended text is English there. If the intended text has the digit `7` and the STT writes "seven", the user may have said सात, so it stays an error. The STT side must be English too: intended "seven" vs STT सात is a translation and stays an error, even though both normalize to 7. So translation errors are not hidden. Errors can only go down: `wer` ≤ `strict_wer`.

Each intended word is also tagged by language: `en` for Latin script, a known short loanword (कोड) or a word the phonetic matcher rescued, `num` for digits, `hi` otherwise. Substitutions and deletions count against the intended word's language, insertions against the inserted word's. Digits are in neither `hi` nor `en`. An English word written in Devanagari that the STT also writes in Devanagari, and that is not in the short-loanword lexicon, is counted as Hindi, so the English share is an undercount.

Languages without `code_switch_languages` (all except `hi`) are scored exactly as before and have none of the fields below.

#### Diagnostic metric

`stt_wer` is a **diagnostic metric** — the benchmark runs and produces valid results even without any WER normalization config for a given language; the metric simply measures raw string distance in that case. You do not need to deeply curate the normalization to get useful benchmark output.

That said, the auto-generated configs can be brittle. The LLM generation step is creative and may miss edge cases, produce inconsistent vocabulary, or generate rules that work for the test cases but fail on real transcripts. **Manual inspection of the generated config is recommended** before treating WER scores as authoritative for a new language. Not all languages can fit in the current normalization framework, and some may require custom rules and code changes.

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

### Hindi (code-switching)

Turn 1: intended `मेरी फ़्लाइट seven बजे है`, STT `मेरी फ़्लाइट 7 बजे है`. Turn 2: intended `मुझे अपना कन्फर्मेशन कोड चाहिए`, STT `मुझे अपना confirmation कोड चाहिए`. Turn 3 has one real error (a dropped `हैं`). The standard fields are as above (omitted here); these are added:

```json
{
  "name": "stt_wer",
  "score": 0.062,
  "details": {
    "wer": 0.062,
    "language": "hi",
    "reference_words": 16,
    "strict_wer": 0.188,
    "code_switch": {
      "languages": ["hi", "en"],
      "code_switched_words": 4,
      "rescued_words": 2,
      "rescues_by_matcher": {"secondary_view": 1, "phonetic": 1},
      "per_language": {
        "hi": {"wer": 0.083, "ref_words": 12, "errors": 1},
        "en": {"wer": 0.0, "ref_words": 4, "errors": 0}
      }
    },
    "per_turn_code_switch_rescues": {
      "1": [{"expected": "seven", "actual": "7", "matcher": "secondary_view"}],
      "2": [{"expected": "कन्फर्मेशन", "actual": "confirmation", "matcher": "phonetic"}]
    }
  }
}
```

- `score` / `wer`: WER after the English re-check. `strict_wer`: WER with the Hindi rules alone.
- `code_switch.code_switched_words`: intended words tagged `en`. `rescued_words`: intended words that stopped counting as errors. `rescues_by_matcher`: number of rescued regions per matcher (`secondary_view` for the English rules, `phonetic` for the sound match).
- `code_switch.per_language`: WER per language tag. `wer` is `null` when the run has no intended words of that language.
- `per_turn_code_switch_rescues`: each rescue, so it can be checked by hand.
- Sub-metrics: `wer_hi`, `wer_en` (each omitted when the run has no intended words of that language), `code_switch_word_rate` (share of intended words that are English) and `code_switch_rescue_rate` (share of intended words rescued).

## Related Metrics

- [transcription_accuracy_key_entities.md](transcription_accuracy_key_entities.md) - Entity-level accuracy (complementary)
- [user_speech_fidelity.md](user_speech_fidelity.md) - Upstream: checks if user TTS matches intended text

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/stt_wer.py`
- **Class**: `STTWERMetric`
- **Base Class**: `CodeMetric`
- **Configuration**: `language` (default: "en")
