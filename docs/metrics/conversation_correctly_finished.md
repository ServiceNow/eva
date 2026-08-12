# Conversation Correctly Finished

> **Diagnostic Metric**: Flags records where the agent failed to respond to the user's final turn, helping isolate agent-side timeouts/no-response failures from other conversation outcomes. Not directly used in final evaluation scores; excluded from pass@k.

## Overview

Deterministic diagnostic metric that flags records where the conversation ended with `inactivity_timeout` and the user was the last speaker by audio timeline — i.e. the agent failed to respond to the user's final turn. Conversations that ended normally, errored out, or ended on inactivity but with the assistant as the last speaker all score 1.0.

### Capabilities Measured

- **Language Model / Pipeline**: Whether the agent produced a response to the user's final turn. A 0.0 score indicates the agent went silent after the user spoke (e.g. generation stalled, tool call hung, or the model chose not to respond) and the session was closed by the inactivity timer.

## How It Works

### Evaluation Method

- **Type**: Deterministic (metadata + audio timeline inspection)
- **Granularity**: Conversation-level

### Input Data

Uses the following MetricContext fields:
- `conversation_ended_reason`: Reason the conversation terminated (e.g. `"goodbye"`, `"inactivity_timeout"`, `"error"`).
- `audio_timestamps_user_turns`: Per-turn `(start, end)` audio intervals for the user.
- `audio_timestamps_assistant_turns`: Per-turn `(start, end)` audio intervals for the assistant.

### Evaluation Methodology

1. Compute `last_audio_speaker` as whichever side (`"user"` or `"assistant"`) has the latest audio end-timestamp across all turns. Returns `None` if neither side recorded audio.
2. Flag the record as a missed turn if `conversation_ended_reason == "inactivity_timeout"` **and** `last_audio_speaker == "user"`.
3. Score 0.0 if flagged, else 1.0.

### Scoring

- **Scale**: Binary (0.0 or 1.0)
  - 1.0: Agent responded to every user turn (normal end, error, or inactivity with assistant as last speaker).
  - 0.0: Conversation ended with `inactivity_timeout` and the user was the last speaker.
- **Normalization**: Already 0-1 scale.
- **Aggregation**: Mean across records (excluded from pass@k).

## Example Output

```json
{
  "name": "conversation_correctly_finished",
  "score": 0.0,
  "normalized_score": 0.0,
  "details": {
    "conversation_ended_reason": "inactivity_timeout",
    "last_audio_speaker": "user",
    "reason": "conversation ended with inactivity_timeout and user was the last speaker"
  }
}
```

## Diagnostic Sub-Metrics

When a record fails (score 0.0), a deterministic classifier assigns **exactly one primary cause** for why the agent went silent, emitted as sub-metrics keyed `conversation_correctly_finished.<cause>_rate`.

**General logic:**
- **One cause per failure.** Causes are checked in a fixed priority order (infra errors first, since they invalidate the run) and the first match wins — causes never double-count. `unknown_reason` is the residual when nothing else matches.
- **Score convention (`_rate`)**: `1.0` = this cause occurred, `0.0` = it did not (lower is better). Every cause flag is emitted on every record (all `0.0` on a clean finish), so the cross-record mean reads as "fraction of conversations with this cause".
- **Signals come from the raw record files** (`agent_perf_stats.csv`, `logs.log`, `pipecat_logs.jsonl`, `user_simulator_events.jsonl`, `audit_log.json`). A missing file just means that cause can't fire — it's noted in `details.notes`, never an error.

**Causes** (in priority order):

| Cause | Detected when |
|-------|---------------|
| `tts_api_error` | `pipecat_logs.jsonl` error frame naming a `*TTSService` (infra → `invalid_run`) |
| `stt_api_error` | `pipecat_logs.jsonl` error frame naming a `*STTService` (infra → `invalid_run`) |
| `llm_api_error` | fatal LLM error line in `logs.log` (allowlist) with no response after it (infra → `invalid_run`) |
| `reasoning_too_long` | last `agent_perf_stats.csv` row: empty `response`, no tool call, model reasoned, `stop_reason == length` (ran out of tokens mid-reasoning) |
| `reasoning_only` | same, but `stop_reason != length` — reasoned (visible text **or** `reasoning_tokens > 0`) yet emitted no answer |
| `stt_empty_transcription` | cascade, last message from assistant, `VAD fired but no transcription` after the last response, plus a `User turn stopped - complete transcript: ''` line (turn closed empty) |
| `stt_missing_transcription` | same, but **no** `User turn stopped` line (turn never closed) |
| `ended_with_user_interruption` | an accepted non-empty user turn, then an interruption, with no response after it |
| `vad_no_turn_detected` | the user's final utterance came after the agent, but no VAD onset fired in its ±1.5s window |
| `unknown_reason` | none of the above (residual; a discovery queue for new causes) |

**Input-characteristic flags** are also emitted on failures. These are *orthogonal* — they describe the user's final utterance, not a cause, so they overlap each other and the causes and do **not** sum to 1.0.

| Flag (`…_rate`) | Fires when |
|-----------------|-----------|
| `final_turn_short` | 1–2 words, after stripping `[annotation]` tags |
| `final_turn_acknowledgement` | leads with `yes/no/ok/sure/…` **and** ≤ 6 words |
| `final_turn_spelled_entity` | a spelled ID/code/name (letter/digit runs ≥3 chars, NATO, "X as in Y", caps codes) that dominates the turn |

## Related Metrics

- [conversation_valid_end.md](conversation_valid_end.md) - Validates the conversation ended via the `end_call` tool (simulator-side quality gate).
- [response_speed.md](response_speed.md) - Measures per-turn latency between user-end and assistant-start.

## Implementation Details

- **File**: `src/eva/metrics/diagnostic/conversation_correctly_finished.py`
- **Class**: `ConversationCorrectlyFinishedMetric`
- **Base Class**: `CodeMetric`
- **Category**: `diagnostic`
- **`exclude_from_pass_at_k`**: `True`
- **Configuration**: None (deterministic)
- **Sub-metric classifier**: `src/eva/utils/conversation_correctly_finished/` — `classifier.py` (extract raw files into signals → fixed-priority cause selection → build sub-metric flags), `causes/*` (per-cause detect + extract), `final_turn.py` (input-characteristic flags).
