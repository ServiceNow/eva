# Agent-Timeout-on-User-Turn Classification

## Overview

Adds a deterministic gate that distinguishes **agent-side failures** from **retry-worthy
failures** when a conversation ends without reaching `goodbye`. Records where the user
spoke last on an `inactivity_timeout` are now classified as real agent failures (the agent
did not respond to the user's final turn) and are **not** retried. This prevents wasting
compute on repeat failures that are guaranteed to repro, while preserving retry behavior
for genuine transient issues (infra errors, assistant-last timeouts).

## Motivation

Realistic conditions like background noise on user audio (see
`docs/changelog_perturbation_feature.md`) expose STT/VAD failure modes — fragmented
transcripts, silent-final-turn, premature LLM fires — that cause the assistant to miss
the user's final utterance. The conversation then drifts into inactivity and is marked
`error/failed` in the CSV, which the orchestrator previously treated as a transient
failure and retried. These retries almost always reproduce the same failure, since the
underlying cause is an agent perception bug, not flakiness.

Investigation on one run (`2026-04-18_18-14-25.851887_nova-3_gpt-5.4-mini_aura-2`) found
18 / 19 non-goodbye records were agent-side perception failures, 1 was an ElevenLabs
SSL infra failure. Only the latter is a legitimate retry candidate.

## Classification rule

For each record that did not end with `goodbye`, the gate reads
`conversation_ended_reason` from `result.json` and decides:

| Reason | `last_audio_speaker` | Outcome |
|---|---|---|
| `goodbye` | — | Gate passes; validation metrics run |
| `inactivity_timeout` | `user` | Gate passes; `agent_timeout_on_user_turn = True` on the context; validation metrics run; record is **treated as passed** at the validation layer. The agent-side failure is surfaced in `metrics.json` (`context.agent_timeout_on_user_turn`), not as a validation failure. **Not retried** (passed records are not retried). |
| `inactivity_timeout` | `assistant` | Retry (`not_finished`) |
| `error` or anything else | — | Retry (`not_finished`) |

**Why passed, not failed:** an agent-timeout-on-user-turn record *has* usable data —
transcripts, audio timestamps, tool calls all exist. The agent's failure to close the
conversation is an agent-behavior bug, not a validation failure. Keeping it in the
"passed" bucket preserves the real validation-metric scores in `metrics.json` (instead
of clobbering them with a synthetic `conversation_finished = 0.0`) and keeps
`ValidationResult.failure_category` as a clean three-value enum
(`passed | validation_failed | not_finished`). Downstream analysis filters on
`context.agent_timeout_on_user_turn` to isolate these records.

`last_audio_speaker` is computed from `audio_timestamps_user_turns` and
`audio_timestamps_assistant_turns` on the metrics processor's `_ProcessorContext`: the
role whose audio interval ends latest wins. Using real-time audio-end timestamps avoids
the delayed-text-event ordering issue where the `assistant_speech` log entry lands
*after* the user's next `user_speech` entry because the user simulator buffers the
assistant audio before transcribing it.

## Flow comparison

**Before:**

```
STEP 1  run conversations
STEP 2  fast gate: check_conversation_finished (goodbye?)
          yes → finished_ids
          no  → not_finished_ids  ──────── retry
STEP 3  ValidationRunner on finished_ids
          fail → failed_validation_ids ─── retry
```

**After:**

```
STEP 1  run conversations
STEP 2  segregate exceptions / incomplete results   → not_finished (retry)
STEP 3  ValidationRunner on everything else — OWNS the gate:
          per record:
            fast: check_conversation_finished (goodbye?) → gate passes
            else: read result.json conversation_ended_reason
              inactivity_timeout + user last  → gate passes + flag
              anything else                   → not_finished (retry)
          run metrics on gate-passed records, reusing cached contexts
          ValidationResult.failure_category ∈
            {passed, validation_failed, not_finished, agent_timeout_on_user_turn}
```

Agent-timeout records are treated as passed at the validation layer. They are not
added to the retry queue (passed records are never retried) and do not generate a
`rerun_history` entry. The agent bug is discoverable through
`metrics.json → context.agent_timeout_on_user_turn`.

## Why the gate moved into `ValidationRunner`

The gate decision needs `audio_timestamps_*_turns`, which is only produced by the metrics
processor. Splitting the gate out of `ValidationRunner` would require a duplicate
`MetricsContextProcessor.process_record()` call (once in the gate, once in
`MetricsRunner._load_context`). To avoid that, the refactor:

1. Gave `ValidationRunner` ownership of the gate plus the processor call.
2. Added `preprocessed_contexts: dict[record_id, _ProcessorContext]` to `MetricsRunner`.
   When set, `_load_context` returns the cached context instead of re-processing.
3. `ValidationRunner` forwards its gate-computed contexts through this cache, so each
   record is processed exactly once per validation attempt.

The fast path still uses `check_conversation_finished` (one-line file read) for goodbye
records — the processor only runs for non-goodbye records that need the audio-timestamp
check.

## Artifact surfaces

- `metrics.json` → `metrics.conversation_correctly_finished` — new diagnostic metric. Standard
  `MetricScore` shape (`score`, `normalized_score`, `details`, `error`). `score = 1.0`
  when the agent responded on every user turn; `score = 0.0` when the conversation
  ended with `inactivity_timeout` and the user was the last speaker (agent failed to
  respond to the user's final turn). `details` carries `conversation_ended_reason`,
  `last_audio_speaker`, and a human-readable `reason`. Filter records with
  `metrics.conversation_correctly_finished.score == 0.0` to isolate agent-timeout records.
  Registered in `src/eva/metrics/diagnostic/` and marked
  `exclude_from_pass_at_k = True` per the diagnostic convention.
- **No stored flag on the processor context or `MetricContext`.** Both the gate and
  the diagnostic metric call the pure helper
  `is_agent_timeout_on_user_turn(conversation_ended_reason, audio_ts_user, audio_ts_asst)`
  in `src/eva/metrics/processor.py` — one source of truth, no duplicated state.
- `ValidationResult` exposes `passed: bool` + `failed_metrics: list[str]`. When
  `passed=False`, empty `failed_metrics` means "gate rejected / not finished"; populated
  `failed_metrics` means "metrics ran and some fell below threshold." Agent-timeout
  records pass the gate (`passed=True`) — the agent-side failure is surfaced via the
  diagnostic metric, not as a validation failure.

## Files changed

| File | Change |
|---|---|
| `src/eva/metrics/processor.py` | `last_audio_speaker()` + `is_agent_timeout_on_user_turn()` pure helpers. Processor populates `conversation_finished` and `conversation_ended_reason` on the context. No stored `agent_timeout_on_user_turn` attribute. |
| `src/eva/metrics/diagnostic/conversation_correctly_finished.py` | New diagnostic metric exposing the classification; 1.0 = responded, 0.0 = agent-timeout-on-user-turn. |
| `src/eva/metrics/diagnostic/__init__.py` | Registers `conversation_correctly_finished`. |
| `src/eva/metrics/runner.py` | `process_records()` public phase-1 API; `run(contexts=...)` accepts pre-computed contexts. |
| `src/eva/orchestrator/validation_runner.py` | Thin classifier: goodbye → gate pass; `is_agent_timeout_on_user_turn(...)` → gate pass + treated as `vr.passed=True`; else → not_finished. |
| `src/eva/orchestrator/runner.py` | Routes records by `vr.passed` + `vr.failed_metrics` only. No `agent_timeout_on_user_turn` branch. |

## Edge cases

- **STT noise hallucination (e.g. run `2026-04-18_…` record 5.1.2):** user stopped
  speaking, agent kept generating TTS on hallucinated transcripts. The processor's user
  turn-5 audio interval extends to the final run timestamp, so `last_audio_speaker`
  returns `user` → correctly classified as agent failure.
- **User intent without audio (record 2.2.5):** simulator emitted `user_speech` text but
  never streamed audio for it. No matching `audio_end elevenlabs_user`, so
  `last_audio_speaker` returns `assistant` → classified as retry. This disagrees with a
  raw ElevenLabs-event heuristic that would include `user_speech` text events. Accepted
  trade-off: using audio timestamps keeps the rule framework-agnostic.
- **Infra errors (record 7.2.6):** `conversation_ended_reason == "error"` short-circuits
  to `not_finished` without invoking the processor (which would raise on empty
  `audit_log.json` anyway).

## Verification

- Unit tests: `tests/unit/orchestrator/test_validation_runner.py` — new fixture helper
  `_write_goodbye_fixture` creates the minimal `elevenlabs_events.jsonl` so mocked-
  metrics tests still pass the gate. All 20 tests green.
- End-to-end: running the new gate on
  `output/2026-04-18_18-14-25.851887_nova-3_gpt-5.4-mini_aura-2/records` classifies the
  19 non-goodbye records as 17 `agent_timeout_on_user_turn` (no retry) + 2
  `not_finished` (2.2.5 asst-last, 7.2.6 infra error) — matching manual inspection.
- Debug: `python scripts/debug_metrics_on_failed_records.py` dumps the full
  `_ProcessorContext` per record under
  `<run_dir>/debug/metrics_on_failed_records/per_record/` for deep inspection.
