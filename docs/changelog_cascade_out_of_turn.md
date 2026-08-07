# Working log: cascade out-of-turn behaviors (Plan 2)

Plan: `docs/superpowers/plans/2026-08-06-cascade-out-of-turn-behaviors.md`
Handoff: `docs/handoff_cascade_plan2.md`

## Deviations from the plan, decided up front

### Structured fields come from a separate follow-up call, not a JSON turn contract

The plan's Tasks 10 and 12 read `self_correction` and `next_interruption` out of a JSON
turn response. That contract does not exist: Plan 1 removed it because demanding "a single
JSON object and nothing else" suppressed the `end_call` tool call and hung conversations
until timeout.

Instead the turn call keeps its exact current shape (plain text + `END_CALL_TOOL`), and a
second call — made *after* the turn's audio is already queued — asks only for the extra
field. Reasoning:

- It structurally cannot regress `end_call`, because the turn call is untouched.
- It costs no conversational latency. The correction is not needed until
  `SELF_CORRECTION_DELAY_MS` (1200ms) after the assistant *starts replying*, which is
  itself well after the caller's audio went out.
- The user additionally required that a self-correction never attach to a final turn, so
  arming is gated on `end_call` being false.

Certainty: high. This is strictly safer than the plan's version and the latency argument is
structural, not empirical.

### Re-added `TranscriptBuffer.current_text()`

Deleted as dead code at the end of Plan 1; Tasks 7, 9, and 12 all call it. Restored with the
`[CURRENTLY SPEAKING, INCOMPLETE]` marker the backchannel prompt's few-shot examples depend on.

Certainty: high — the prompt examples are meaningless without the marker.

### Task 13 stages explicitly rather than `git add -A`

The handoff forbids `git add -A` at the repo root (many unrelated untracked files).

## Progress

- [ ] Task 1: behavior constants and vocabularies
- [ ] Task 2: behavior flags on the config
- [ ] Task 3: check-tick predicate
- [ ] Task 4: decision prompts (+ `current_text()`)
- [ ] Task 5: decision checks
- [ ] Task 6: phrase cache
- [ ] Task 7: backchannel behavior
- [ ] Task 8: streaming TTS
- [ ] Task 9: reactive interruption
- [ ] Task 10: self-correction
- [ ] Task 11: ambient noise mixing
- [ ] Task 12: speculative generation
- [ ] Task 13: live ablation verification
