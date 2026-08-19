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

- [x] Task 1: behavior constants and vocabularies
- [x] Task 2: behavior flags on the config
- [x] Task 3: check-tick predicate
- [x] Task 4: decision prompts (+ `current_text()`)
- [x] Task 5: decision checks
- [x] Task 6: phrase cache
- [x] Task 7: backchannel behavior
- [x] Task 8: streaming TTS
- [x] Task 9: reactive interruption
- [x] Task 10: self-correction
- [x] Task 11: ambient noise mixing
- [x] Task 12: speculative generation
- [ ] Task 13: live ablation verification

## Further deviations found during implementation

### Task 10's ordering had to be rebuilt for the separate-call design

With a separate follow-up call the turn call has already produced the *correct*
utterance, so asking for "a correction" would have flipped a right answer into a wrong
one — inverting the plan's invariant and putting `must_have_criteria` at risk. Instead
the correction prompt asks for a deliberately WRONG variant of the already-generated
line; the slip is spoken as the turn and the model's original goal-consistent line is
armed as the correction. Wrong-then-right ordering is preserved exactly, and the
goal-consistent line is still what lands last. Certainty: high.

### Task 11's tests contradicted Plan 1's always-send-silence invariant

The plan's `run_tick` edit made sending conditional (`if outgoing: await send(...)`) and
its test asserted `media == []` with no perturbator. `RealtimeWSAdapter` deliberately
sends a full tick of frames every tick — real audio or synthesized silence — because
"gaps with no frames at all are what caused turn detection to misfire". Implemented as
mixing noise *into* what is already sent, never as gating whether to send. Tests assert
the invariant instead. Certainty: high — this is documented in the adapter's own docstring.

### `_warn_unsupported_perturbation` removed

It warned that cascade drops `background_noise`/`snr_db`/`connection_degradation`.
Task 11 makes all three take effect via `AudioPerturbator.apply()`, so the warning became
a false statement rather than a stale one. Certainty: high.

### Decision prompts are loaded with `get_template`, not a round-tripped placeholder

Plan Task 7 Step 5 suggested calling `get_prompt(..., conversation_history="{conversation_history}")`
so the placeholder survives for `ListenerDecisions` to fill later, and flagged that it might
not round-trip. `PromptManager.get_template()` returns the raw unformatted template, which
removes the failure mode entirely. Certainty: high.

### Interruption consumes the transcript rather than peeking at it

The plan's `_play_interruption` appends `buffer.current_text()` to history but leaves it in
the buffer, so the same assistant prefix is appended again at the next ordinary turn.
Consumed instead, and logged once via `_on_assistant_speaks`. Certainty: high.

### Plan 1's `cascade_` guard test rewritten

`test_no_cascade_specific_prompts_remain_in_the_prompt_file` asserted the substring
`cascade_` never appears in `simulation.yaml`. The invariant it protected is that no
cascade-only contract is layered onto the *turn call's* system prompt. Plan 2's prompts are
used only in their own standalone calls, so the test now asserts the real invariant:
`_messages()[0]["content"] == _build_prompt()`. Certainty: high — strictly stronger guard.

## Defects found by the Task 13 ablation runs

Four defects. All four surfaced only against live services; the unit suite was green
throughout, which is the same pattern the Plan 1 handoff warned about.

### 1. A backchannel consumed the caller's turn — FIXED (cb834b75)

`TickScheduler.run_tick` set `_awaiting_reply = True` on *any* outgoing audio. A continuer
earns no reply, so `may_take_turn()` blocked permanently and the call died at the
inactivity timeout. Live: 7/8 conversations vs 4/9 at baseline. Fixed with
`enqueue_backchannel()`, which tracks continuer bytes at the head of the playout queue.

**Loose end, deliberately not fixed:** post-fix the backchannel run still ends 6
timeout / 3 goodbye against baseline's 4/5, and the backchannel count fell 20 -> 6
between runs unexplained. Candidate second mechanism: a backchannel still resets
`_ticks_since_caller_speech`, delaying the caller's own next turn by the full
`WAIT_TO_RESPOND_SELF_MS`. Unproven — may be n=9 noise. Do not call defect 1 closed.

### 2. Slip was structurally unmeasurable — FIXED (7dba2ba7)

`interrupt_slip_ms` differenced tick counters, but `run_tick` is only pumped by the `_run`
loop and `_play_interruption` is awaited from inside it, so `scheduler.tick` provably cannot
advance during the generation. All 525 logged interruptions reported `slip_ms=0` with
`intended_tick == actual_tick`, and `should_drop_interrupt`'s slip branch never engaged.
Now measured with `time.monotonic()`. Certainty: high — deductive, not statistical.

### 3. Self-correction was unreachable — FIXED (7dba2ba7)

`random.Random(0)` was reseeded identically per conversation. Its first draw below
`SELF_CORRECTION_RATE` (0.15) is #26, while a conversation runs ~7 turns, so no conversation
ever armed a correction — zero events across both configs that enabled it. Verified the CLI
flag *did* propagate (`config.json` showed `enable_self_correction: true`) before blaming the
RNG, specifically to avoid fixing the wrong layer. Now seeded from the record id via
`crc32`, keeping runs reproducible while differing across conversations.

### 4. Degraded interruption runs — root cause was NOT what it first looked like

Initial reading of the aggregates (62-107 interruptions against ~2.6 caller turns) suggested
runaway barge-ins and a missing cooldown. **That diagnosis was wrong.** Reading a single
conversation timeline end to end showed two different causes:

**4a. The interruption path swallowed the hang-up — FIXED (7dba2ba7).** `_play_interruption`
did `utterance, _end_call = extract_turn(message)`, discarding the flag, while `_take_turn`
used it. A caller deciding to hang up mid-assistant-turn therefore emitted nothing (the model
called the tool instead of speaking) and the call could never end. Observed directly: after
`'Thanks. Goodbye.'` at tick 640, six interruptions with empty text at ticks 660-800 while
the assistant looped "Confirmed... your account is unlocked" and then "Sure. What can I help
you with?". This — not barge-in spam — is what depressed `task_completion` to 0.500.

**4b. The interrupt check preempts ordinary turn-taking — NOT FIXED, design question.**
The gaps between logged "interruptions" were 10s, 18s, 50s, and their content was ordinary
replies ("Employee ID is E M P zero four eight two seven one"). These are normal turns
routed through the interruption path, not barge-ins, which is why `caller_turn` appeared to
collapse — the turns were relabelled. Structurally: `_run` only reaches `_take_turn` when the
assistant is silent, but the check fires while it is speaking, so with interruptions enabled
the caller preferentially speaks via the check at 2s granularity instead of waiting out the
1s silence gate. Knock-on effects: turns taking the interrupt path bypass
`_maybe_arm_self_correction` and `_prerender_candidate`, and log as `interruption` rather
than `caller_turn`, which affects metric turn numbering. Deciding how the check and the turn
gate should interact is a design change to Plan 2, left for the author.

### Method note

Both times an initial diagnosis was wrong (the "service outage" that was really machine
sleep, and the "interruption storm" that was really a swallowed hang-up), the error came from
reading *aggregate counts* and inferring a mechanism. Both were settled immediately by
reading *one sequence end to end*. Prefer a single full timeline over a summary table.

### Still outstanding

- 4b (above), and defect 1's unexplained residual.
- `_run_checks` awaits two LLM calls inline in the tick loop with no timeout, so a slow
  decision provider stalls the wire. Fixing it means moving the checks off the tick loop.
- Task 13 steps 2-6 need re-running: step 5 has never produced data, and the `interrupt` and
  `all-on` rows from the last pass aggregate retry attempts (12 event files, not 9).
