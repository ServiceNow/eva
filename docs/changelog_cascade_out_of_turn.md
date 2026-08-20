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
`_maybe_arm_self_correction` and `_prerender_candidate`, so enabling interruptions silently
disables part of the other two behaviours. They also log as `interruption` rather than
`caller_turn` — which does **not** affect metrics (verified: nothing under `src/eva/metrics/`
reads `caller_turn`, and turns are numbered from `audio_start(simulated_user)`, which fires on
both paths), only the human-readable event log and the ablation analysis. Deciding how the check and the turn
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

---

## Session 2: further defects, and the measurement that matters

### 5. inactivity_timeout measured cumulative, not contiguous, silence — FIXED (abd1a42a)

**A Plan 1 bug, not a Plan 2 one, and it has distorted every run including baseline.**

`_assistant_is_inactive` was only ever reached on ticks where the assistant was *silent*,
because the speech branch `continue`d before it. That made its own reset line unreachable in
production, so `_ticks_assistant_silent` accumulated quiet ticks across the whole call and
never reset. "Silent for 120s" therefore meant "quiet ticks have totalled 120s at some point",
which any long healthy conversation eventually satisfies.

Measured live: one conversation was killed for `inactivity_timeout` after **73.7s** of actual
contiguous silence, in a 197.6s call.

The unit test that appeared to cover this passes because it calls the method *directly* with a
speech tick — exercising a branch the real loop never reaches. Worth remembering as a shape:
a test can cover a line and still not cover the path.

Now called on every tick. Certainty: high — mechanism read from source and confirmed against
event timestamps.

### 6. Assistant turn boundaries were any 200ms gap — FIXED (abd1a42a)

The interruption cap allows one barge-in per assistant turn, but "new turn" was implemented as
"a single tick with no assistant audio". A pause between sentences therefore re-armed the cap
mid-utterance. Now uses `is_new_assistant_turn`, requiring the same 1s (`WAIT_TO_RESPOND_OTHER_MS`)
the turn gate already applies. Note this also means earlier per-turn rate arithmetic used
*speech segments* as the denominator, which inflated it.

### 7. The opener was emitted before the content existed — FIXED (abd1a42a)

Plan 2's design plays a pre-rendered opener the instant the decision fires, to hide ~1s of
generation latency. But that commits the caller to barging in before knowing whether it has
anything to say. Both a hang-up and a stale drop then left an orphaned "Actually—" on the wire.

Content is now generated *first*; the opener and content are queued together only if the line
is worth speaking. Dropping now costs zero audio. The cost is that the barge-in lands ~1s after
the decision rather than ~200ms — which is what makes the slip measurement below meaningful
rather than cosmetic.

### The interrupt rate gate: added, then removed (97448885, then 3cb17e37)

Recorded because the reasoning is the useful part. A rate gate was added on the belief that the
decision "always said YES". **It does not.** Measured: in one call the check was asked ~101
times and answered YES 14 times (~14%) — correctly NO through most of each assistant turn, YES
once near the end. The defect was *where* it fired, not how often: "the caller has heard enough
to reply" and "the assistant is finishing" are nearly the same instant, so barge-ins displaced
ordinary turn-taking rather than being excessive.

So the gate was a random suppressor discarding decisions that were correct, masking a judgement
problem. Passing the caller's goal to the decision (1d139c07) addresses the judgement at source,
which made the gate a second suppressor stacked on a fix. Removed. The per-turn eligibility flag
remains — that is the one-barge-in-per-turn cap, which is structurally right regardless.

### Goal-aware interrupt decision (1d139c07)

`summarize_goal` passes high-level goal, must/nice-to-have criteria, option-evaluation steps,
and the resolution/failure/escalation conditions. Field meanings live in the prompt template so
they stay static across calls; the block sits ahead of the conversation history so it is
prompt-cacheable. `edge_cases` and `information_required` are omitted — long, and about how to
answer questions rather than whether the goal is finished. Renders ~605 tokens on a real ITSM
record, of which `negotiation_behavior` is ~60%; drop that field first if cost bites.

## The headline result: reactive interruption does not work on the real-time path

Run: 3 ITSM records x 3 trials, concurrency 3 (deliberately low — slip is a wall-clock
measurement and contention would inflate it), no rate gate, generate-first.

| | |
|---|---|
| barge-ins fired | 35 (across 75 assistant speech segments, ~47%) |
| kept | **4 (11%)** |
| dropped, slip > 1500ms | **31 (89%)** |
| dropped, assistant already stopped | 0 |
| slip_ms | min 1295, **median 1902**, max 2991 |

Median generation latency is **1902ms against a 1500ms staleness budget**, and every single
drop was for lateness — not one because the assistant had finished. On `RealtimeWSAdapter`,
reactive interruption fires and then fails to land roughly nine times in ten.

This is exactly the number Task 13 step 3 was designed to produce and never could while slip
was measured on a frozen tick counter. The plan states that a high drop rate is "the signal to
reconsider tick-driving more frameworks (Plan 3)" — that signal has now arrived quantified. On
a tick-driven adapter the assistant is frozen while the caller thinks, so slip vanishes and all
35 would land.

**Before concluding the design is unworkable, try a fast `decision_llm`.** It currently defaults
to `user-llm`, a full-size model, and both the check and the content generation run on it.
A small fast model could plausibly halve slip. Untested.

## Still outstanding

- **Task 13 remains the only incomplete task.** Steps 2-6 all need re-running against the
  current build: every earlier number describes code that no longer exists.
- Step 5 (self-correction validity) has **never** produced data, and it is the check the plan
  calls the most important — it is also where this implementation departs furthest from the plan.
- 4b (the interrupt check preempting ordinary turn-taking) — still a design question.
- Defect 1's unexplained residual (backchannel timeouts still above baseline).
- `_run_checks` awaits two LLM calls inline in the tick loop with **no timeout**, so a slow
  decision provider stalls the wire.
- The last interrupt run still showed 5 `inactivity_timeout` and 2 `unknown` of 10 *after* the
  contiguous-silence fix. Cause unknown — deliberately not guessed at.
- `gpt-5.4` has two deployments in `.env`, one with a stale `sk-svcacct-` key, so it fails
  preflight at random. Pre-existing and unrelated; ablations use `gpt-5.2` to avoid it.

## Method notes worth keeping

- Every diagnosis that turned out wrong came from reading **aggregate counts** and inferring a
  mechanism. Every one was settled immediately by reading **one sequence end to end**. Prefer a
  single full timeline over a summary table.
- Scripted text-surgery on source caused a fourth defect this session: `t.index("def test_...")`
  matched inside `async def` and silently stripped the keyword, breaking test collection. The
  handoff already recorded three from the same cause. Use targeted edits.
