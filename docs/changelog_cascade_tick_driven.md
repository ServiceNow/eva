# Working log: cascade tick-driven adapter (Plan 3)

Plan: `docs/superpowers/plans/2026-08-06-cascade-tick-driven-adapter.md`

Implemented Tasks 1-8 on `worktree-user-sim-phase-3`, branched off
`feat/cascade-user-simulator` (Plan 1 + Plan 2 as merged there). Task 9's steps
1-4 were run live; step 6 was blocked externally and step 5 was deliberately
deferred — see the last two sections, which also record two design errors in the
plan that only a live run exposed.

## Deviations from the plan, and why

### The worktree was branched off `main`, not off the plan's branch

`.claude/worktrees/user-sim-phase-3` was created at `e0041e3d` (main), so none of
Plan 1's or Plan 2's code was present — `src/eva/user_simulator/cascade/` did not
exist. Reset the worktree branch to `feat/cascade-user-simulator` (`3cb17e37`)
before starting. Nothing was lost; the branch had no commits of its own.

### Tests in this worktree import the *main* checkout's `src/`

The editable install resolves `eva` to `/Users/tara.bogavelli/third_eva/EVA-Bench3/src`
under pytest, so a bare `pytest` in the worktree silently tests the other agent's
Plan 2 working tree instead of these changes. Every test run here used
`PYTHONPATH=<worktree>/src`. Anyone re-verifying this branch must do the same or
the results are meaningless.

### `TickDrivenAdapter` subclasses `RealtimeWSAdapter` instead of being a peer

The plan sketched a standalone class calling `RealtimeWSAdapter._pcm16k_to_mulaw8k`
as though it were static. It is not: both resamplers carry per-instance
`audioop.ratecv` filter state, and calling them unbound would either crash or
produce discontinuous audio. Subclassing reuses the handshake, receive loop,
error propagation, speech-event frames and resamplers, and overrides only the
timing — which is the one thing this plan is actually about.

Consequences: `perturbator` is accepted (and inherited) rather than rejected, so
the framework-selection site in `simulator.py` passes it unconditionally instead
of using the plan's `**({...} if adapter_cls is RealtimeWSAdapter else {})`
conditional. Perturbation is applied exactly as on the real-time path (ambient
noise replaces the silence a quiet tick already sends) — see design error 2 below
for why the plan's "emit nothing when silent" rule had to go.

### Nothing armed a barge-in, so Task 8 would have been dead code

Task 8 specified the adapter half and the server half but no caller. The
interruption is enqueued as *audio* by `_play_interruption` and only reaches the
wire on a later tick, so `run_tick(barge_in=True)` had no natural call site.
Added `TickScheduler.arm_barge_in()`: it marks the next tick that actually puts
caller audio on the wire, so the truncation carries the played position as of
*that* tick rather than as of the decision. `_play_interruption` arms it on both
of its enqueue paths (speculative and freshly generated). Without this the
truncate frame would never have been sent.

### Truncation target read off the audio delta, not `response.output_item.added`

The plan said to track `active_item_id` wherever `response.output_item.added` is
handled — that case does not exist in `openai_realtime_server.py`. The audio
delta event carries the same `item_id` and is already dispatched, so the item is
recorded there and no new event handler was added.

Verified the SDK entry point rather than trusting the plan's spelling:
`openai.resources.realtime.realtime.AsyncRealtimeConversationItemResource.truncate(*,
audio_end_ms, content_index, item_id, event_id)`. The call matches. It is wrapped
in a `try/except` that logs and returns — a failed truncation should degrade
fidelity, not kill the conversation.

### `USER_ACTIVE_GUARD_S` removal

Removed the constant and `_last_user_audio_mono` as the plan specified, replaced
by `USER_ACTIVE_GUARD_DELTAS = 25` plus `note_user_audio()` /
`note_assistant_delta()` / `user_recently_active()`. The stale comment block that
documented the old constant was rewritten rather than left pointing at a name
that no longer exists.

### Test-fixture adaptations

- `AgentConfig` requires `name` and `role`, which the plan's fixture omitted, and
  `tool_module_path` must resolve to a real module (`eva.assistant.tools.itsm_tools`);
  `test_tools` does not exist. `ToolExecutor` also reads `scenario_db_path` at
  construction, so the fixture writes an empty `db.json`.
- The plan's adapter tests queued inbound frames without calling `start()`, so no
  receive loop existed to drain them. They now `start()` and `_settle()`.
- `1s` of mulaw resamples to 31998 PCM bytes, not 32000 — `ratecv` filter state
  eats two samples. The plan's `[BYTES_PER_TICK] * 5` assertion is unreachable;
  the test now asserts four full ticks plus a short-but-nonzero fifth, and that
  every *released* chunk is exactly tick-sized (silence-padded).
- `Adapter.run_tick` gained the `barge_in` keyword, so the three test doubles
  implementing the ABC were widened to match, and `_InterruptScheduler` gained
  `arm_barge_in`.
- `test_worker_uses_configured_factory_and_timeout` pins the exact
  `create_user_simulator` kwargs, so `framework=` was added to its expectation.

## Verification actually performed

`PYTHONPATH=<worktree>/src uv run pytest tests/unit -q` → **2119 passed, 52
skipped, 3 xfailed**. `pre-commit run --all-files` → all hooks pass.

New tests: 1 base-server pacing, 6 openai-realtime (pacing both ways, delta-based
activity guard, truncation target/no-op/item tracking), 4 `TickResult`, 8
tick-driven adapter (unpaced send, per-tick release, silent tick still sends a
full tick, played position, buffered ticks release without delay, quiet tick waits
out the grace, early return mid-grace, barge-in position, barge-in discard),
2 scheduler barge-in arming, 4 framework selection, 3 worker pacing.

## Two design errors in the plan, found only by running it

The unit tests all passed while the path was completely non-functional. Both
faults were invisible to them because both are about what happens across *many*
ticks against a real provider.

### 1. Removing all pacing left nothing to wait for the provider

First live run ended after 160ms with `reason: timeout` and an empty transcript.
`max_ticks = timeout * 1000 / TICK_DURATION_MS` ticks ran to exhaustion instantly:
the real-time adapter's 200ms per-tick floor was the only thing that had ever
given the assistant time to produce anything, and the plan removed it without
replacement.

Added `QUIET_TICK_GRACE_S` (one tick, 200ms) and `_await_tick_of_audio()`: a tick
waits for a full tick of assistant audio, returning the moment it is available.
This is not pacing — it never delays audio that has already arrived, so a provider
generating faster than real time is still drained as fast as it produces and
caller compute still costs the conversation nothing. It only bounds how long "the
assistant has said nothing yet" takes to establish. The wait is driven by an
`asyncio.Event` set from a new `_on_inbound_audio()` hook on `RealtimeWSAdapter`
(a no-op there).

### 2. "Nothing is emitted on a silent tick" was backwards

Second live run reached a real turn — assistant greeted, caller replied at tick 23
— and then died at the 40s liveness check with the assistant never answering.

The plan asserts that emitting nothing freezes the assistant, and treats that as
the mechanism. Emitting nothing does freeze it, but it also starves the provider's
VAD of the trailing silence that *ends the caller's turn*, so no response is ever
generated. The freeze this plan is built on comes from the caller not calling
`run_tick` while it thinks; it does not require, and is actively broken by, a tick
that puts nothing on the wire. That is why the real-time adapter sends a full tick
every tick.

`run_tick` now sends a full tick of audio — real or silence, perturbation applied
as on the real-time path — with no pacing sleeps and no minimum tick duration.
The plan's `test_nothing_is_emitted_on_a_silent_tick` was inverted accordingly,
and its two "no wait" tests were reframed: they now assert that ticks with audio
*already buffered* release without delay, plus new tests that a quiet tick waits
out the grace and that a tick returns early when audio arrives mid-grace.

## Task 9 — steps 1-4 and 6 run; step 5 not run

One ITSM record (record 1), `gpt-realtime-mini`, behaviors off, metrics enabled.
`--metrics=` was used only for the first diagnostic run.

**Steps 1-2 (artifacts, tools, DB).** Tick-driven run completed, ended `goodbye`,
7 turns. Full artifact set produced and identical to the baseline's:
`audit_log.json`, `transcript.jsonl`, `audio_user.wav`, `audio_assistant.wav`,
`audio_user_clean.wav`, `audio_mixed.wav`, `framework_logs.jsonl`,
`pipecat_metrics.jsonl`, `initial_scenario_db.json`, `final_scenario_db.json`.
Audit log holds 2 `tool_call` / 2 `tool_response` entries
(`verify_employee_auth`, `attempt_account_unlock`), and the DB diff is exactly the
intended mutation: `active_directory.locked True -> False`, `lock_reason
too_many_attempts -> None`, plus the two session-auth fields.

**Step 3 (latency).** Compared against the same record on the same framework with
`TICK_DRIVEN_FRAMEWORKS` temporarily emptied — a like-for-like real-time baseline,
which the ElevenLabs run could not provide:

| | tick-driven | real-time |
|---|---|---|
| completed / end reason | yes / goodbye | yes / goodbye |
| `model_response` p50 | 765 ms | 780 ms | 
| `model_response` mean (n) | 823 ms (4) | 1178 ms (2) |
| tools called | both | both |

p50 is within 2%. The mean gap is small-n noise (4 samples vs 2), not distortion —
the measurement window (caller stops → first assistant byte) contains no caller
compute either way.

**Step 4 (turn ordering).** Timestamps monotonic on both paths. The tick-driven
transcript alternates cleanly; the real-time baseline has two caller turns landing
before the assistant's reply block and answers "which system?" *before* the caller
says "Active Directory" — i.e. tick-driving visibly improved ordering, which is the
direction this plan predicts. The duplicated/concatenated assistant entries appear
**identically on the baseline**, so they are a pre-existing `openai_realtime`
transcript artifact, not something this plan introduced.

**Step 6 (ElevenLabs regression).** Could not be completed: the ElevenLabs
simulator failed to connect to ElevenLabs' own service
(`EOFError: connection closed while reading HTTP status line`) on all three
attempts, before any of this plan's code ran. External credential/connectivity
issue, unrelated to these changes. The paced server branch was instead exercised
by the Step 3 baseline run, which used `paced_output=True` and completed normally
— so the pacing path is verified working, just not with the ElevenLabs caller.

**Validation "failure" on every run, including the baseline and the ElevenLabs
attempt**, is `user_speech_fidelity` erroring with
`Unable to load vertex credentials from environment`. Environmental, identical
across all paths. `conversation_valid_end` and `user_behavioral_fidelity` both
scored 1.0 on the tick-driven run.

**Step 5 (interruption fidelity) NOT run**, by decision. It needs Plan 2's
interruption path to be stable — still being fixed in the other worktree, whose
changelog lists defect 4b (the interrupt check preempting ordinary turn-taking) as
an open design question. Running it before 4b is settled would measure Plan 2's
turn-routing bug rather than this plan's clock. It remains the deciding evidence
for porting `gemini_live` and `grok_voice`; expected result is `slip_ms == 0` and
`dropped == false` tick-driven against non-zero slip and some drops real-time.
