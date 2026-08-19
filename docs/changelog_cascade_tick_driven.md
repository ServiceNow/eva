# Working log: cascade tick-driven adapter (Plan 3)

Plan: `docs/superpowers/plans/2026-08-06-cascade-tick-driven-adapter.md`

Implemented Tasks 1-8 on `worktree-user-sim-phase-3`, branched off
`feat/cascade-user-simulator` (Plan 1 + Plan 2 as merged there). Task 9 is live
end-to-end verification and has **not** been run — see the last section.

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
conditional. Perturbation is applied only to ticks that carry real caller audio —
mixing ambient noise into a stalled tick would emit frames and unfreeze the
assistant, defeating the whole mechanism.

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

`PYTHONPATH=<worktree>/src uv run pytest tests/unit -q` → **2117 passed, 52
skipped, 3 xfailed**. `pre-commit run --all-files` → all hooks pass.

New tests: 1 base-server pacing, 6 openai-realtime (pacing both ways, delta-based
activity guard, truncation target/no-op/item tracking), 4 `TickResult`, 8
tick-driven adapter (unpaced send, per-tick release, silent-tick emits nothing,
played position, no tick-duration floor, barge-in position, barge-in discard),
2 scheduler barge-in arming, 4 framework selection, 3 worker pacing.

## Task 9 — NOT DONE

Every step needs live paid runs against OpenAI Realtime plus the cascade caller's
STT/LLM/TTS providers, and step 5 additionally needs Plan 2's interruption path to
be stable — it is still being fixed in the other worktree, and that changelog
lists defect 4b (the interrupt check preempting ordinary turn-taking) as an open
design question. Running the fidelity comparison before 4b is settled would
measure Plan 2's turn-routing bug, not this plan's clock.

Step 5 is the one that matters: it is the entire justification for this plan and
the evidence for or against porting `gemini_live` and `grok_voice`. Expected
result is `slip_ms == 0` and `dropped == false` on the tick-driven path against
non-zero slip and some drops on the real-time path.
