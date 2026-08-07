# Cascade STT reliability and provider abstraction

Working log for the fix to Defect B (verbatim caller repeats) and the LiveKit STT port.
Reasoning and certainty are recorded per change.

## Background: what actually broke

Live run `cascade-repro-1` produced three verbatim repeats of the same caller utterance.
The assistant noticed: *"I'm hearing the same phrase repeated, so the line may be
transcribing you incorrectly."*

Evidence chain:

- `user_simulator_events.jsonl` shows three consecutive `caller_turn` events (ticks 312,
  390, 491) with **no `assistant_speech` between them**, while `transcript.jsonl` proves
  the assistant spoke twice in that window, 14-20s before each repeat.
- The Scribe reconnect at 12:01:47 happened **after** all three repeats, so it is not the cause.
- The next successful commit contained **only** the 6th assistant utterance, with no trace
  of the two missing ones. A merely slow commit would have landed in `committed` and
  appeared prepended on the next `take_committed()`. It did not. So those utterances were
  **never committed at all** - this is not a read-too-early race.

Mechanism of the repeat itself is confirmed at `simulator.py:179`: `_take_turn` reads
`take_committed()`, and when it returns empty it still calls the caller LLM with unchanged
history. Same messages in, same utterance out.

## The reframing that drives this work

Provider-side turn detection was initially blamed. That was wrong, and the correction
matters for provider choice:

- `TickScheduler.may_take_turn()` decides when the caller speaks, from counting silent
  ticks of assistant **audio**. STT never participates. So provider endpointing cannot
  corrupt who-speaks-when.
- We already run `commit_strategy=manual` (no provider VAD) and hit the failure anyway.

The real root cause is upstream of any provider: **we read the transcript buffer
optimistically - no wait, no acknowledgement that the commit was processed, no fallback,
and no check that we heard anything at all.**

Three failure shapes at read time:

| State at read | Consequence |
|---|---|
| Nothing committed | Stale history -> verbatim repeat (Defect B) |
| Partially committed | Caller replies to half a sentence, silently. **Most insidious** |
| Fully committed | Correct |

## Changes

### 1. Bounded wait for the committed transcript

**Certainty: high.** Implemented by *skipping* the turn and retrying on later ticks rather
than a blocking sleep - which fits the tick architecture and costs nothing, since each
retry just re-reads the buffer. Bounded by a tick counter so it cannot wait forever.

Latency budget is ample: the caller's own LLM call takes ~10s observed, so absorbing a few
hundred ms of STT finalization is free.

### 2. Fall back to the in-flight partial

**Certainty: medium-high.** Approximate text beats stale text by a wide margin. Precedent:
tau-voice drives its interrupt/backchannel decisions off a linearly interpolated,
mid-word-truncating approximation (`get_proportional_text`, transcript_utils.py:7-25) and
that is good enough to have shipped.

Open question: whether partials were actually flowing in the failing run. `commit()` clears
`in_flight`, and reconnect clears it too. Instrumentation now logs `in_flight` on this exact
path, but the defect has not recurred in the runs since, so this is **unmeasured**.

### 3. Never generate a turn from unchanged history

**Certainty: high on the rule, medium on the remedy.** The rule - never call the caller LLM
with history that did not change - is unambiguous.

For the remedy, three options were considered:

- *Stall until text arrives* - converts a corrupt-transcript bug into a dead-air bug.
- *Speak anyway* - the current behavior, i.e. the defect.
- **Chosen: tell the caller it did not hear.** Inject an explicit note so it says "Sorry, I
  didn't catch that." This is what a real caller does, keeps the conversation alive, records
  the failure honestly in the transcript, and can never emit a stale repeat.

### 4-6. Provider-agnostic STT interface, LiveKit implementation, config + loss metric

**Certainty: medium.** Motivation is that LiveKit's base `RecognizeStream` provides
`flush()`/`end_input()` (livekit-agents `stt.py:349-571`) - a flush sentinel *in the stream*
rather than our out-of-band `{"commit": true}` flag - plus typed events and a uniform
interface across 28 providers. Adopting it deletes our hand-rolled reconnect/idle-close
machinery.

Caveats recorded before committing:

- **Dependency cost is the real risk.** Any LiveKit STT plugin pulls `livekit-agents` ->
  pinned `livekit==1.1.14`, a compiled native Rust/WebRTC wheel, plus `av` (FFmpeg
  bindings), `sounddevice` and OpenTelemetry - for a codepath that never opens a room.
  The `.venv.x86-broken/` directory in the tree makes this a concrete, not theoretical, risk.
- **Swappability is narrower than the catalogue suggests.** The uniform interface does not
  expose whether provider turn detection can be disabled. That varies per plugin and is only
  visible in each plugin's source.
- `ink-2` has **mandatory** turn detection (only `turn_start_threshold`,
  `turn_eager_end_threshold`, `turn_end_threshold`, `turn_end_timeout_ms`; no off switch)
  and is **English-only** per its docstring. `ink-whisper` has no interim results.
  Neither Cartesia model satisfies both requirements today.

The loss metric in change 6 exists so provider comparison is settled with data rather than
anecdote.

## LiveKit spike results (measured, not inferred)

Scratch venv, `livekit-agents` 1.6.8 + 4 plugins. One recorded 14s assistant utterance fed at
our 200ms tick cadence, then `flush()`.

| Provider | interims (R2) | `flush()` honored | auto-final | verdict |
|---|---|---|---|---|
| `elevenlabs/scribe_v2_realtime` | 14 | **yes, 0.15s** | none (endpointing off) | full manual control |
| `cartesia/ink-2` | 42 | **no - explicitly ignored** | +1.20s after speech end | provider-driven only |
| `deepgram/flux-general-en` | - | - | - | blocked: WS handshake **HTTP 402** |
| `deepgram/nova-3` | - | - | - | same closure, same account |
| `assemblyai/universal-streaming-multilingual` | - | - | - | no key in `.env` |

Cartesia logs it outright: `Cartesia STT stream.flush() was ignored.`

**The decisive number: ink-2 auto-finalizes +1.20s after speech ends, while
`WAIT_TO_RESPOND_OTHER_MS` lets the caller take its turn at 1.0s.** So the final lands ~200ms
*after* we read the buffer - the late-final case, by a small margin. A bounded wait of ~500ms
covers it comfortably. This makes ink-2 viable *provided* changes 1-3 land first.

**This also settles Defect B.** Scribe answers a commit in 150ms, so Defect B is an
intermittent dropout, not systematic slowness. That was the top falsification test - and it
says fix the race in place rather than redesign the transport. Changes 1-3 do exactly that
and are provider-independent.

Integration details learned (needed by the real implementation):

- Standalone use requires `async with livekit.agents.utils.http_context.open():` or an
  explicitly passed `aiohttp.ClientSession`; plugins otherwise raise "http session outside of
  a job context".
- Deepgram Flux is `deepgram.STTv2`, not `deepgram.STT` (which is the nova path).
- Install on arm64/py3.12: 74 packages, 268MB, all prebuilt wheels, no compilation. x86 CI
  still unverified.

## STT requirements these changes must preserve

1. The caller owns the turn boundary; STT reports what was said.
2. In-flight partials for Plan 2's interrupt/backchannel checks (only when those are on).
3. Audio-only - we are a client on a Twilio WebSocket, with no access to the assistant's text.
4. A self-hostable option (NVIDIA Riva/NIM is first-party and runs offline).
5. No silent transcript loss. Violated by Defect B; the reason for changes 1-3.
6. Non-English support - the simulator takes a `language` param.

## Related fixes landed alongside

- **Deadlock (separate defect, 2/7 runs).** Caller said goodbye, assistant treated it as
  terminal, `_awaiting_reply` never cleared, caller could never reach its `end_call` turn,
  run stalled 5 minutes to pipecat's idle timeout. Fixed with `ASSISTANT_UNRESPONSIVE_MS`
  (90s) in `may_take_turn()`, plus removal of contradictions in `END_CALL_DESCRIPTION`.
  Threshold chosen from measurement: longest *legitimate* assistant gap observed live was
  220 ticks (44s), so 25s would have misfired mid-conversation in 2 of 5 runs.
- **Swallowed Scribe errors.** The receive loop dropped any message lacking a `text` field,
  which includes error messages. Now logged explicitly.
- **Diagnostic instrumentation** for the commit boundary: per-commit audio fed vs non-silent
  seconds, every Scribe message type, and a warning when a turn is taken having heard nothing.
