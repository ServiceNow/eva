# EVA assistant server for the aai voice-agent framework

**Date:** 2026-07-28
**Status:** Approved — ready for implementation planning
**Repos:** `~/Code/eva` (this change), `~/Code/aai` (host mode, already implemented)
**Prior art:** `~/Code/tau2-bench` `aai` audio-native provider

## Goal

Evaluate the **aai** voice-agent framework on EVA, scoring it on both EVA-A (accuracy)
and EVA-X (experience) across EVA's existing domains and scenarios.

EVA integrates a system under test as an **assistant server**: a process that exposes a
Twilio-framed WebSocket which EVA's user simulator calls into, and that writes a fixed
set of output files the metrics read. `docs/assistant_server_contract.md` is the
normative reference. This spec adds such a server for aai, plus a reusable base that a
second WebSocket-backed integration can sit on later.

## Why aai host mode is the integration point

aai agents are normally defined in TypeScript and execute their tools **server-side in a
sandbox** against `ctx.kv`. EVA requires the inverse: the agent runs the *domain's*
prompt and tool *schemas*, while **EVA's `ToolExecutor` executes the tools** against the
scenario database.

This matters because EVA's accuracy metrics diff `initial_scenario_db.json` against
`final_scenario_db.json`. If aai executed tools internally, the scenario database would
never mutate and every task-completion metric would read as "nothing happened".

aai's **host mode** (`packages/aai/host/host-mode.ts`) bridges this: the client's first
`config` frame carries `host{systemPrompt, greeting, tools}`, aai builds a fresh
single-use `Runtime` from it, and each tool call is relayed to the client as a
`tool_call` frame that resolves when the matching `tool_result` arrives. This is the same
mechanism the tau2 `aai` provider drives.

## Confirmed facts (from code inspection)

**aai side** (`~/Code/aai/agent`):
- WS endpoint `ws://localhost:3000/websocket`; host mode selected by `?host=1`
  (`wantsHostMode(rawUrl)` in `packages/aai-server/orchestrator.ts`).
- Host mode is gated by `AAI_ALLOW_HOST` (`isHostAllowed`, `packages/aai/host/host-mode.ts`).
- Handshake: client sends `config` with `host{systemPrompt, greeting?, tools[]}` plus
  `audioFormat:"pcm16"`, `sampleRate`, `ttsSampleRate`; schema in `packages/aai/sdk/protocol.ts`.
- Client→server `tool_result{toolCallId, result, error?}`; `result` is a string on the wire
  and a JSON string is unwrapped before reaching the model.
- Audio: binary PCM16 frames, **16000 Hz in / 24000 Hz out** by default.
- Server→client JSON events: `config`, `speech_started`, `speech_stopped`,
  `user_transcript`, `agent_transcript`, `tool_call`, `tool_call_done`, `reply_done`,
  `audio_done`, `cancelled`, `reset`, `idle_timeout`, `error`, `custom_event`.
- Host agents default to `maxSteps` 30, so multi-tool turns are supported.
- **No usage/token events exist.**

**EVA side** (`~/Code/eva`):
- `AbstractAssistantServer` (`src/eva/assistant/base_server.py`, 368 lines) provides
  `audit_log`, `tool_handler`, `execute_tool()`, the three audio buffers, and
  `save_outputs()`. Subclasses implement `start()` and `stop()`.
- The user simulator speaks Twilio Media Streams framing: μ-law 8 kHz, plus
  `user_speech_start` / `user_speech_stop` events carrying wall-clock timestamps.
- `src/eva/assistant/audio_bridge.py` already provides `mulaw_8k_to_pcm16_16k` and
  `pcm16_24k_to_mulaw_8k` — exactly aai's rates — along with `sync_buffer_to_position`,
  `FrameworkLogWriter`, and `MetricsLogWriter`.
- Framework selection is a `Literal` at `src/eva/models/config.py:506`; the server
  registry is `_get_server_class()` at `src/eva/orchestrator/worker.py:26`.
- Prompt and tool definitions come from `AgentConfig` (`configs/agents/*.yaml`), so no
  per-domain work is needed — airline, itsm, and medical_hr all work from one server.
- The existing five servers each hand-roll the Twilio bridge and pacing loop
  (600–850 lines apiece).
- **No `.env` is present locally**, so a live end-to-end run cannot be performed as part
  of this work.

## Architecture

```
user simulator ──Twilio μ-law 8k──▶ AAIAssistantServer ──PCM16 16k───▶ aai host
(ElevenLabs /   ◀──Twilio μ-law 8k── localhost:{port}/ws ◀──PCM16 24k── :3000/websocket?host=1
 OpenAI caller)                              │
                                       execute_tool()
                                             ▼
                                      scenario database
```

Two WebSocket hops with the server in the middle. Audio converts at each boundary; tool
calls travel inward from aai and resolve against EVA's scenario database.

**Note the inversion from tau2.** In tau2 the aai code is a *client* pulled along by a
discrete-time tick loop. In EVA it is a *server* between two live sockets, responsible
for pacing its own output at wall-clock 20 ms / 160-byte cadence — the user simulator
infers turn boundaries from arrival timing, so pacing errors corrupt the evaluation
rather than merely sounding wrong. What ports from tau2 is the protocol layer, not the
driving loop.

### aai host provisioning

EVA connects out to an already-running host at `AAI_WS_URL`
(default `ws://localhost:3000/websocket?host=1`), started separately with
`AAI_ALLOW_HOST=1`. Because host mode builds a fresh single-use `Runtime` per
connection, EVA's `max_concurrent_conversations` maps to N parallel host sessions
without further coordination.

Rejected: having EVA spawn the aai server as a subprocess, or as a compose service.
Both place Node lifecycle and port allocation inside a Python eval worker for no
evaluation benefit.

## Components

All work is **additive**. The five existing servers are left untouched so that
leaderboard-validated pipelines cannot regress.

### New modules

| File | Purpose |
|---|---|
| `src/eva/assistant/ws_bridge_server.py` | `WebSocketBridgeAssistantServer` — shared base: `start`/`stop`, Twilio parse/encode, 20 ms pacing task, buffer sync, turn bookkeeping, latency and audit writes. Protocol-agnostic. |
| `src/eva/assistant/aai_events.py` | aai protocol event models + `parse_aai_event`, ported from tau2's `events.py`. |
| `src/eva/assistant/aai_session.py` | `AAIHostSession` — WS client implementing the backend seam: host handshake, binary audio, `tool_result` relay. |
| `src/eva/assistant/aai_server.py` | `AAIAssistantServer` — builds the system prompt from `AgentConfig` and the flat aai tool schemas. Thin. |

Flat modules rather than a package, matching EVA's existing `*_server.py` convention.

### Edits

| File | Change |
|---|---|
| `src/eva/models/config.py:506` | Add `"aai"` to the `framework` Literal and its description. |
| `src/eva/orchestrator/worker.py` | Add a lazy-import branch to `_get_server_class()` and update the `Supported:` error string. |
| `.env.example` | Document `AAI_WS_URL` and the `AAI_ALLOW_HOST` requirement. |
| `docs/aai_integration.md` *(new)* | Setup, run command, and known gaps. |

No changes to `configs/agents/*.yaml`.

## The backend seam

```python
class VoiceBackendSession(Protocol):
    backend_input_rate: int    # 16000 for aai
    backend_output_rate: int   # 24000 for aai

    async def send_audio(self, pcm: bytes) -> None: ...
    async def send_tool_result(self, call_id: str, result: dict) -> None: ...
    def events(self) -> AsyncIterator[BridgeEvent]: ...
    async def aclose(self) -> None: ...
```

`BridgeEvent` is a union of `AudioChunk`, `AssistantTranscript`, `UserTranscript`,
`ToolCall`, `SpeechStarted`, `TurnDone`, and `BackendError`.

The base selects audio converters from the declared sample rates and owns every
contract-shaped concern; a subclass owns only its wire protocol. Adding a second
WebSocket-backed integration is therefore one new `*_session.py` plus a small server
subclass, with no base changes.

Each unit is independently testable: `aai_events` is pure parsing, `aai_session` is
protocol framing over a socket, `ws_bridge_server` is the EVA contract over an abstract
session, and `aai_server` is prompt and schema construction.

## Event mapping

| aai event | Bridge event | Contract action |
|---|---|---|
| binary frame | `AudioChunk` | On first chunk of a turn: `fw_log.turn_start()` and `write_latency("model_response", …)` (bounds-checked `0 < ms < 30_000`). Convert 24 k → μ-law, enqueue for paced send. `sync_buffer_to_position(user_audio_buffer, …)` then extend `assistant_audio_buffer`. |
| `agent_transcript` | `AssistantTranscript` | Accumulate with full-text-overwrite semantics, as tau2 does. |
| `user_transcript` | `UserTranscript` | `append_user_input(text, timestamp_ms=<user_speech_start>)`. |
| `tool_call` | `ToolCall` | `await self.execute_tool(name, args)` → `send_tool_result`. |
| `speech_started` | `SpeechStarted` | Barge-in: drain the pacing queue, `turn_end(was_interrupted=True)`, record the partial transcript suffixed `[interrupted]`. |
| `reply_done` / `audio_done` | `TurnDone` | `append_assistant_output(text)`, `fw_log.llm_response(text)`, `turn_end(False)`. |
| `error` | `BackendError` | Log. |
| `idle_timeout` | `BackendError` | Log and end the session. |
| `tool_call_done`, `cancelled`, `reset`, `custom_event`, `speech_stopped` | — | Parsed, no contract action. |

### Timestamps and VAD

Two VAD systems coexist: the simulator's (`user_speech_start` / `user_speech_stop`, with
wall-clock timestamps) and aai's (`speech_started` / `speech_stopped`). The contract
specifies the simulator's `user_speech_stop` as the latency denominator, so that remains
the sole source for `model_response` latency. aai's events drive barge-in only. Mixing
the two would silently skew `model_response_latency`.

### Greeting

`host.greeting` is set to `get_initial_message(language)`, so aai opens the call. This
matches EVA's expectation that the assistant speaks first.

### Tools

`self.agent.tools` maps to aai's flat `ToolSchema` shape —
`{name, description, parameters}` with `parameters` as JSON Schema. Analogous to
`_build_realtime_tools()` in `openai_realtime_server.py`, but flat rather than nested.

## Decisions and known gaps

- **No token usage.** aai emits no usage events, so `write_token_usage()` is omitted.
  `pipecat_metrics.jsonl` is still written with latency entries, so
  `ConversationResult.model_response_latency` populates normally. Documented as a gap.
- **Model identity.** The contract requires `s2s_params["model"]`. Configuration is
  `s2s: aai-host` with `s2s_params: {model: "aai-host", ws_url: …}`.
- **Relay tool timeout.** aai rejects a relayed call after
  `DEFAULT_RELAY_TOOL_TIMEOUT_MS`. EVA tool execution is local and fast, so the default
  stands; a timeout surfaces as a tool error and the turn continues.

## Error handling

- **Handshake failure** (host mode disabled, bad config): fail `start()` loudly with the
  server's rejection reason. A silent fallback would produce a scored-but-meaningless run.
- **Backend disconnect mid-conversation**: end the session, still call `save_outputs()`
  so partial artifacts and the scenario database are written for inspection.
- **Malformed events**: `parse_aai_event` yields an unknown-event model that is logged
  and skipped, never raised — one unrecognized frame must not abort a conversation.
- **Tool execution failure**: `execute_tool()` already records the error result in the
  audit log; relay it to aai as a `tool_result` error so the model can recover.

## Testing

Unit tests in `tests/unit/assistant/`:

- `test_aai_events.py` — event parsing, ported from tau2.
- `test_aai_session.py` — handshake frame contents, `tool_result` framing, relay timeout.
- `test_ws_bridge_server.py` — Twilio round-trip, pacing cadence, buffer drift ≤ 500 ms,
  latency sanity bounds, barge-in handling.
- `test_aai_server.py` — prompt and tool-schema construction, and registration assertions
  covering both the `framework` Literal and `_get_server_class("aai")`.

Plus an **in-process fake aai backend** speaking the real protocol, giving a full-session
test with no paid APIs and no Node process. This is the verification driven to green.

**Not verifiable here:** the live end-to-end run, which needs API keys absent from this
checkout plus a running aai host:

```bash
AAI_ALLOW_HOST=1 EVA_FRAMEWORK=aai EVA_MODEL__S2S=aai-host \
EVA_USER_SIMULATOR__PROVIDER=openai_realtime EVA_DOMAIN=itsm \
EVA_RECORD_IDS=15 EVA_MAX_CONCURRENT_CONVERSATIONS=1 eva
```

This command is documented in `docs/aai_integration.md` for the user to run.

## Non-goals

- Migrating the five existing servers onto the new base.
- Building the second (AssemblyAI-hosted) integration. The base is shaped to accept one;
  no speculative hooks are added for it.
- Perturbation-suite or leaderboard-submission plumbing for aai.
- Any change to aai's sandbox or tool-execution model for non-host agents.
