# Backend/Role migration — working notes

Branch-only scratch notes for the multi-backend migration. Not a spec; the spec
is `docs/refactor-step1.md`. This file is a running log so context survives
conversation compaction. Delete before the final merge if desired.

## Goal / shape

Split the two duplicated provider stacks (assistant-side `AbstractAssistantServer`
subclasses, user-side `AbstractUserSimulator` subclasses) into:

- **`Backend`** (`eva.backend.base.Backend`): role-agnostic, provider-specific
  adapter. One per provider integration. Uniform `open/send/receive/close` +
  `trigger_response`. Emits **normalized** `BackendEvent`s (no raw provider
  events leak to roles).
- **`AssistantRole`** / **`UserRole`** (`eva.role.*`): exactly ONE concrete
  generic class each. Holds a `Backend`; owns role-common concerns (prompt,
  tools, logging, recording, transport to counterparty). Any backend works with
  either role.
- **`BackendFactory`**: single concrete class (no ABC — only one factory),
  worker-owned, stateless, lazy per-provider imports. `create(name, config)`
  returns a `Backend` for a migrated provider, else `None`.

## Dispatch (post loose-finalization)

No boolean gate, **no config-type gate**. Both sides route to the Role/Backend
path **iff `create()` returns a backend** (None ⇒ legacy fallback), dispatching
purely on the provider name:

- Assistant (`_start_assistant`): `if backend := _BACKEND_FACTORY.create(framework, args)`.
- User (`_start_user_simulator`): dump the user-sim config wholesale
  (`sim.model_dump()`) into a caller blob, read voice from the dumped dict (not
  attribute access — no coupling to a concrete config type), then
  `if backend := _BACKEND_FACTORY.create(sim.provider, args)` ⇒ UserRole, else
  legacy. No `isinstance` — any config whose provider is factory-backed uses the
  new path.

Migrating a provider = add a lazy-import branch in `BackendFactory.create` (+ if
it's a user-sim provider, allow its name in the user-sim config `provider`). It
then auto-routes; the legacy branch in `_get_server_class` /
`create_user_simulator` becomes dead (harmless fallback until a later sweep).

### Backend config validation = construction (single source of truth)
No backend-specific credential/field validation in `config.py`. A native-S2S
backend is validated by **constructing it via the `BackendFactory`** — the
backend's own `__init__` checks required fields/keys (api_key with env fallback,
model, accent rejection). This runs in `orchestrator.preflight._preflight_backends`:
- **Always runs** (cheap, no network) — `--no-preflight` only skips the live
  model probes, never this.
- Covers the **S2S assistant framework** (when `pipeline_type == S2S`) and the
  **user-sim provider**. Non-factory providers (`create()` → None: ElevenLabs
  Conversational AI, cascade pipecat) are skipped — validated by the live probes /
  legacy paths, not here. So it validates exactly the migrated backends; it does
  NOT hard-fail legacy (revisit if we want that forcing function later).
- A misconfigured factory backend (missing key/model) → `PreflightError`.
- Removed from `config.py`: the S2S `_validate_service_params` case (assistant)
  and `_check_s2s_simulator_credentials` (user). Trade-off: a missing S2S key is
  now a `PreflightError`, not a pydantic `ValidationError`. STT/TTS/LLM/audio-LLM
  param validation stays in config (not backends).

### User-sim config
`OpenAIRealtimeSimulatorConfig` → **`S2SSimulatorConfig`** (generic native-S2S
config), `provider: Literal["openai_realtime", "grok_voice"]`. Grok is
OpenAI-Realtime-compatible so it shares the config; defaults target OpenAI, Grok
users override `model` + voices. Per-provider API key comes from the env
(OPENAI_API_KEY / XAI_API_KEY) — fallback lives in the backend, with a fail-fast
mirror in `RunConfig._check_s2s_simulator_credentials`
(`_S2S_PROVIDER_API_KEY_ENV` map). Accent-perturbation validator generalized to
"native S2S" (still ElevenLabs-only).

## Migration status

- [x] `openai_realtime` — assistant + user. Baseline, tested end-to-end.
- [x] `grok_voice` — **assistant + caller DONE** (`src/eva/backend/grok_voice.py`,
      in factory). Thin subclass of `OpenAIRealtimeBackend` (xAI is
      OpenAI-Realtime-compatible), mirroring `grok_voice_server.py`. Works as a
      caller for free (inherits the OpenAI backend's pcmu/send/trigger_response
      surface) once the user-sim config accepts `provider="grok_voice"` (see
      user-sim config below). api_key falls back to `XAI_API_KEY`.
- [~] `gemini_live` — assistant was implemented then **parked** to avoid a
      conflict with a separate in-flight Gemini change. Working file saved at
      `output/tmp/gemini_live.py` (git-ignored) to restore later. When resuming:
      re-add the file, its factory branch, and re-plumb `language` into the
      assistant `backend_args` (Gemini's `language_code`). Caller still deferred
      (manual turn-taking / no VAD speech-boundary events don't map onto
      UserRole's gating).
- [ ] `elevenlabs`, cascade/pipecat — later.

### Grok as caller — native-VAD path (manual not supported)
Root cause (proven via backend-tagged DEBUG logs): **xAI ignores manual turn-taking**
(`create_response:false` AND `interrupt_response:false`) — it always auto-responds via
its own VAD. In the freeze runs our code never called `trigger_response` (0
`caller_response_created`) yet grok emitted `response.created`; it also wedged when its
VAD re-fired `speech_started` on still-streaming audio right at `response.created`.

So the caller now branches on a capability instead of forcing manual on everyone:
- `BackendCapabilities.supports_manual_response` — True (OpenAI: honors manual gating),
  False (Grok: native VAD only).
- `UserRole`: manual-capable → today's "respond now" gating (OpenAI **unchanged**);
  native-VAD → **no manual trigger**, we just consume grok's auto
  `OUTPUT_TURN_STARTED`/`AUDIO_OUTPUT`/`TURN_END` (symmetric with the assistant).
- `GrokVoiceBackend` widens the caller VAD silence (`GROK_CALLER_MIN_SILENCE_MS=1200`,
  only when `manual_turn_taking` is in config, i.e. caller use) so its native VAD doesn't
  segment mid-turn and auto-respond into still-arriving audio (the turn-0 wedge). Also
  drops inbound audio while `responding` as defense. Manual/`interrupt_response` config is
  moot for grok (ignored), so it inherits the role-declared values.
- Tuning knob: `GROK_CALLER_MIN_SILENCE_MS`. If grok still auto-responds mid-turn, raise
  it; if it's too laggy, lower it. If widening proves insufficient, the fallback is the
  clean-turn feeding-gate (stop feeding at `speech_stopped` until `TURN_END`).

### Grok — implementation notes
- `GrokVoiceBackend(OpenAIRealtimeBackend)` overrides only: default `base_url`
  (x.ai), default `voice` (`eve`), api_key **required** (no OPENAI_API_KEY
  fallback), and buffered input transcription. Everything else inherited.
- xAI fires `input_audio_transcription.completed` repeatedly with growing text;
  the backend buffers it on the session and emits ONE final input `TRANSCRIPT`
  at the turn boundary (next `speech_started` / `response.done`) — matching the
  old server's deferred-transcript flush. Without this the role would append
  multiple progressive user-input records.
- Generic seam added to `OpenAIRealtimeBackend`: `_SESSION_CLS` class attr that
  `open()` instantiates, so a subclass can carry extra per-turn session state
  (Grok's `pending_input_transcript`). No behavior change for OpenAI.
- The `grok_voice_server.py` docstring claims it overrides `_build_session_config`
  to drop `transcription.model`, but the current class does NOT — so no
  transcription-selector change was needed (matched actual code, not the docstring).

## Known looseness (intentional, revisit as we migrate)

- **User caller-config assembly is still OpenAI-Realtime-shaped** in
  `_start_user_simulator` (`male_voice`/`female_voice`, `CALLER_BACKEND_DEFAULTS`).
  The real generic top-level user JSON isn't designed yet; isinstance-narrow to
  `OpenAIRealtimeSimulatorConfig` for now.
- **Cascade/pipeline configs** not adapted to the backend-args dump style yet.
- `AssistantRole` may grow features not currently in the S2S path as we migrate
  cascade.

## Behavior deltas vs legacy (all intentional, sub-macro)

- `end_call` args logged as dict vs string.
- `connected` event drops OpenAI-specific labels.
- 24k-fixed mulaw converters (noted inline in role/user.py).
- `FrameworkLogWriter` S2S methods restored in `observers.py` (was a real `main`
  regression from commit 4bf4881, bundled into this refactor).

## Key files

- `src/eva/backend/base.py` — `Backend` ABC, `BackendEvent`/`BackendEventType`,
  `BackendSession`, `ToolCallRequest`/`ToolCallResult`.
- `src/eva/backend/openai_realtime.py` — reference backend + normalizer.
- `src/eva/backend/factory.py` / `default_factory.py` — factory.
- `src/eva/role/{base,assistant,user}.py` — roles.
- `src/eva/orchestrator/worker.py` — dispatch (`_start_assistant`,
  `_start_user_simulator`, `_run_conversation`).
- `tests/unit/test_openai_realtime_backend.py` — backend unit tests (no network).
