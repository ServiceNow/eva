# EVA Refactor — Step 1: Server / Role Separation

## Goal
Split the two parallel, duplicated provider stacks (assistant-side and user-side) into:
- **Server classes** — pure API / session / data-exchange for one backend. No role knowledge.
- **Role classes** — `AssistantRole` and `UserRole`, each owning its own prompt, tools, and goal/persona, and each **holding a Server instance created at runtime** via a factory.

This removes duplication, and — the payoff — lets *any* backend act as either role, which is what unlocks "any model as a user simulator" and sets up the later mediator/interruption work.

> **Naming flag:** we've been calling the API class a "server," but today `AbstractAssistantServer` is literally a FastAPI/uvicorn network server. Consider naming the new abstraction `Backend` or `Transport` to avoid confusion, and reserve "server" for the actual socket if one survives. Decide this early.

## Current state (what exists today)
**Assistant side** — `assistant/base_server.py` `AbstractAssistantServer`: FastAPI+uvicorn WS server at `ws://localhost:{port}/ws` (Twilio frames), owns `ToolExecutor`, `AuditLog`, audio buffers, output saving, system-prompt building. Subclasses: `pipecat_server`, `openai_realtime_server`, `gemini_live_server`, `grok_voice_server`, `elevenlabs_server`. Plus `assistant/services/llm.py`, `assistant/agentic/`, `assistant/pipeline/`, `assistant/tools/`, `audio_bridge.py`, `audio_buffer.py`.

**User side** — `user_simulator/base.py` `AbstractUserSimulator`: a **client** that connects to `server_url`, owns persona/goal/prompt building, perturbation, event logging, audio recording. Subclasses: `elevenlabs.py`, `openai_realtime.py`. Plus `factory.py`, `perturbation.py`, its own `audio_bridge.py`.

**The duplication to kill:**
- `audio_bridge.py` exists twice (assistant + user).
- OpenAI Realtime (and ElevenLabs) integration exists on **both** sides.
- Prompt building, audio recording, and tool handling are re-implemented per side.

## Target architecture

```
Backend (abstract)                # pure API/data exchange, no role knowledge
├── session lifecycle (open/close)
├── send_audio(frames) / receive_audio()  → streaming in/out
├── surface tool-call requests            → role decides + executes
├── provider events (end-of-turn, etc.)
├── capabilities (see below)
└── impls: OpenAIRealtimeBackend, GeminiLiveBackend, GrokBackend,
           ElevenLabsBackend, CascadeBackend (STT→LLM→TTS)

Role (abstract)                   # owns prompt, tools, goal/persona
├── self.backend: Backend         # created at runtime via factory
├── build_prompt()
├── tools / tool execution policy
└── AssistantRole  |  UserRole

BackendFactory.create(name, config) -> Backend
```

**Key moves:**
1. Extract everything provider-specific and role-agnostic out of both `AbstractAssistantServer` and `AbstractUserSimulator` into `Backend` implementations.
2. Move role-specific concerns (assistant: agent config, tools, ToolExecutor; user: goal, persona, starting utterance) into `AssistantRole` / `UserRole`.
3. Each role instantiates its backend at construction via `BackendFactory`.
4. Collapse the two `audio_bridge.py` into one shared module.
5. Consolidate audio recording / output-saving into one shared helper used by both roles (currently split across `base_server` and `user_simulator/base`).

**Declare backend capabilities now (cheap, needed later):**
```
emits_continuous_audio: bool     # cascade/S2S streaming vs discrete utterance
supports_streaming_interruption: bool
owns_playout_clock: bool         # true for cascade → required for user barge-in later
```
Don't build turn-taking on these yet — just surface them so the later mediator can branch on them without another refactor.

## Out of scope for this step (do NOT build)
- The master/mediator layer, VAD/energy detection, streaming-STT decision loop.
- Interruptions, backchannels, yield, the judge gate.
- Transport decision (WS-in-the-middle vs in-process bus) — the mediator step decides that. **But** design the `Backend` interface so it does **not** hard-assume "one side is a network server the other connects to." Keep send/receive symmetric so a mediator can sit between later.
- Parallelism (process-per-conversation / thread-offload) — separate, measured, later.

## Constraints & gotchas
- **Preserve all current outputs** — `audit_log.json`, `transcript.jsonl`, `user_simulator_events.jsonl`, audio WAVs, scenario DBs. This refactor must be behavior-preserving; existing runs and metrics should produce identical artifacts.
- **Twilio frame format / sample rates** — currently 24kHz assistant path, telephony on user path; keep conversions in the shared audio module, not per-backend.
- **Tool execution stays role-side** — `ToolExecutor` + audit logging belongs to `AssistantRole`; don't push it into `Backend`. (Note the existing split: `AgenticSystem` has its own tool loop vs `execute_tool` on the server — reconcile into one path owned by the role.)
- **ElevenLabs Agents is end-to-end** — its backend won't cleanly expose the same seams as a cascade backend (no separable STT/LLM/TTS). Expect it to implement a thinner slice of the interface; the capability flags cover that.

## Definition of done
- One `Backend` abstraction + factory; each provider integration exists **once**.
- `AssistantRole` and `UserRole` each hold a runtime-created backend.
- Single shared `audio_bridge` and shared recording/output path.
- All existing artifacts reproduced byte-for-byte (or with a documented, intentional diff) on a smoke-test record per domain.
- Capability flags present but unused.

---

*Step 1 of the larger EVA user-simulator interruption/turn-taking plan. This step is a behavior-preserving code refactor; the mediator, interruption policy, and judge-gate work build on top of it in later steps.*
