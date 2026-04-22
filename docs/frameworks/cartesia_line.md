# Cartesia Line: EVA-Bench Feasibility Analysis

> **Status**: Not recommended for integration at this time. Critical blockers prevent full evaluation pipeline compatibility.

---

## Table of Contents

1. [Framework Overview](#1-framework-overview)
2. [Architecture Analysis](#2-architecture-analysis)
3. [What IS Possible](#3-what-is-possible)
4. [Critical Blockers](#4-critical-blockers)
5. [Potential Workarounds](#5-potential-workarounds)
6. [What Would Need to Change in EVA-Bench](#6-what-would-need-to-change-in-eva-bench)
7. [Recommendation](#7-recommendation)
8. [Evaluation Targets If Feasible](#8-evaluation-targets-if-feasible)

---

## 1. Framework Overview

Cartesia Line is a voice agent platform with a **split architecture**: the Cartesia "harness" handles all audio processing (STT via Ink, TTS via Sonic, VAD, turn-taking), while the developer's agent code runs as a separate FastAPI/WebSocket server that only receives and sends **text events**.

- **SDK**: `cartesia-line` Python package (v0.2.6a0, alpha)
- **Agent definition**: `VoiceAgentApp` wrapping an `LlmAgent` with `LlmConfig`
- **LLM routing**: Via LiteLLM (100+ providers)
- **Tool execution**: `@loopback_tool` decorator, executes locally in agent code
- **Deployment**: `cartesia chat` (local), `cartesia deploy` (managed cloud), `cartesia connect` (self-hosted)

### Key Architectural Constraint

**Agent code never touches raw audio.** The harness owns the entire audio pipeline (microphone/speaker input, STT, TTS, VAD, turn-taking). The agent server only receives text transcriptions and sends text responses.

### SDK Event Model

**Input events** (harness to agent):
- `CallStarted`, `CallEnded`
- `UserTurnStarted`, `UserTurnEnded` (contains `UserTextSent` list)
- `UserTextSent`, `UserDtmfSent`
- `AgentHandedOff`

**Output events** (agent to harness):
- `AgentSendText(text, interruptible=True)`
- `AgentEndCall`, `AgentTransferCall`
- `AgentToolCalled(tool_call_id, tool_name, tool_args)`
- `AgentToolReturned(tool_call_id, tool_name, tool_args, result)`
- `LogMetric(name, value)`, `LogMessage(name, level, message, metadata)`

---

## 2. Architecture Analysis

### EVA-Bench Architecture (Current)

EVA-Bench uses a tightly integrated pipeline where the `PipecatAssistantServer` owns the full audio path:

```
[ElevenLabs User Simulator]
        |
        | Twilio WebSocket frames (8 kHz mu-law)
        v
[PipecatAssistantServer (Pipecat pipeline)]
        |
        | STT -> LLM -> TTS (all local/controlled)
        | Records: audio_user.wav, audio_assistant.wav, audio_mixed.wav
        | Logs: audit_log.json, framework_logs.jsonl, framework_metrics.jsonl
        v
[Output files for MetricsContextProcessor]
```

The `AbstractAssistantServer` contract (see `docs/assistant_server_contract.md`) requires:
- A local WebSocket endpoint (`ws://localhost:{port}/ws`) accepting Twilio-format frames
- Audio capture of both tracks (user, assistant, mixed) as WAV files
- Detailed framework logs with timestamps for turn boundaries, TTS text, and LLM responses
- Integration with the ElevenLabs user simulator, which produces `elevenlabs_events.jsonl`

### Cartesia Line Architecture

Cartesia Line splits the system across two processes that communicate via text-only WebSocket:

```
[Phone/Browser/CLI audio source]
        |
        | Raw audio
        v
[Cartesia Harness (managed)]
        | - Ink STT
        | - Sonic TTS
        | - VAD + turn-taking
        | - Audio recording (?)
        |
        | Text events only (UserTextSent, AgentSendText)
        v
[Agent Server (your FastAPI code)]
        | - LLM calls (via LiteLLM)
        | - Tool execution (@loopback_tool)
        | - No audio access
```

### Fundamental Mismatch

The EVA-Bench `AbstractAssistantServer` contract requires the agent server to:

1. **Accept raw audio** from the user simulator via WebSocket -- Cartesia agent code never sees audio
2. **Record audio tracks** (user, assistant, mixed) -- Cartesia agent code has no audio buffers
3. **Expose a Twilio-format WebSocket** for the ElevenLabs user simulator -- Cartesia expects its own harness as the audio source
4. **Produce audio timestamps** for turn-taking metrics -- Cartesia agent code only sees text events

These are not implementation gaps that can be bridged with adapter code. They reflect a fundamental architectural difference: EVA-Bench assumes the agent server is the audio owner, while Cartesia Line assumes the harness is the audio owner.

---

## 3. What IS Possible

Despite the architectural mismatch, several EVA-Bench output files can be produced from Cartesia Line's text event stream.

### 3.1 audit_log.json -- YES

The audit log is a text-level record of the conversation. Cartesia Line's events map cleanly:

| EVA-Bench Audit Log Entry | Cartesia Line Source |
|---|---|
| `append_user_input(content, timestamp_ms)` | `UserTextSent.text` within `UserTurnEnded` event |
| `append_assistant_output(content, tool_calls)` | `AgentSendText.text` events |
| `append_tool_call(tool_name, params, result)` | `AgentToolCalled` / `AgentToolReturned` events |
| `append_llm_call(llm_call)` | Requires instrumentation (see below) |

**LLM call logging**: `LlmAgent` uses LiteLLM internally to make LLM calls. The agent code does not directly see LLM prompts and responses. However, since LiteLLM is used, it may be possible to hook into LiteLLM callbacks or use `LogMetric`/`LogMessage` events to capture LLM call details (prompt, response, latency, model, token usage). This would require SDK-level instrumentation that is not currently documented.

### 3.2 framework_logs.jsonl -- PARTIAL

| Event Type | Cartesia Line Source | Status |
|---|---|---|
| `turn_start` | `UserTurnStarted` event timestamp | YES |
| `turn_end` | Inferred from `AgentSendText` completion or next `UserTurnStarted` | PARTIAL -- no explicit assistant turn end event |
| `tts_text` | `AgentSendText.text` -- this is the text sent to TTS | YES |
| `llm_response` | Not directly available -- LLM is internal to `LlmAgent` | NO (without instrumentation) |

The `tts_text` equivalent is available because `AgentSendText` contains the exact text that the harness will synthesize via Sonic TTS. This is sufficient for the `intended_assistant_turns` variable in `MetricContext`, which is used by most judge metrics (faithfulness, conciseness, conversation progression, etc.).

The `llm_response` event type is a fallback when `tts_text` is not available (per the contract: "If both tts_text and llm_response entries are present, the processor discards all llm_response entries and uses only tts_text"). Since `tts_text` is available, the absence of `llm_response` is acceptable.

### 3.3 Tool Execution -- YES

Cartesia Line's `@loopback_tool` decorator executes tools locally in the agent process. This aligns with EVA-Bench's requirement that all tool calls go through the local `ToolExecutor`:

```python
@loopback_tool
async def search_booking(ctx, confirmation_number: Annotated[str, "The booking confirmation number"]):
    """Search for a booking by confirmation number."""
    # Route to EVA-Bench ToolExecutor
    result = tool_handler.execute("get_reservation", {"confirmation_number": confirmation_number})
    audit_log.append_tool_call("get_reservation", {"confirmation_number": confirmation_number}, result)
    return result
```

Tool calls would need to be wrapped to route through `ToolExecutor` and log to `AuditLog`, but the execution model is compatible.

### 3.4 Scenario Database Files -- YES

`initial_scenario_db.json` and `final_scenario_db.json` depend only on `ToolExecutor` state, which is fully within the agent code. These can be saved at `CallStarted` and `CallEnded` respectively.

### 3.5 System Prompt -- YES

`LlmConfig(system_prompt=...)` accepts an arbitrary system prompt. EVA-Bench's `build_system_prompt()` output can be passed directly.

### 3.6 LLM Choice -- YES

`LlmAgent(model=...)` accepts any LiteLLM-supported model identifier. This covers all 100+ providers supported by LiteLLM, matching EVA-Bench's existing model configuration approach.

### 3.7 transcript.jsonl -- YES

Can be produced from `UserTextSent` and `AgentSendText` events with their timestamps.

### 3.8 agent_perf_stats.csv -- PARTIAL

Requires LLM call-level data (prompt tokens, completion tokens, latency, cost). If LiteLLM callbacks can be hooked, this is producible. Otherwise, only partial data from `LogMetric` events may be available.

---

## 4. Critical Blockers

### 4.1 No Raw Audio Access

**Severity**: Blocking

The `AbstractAssistantServer` contract requires producing three WAV files:
- `audio_assistant.wav` -- assistant audio track
- `audio_user.wav` -- user audio track
- `audio_mixed.wav` -- mixed overlay of both tracks

These files are consumed by audio judge metrics (`agent_speech_fidelity`, `user_speech_fidelity`) which evaluate whether the spoken audio matches the intended text. The Cartesia agent server never receives or produces audio bytes -- the harness handles all audio I/O.

**Impact**: `agent_speech_fidelity` and `user_speech_fidelity` metrics cannot be computed.

### 4.2 No Audio Timestamps

**Severity**: Blocking

The `MetricContext` requires:
- `audio_timestamps_user_turns` -- `(start, end)` tuples for user audio segments
- `audio_timestamps_assistant_turns` -- `(start, end)` tuples for assistant audio segments
- `assistant_interrupted_turns` -- detected from audio overlap analysis
- `user_interrupted_turns` -- detected from audio overlap analysis

These are sourced from `elevenlabs_events.jsonl`, which is produced by the ElevenLabs user simulator based on real audio stream analysis. The Cartesia agent server has no audio timestamps -- only text event timestamps.

**Impact**: `turn_taking` metric cannot be computed (requires audio timestamps per turn). Interruption detection is unavailable.

### 4.3 Cannot Produce elevenlabs_events.jsonl

**Severity**: Blocking

`elevenlabs_events.jsonl` is normally produced by the ElevenLabs user simulator. It contains:
- `user_speech` / `assistant_speech` -- transcriptions from the user simulator's perspective
- `audio_start` / `audio_end` -- audio session boundaries that **drive turn numbering**

Turn numbering across all three log sources is aligned by `audio_start` events with `audio_user: "elevenlabs_user"`. Without this file, the `MetricsContextProcessor` cannot assign events to the correct turn indices.

**Impact**: All per-turn metrics lose their turn alignment mechanism. The `transcribed_assistant_turns` variable (what the user simulator actually heard) is unavailable, preventing speech fidelity comparison.

### 4.4 User Simulator Incompatible

**Severity**: Blocking

The ElevenLabs user simulator connects to the assistant server via a Twilio-format WebSocket, sending and receiving 8 kHz mu-law audio frames. The Cartesia harness does not accept external WebSocket audio input -- it expects audio from its own managed sources (phone, browser, or CLI).

The existing `UserSimulator` class (`src/eva/user_simulator/client.py`) creates a `BotToBotAudioInterface` that bridges ElevenLabs audio to the assistant's WebSocket. This interface is fundamentally incompatible with Cartesia Line's connection model.

**Impact**: No way to run the standard EVA-Bench evaluation loop. Cannot connect the user simulator to the agent.

### 4.5 STT/TTS Locked to Cartesia Services

**Severity**: Significant (not strictly blocking for all metrics)

The harness uses Cartesia's own STT (Ink) and TTS (Sonic) services exclusively. There is no documented way to substitute alternative STT or TTS providers. This means:
- Cannot evaluate the same LLM with different STT/TTS combinations
- Cannot isolate LLM performance from Cartesia's audio pipeline quality
- EVA-Bench's `pipeline_config` (which specifies STT, LLM, and TTS independently) cannot be honored

**Impact**: Limits the evaluation to Cartesia's fixed audio pipeline. Cannot do controlled comparisons across STT/TTS providers.

### 4.6 Alpha SDK

**Severity**: Risk factor

The `cartesia-line` package is at version 0.2.6a0 (alpha). The API surface, event model, and deployment patterns may change without backward compatibility guarantees. Any integration work could be invalidated by a breaking SDK update.

### 4.7 response_latencies.json Unavailable from Agent Code

**Severity**: Blocking for response_speed metric

Response latency in EVA-Bench measures end-to-end time from user speech end to first assistant audio byte. The Cartesia agent server only sees text events -- it cannot measure audio-level latency. Text-event timestamps (`UserTurnEnded` to first `AgentSendText`) would capture LLM processing time but miss STT and TTS latency, which are handled by the harness.

**Impact**: `response_speed` metric cannot be accurately computed. The measured latency would exclude STT and TTS time, underreporting true end-to-end latency.

### 4.8 framework_metrics.jsonl Unavailable

**Severity**: Moderate

This file captures per-component latencies (STT processing time, TTS TTFB, LLM token usage). Since the Cartesia harness owns STT and TTS, these per-component metrics are not accessible from the agent server. `LogMetric` events could theoretically carry some of this data if the harness exposes it, but this is not documented.

**Impact**: Cannot compute per-component latency breakdowns. `ConversationWorker` statistics will be incomplete.

---

## 5. Potential Workarounds

### 5.1 Audio Recording via Cartesia Platform

Cartesia's managed platform may record calls server-side. If an API exists to retrieve call recordings after the conversation ends, these could be downloaded and split into user/assistant tracks. This is speculative -- no such API is documented in the current SDK.

**Feasibility**: Unknown. Would require enterprise-tier access or a platform API that may not exist.

### 5.2 Audio Timestamp Inference

`UserTurnStarted` and `UserTurnEnded` timestamps could serve as rough proxies for audio timing. However:
- These are text-level events, not audio-level -- there is latency between when the user starts speaking and when the harness fires `UserTurnStarted` (after STT processes enough audio)
- No equivalent assistant audio start/end events exist
- Interruption detection requires audio overlap analysis, which is impossible from text events alone

**Feasibility**: Low. Timestamps would be systematically late (by STT processing time) and miss interruptions entirely.

### 5.3 Using `cartesia chat` as an Alternative User Simulator

The `cartesia chat 8000` CLI provides a local testing harness that simulates a user via browser microphone. In principle, a text-based or audio-based test could be driven through this interface. However:
- It requires manual interaction (browser-based) or at minimum a programmatic audio source
- It does not produce `elevenlabs_events.jsonl`
- It does not support the structured user goals and personas that EVA-Bench's user simulator follows
- There is no way to programmatically inject a user persona or goal

**Feasibility**: Not viable for automated evaluation.

### 5.4 Custom Audio Proxy

A custom proxy server could sit between the ElevenLabs user simulator and the Cartesia harness:

```
[ElevenLabs User Simulator]
        |
        | Twilio WebSocket (8 kHz mu-law)
        v
[Custom Audio Proxy]
        |
        | Cartesia harness audio format
        v
[Cartesia Harness]
        |
        | Text events
        v
[Agent Server]
```

The proxy would:
1. Accept Twilio-format WebSocket connections from the user simulator
2. Convert audio to the format expected by Cartesia's harness
3. Forward audio to the Cartesia harness (acting as a phone/browser source)
4. Capture both audio streams for WAV file generation
5. Produce `elevenlabs_events.jsonl` equivalent from the audio it observes

**Feasibility**: Technically possible but requires reverse-engineering the Cartesia harness's audio input protocol, which is not publicly documented. The harness expects connections from phones (via SIP/PSTN) or browsers (via WebRTC), not arbitrary WebSocket clients. This would be a substantial engineering effort with fragile dependencies on undocumented harness behavior.

### 5.5 LiteLLM Callback Hooks

Since `LlmAgent` uses LiteLLM internally, LiteLLM's callback system (`litellm.callbacks`) could potentially be used to intercept LLM calls and capture prompts, responses, token usage, and latency. This would enable:
- `llm_prompts` array in `audit_log.json`
- `agent_perf_stats.csv`
- `LLMTokenUsageMetricsData` in `framework_metrics.jsonl`

**Feasibility**: Moderate. Depends on whether `LlmAgent` uses LiteLLM in a way that respects global callbacks. Worth investigating if other blockers are resolved.

---

## 6. What Would Need to Change in EVA-Bench

If Cartesia Line integration were pursued despite the blockers, the following EVA-Bench components would need modification:

### 6.1 New User Simulator

A new user simulator class would be needed that works with Cartesia's harness rather than connecting via Twilio WebSocket. Options:
- A text-only user simulator that connects to the agent server directly (bypassing audio entirely)
- A SIP/PSTN-based simulator that calls the Cartesia harness as a phone participant
- A WebRTC-based simulator that connects as a browser participant

None of these exist today. A text-only simulator would be the simplest but would eliminate all audio-dependent metrics.

### 6.2 MetricsContextProcessor Without elevenlabs_events.jsonl

The processor currently requires `elevenlabs_events.jsonl` for turn numbering, `transcribed_assistant_turns`, `intended_user_turns`, and audio timestamps. Without this file:
- Turn numbering would need to be derived from `audit_log.json` and `framework_logs.jsonl` alone
- `transcribed_assistant_turns` would be unavailable (no independent transcription of assistant speech)
- Audio timestamp fields would be empty
- Interruption detection would be disabled

This would require a significant refactor of `MetricsContextProcessor` to support a two-source (rather than three-source) log joining mode.

### 6.3 Metrics That Require Audio

The following metrics would need alternative data sources or would need to be skipped:

| Metric | Current Requirement | Status with Cartesia Line |
|---|---|---|
| `agent_speech_fidelity` | `audio_assistant.wav` | Cannot compute -- no audio |
| `user_speech_fidelity` | `audio_user.wav` | Cannot compute -- no audio |
| `turn_taking` | Audio timestamps per turn | Cannot compute -- no timestamps |
| `response_speed` | End-to-end audio latency | Inaccurate -- text-level only |
| `stt_wer` | Transcribed vs intended user text | Partially available from `UserTextSent` (but this is the harness's STT output, not independently verifiable) |

Metrics that would work normally:

| Metric | Status |
|---|---|
| `task_completion` | YES -- depends only on scenario DB state |
| `faithfulness` | YES -- depends on conversation trace |
| `conciseness` | YES -- depends on conversation trace |
| `conversation_progression` | YES -- depends on conversation trace |
| `conversation_valid_end` | YES -- depends on conversation trace |
| `tool_call_validity` | YES -- depends on audit log |
| `speakability` | YES -- depends on intended assistant text |
| `authentication_success` | YES -- depends on conversation trace |
| `transcription_accuracy_key_entities` | PARTIAL -- depends on transcribed user turns (available from Ink STT) |

### 6.4 Response Latency from Cartesia Platform

If Cartesia's platform exposes per-call latency metrics (e.g., via `LogMetric` events or a post-call analytics API), these could replace Pipecat's `UserBotLatencyObserver`. However, the latency breakdown would be different -- Cartesia may report end-to-end latency including their infrastructure, rather than the component-level breakdown EVA-Bench currently uses.

---

## 7. Recommendation

**Not recommended for integration at this time.**

The Cartesia Line architecture is fundamentally incompatible with EVA-Bench's evaluation pipeline. The split between harness (audio owner) and agent server (text only) means that the agent code cannot produce the audio recordings, audio timestamps, or user simulator connectivity that EVA-Bench requires.

Of the 12 required output files in the `AbstractAssistantServer` contract, only 5 can be fully produced, 3 can be partially produced, and 4 cannot be produced at all:

| File | Status |
|---|---|
| `audit_log.json` | Fully producible (with LLM call instrumentation work) |
| `framework_logs.jsonl` | Partially producible (tts_text yes, llm_response and turn_end require work) |
| `transcript.jsonl` | Fully producible |
| `initial_scenario_db.json` | Fully producible |
| `final_scenario_db.json` | Fully producible |
| `audio_assistant.wav` | **Not producible** |
| `audio_user.wav` | **Not producible** |
| `audio_mixed.wav` | **Not producible** |
| `elevenlabs_events.jsonl` | **Not producible** |
| `response_latencies.json` | Partially producible (text-level latency only, missing STT/TTS time) |
| `framework_metrics.jsonl` | Partially producible (LLM metrics only via callbacks, no STT/TTS metrics) |
| `agent_perf_stats.csv` | Partially producible (via LiteLLM callbacks) |

### Conditions for Re-evaluation

Defer integration until Cartesia adds at least the following capabilities:

1. **WebSocket audio input to the harness** -- allowing the ElevenLabs user simulator (or a similar audio source) to connect programmatically, rather than requiring phone/browser sources
2. **Audio recording API** -- allowing retrieval of call recordings (user track, assistant track) after a conversation ends
3. **Audio timestamp events** -- exposing `audio_start`/`audio_end` events for both speakers in the agent SDK event stream
4. **SDK stability** -- moving out of alpha to a stable release with backward compatibility guarantees

If Cartesia exposes a WebSocket or SIP audio input and provides call recording retrieval, a custom audio proxy approach (section 5.4) becomes viable without reverse-engineering undocumented protocols.

---

## 8. Evaluation Targets If Feasible

If the blockers are resolved, Cartesia Line integration would enable evaluation of:

### Cartesia Sonic TTS Quality
- `agent_speech_fidelity` -- how faithfully Sonic TTS renders the intended text
- `speakability` -- whether the LLM generates text that is appropriate for Sonic's synthesis

### Cartesia Ink STT Quality
- `stt_wer` -- word error rate of Ink STT on user speech
- `transcription_accuracy_key_entities` -- whether Ink correctly transcribes confirmation codes, flight numbers, passenger names, and other critical entities

### Cartesia Turn-Taking Model
- `turn_taking` -- how well Cartesia's proprietary turn-taking model (within the harness) times assistant responses
- `response_speed` -- end-to-end latency including Cartesia's pipeline

### LLM Performance Through Cartesia's Pipeline
- All text-based judge metrics (faithfulness, conciseness, conversation progression, task completion) to assess whether Cartesia's pipeline affects LLM decision quality
- Tool call validity to verify that Cartesia's event model does not introduce tool execution errors

### Comparison Value
The most valuable comparison would be the same LLM evaluated through both EVA-Bench's Pipecat pipeline and Cartesia Line's pipeline, isolating the effect of the framework and audio services on overall agent quality. This directly addresses EVA-Bench's documented limitation that "EVA scores reflect a specific Pipecat-based deployment configuration" (see `docs/limitations.md`).
