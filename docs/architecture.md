# EVA Architecture Guide

EVA is an end-to-end evaluation framework for conversational voice agents. It uses a **bot-to-bot architecture** where two AI systems — a user simulator and a voice assistant — have real spoken conversations over WebSocket audio, followed by automated validation and multi-dimensional metrics scoring. This document is the comprehensive architectural reference. For quick-start instructions see the [README](../README.md); for specific topics see the [docs index](README.md).

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
                                  EVA Framework
+------------------------------------------------------------------------+
|                                                                        |
|  +--------------+    +-------------------------+    +----------------+ |
|  |              |    |  Conversation Execution  |    |                | |
|  | Orchestration|--->|                         |--->|   Evaluation   | |
|  |              |    |  +--------+  +--------+ |    |                | |
|  |  CLI         |    |  | User   |  |  Asst  | |    |  Validate     | |
|  |  RunConfig   |    |  | Sim    |  | Server | |    |  Metrics      | |
|  |  Runner      |    |  |(Eleven |  |(Pipecat| |    |  Aggregate    | |
|  |  Workers     |    |  | Labs)  |  |)       | |    |  Pass@k       | |
|  |              |    |  +---+----+  +---+----+ |    |                | |
|  +--------------+    |      |  WebSocket |      |    +----------------+ |
|                      |      +---audio----+      |                      |
|                      +-------------------------+                      |
+------------------------------------------------------------------------+

Dataset JSONL --> BenchmarkRunner --> [N ConversationWorkers] --> Logs
                                                                   |
                             Summary <-- Aggregation <-- Metrics <-+
```

### 1.2 Execution Lifecycle

A complete benchmark run proceeds through these steps:

1. **CLI invocation** — `main.py` calls `cli.py`, which parses `RunConfig` from CLI args, environment variables, and `.env` file.
2. **Load dataset** — `run_benchmark()` loads `EvaluationRecord` entries from the dataset JSONL file.
3. **Filter records** — `BenchmarkRunner` applies debug mode or `--record-ids` filtering.
4. **Spawn workers** — For each record (and each trial if `num_trials > 1`), a `ConversationWorker` is created and assigned a port from the pool.
5. **Run conversations** — Workers run concurrently (up to `max_concurrent_conversations`). Each worker starts an `AssistantServer`, connects a `UserSimulator`, and runs a spoken conversation.
6. **Inline gate** — After each conversation, `check_conversation_finished()` determines if the conversation ended naturally.
7. **Validation** — `ValidationRunner` runs validation metrics on finished conversations (`user_behavioral_fidelity`, `user_speech_fidelity`).
8. **Retry** — Records that fail the gate or validation are archived and rerun, up to `max_rerun_attempts`.
9. **Metrics** — `MetricsRunner` computes the full metrics suite on all successful records.
10. **Aggregation** — EVA composite scores (EVA-A, EVA-X, EVA-overall) and pass@k statistics are computed.
11. **Summary** — Final `summary.json` is written with per-metric and aggregate results.

### 1.3 Module Map

| Directory | Responsibility |
|---|---|
| `src/eva/orchestrator/` | Benchmark scheduling, worker lifecycle, port pool, validation gating, retry loop |
| `src/eva/assistant/` | Pipecat voice server, pipeline assembly, agentic system, tool execution, audit logging |
| `src/eva/user_simulator/` | ElevenLabs user simulator client, bot-to-bot audio bridge, event logging |
| `src/eva/metrics/` | Metric base classes, registry, runner, context processor, individual metrics |
| `src/eva/models/` | Pydantic data models for config, records, results, agents |
| `src/eva/utils/` | Shared utilities: hashing, pass@k, logging, prompt management, LLM routing |

---

## 2. Orchestration Layer

### 2.1 Entry Points

```
main.py  ──>  cli.py:main()  ──>  run_benchmark.py:run_benchmark()  ──>  BenchmarkRunner
```

- `main.py` — Thin CLI entry point.
- `cli.py` — Parses `RunConfig` from CLI args, env vars, and `.env` file using pydantic-settings. Priority: CLI > env > `.env` > defaults.
- `run_benchmark.py` — Loads the dataset, instantiates `BenchmarkRunner`, and calls `runner.run(records)`.

### 2.2 BenchmarkRunner

**File:** `src/eva/orchestrator/runner.py`

`BenchmarkRunner` is the main orchestrator. It manages:

- **Port pool** — `PortPool` allocates WebSocket ports from a configurable range (`base_port` + `port_pool_size`) so multiple conversations can run in parallel on different ports.
- **Concurrency** — An asyncio semaphore limits the number of simultaneous conversations to `max_concurrent_conversations`.
- **Trial support** — When `num_trials > 1`, each record is run multiple times (e.g., `1.2.1/trial_0`, `1.2.1/trial_1`) for pass@k evaluation.
- **Retry loop** — Up to `max_rerun_attempts` total attempts per record. Each attempt: run conversation → check finished → run validation → archive failures and retry.
- **Config serialization** — Writes `config.json` to the output directory for reproducibility.

### 2.3 ConversationWorker

**File:** `src/eva/orchestrator/worker.py`

Each `ConversationWorker` manages a single conversation:

```
+-------------------------------------------------------+
|                  ConversationWorker                    |
|                                                       |
|  1. Allocate port from PortPool                       |
|  2. Start AssistantServer on port                     |
|  3. Create UserSimulator (connects to ws://port/ws)   |
|  4. Run conversation (await user_sim.run())           |
|  5. Collect stats from assistant server               |
|  6. Compute scenario DB hashes + latency stats        |
|  7. Cleanup: stop server, release port                |
|                                                       |
|  Output: ConversationResult                           |
+-------------------------------------------------------+
```

The worker creates per-record log files (`logs.log`) using asyncio task-local context variables so concurrent workers don't contaminate each other's logs.

### 2.4 Retry and Validation Gating

The retry loop inside `BenchmarkRunner.run()` works as follows:

1. **Run conversations** for all pending output IDs concurrently.
2. **Inline gate** — Check `conversation_finished` by parsing `elevenlabs_events.jsonl` for a `connection_state` event with `reason="goodbye"`. Records that didn't finish are immediately marked for retry.
3. **Validation metrics** — Run `user_behavioral_fidelity` and `user_speech_fidelity` on records that passed the inline gate.
4. **Archive failures** — Failed records are moved to `_attempt_N` subdirectories.
5. **Retry** — Failed records are rerun in the next attempt.

After all attempts are exhausted, only records that passed both gates proceed to the full metrics suite.

---

## 3. Bot-to-Bot Audio Architecture

This is the core architectural innovation of EVA. Two AI systems have a fully spoken conversation — the assistant (being evaluated) and a simulated user (playing the customer role) — communicating via real-time audio over a WebSocket connection.

### 3.1 Conceptual Overview

```
+------------------------+                       +------------------------+
|  ElevenLabs User       |      WebSocket        |  Pipecat Assistant     |
|  Simulator             |   (JSON + base64      |  Server                |
|  (Customer Role)       |    mulaw audio)       |  (Agent Role)          |
|                        |                       |                        |
|  - Conversational AI   | <----- audio -------> |  - FastAPI + Uvicorn   |
|  - Persona + Goal      |                       |  - STT -> LLM -> TTS  |
|  - end_call tool       |                       |  - Tool execution      |
|                        |                       |  - Audit logging       |
+-----------+------------+                       +-----------+------------+
            |                                                |
            |           BotToBotAudioInterface                |
            |           (bridges the two systems)             |
            +------------------------------------------------+
```

The user simulator generates speech (as a customer with a specific persona and goal), the assistant processes that speech and responds (using tools, following instructions), and the cycle continues until the conversation ends.

### 3.2 The Assistant Server

**File:** `src/eva/assistant/server.py`

The `AssistantServer` is a Pipecat-based WebSocket server built on FastAPI + Uvicorn:

- **WebSocket endpoint** — Listens at `/ws` (and `/` for compatibility).
- **Frame serialization** — Uses `TwilioFrameSerializer` for JSON + base64 mulaw encoding, a Twilio-compatible telephony protocol.
- **Pipeline assembly** — Creates a Pipecat `Pipeline` of frame processors based on the configured pipeline mode (cascade, S2S, or audio-LLM). See [Section 4](#4-assistant-pipeline-architectures) for details.
- **VAD** — `SileroVADAnalyzer` for voice activity detection with configurable stop threshold (`VAD_STOP_SECS = 0.2s`).
- **Smart turn detection** — `LocalSmartTurnAnalyzerV3` determines when the user has finished their turn (not just paused).
- **Initial greeting** — On client connection, the assistant speaks `"Hello! How can I help you today?"` via `TTSSpeakFrame` (cascade) or `LLMRunFrame` (S2S).
- **Observers** — Multiple Pipecat observers collect data during the conversation:
  - `BenchmarkLogObserver` — Writes `pipecat_events.jsonl` with TTS text, STT transcripts, and turn boundaries.
  - `MetricsFileObserver` — Writes `pipecat_metrics.jsonl` with TTFB and processing metrics.
  - `UserBotLatencyObserver` — Measures end-to-end response latency (user speech end → assistant speech start).
  - `MetricsLogObserver` — Console metrics output.
- **Event handlers** — Transport events (`on_client_connected`, `on_client_disconnected`), audio buffer events (`on_audio_data`, `on_track_audio_data`), turn tracking events, and `on_user_turn_stopped` which triggers the agentic system to process the user's complete transcript.

### 3.3 The User Simulator

**File:** `src/eva/user_simulator/client.py`

The `UserSimulator` uses the ElevenLabs Conversational AI SDK to simulate a customer:

- **Prompt construction** — A dynamic prompt is built from the `EvaluationRecord` using `PromptManager`:
  - `high_level_user_goal` — What the user wants to accomplish
  - `decision_tree` — Must-have criteria, negotiation behavior, escalation, resolution/failure conditions, edge cases
  - `information_required` — Facts the user knows (names, dates, confirmation numbers)
  - `user_persona` — Personality and communication style
  - `starting_utterance` — The user's opening line
  - `current_date_time` — Simulated date/time context
  - `user_simulator_context` — Domain-specific context from agent config

- **ElevenLabs Conversation** — Created with:
  - Agent ID from environment variable (per persona: `ELEVENLABS_USER_AGENT_ID_USER_PERSONA_{id}`)
  - `BotToBotAudioInterface` as the custom audio interface
  - Callbacks for speech events: `callback_agent_response` (user speaks), `callback_user_transcript` (assistant speaks)
  - Note: ElevenLabs naming is inverted from EVA's perspective — ElevenLabs' "agent" is EVA's "user", and ElevenLabs' "user" is EVA's "assistant"

- **Conversation lifecycle:**
  1. Create `BotToBotAudioInterface` and connect to assistant WebSocket
  2. Create ElevenLabs `Conversation` with audio interface
  3. Start session (blocking call, run in executor)
  4. Start keep-alive task (pings every 10s to prevent ElevenLabs timeout)
  5. Wait for conversation to end or timeout
  6. End session, check for `end_call` tool via ElevenLabs API
  7. Wait 4s grace period for assistant STT to finish processing
  8. Close WebSocket

- **End detection:**
  - `end_call` tool — ElevenLabs' user simulator can invoke this tool to signal conversation end (detected post-hoc via API)
  - Keep-alive timeout — 12 consecutive keep-alive pings without activity triggers end (2 minutes of inactivity)
  - Conversation timeout — Configurable via `conversation_timeout_seconds`

- **Event logging** — `ElevenLabsEventLogger` (`src/eva/user_simulator/event_logger.py`) records all events to `elevenlabs_events.jsonl` in JSONL format: user speech, assistant speech transcription, audio timing, connection state changes.

### 3.4 BotToBotAudioInterface — The Bridge

**File:** `src/eva/user_simulator/audio_interface.py`

This is the critical component that bridges the two AI systems. It implements the ElevenLabs `AudioInterface` abstract class to connect ElevenLabs' audio I/O to the assistant's WebSocket.

#### Bidirectional Audio Flow

```
USER SPEAKS (outbound):

  ElevenLabs             BotToBotAudioInterface              Assistant
  output()                                                   Server
     |                                                          ^
     |  16kHz PCM          _send_to_assistant()                 |
     +-------> send_queue -------> downsample 16kHz -> 8kHz     |
                                   encode PCM -> mulaw          |
                                   base64 encode                |
                                   ws.send(media frame) --------+
                                                        TwilioFrameSerializer
                                                        decodes


ASSISTANT SPEAKS (inbound):

  ElevenLabs             BotToBotAudioInterface              Assistant
  hears audio                                                Server
     ^                                                          |
     |  mulaw 8kHz         _continuous_input_stream()           |
     +------- input_callback <--- drain buffer every 250ms      |
              (250ms chunks)      pad with silence if empty      |
                                       ^                        |
                                       |                        |
                           _receive_from_assistant()            |
                           audio_buffer <--- ws.recv() <--------+
                           (decode base64,    mulaw 8kHz, base64 JSON
                            buffer mulaw)
```

#### Three Async Tasks

The interface runs three concurrent async tasks:

1. **`_receive_from_assistant()`** — Reads JSON messages from the WebSocket. On `media` events, decodes base64 mulaw audio and pushes to `audio_buffer` queue. Tracks assistant audio start/end timing.

2. **`_send_to_assistant()`** — Drains the `send_queue` (filled by `output()`), converts PCM 16kHz audio to mulaw 8kHz, and sends in 20ms chunks as WebSocket media frames. Between active speech, sends silence frames to maintain the audio stream (assistant needs continuous audio for VAD).

3. **`_continuous_input_stream()`** — Feeds audio to ElevenLabs at a steady 250ms rate (simulating a microphone). Collects audio from `audio_buffer`; when the buffer is empty, pads with mulaw silence (`0xFF`). Also handles silence detection for assistant audio end and triggers catch-up silence sending.

#### Key Methods

- **`output(audio)`** — Called by ElevenLabs when the simulated user generates speech. Queues raw PCM audio for sending.
- **`interrupt()`** — Called by ElevenLabs when it wants to interrupt playback. Deliberately a **no-op** — the user should keep talking even when the assistant responds.
- **`start(input_callback)`** — Called by ElevenLabs to register the callback for receiving assistant audio.
- **`stop()`** — Called by ElevenLabs to signal conversation end. Does NOT close the WebSocket immediately — the assistant's STT needs the connection alive to finish processing the last utterance.
- **`start_async()`** — Opens WebSocket, sends `connected` and `start` protocol messages, launches the three async tasks.
- **`stop_async()`** — Cancels tasks, sends `stop` message, closes WebSocket.

### 3.5 Audio Encoding and Format

| Stage | Format | Sample Rate | Details |
|---|---|---|---|
| ElevenLabs output | 16-bit PCM mono | 16kHz | Raw audio from user simulator TTS |
| Bridge conversion | mulaw | 8kHz | `audioop.ratecv()` downsample + `audioop.lin2ulaw()` encode |
| WebSocket transport | base64-encoded mulaw | 8kHz | JSON `{"event":"media","media":{"payload":"..."}}` |
| Assistant input | mulaw | 8kHz | Decoded by `TwilioFrameSerializer` |
| Assistant output | mulaw | 8kHz | Encoded by `TwilioFrameSerializer` |
| ElevenLabs input | mulaw | 8kHz | Passed directly (configured with `ELEVENLABS_INPUT_FORMAT = "mulaw"`) |

**Key constants** (from `audio_interface.py`):

| Constant | Value | Meaning |
|---|---|---|
| `PCM_SAMPLE_WIDTH` | 2 | 16-bit PCM = 2 bytes per sample |
| `ASSISTANT_SAMPLE_RATE` | 8000 | Assistant uses 8kHz mulaw |
| `ELEVENLABS_OUTPUT_RATE` | 16000 | ElevenLabs outputs 16kHz PCM |
| `SEND_CHUNK_DURATION_MS` | 20 | Send audio in 20ms chunks for real-time streaming |
| `SEND_CHUNK_SIZE_PCM` | 640 | 16000 * 0.02 * 2 = 640 bytes per chunk |
| `INPUT_CHUNK_DURATION` | 0.25 | Feed ElevenLabs every 250ms (simulating microphone) |
| `INPUT_FRAMES_PER_BUFFER` | 4000 | 250ms at 16kHz = 4000 samples |

### 3.6 WebSocket Protocol

The WebSocket uses a JSON-based protocol called `voice-bench-v1`, compatible with Twilio's media streams format:

| Event | Direction | Purpose |
|---|---|---|
| `connected` | Client → Server | Initial handshake with protocol version and conversation ID |
| `start` | Client → Server | Signal that audio streaming is beginning |
| `media` | Bidirectional | Audio data with base64-encoded mulaw payload in `media.payload` |
| `stop` | Client → Server | Signal that audio streaming is ending |

The `TwilioFrameSerializer` on the assistant side handles encoding/decoding this protocol into Pipecat audio frames.

### 3.7 Turn Management and Silence Detection

Turn management is critical for natural conversation flow. The system uses multiple mechanisms:

**On the assistant side (Pipecat):**
- `SileroVADAnalyzer` detects voice activity with `stop_secs = 0.2s` (200ms of silence triggers stop).
- `LocalSmartTurnAnalyzerV3` determines if the user has completed their turn (vs. mid-sentence pause) using the transcript.
- `LLMContextAggregatorPair` manages user/assistant context aggregation and fires `on_user_turn_stopped` when a complete user turn is available.

**On the bridge side (`BotToBotAudioInterface`):**
- **Silence detection threshold** — `SILENCE_DETECTION_THRESHOLD_S = 0.2s`. When 200ms passes without receiving assistant audio, the bridge considers assistant speech ended.
- **Catch-up silence** — After assistant audio ends, `ASSISTANT_CATCHUP_SILENCE_CHUNKS = 10` chunks (200ms) of silence are sent to the assistant to help its VAD detect end-of-speech cleanly.
- **User end detection delay** — `USER_END_DETECTION_DELAY_INTERVALS = 30` (600ms at 20ms per interval). A longer delay for user audio end to avoid splitting natural pauses.
- **Continuous silence sending** — When the user has stopped speaking and the assistant hasn't started, silence frames are continuously sent to the assistant (18 chunks per 250ms cycle) so its VAD stays responsive.
- **`_should_send_assistant_silence()` / `_should_send_user_silence()`** — Logic to determine which type of silence to send based on which party stopped speaking most recently. In interruption scenarios where both parties have recent end timestamps, the one who ended more recently determines the silence direction.

### 3.8 Interruption Handling

Interruptions happen when one party speaks while the other is still talking:

- **User interrupts assistant** — The user starts speaking while assistant audio is still being received. The `_on_user_audio_start()` handler tracks this via `_user_audio_active` and `_assistant_audio_active` flags. The Pipecat pipeline has `allow_interruptions=True`, so the assistant stops its current response and processes the new user input.
- **Assistant interrupts user** — The assistant starts responding while the user is still speaking. The `interrupt()` method on `BotToBotAudioInterface` is deliberately a no-op — the user simulator continues speaking regardless.
- **Both ended recently** — When both `_user_audio_ended_time` and `_assistant_audio_ended_time` are set, the silence logic uses timestamp comparison to determine who ended more recently and what silence to send.

### 3.9 Conversation Lifecycle

```
1. ConversationWorker._start_assistant()
   +-- AssistantServer.start()
       +-- FastAPI/Uvicorn server starts on port N
           +-- Listening at ws://localhost:N/ws

2. ConversationWorker._start_user_simulator()
   +-- UserSimulator created (not yet connected)

3. ConversationWorker._run_conversation()
   +-- UserSimulator.run_conversation()
       |-- BotToBotAudioInterface.start_async()
       |   |-- websockets.connect(ws://localhost:N/ws)
       |   |-- Send {"event":"connected","protocol":"voice-bench-v1"}
       |   |-- Send {"event":"start"}
       |   +-- Launch 3 async tasks (receive, send, input_stream)
       |
       |-- ElevenLabs Conversation.start_session()
       |   +-- Calls audio_interface.start(input_callback)
       |       +-- Three tasks now actively streaming audio
       |
       |-- Assistant: on_client_connected -> TTSSpeakFrame("Hello!...")
       |   +-- Assistant speaks greeting -> audio flows through bridge
       |       +-- ElevenLabs hears greeting -> user responds
       |
       |   ... conversation continues turn by turn ...
       |
       |-- End detection:
       |   |-- ElevenLabs invokes end_call tool -> conversation ends
       |   |-- OR: keep-alive timeout (12 consecutive pings, ~2min)
       |   +-- OR: conversation_timeout_seconds exceeded
       |
       |-- Conversation.end_session()
       |-- Check end_call via ElevenLabs API (with polling/backoff)
       |-- Wait 4s grace period for assistant STT
       +-- BotToBotAudioInterface.stop_async()
           |-- Cancel async tasks
           |-- Send {"event":"stop"}
           +-- Close WebSocket

4. ConversationWorker._cleanup()
   +-- AssistantServer.stop()
       |-- Cancel PipelineTask
       |-- Stop Uvicorn server
       +-- Save outputs (audio WAV files, audit log, latencies)
```

---

## 4. Assistant Pipeline Architectures

The assistant supports three distinct pipeline modes, selected by the `RunConfig.model` discriminated union type.

### 4.1 Cascade (STT + LLM + TTS)

**Config type:** `PipelineConfig`

The standard pipeline where each stage is a separate service:

```
transport.input()
    |
    v
STT Service (e.g., Deepgram, OpenAI Whisper)
    |  TranscriptionFrame
    v
UserObserver (metrics collection)
    |
    v
user_aggregator (LLMContextAggregator)
    |  Fires on_user_turn_stopped when turn complete
    v
BenchmarkAgentProcessor
    |  Calls AgenticSystem: LLM + tool loop
    |  Emits TTSSpeakFrame with response text
    v
TTS Service (e.g., Cartesia, ElevenLabs)
    |  Audio frames
    v
transport.output()
    |
    v
AudioBufferProcessor (records audio)
    |
    v
assistant_aggregator
```

**Key files:**
- `src/eva/assistant/server.py` — Pipeline assembly in `_create_pipeline()`
- `src/eva/assistant/pipeline/agent_processor.py` — `BenchmarkAgentProcessor` that bridges Pipecat frames to the `AgenticSystem`

**Turn detection:**
- `VADUserTurnStartStrategy` — VAD triggers user turn start
- `TurnAnalyzerUserTurnStopStrategy` with `LocalSmartTurnAnalyzerV3` — Analyzes the transcript to determine if the user has finished their turn

### 4.2 Speech-to-Speech (S2S)

**Config type:** `SpeechToSpeechConfig`

Direct audio-in, audio-out via realtime LLM services (e.g., GPT Realtime, Gemini Live):

```
transport.input()
    |
    v
UserAudioCollector (buffers raw audio)
    |
    v
UserObserver (metrics)
    |
    v
user_aggregator
    |
    v
InstrumentedRealtimeLLMService (or base realtime LLM)
    |  Handles audio directly, tool calls via _realtime_tool_handler
    v
transport.output()
    |
    v
AudioBufferProcessor + assistant_aggregator
```

The realtime LLM service processes audio directly — there is no separate STT or TTS. Tool calls are handled via `_realtime_tool_handler`, which wraps tool execution to record calls/responses in the audit log.

**Key file:** `src/eva/assistant/pipeline/realtime_llm.py`

### 4.3 Audio-LLM (S2T+TTS)

**Config type:** `AudioLLMConfig`

For models that accept audio input but produce text output (e.g., Ultravox). A separate TTS service synthesizes the text response:

```
transport.input()
    |
    v
AudioLLMUserAudioCollector (buffers audio with pre-speech buffer)
    |
    v
UserObserver (metrics)
    |
    v
user_aggregator
    |
    v
ParallelPipeline
    +-- Branch 1: InputTranscriptionContextFilter -> AudioTranscriptionProcessor
    |   (parallel transcription for logging -- not used by the model)
    +-- Branch 2: AudioLLMProcessor
        (sends buffered audio to model, processes text response, handles tools)
    |
    v
TTS Service
    |
    v
transport.output() + AudioBufferProcessor + assistant_aggregator
```

**Key file:** `src/eva/assistant/pipeline/audio_llm_processor.py`

### 4.4 Pipeline Configuration

The three modes are selected via a discriminated union in `src/eva/models/config.py`:

```python
ModelConfigUnion = Annotated[
    PipelineConfig | SpeechToSpeechConfig | AudioLLMConfig,
    Discriminator(...)
]
```

- `PipelineConfig` — Has `llm`, `stt`, `tts` fields. Standard cascade.
- `SpeechToSpeechConfig` — Has `s2s` field. Direct audio model.
- `AudioLLMConfig` — Has `audio_llm` and `tts` fields. Audio input, text+TTS output.

---

## 5. Agentic System and Tool Execution

### 5.1 AgenticSystem

**File:** `src/eva/assistant/agentic/system.py`

The `AgenticSystem` orchestrates the LLM conversation with tool calling:

1. **Receive user text** — From `BenchmarkAgentProcessor.process_complete_user_turn()` (cascade) or `AudioLLMProcessor` (audio-LLM).
2. **Build context** — System prompt (from `AgentConfig.instructions` with `current_date_time` interpolation), conversation history, and available tools.
3. **Call LLM** — Via `LiteLLMClient` which routes through LiteLLM for provider abstraction.
4. **Process tool calls** — If the LLM returns tool calls, execute them sequentially via `ToolExecutor`, append results to context, and call the LLM again. This loops until the LLM produces a text response (no more tool calls).
5. **Stream response** — Response tokens are streamed back to the pipeline.
6. **Special handling** — `transfer_to_agent` tool call terminates the conversation (the assistant is handing off to a live agent).

All interactions are recorded in the `AuditLog`.

### 5.2 ToolExecutor

**File:** `src/eva/assistant/tools/tool_executor.py`

`ToolExecutor` executes tools as Python functions against a scenario database:

- **Architecture** — One Python module per domain (e.g., `eva.assistant.tools.airline_tools`). The module is loaded dynamically via `importlib`.
- **Scenario database** — A JSON file per scenario (e.g., `data/airline_scenarios/1.2.1.json`) containing test data (reservations, flights, policies, etc.).
- **Execution** — `execute(tool_name, arguments)` calls the corresponding Python function with `params` (the arguments), `db` (the scenario database), and `call_index` (how many times this tool has been called). The function can query and mutate the database.
- **State tracking** — The database is mutated in-memory during the conversation. At the end, it's saved as `final_scenario_db.json` for comparison against `expected_scenario_db` in the `task_completion` metric.
- **Schema validation** — Tool schemas are loaded from the agent YAML config and exposed to the LLM in OpenAI function-calling format.

### 5.3 AuditLog

**File:** `src/eva/assistant/agentic/audit_log.py`

The `AuditLog` is a comprehensive record of everything that happens during a conversation:

- **LLM calls** — Prompt, response, latency, token counts, cost, model info
- **Tool calls** — Tool name, parameters, call index
- **Tool responses** — Tool name, response data, status
- **Conversation messages** — User inputs and assistant outputs with timestamps

Written to `audit_log.json` in the record output directory.

### 5.4 Conversation Termination

Conversations can end through several mechanisms:

| Mechanism | Side | How |
|---|---|---|
| `transfer_to_agent` tool | Assistant | LLM decides to transfer to a live agent. `AgenticSystem` detects this and signals end. |
| `end_call` tool | User Sim | ElevenLabs user simulator invokes this tool to end the call. Detected post-hoc via ElevenLabs API. |
| Goodbye detection | User Sim | ElevenLabs detects natural conversation end. |
| Keep-alive timeout | User Sim | 12 consecutive keep-alive pings without activity (~2 minutes). |
| Conversation timeout | Worker | `conversation_timeout_seconds` exceeded. |
| Error | Either | Unrecoverable error in either system. |

---

## 6. Logging and Data Flow

### 6.1 Three Log Sources

Each conversation produces three independent log files from different components:

| Log File | Producer | Contents |
|---|---|---|
| `audit_log.json` | `AgenticSystem` / `AuditLog` | LLM calls (prompts, responses, latency, tokens), tool calls and responses, user/assistant message history |
| `pipecat_events.jsonl` | `BenchmarkLogObserver` | Pipecat pipeline events: TTS input text, STT transcripts, turn start/stop, frame processing. Uses `WallClock` for consistent timestamps. |
| `elevenlabs_events.jsonl` | `ElevenLabsEventLogger` | ElevenLabs events: user intended speech, assistant speech transcription (what user sim heard), audio start/end timing, connection state changes |

These three sources were never designed to work together — they have different timing, formats, and quirks. The `MetricsContextProcessor` joins them into a unified timeline.

### 6.2 MetricsContextProcessor

**File:** `src/eva/metrics/processor.py`

The `MetricsContextProcessor` parses and joins the three log sources into a `MetricContext` — the data capsule that all metrics receive. Key processing:

- **Turn numbering** — Turn 0 is the assistant greeting. Subsequent turns increment on user audio start. Same turn ID across user and assistant dictionaries means the assistant's reply to that user turn.
- **Intended vs. transcribed** — Maintains both what speakers *intended* to say (TTS input / user sim prompt) and what was *actually heard* (STT output). This distinction is critical for evaluating speech fidelity.
- **Conversation trace** — A unified chronological view interleaving user turns, assistant turns, and tool calls/responses.
- **Interruption detection** — Identifies turns where audio overlapped.
- **Robustness** — Handles delayed/out-of-order timestamps, duplicated entries, false speech detection, and missing events through multiple fallback paths.

For full details on `MetricContext` fields, see [MetricContext Documentation](metric_context.md).

### 6.3 End-to-End Data Flow

```
+--------------------+
|   Dataset JSONL    |--> EvaluationRecord (user_goal, persona, scenario_db)
+--------+-----------+
         |
         v
+--------------------+
| ConversationWorker |
|                    |    Outputs:
|  AssistantServer --+--> audit_log.json, pipecat_events.jsonl
|       |            |    response_latencies.json, audio WAV files
|       v            |    initial_scenario_db.json, final_scenario_db.json
|  UserSimulator   --+--> elevenlabs_events.jsonl
|                    |
|  ConversationResult +--> result.json
+--------+-----------+
         |
         v
+--------------------+
|   MetricsRunner    |
|                    |
|  MetricsContext    |    Joins 3 log sources + dataset ground truth
|  Processor         |    + scenario DB states + agent config
|       |            |
|       v            |
|  Run metrics       |    Each metric receives MetricContext
|  concurrently      |    and returns MetricScore
|       |            |
|       v            |
|  metrics.json      |    Per-record metric scores + context snapshot
+--------+-----------+
         |
         v
+--------------------+
|    Aggregation     |
|                    |
|  EVA Composites    |--> EVA-A, EVA-X, EVA-overall (pass + mean)
|  Pass@k stats      |--> Multi-trial statistical evaluation
|                    |
|  summary.json      |--> Final aggregate results
+--------------------+
```

---

## 7. Validation Pipeline

**File:** `src/eva/orchestrator/validation_runner.py`

### 7.1 Two-Stage Design

Validation uses a fast-gate pattern to avoid running expensive LLM judge metrics on conversations that clearly failed:

1. **Stage 1 (Fast Gate)** — `conversation_finished`: A deterministic code check that parses `elevenlabs_events.jsonl` for a `connection_state` event with `reason="goodbye"`. Binary pass/fail, no LLM needed. Records that fail are immediately marked `not_finished`.

2. **Stage 2 (Detailed Validation)** — Only runs on records that passed Stage 1:
   - `user_behavioral_fidelity` — LLM judge that checks if the user simulator behaved correctly (no goal drift, premature resolution, information volunteering, etc.)
   - `user_speech_fidelity` — Audio judge that checks user speech quality per turn

### 7.2 Validation Metrics

| Metric | Type | What It Checks |
|---|---|---|
| `conversation_finished` | Code | Did the conversation end naturally with a goodbye? |
| `user_behavioral_fidelity` | Text Judge | Did the user simulator follow its instructions without corruption? |
| `user_speech_fidelity` | Audio Judge | Was the user's speech clear and faithful to intended text? |

### 7.3 Failure Categories

`ValidationResult` classifies each record into one of three categories:

- **`not_finished`** — Conversation didn't end naturally (timeout, error, no goodbye)
- **`validation_failed`** — Conversation finished but failed a validation metric (user behavior was corrupted or speech quality was poor)
- **`passed`** — All validation metrics passed; record proceeds to full metrics

---

## 8. Metrics System

### 8.1 Metric Categories

EVA organizes metrics into four categories, evaluated in a layered fashion:

| Category | Purpose | Evaluated When |
|---|---|---|
| **Validation** | Quality gate — is this conversation usable? | Before other metrics (fast gate + detailed validation) |
| **Accuracy** | Did the agent do the right thing? | On validated records only |
| **Experience** | Was the interaction natural and appropriate? | On validated records only |
| **Diagnostic** | Debugging and component-level analysis | On validated records only; excluded from composite scores |

For individual metric details, scoring rubrics, and judge prompts, see [Metrics Documentation](metrics/README.md).

### 8.2 How Metrics Run

**Files:** `src/eva/metrics/runner.py`, `src/eva/metrics/base.py`, `src/eva/metrics/registry.py`

**Metric base class hierarchy:**

| Base Class | Type | Evaluation Method |
|---|---|---|
| `CodeMetric` | `CODE` | Synchronous, rule-based (no LLM). E.g., hash comparison, log parsing. |
| `TextJudgeMetric` | `TEXT_JUDGE` | LLM judge with text input. Sends conversation transcript to an LLM with a judge prompt. |
| `ConversationTextJudgeMetric` | `TEXT_JUDGE` | Whole-conversation text judge. Formats the full transcript and calls an LLM judge. |
| `PerTurnConversationJudgeMetric` | `TEXT_JUDGE` | Per-turn text judge in a single LLM call. Returns per-turn ratings, aggregated via mean or harmonic mean. |
| `AudioJudgeMetric` | `AUDIO_JUDGE` | LLM judge with audio input. Sends audio files to Gemini for per-turn evaluation. |

**Registry** — Metrics are auto-discovered via the `@register_metric` decorator. `get_global_registry()` returns all registered metrics.

**MetricsRunner lifecycle:**

1. **Discover records** — Find all record directories in the output folder (including archived attempts for pass@k).
2. **Build MetricContext** — For each record, `MetricsContextProcessor` joins the three log sources with dataset ground truth, scenario DB states, and agent config.
3. **Run metrics concurrently** — All requested metrics are run in parallel via `asyncio`.
4. **Smart caching** — Existing `metrics.json` is read; only missing metrics are computed. In rerun mode, only failed metrics are recomputed.
5. **Save results** — `metrics.json` is written per record with all metric scores and a context snapshot.
6. **Aggregate** — Compute EVA composites and pass@k across all records.
7. **Write summary** — `summary.json` with per-metric statistics and aggregate scores.

---

## 9. Scoring and Aggregation

**File:** `src/eva/metrics/aggregation.py`

### 9.1 EVA Composites

EVA defines composite scores across two dimensions:

**EVA-A (Accuracy):**

| Metric | Pass Threshold |
|---|---|
| `task_completion` | `== 1.0` |
| `faithfulness` | `>= 0.5` |
| `agent_speech_fidelity` | `>= 0.95` |

**EVA-X (Experience):**

| Metric | Pass Threshold |
|---|---|
| `conversation_progression` | `>= 0.5` |
| `turn_taking` | `>= 0.5` |
| `conciseness` | `>= 0.5` |

**Composite variants:**
- **`EVA-A_pass`** — Binary: 1.0 if all accuracy thresholds met, else 0.0
- **`EVA-A_mean`** — Arithmetic mean of accuracy metric scores
- **`EVA-X_pass`** — Binary: 1.0 if all experience thresholds met, else 0.0
- **`EVA-X_mean`** — Arithmetic mean of experience metric scores
- **`EVA-overall_pass`** — Derived: 1.0 only if both `EVA-A_pass` and `EVA-X_pass` are 1.0
- **`EVA-overall_mean`** — Arithmetic mean of all six component metrics

### 9.2 Pass@k Framework

**File:** `src/eva/utils/pass_at_k.py`

When `num_trials > 1`, each record is evaluated multiple times. Pass@k provides statistical evaluation:

- **pass@k** = `1 - C(n-c, k) / C(n, k)` — Probability that at least 1 of k randomly drawn samples passes. Measures whether the model *can* succeed.
- **pass^k** = `C(c, k) / C(n, k)` — Probability that *all* k randomly drawn samples pass. Measures consistency/reliability.

Where `n` = total trials, `c` = passing trials, `k` = number of draws.

---

## 10. Configuration

### 10.1 RunConfig

**File:** `src/eva/models/config.py`

`RunConfig` uses pydantic-settings with priority: CLI > env vars > `.env` > defaults.

Key fields:

| Field | Env Var | Description |
|---|---|---|
| `model` | `EVA_MODEL__*` | Pipeline configuration (discriminated union) |
| `domain` | `EVA_DOMAIN` | Domain name (e.g., `airline`) |
| `dataset_path` | `EVA_DATASET_PATH` | Path to dataset JSONL |
| `agent_config_path` | `EVA_AGENT_CONFIG_PATH` | Path to agent YAML |
| `output_dir` | `EVA_OUTPUT_DIR` | Output directory |
| `max_concurrent_conversations` | `EVA_MAX_CONCURRENT_CONVERSATIONS` | Parallel conversation limit |
| `conversation_timeout_seconds` | `EVA_CONVERSATION_TIMEOUT_SECONDS` | Per-conversation timeout |
| `max_rerun_attempts` | `EVA_MAX_RERUN_ATTEMPTS` | Retry attempts for failed records |
| `num_trials` | `EVA_NUM_TRIALS` | Number of trials per record (for pass@k) |
| `metrics` | `EVA_METRICS` | List of metrics to compute |
| `record_ids` | `EVA_RECORD_IDS` | Specific record IDs to run |
| `debug` | `EVA_DEBUG` | Debug mode (run 1 record) |

### 10.2 AgentConfig

**File:** `src/eva/models/agents.py`, loaded from YAML (e.g., `configs/agents/airline_agent.yaml`)

| Field | Description |
|---|---|
| `description` | Agent description |
| `instructions` | System prompt template (supports `{current_date_time}` interpolation) |
| `tools` | List of tool definitions with schemas in OpenAI function-calling format |
| `role` | Agent role (e.g., "customer service representative") |
| `personality` | Agent personality traits |
| `tool_module_path` | Python module path for tool implementations |
| `user_simulator_context` | Domain-specific context line passed to user simulator prompt |

### 10.3 EvaluationRecord

**File:** `src/eva/models/record.py`, loaded from dataset JSONL

| Field | Description |
|---|---|
| `id` | Unique record identifier (e.g., `1.2.1`) |
| `user_goal` | Structured goal: high-level description, starting utterance, decision tree, required information |
| `user_config` | Persona configuration including ElevenLabs agent ID reference |
| `scenario_db` | Path to scenario database JSON |
| `expected_scenario_db` | Expected final database state after successful task completion |
| `current_date_time` | Simulated date/time for the conversation |
| `subflow_in_depth` | Subflow detail for the record |
| `expected_flow` | Subflow description |

### 10.4 Environment Variables

Key environment variables (see `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `ELEVENLABS_API_KEY` | Yes | ElevenLabs API key for user simulator |
| `ELEVENLABS_USER_AGENT_ID_USER_PERSONA_{N}` | Yes | ElevenLabs agent ID per user persona |
| `OPENAI_API_KEY` | Depends | Required for OpenAI models |
| `GOOGLE_APPLICATION_CREDENTIALS` | Depends | Required for Gemini audio judges |
| `EVA_DOMAIN` | Yes | Domain name |
| `EVA_MODEL__LLM` | Yes | LLM model name |

---

## 11. Text-Only Mode

**File:** `scripts/run_text_only.py`

For debugging LLM behavior without the audio pipeline, EVA provides a text-only mode:

- Bypasses all audio: no WebSocket, no TTS, no STT, no ElevenLabs
- Runs a pure text loop: user simulator prompt → LLM generates user text → `AgenticSystem` processes → LLM generates assistant response → repeat
- Compatible metrics subset: `task_completion`, `tool_call_validity`, `authentication_success`, `faithfulness`, `conversation_progression`, `conciseness`, `speakability`, `user_behavioral_fidelity`
- Useful for rapid iteration on agent instructions, tool logic, and conversation flow

---

## 12. Output Structure

A benchmark run produces the following directory structure:

```
output/{run_id}/
├── config.json                      # RunConfig serialized for reproducibility
├── summary.json                     # Aggregate results + pass@k statistics
│
└── records/
    ├── {record_id}/                 # One directory per record (or per trial)
    │   ├── result.json              # ConversationResult (completion status, duration, hashes)
    │   ├── metrics.json             # All metric scores + context snapshot
    │   ├── audit_log.json           # LLM calls, tool calls, conversation messages
    │   ├── pipecat_events.jsonl     # Pipecat pipeline events (TTS text, STT, turns)
    │   ├── elevenlabs_events.jsonl  # ElevenLabs events (speech, audio timing)
    │   ├── pipecat_metrics.jsonl    # Pipecat TTFB and processing metrics
    │   ├── response_latencies.json  # User→assistant response latencies
    │   ├── initial_scenario_db.json # Scenario database state at conversation start
    │   ├── final_scenario_db.json   # Scenario database state at conversation end
    │   ├── audio_assistant.wav      # Assistant audio channel (mono)
    │   ├── audio_user.wav           # User audio channel (mono)
    │   ├── audio_mixed.wav          # Mixed stereo audio (left=assistant, right=user)
    │   ├── transcript.jsonl         # Real-time transcript messages
    │   ├── logs.log                 # Per-record log file
    │   └── elevenlabs_conversation_details.json  # Full ElevenLabs API response
    │
    ├── {record_id}/trial_0/         # When num_trials > 1
    │   └── ...
    ├── {record_id}/trial_1/
    │   └── ...
    │
    └── {record_id}/_attempt_1/      # Archived failed attempts
        └── ...
```

---

## Appendix: Source File Reference

| File | Responsibility |
|---|---|
| **Orchestration** | |
| `src/eva/cli.py` | CLI entry point, `RunConfig` parsing |
| `src/eva/run_benchmark.py` | `run_benchmark()` top-level function |
| `src/eva/orchestrator/runner.py` | `BenchmarkRunner` — scheduling, retry loop, aggregation |
| `src/eva/orchestrator/worker.py` | `ConversationWorker` — single conversation lifecycle |
| `src/eva/orchestrator/port_pool.py` | `PortPool` — WebSocket port allocation |
| `src/eva/orchestrator/validation_runner.py` | `ValidationRunner` — two-stage validation |
| **Assistant** | |
| `src/eva/assistant/server.py` | `AssistantServer` — Pipecat WebSocket server, pipeline assembly |
| `src/eva/assistant/pipeline/agent_processor.py` | `BenchmarkAgentProcessor` — bridges Pipecat to agentic system |
| `src/eva/assistant/pipeline/realtime_llm.py` | `InstrumentedRealtimeLLMService` — S2S with audit logging |
| `src/eva/assistant/pipeline/audio_llm_processor.py` | `AudioLLMProcessor` — audio-LLM pipeline |
| `src/eva/assistant/pipeline/services.py` | Factory functions for STT, TTS, LLM services |
| `src/eva/assistant/pipeline/observers.py` | `BenchmarkLogObserver`, `MetricsFileObserver`, `WallClock` |
| `src/eva/assistant/agentic/system.py` | `AgenticSystem` — LLM conversation loop with tool calling |
| `src/eva/assistant/agentic/audit_log.py` | `AuditLog` — comprehensive conversation recording |
| `src/eva/assistant/tools/tool_executor.py` | `ToolExecutor` — Python function-based tool execution |
| `src/eva/assistant/tools/airline_tools.py` | Airline domain tool implementations |
| `src/eva/assistant/services/llm.py` | `LiteLLMClient` — LLM provider abstraction |
| **User Simulator** | |
| `src/eva/user_simulator/client.py` | `UserSimulator` — ElevenLabs conversation client |
| `src/eva/user_simulator/audio_interface.py` | `BotToBotAudioInterface` — WebSocket audio bridge |
| `src/eva/user_simulator/event_logger.py` | `ElevenLabsEventLogger` — JSONL event logging |
| **Metrics** | |
| `src/eva/metrics/base.py` | Metric base classes (`CodeMetric`, `TextJudgeMetric`, `AudioJudgeMetric`, etc.) |
| `src/eva/metrics/registry.py` | `MetricRegistry` — auto-discovery via `@register_metric` |
| `src/eva/metrics/runner.py` | `MetricsRunner` — orchestrates metric computation |
| `src/eva/metrics/processor.py` | `MetricsContextProcessor` — joins log sources into `MetricContext` |
| `src/eva/metrics/aggregation.py` | EVA composite definitions and aggregation logic |
| `src/eva/metrics/speech_fidelity_base.py` | Shared base for audio fidelity metrics |
| **Models** | |
| `src/eva/models/config.py` | `RunConfig`, `PipelineConfig`, `SpeechToSpeechConfig`, `AudioLLMConfig` |
| `src/eva/models/agents.py` | `AgentConfig`, `AgentTool` |
| `src/eva/models/record.py` | `EvaluationRecord` |
| `src/eva/models/results.py` | `ConversationResult`, `MetricScore`, `RecordMetrics`, `RunResult` |
| **Utilities** | |
| `src/eva/utils/hash_utils.py` | Deterministic JSON hashing, DB diff computation |
| `src/eva/utils/pass_at_k.py` | Pass@k and pass^k statistical computation |
| `src/eva/utils/prompt_manager.py` | Template-based prompt construction |
| `src/eva/utils/conversation_checks.py` | `check_conversation_finished()` fast check |
| `src/eva/utils/log_processing.py` | Log parsing utilities |
| `src/eva/utils/logging.py` | Per-record logging with asyncio task-local context |
