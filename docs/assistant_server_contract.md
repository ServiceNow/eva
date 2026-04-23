# Adding a New Assistant Server

This document is a reference for implementing a new framework integration
for EVA evaluation. It covers requirements, shared utilities, and integration
points the server must satisfy.

## Overview

Any new assistant server implementation must evaluate voice assistants by running structured conversations between a
**user simulator** and an **assistant server**. The user simulator connects over a
local WebSocket using the Twilio media stream format. The server's job is to bridge
that WebSocket to the native API or pipeline used by the framework, then produce a
standard set of output files at the end of the conversation.

All servers inherit from `AbstractAssistantServer`
(`src/eva/assistant/base_server.py`). The base class provides shared components
(audit log, tool executor, audio buffers, output helpers). Implement two abstract
methods — `start()` and `stop()` — the rest of the contract flows from them.

The orchestrator (`src/eva/orchestrator/worker.py`) instantiates the server, calls
`start()`, runs the conversation, calls `stop()`, then reads the output files to
produce `ConversationResult`. The server name must be registered in
`_get_server_class()` in that file.

---

## Quick checklist

- [ ] Subclass `AbstractAssistantServer`
- [ ] Assert `isinstance(self.pipeline_config, SpeechToSpeechConfig)` (or the correct
      config type) in `__init__`
- [ ] Expose `ws://localhost:{self.port}/ws` accepting Twilio-framed audio
- [ ] Override `_audio_sample_rate` to match the recording sample rate
- [ ] Populate `self.user_audio_buffer` and `self.assistant_audio_buffer` during
      streaming, calling `sync_buffer_to_position` before each extend
- [ ] Write user turns to `self.audit_log` via `append_user_input()`
- [ ] Write assistant turns via `append_assistant_output()`
- [ ] Call `self.execute_tool()` for every tool call — never call
      `self.tool_handler.execute()` directly
- [ ] Write `model_response` latency via `self._metrics_log.write_latency()` on the
      first audio chunk of each turn
- [ ] Write token usage via `self._metrics_log.write_token_usage()`
- [ ] Call `await super().save_outputs()` inside `stop()` implementation
- [ ] Register the server in `_get_server_class()` in `worker.py`

---

## 1. Constructor

Call `super().__init__(**kwargs)` first. The base class sets up:

| Attribute | Type | Description |
|---|---|---|
| `self.audit_log` | `AuditLog` | Conversation event log — every user/assistant turn and tool call must go here |
| `self.tool_handler` | `ToolExecutor` | Executes tool calls against the scenario database |
| `self.user_audio_buffer` | `bytearray` | Accumulates raw user PCM during streaming |
| `self.assistant_audio_buffer` | `bytearray` | Accumulates raw assistant PCM during streaming |
| `self._audio_buffer` | `bytearray` | Mixed audio (leave empty — base class mixes automatically) |
| `self._audio_sample_rate` | `int` | Recording sample rate, default 24000 |

After calling `super().__init__()`, narrow the config type and initialize state:

```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    if isinstance(self.pipeline_config, SpeechToSpeechConfig):
        s2s_params = self.pipeline_config.s2s_params
    else:
        logger.error("Pipeline config is not SpeechToSpeechConfig")
        return
    self._model = s2s_params["model"] # model is required in the s2s params config
    self._audio_sample_rate = SAMPLE_RATE  # match  recording rate
    self._fw_log: FrameworkLogWriter | None = None
    self._metrics_log: MetricsLogWriter | None = None
    # ... other setup
```

The assertion fails fast with a clear message rather than an obscure `AttributeError`
later.

---

## 2. `start()` — server startup

`start()` must return **after** the server is ready to accept connections (i.e.
non-blocking with respect to the conversation).

The standard pattern used by all existing servers:

```python
async def start(self) -> None:
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self._fw_log = FrameworkLogWriter(self.output_dir)
    self._metrics_log = MetricsLogWriter(self.output_dir)

    self._app = FastAPI()

    @self._app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        await self._handle_session(websocket)

    @self._app.websocket("/")  # also accept root path
    async def ws_root(websocket: WebSocket):
        await websocket.accept()
        await self._handle_session(websocket)

    config = uvicorn.Config(self._app, host="0.0.0.0", port=self.port, log_level="warning")
    self._server = uvicorn.Server(config)
    self._server_task = asyncio.create_task(self._server.serve())
    # Wait until uvicorn is ready before returning
    while not self._server.started:
        await asyncio.sleep(0.05)
    self._running = True
```

The user simulator always connects to `ws://localhost:{port}/ws`, so both `/ws` and
`/` endpoints must be registered.

---

## 3. WebSocket transport — Twilio frame format

The user simulator sends and receives audio in the
[Twilio Media Streams](https://www.twilio.com/docs/voice/media-streams) format: JSON
envelopes over WebSocket. Helper functions in `audio_bridge.py` handle all encoding
and decoding.

### Incoming messages from the simulator

| Event type | What it means | Helper |
|---|---|---|
| `start` | Stream opened, contains `streamSid` | Parse `msg["start"]["streamSid"]` |
| `media` | Audio chunk (8 kHz mulaw, base64) | `parse_twilio_media_message(raw)` → `bytes` |
| `stop` | Stream closed | Break receive loop |
| `user_speech_start` | User began speaking (wall-clock timestamp) | `msg["timestamp_ms"]` |
| `user_speech_stop` | User stopped speaking (wall-clock timestamp) | `msg["timestamp_ms"]` |

The `user_speech_start` and `user_speech_stop` events carry accurate wall-clock
timestamps from the user simulator's VAD. Use them for latency measurement (see
section 7). If SDK provides user start/stop events, prefer those over the simulator's
events.

### Sending audio back to the simulator

```python
from eva.assistant.audio_bridge import create_twilio_media_message
msg = create_twilio_media_message(stream_sid, mulaw_chunk)
await websocket.send_text(msg)
```

Audio output must be 8 kHz mulaw. Send it in 160-byte chunks (20 ms each) at
real-time pace. Both `openai_realtime_server.py` and `gemini_live_server.py` use a
dedicated `_pace_audio_output` asyncio task to drain a queue at this rate — copy that
pattern. If audio is sent too fast or too slow, the user simulator's may incorrectly detect
turn boundaries.

### Audio conversion utilities (`audio_bridge.py`)

| Function | Converts |
|---|---|
| `mulaw_8k_to_pcm16_24k` | Twilio input → OpenAI / most models (24 kHz PCM16) |
| `mulaw_8k_to_pcm16_16k` | Twilio input → Gemini Live (16 kHz PCM16) |
| `pcm16_24k_to_mulaw_8k` | 24 kHz PCM16 → Twilio output |

Use `soxr`-based `pcm16_24k_to_mulaw_8k` for the 24→8 kHz path; plain
`audioop.ratecv` produces muffled audio due to lack of anti-aliasing.

---

## 4. Audio buffers

The base class declares three bytearrays. Populate `user_audio_buffer` and
`assistant_audio_buffer` during streaming. The base class mixes them automatically in
`_save_audio()`.

**Time alignment responsibility.** Both buffers represent a timeline in
samples. If user audio is at position T and assistant audio is at position T−Δ, the
resulting mixed WAV will have the audio offset by Δ. Before extending either buffer,
pad the *other* buffer to the same position:

```python
from eva.assistant.audio_bridge import sync_buffer_to_position

# When user audio arrives and the model is not speaking:
if not model_is_speaking:
    sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
self.user_audio_buffer.extend(pcm_chunk)

# When model audio arrives and the user is not speaking:
if not user_is_speaking:
    sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
self.assistant_audio_buffer.extend(pcm_chunk)
```

If at `stop()` time the buffers differ in length by more than 500 ms, `_save_audio()`
logs a warning. This does not fail the run but indicates a synchronisation bug.

Set `self._audio_sample_rate` to the expected sample rate. All three WAV files
(`audio_user.wav`, `audio_assistant.wav`, `audio_mixed.wav`) are written at this
rate. The conversion to mulaw for the WebSocket output is separate — do that before
sending, not before recording.

---

## 5. Audit log

The audit log is the source of truth for the conversation transcript and all
downstream metrics. Every server must write to it correctly.

| Method | When to call |
|---|---|
| `self.audit_log.append_user_input(text, timestamp_ms=...)` | When a user turn is complete and transcribed |
| `self.audit_log.append_assistant_output(text, tool_calls=..., timestamp_ms=...)` | When an assistant turn is complete |

Pass `timestamp_ms` as a string containing epoch milliseconds. Use the wall-clock
time the speech *started* for user entries (from `user_speech_start`), and the
wall-clock time of the first audio chunk for assistant entries. If server emits VAD events, may use those timestamps for user speech started, otherwise use the wall-clock time when the first audio chunk is received. Do not use transcription time as the timestamp as this may describe when transcription was received rather than when speech was emitted.


Do not call `append_llm_call()` for s2s/realtime models. That method is for cascade
pipelines where there is a separate LLM API call with a request/response pair to
record.

Tool calls are handled automatically by `execute_tool()` — do not write tool entries
to the audit log manually.

---

## 6. Tool execution

When the model requests a tool call, call `execute_tool()` on the base class:

```python
result = await self.execute_tool(tool_name, arguments)
# then send result back to the model API
```

This:
1. Appends a `tool_call` entry to the audit log (with timestamp)
2. Executes the tool against the scenario database via `self.tool_handler`
3. Appends a `tool_response` entry to the audit log (with timestamp)
4. Returns the result dict

Do **not** call `self.tool_handler.execute()` directly — that bypasses audit logging.

Arguments arriving as a JSON string must be decoded before passing:

```python
try:
    arguments = json.loads(raw_args_str)
except json.JSONDecodeError:
    arguments = {}
result = await self.execute_tool(tool_name, arguments)
```

---

## 7. Metrics logging

Instantiate `FrameworkLogWriter` and `MetricsLogWriter` in `start()` and store them
as instance attributes. Both write to files inside `self.output_dir`.

### Framework log (`framework_logs.jsonl`)

Record turn boundaries and assistant text for downstream processors:

```python
self._fw_log.turn_start(timestamp_ms=...)                    # when user turn begins; timestamp_ms is int | None (Optional - not used in processing)
self._fw_log.turn_end(was_interrupted=False, timestamp_ms=...) # when assistant finishes; timestamp_ms is int | None (Optional - not used in processing)
self._fw_log.llm_response(text)                              # full assistant response text
self._fw_log.tts_text(text)                                  # text sent to TTS API
self._fw_log.s2s_transcript(text, timestamp_ms=...)          # S2S transcript (what was actually spoken for s2s models that were interrupted); timestamp_ms is int | None (Optional - not used in processing)
```

### Metrics log (`pipecat_metrics.jsonl`)

#### Model response latency

Write one entry per turn when the first audio chunk arrives from the model:

```python
# latency_ms = first_audio_wall_ms - user_speech_stop_wall_ms
latency_s = latency_ms / 1000
self._metrics_log.write_latency("model_response", latency_s, self._model)
```

`user_speech_stop_wall_ms` comes from the `user_speech_stop` WebSocket event sent
by the audio interface or from a VAD event. Omit this call on turns where `user_speech_stop` was not
received (e.g. the opening greeting, which is model-initiated).

Sanity-check before writing: `0 < latency_ms < 30_000`.

#### Token usage

Write after each complete model response:

```python
self._metrics_log.write_token_usage(
    processor="my_framework",
    model=self._model,
    prompt_tokens=input_tokens,
    completion_tokens=output_tokens,
)
```

---

## 8. `stop()` — shutdown and output

`stop()` must:

1. Signal the session loop to exit (set `self._running = False`)
2. Cancel in-flight asyncio tasks
3. Shut down the uvicorn server
4. Call `await self.save_outputs()`

```python
async def stop(self) -> None:
    self._running = False
    if self._server:
        self._server.should_exit = True
    if self._server_task:
        self._server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._server_task
    await self.save_outputs()
```

`save_outputs()` (base class) writes:

| File | Contents |
|---|---|
| `audit_log.json` | Full conversation event log |
| `transcript.jsonl` | Simplified user/assistant transcript |
| `audio_user.wav` | User audio track |
| `audio_assistant.wav` | Assistant audio track |
| `audio_mixed.wav` | Mixed stereo-equivalent recording |
| `initial_scenario_db.json` | Scenario database before the conversation |
| `final_scenario_db.json` | Scenario database after tool mutations |

If the framework produces additional output (e.g. raw event logs), override
`save_outputs()` and call `super()`:

```python
async def save_outputs(self) -> None:
    await super().save_outputs()
    # write framework-specific files here
```

---

## 9. Config type

`pipeline_config` arrives as a union type from the orchestrator. For s2s models it
will be `SpeechToSpeechConfig`, which exposes:

```python
self.pipeline_config.s2s            # model identifier string
self.pipeline_config.s2s_params     # dict of additional params (api_key, voice, model, etc.)
```

Return if the config is not `SpeechToSpeechConfig`.

Server should be documented in the relevant `configs/` YAML with `framework:
my_framework` and `model: {s2s: my-model-id, s2s_params: {...}}`.

---

## 10. Registering the server

Add class to `_get_server_class()` in `src/eva/orchestrator/worker.py`:

```python
elif framework == "my_framework":
    from eva.assistant.my_framework_server import MyFrameworkAssistantServer
    return MyFrameworkAssistantServer
```

Use a lazy import (inside the `elif` branch) to avoid loading framework-specific
dependencies when running other frameworks.

---

## 11. Minimal skeleton

```python
import asyncio
import contextlib
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket

from eva.assistant.audio_bridge import (
    FrameworkLogWriter,
    MetricsLogWriter,
    create_twilio_media_message,
    mulaw_8k_to_pcm16_24k,
    parse_twilio_media_message,
    pcm16_24k_to_mulaw_8k,
    sync_buffer_to_position,
)
from eva.assistant.base_server import INITIAL_MESSAGE, AbstractAssistantServer
from eva.models.config import SpeechToSpeechConfig


class MyFrameworkAssistantServer(AbstractAssistantServer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.pipeline_config, SpeechToSpeechConfig):
            s2s_params = self.pipeline_config.s2s_params
        else:
            logger.error("Pipeline config is not SpeechToSpeechConfig")
            return
        self._model = s2s_params["model"]
        self._app: FastAPI | None = None
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None
        self._running = False
        self._fw_log: FrameworkLogWriter | None = None
        self._metrics_log: MetricsLogWriter | None = None

    async def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)

        self._app = FastAPI()

        @self._app.websocket("/ws")
        async def ws(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        @self._app.websocket("/")
        async def ws_root(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        config = uvicorn.Config(self._app, host="0.0.0.0", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            await asyncio.sleep(0.05)
        self._running = True

    async def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.should_exit = True
        if self._server_task:
            self._server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_task
        await self.save_outputs()

    async def _handle_session(self, websocket: WebSocket) -> None:
        stream_sid = self.conversation_id
        _user_speech_stop_ts: str | None = None
        audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # Connect to framework's API
        async with my_framework.connect(model=self._model) as session:

            async def _forward_user_audio() -> None:
                nonlocal stream_sid, _user_speech_stop_ts
                while self._running:
                    raw = await websocket.receive_text()
                    msg = json.loads(raw)
                    event = msg.get("event")

                    if event == "start":
                        stream_sid = msg["start"]["streamSid"]
                    elif event == "stop":
                        break
                    elif event == "user_speech_stop":
                        _user_speech_stop_ts = msg.get("timestamp_ms")
                    elif event == "media":
                        pcm = mulaw_8k_to_pcm16_24k(parse_twilio_media_message(raw))
                        sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
                        self.user_audio_buffer.extend(pcm)
                        await session.send_audio(pcm)

            async def _process_model_events() -> None:
                nonlocal _user_speech_stop_ts
                _first_audio = True
                async for event in session:
                    if event.type == "audio":
                        pcm = event.audio_bytes  # 24 kHz PCM16
                        if _first_audio:
                            _first_audio = False
                            wall_ms = str(int(time.time() * 1000))
                            self._fw_log.turn_start()
                            if _user_speech_stop_ts and self._metrics_log:
                                latency_ms = int(wall_ms) - int(_user_speech_stop_ts)
                                if 0 < latency_ms < 30_000:
                                    self._metrics_log.write_latency("model_response", latency_ms / 1000, self._model)
                            _user_speech_stop_ts = None

                        sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
                        self.assistant_audio_buffer.extend(pcm)
                        await audio_output_queue.put(pcm16_24k_to_mulaw_8k(pcm))

                    elif event.type == "transcript":
                        self.audit_log.append_assistant_output(event.text, timestamp_ms=str(int(time.time() * 1000)))
                        self._fw_log.llm_response(event.text)
                        self._fw_log.turn_end(was_interrupted=False)
                        _first_audio = True

                    elif event.type == "tool_call":
                        result = await self.execute_tool(event.function_name, event.arguments)
                        await session.send_tool_result(event.call_id, result)

                    elif event.type == "usage":
                        self._metrics_log.write_token_usage(
                            "my_framework", self._model, event.input_tokens, event.output_tokens
                        )

            async def _pace_audio_output() -> None:
                import time as _time
                next_send = _time.monotonic()
                while self._running:
                    chunk = await audio_output_queue.get()
                    await websocket.send_text(create_twilio_media_message(stream_sid, chunk))
                    next_send += 0.02  # 20 ms per 160-byte mulaw chunk
                    sleep = next_send - _time.monotonic()
                    if sleep > 0:
                        await asyncio.sleep(sleep)

            await asyncio.gather(
                _forward_user_audio(),
                _process_model_events(),
                _pace_audio_output(),
                return_exceptions=True,
            )
```

---

## 12. What the orchestrator expects after `stop()`

`ConversationWorker` reads the following after calling `stop()`. Missing files cause
the run to fail or produce `None` latency fields in the result.

| File | Required | Used for |
|---|---|---|
| `audit_log.json` | Yes | All quality metrics |
| `transcript.jsonl` | Yes | Turn-level metrics |
| `initial_scenario_db.json` | Yes | Task completion metrics |
| `final_scenario_db.json` | Yes | Task completion metrics |
| `audio_mixed.wav` | No | Human review |
| `audio_user.wav` | Yes | STT accuracy metrics |
| `audio_assistant.wav` | Yes | TTS quality metrics |
| `framework_logs.jsonl` | Yes | Turn boundary metrics |
| `pipecat_metrics.jsonl` | Yes | `model_response_latency` in `ConversationResult` |
