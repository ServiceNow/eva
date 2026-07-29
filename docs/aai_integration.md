# Evaluating the aai Voice Agent

EVA can evaluate the **aai** voice-agent framework via aai's **host mode**, in
which EVA supplies the domain system prompt, greeting, and tool schemas per
session and aai relays each tool call back to EVA for execution.

Tool execution must happen inside EVA: the accuracy metrics diff
`initial_scenario_db.json` against `final_scenario_db.json`, so a backend that
ran tools internally would score as if nothing happened.

## Prerequisites

1. **A running aai host with host mode enabled.** From your aai checkout:

   ```bash
   AAI_ALLOW_HOST=1 pnpm dev
   ```

   The default endpoint is `ws://localhost:3000/websocket`; EVA appends
   `?host=1` to select host mode. Without `AAI_ALLOW_HOST`, the host rejects the
   handshake and EVA aborts the conversation with a logged error.

   **Or a deployed agent**, at `wss://<host>/<slug>/websocket`. The platform
   gates host mode differently: an agent's WebSocket is deliberately
   unauthenticated, so allowing prompt and tool overrides on that footing would
   turn any deployed agent into an open LLM proxy billed to its owner. It
   therefore requires the owner's platform API key as a bearer token on the
   upgrade — set `s2s_params.api_key` or `AAI_API_KEY`. `401` means no key was
   sent; `403` means the key does not own that slug. No `AAI_ALLOW_HOST`
   involved.

2. **EVA's usual keys** in `.env` — a user simulator (ElevenLabs or OpenAI
   Realtime) plus the judge-model credentials the metrics need. See the main
   README.

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| `EVA_FRAMEWORK=aai` | — | Selects this server. |
| `EVA_MODEL__S2S=aai-host` | — | Marks the run as speech-to-speech. |
| `AAI_WS_URL` | `ws://localhost:3000/websocket` | aai host endpoint. |
| `s2s_params.ws_url` | — | Per-run override, takes precedence over `AAI_WS_URL`. |
| `s2s_params.api_key` / `AAI_API_KEY` | — | Agent owner's platform key. Required for a **deployed** agent, ignored by `aai dev`. |
| `s2s_params.model` | `aai-host` | Label recorded in the metrics log. |
| `s2s_params.input_sample_rate` | `16000` | PCM16 rate sent to aai. |
| `s2s_params.output_sample_rate` | `24000` | PCM16 rate received from aai. |

## Running

```bash
EVA_FRAMEWORK=aai EVA_MODEL__S2S=aai-host \
EVA_USER_SIMULATOR__PROVIDER=openai_realtime EVA_DOMAIN=itsm \
EVA_RECORD_IDS=15 EVA_MAX_CONCURRENT_CONVERSATIONS=1 eva
```

`AAI_ALLOW_HOST` belongs on the *host* process, not here — EVA never reads it.

`EVA_MAX_CONCURRENT_CONVERSATIONS` above 1 works: host mode builds a fresh
single-use runtime per connection, so each conversation gets its own session.

All three domains (airline, itsm, medical_hr) work without further setup — the
system prompt and tool schemas come from `configs/agents/*.yaml`.

## Architecture

```
user simulator ──Twilio μ-law 8k──▶ AAIAssistantServer ──PCM16 16k───▶ aai host
                ◀──Twilio μ-law 8k── localhost:{port}/ws ◀──PCM16 24k── ?host=1
                                             │
                                       execute_tool()
                                             ▼
                                      scenario database
```

| Module | Responsibility |
|---|---|
| `assistant/ws_bridge_server.py` | Shared bridge: Twilio framing, 20 ms pacing, recording buffers, turn bookkeeping, metrics. |
| `assistant/bridge_events.py` | Backend-agnostic event vocabulary and the session protocol. |
| `assistant/aai_session.py` | aai host handshake, audio frames, `tool_result` relay. |
| `assistant/aai_events.py` | aai wire-event models. |
| `assistant/aai_server.py` | Prompt and tool-schema construction. |

Adding another WebSocket voice backend means writing one session adapter plus a
thin server subclass — no changes to the bridge.

### Turn taking and latency

Two VAD systems are in play. The user simulator's `user_speech_start` /
`user_speech_stop` events are the only source used for `model_response` latency,
as the assistant-server contract specifies. Both carry wall-clock (`time.time()`)
milliseconds; the bridge subtracts them from its own `time.time()` reading, so
the simulator converts its monotonic send-loop clock before emitting
(`user_simulator/audio_bridge.py:_on_user_audio_end`). Emitting the raw
monotonic value makes every latency register as the clock gap (~1.8e12 ms) and
get discarded.
aai's own `speech_started` drives barge-in. Mixing the two would silently skew
the latency metric.

## Known gaps

- **No token usage.** aai emits no usage events, so `pipecat_metrics.jsonl`
  contains latency entries only. `model_response_latency` is unaffected;
  token-based cost reporting is unavailable for aai runs.
- **Backend output must be 24 kHz.** Only a 24 kHz → μ-law converter exists in
  `audio_bridge.py`. Another rate raises at session start with a clear message
  rather than producing distorted audio.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `aai host did not acknowledge the host-mode config` | `AAI_ALLOW_HOST` not set, or the host is not listening. |
| `HTTP 401` on connect | Deployed agent, no API key sent — set `s2s_params.api_key` or `AAI_API_KEY`. |
| `HTTP 403` on connect | The API key is valid but does not own that agent slug. |
| `llm: Failed after 3 attempts … Internal Server Error` | The *agent's own* LLM provider is failing. Host mode runs on the deployed agent's credentials and pipeline; nothing to fix in EVA. |
| `aai host rejected host mode` | Host mode disabled server-side, or the tool schemas failed validation (each tool needs a non-empty description). |
| `Could not connect to aai host` | Wrong `AAI_WS_URL`, or the host is not running. |
| Empty transcript, conversation scored as failure | Check the log for `Failed to open backend session` — the handshake aborted. |

## Tests

```bash
uv run --extra dev python -m pytest tests/unit/assistant/test_aai_events.py \
  tests/unit/assistant/test_aai_session.py \
  tests/unit/assistant/test_aai_server.py \
  tests/unit/assistant/test_ws_bridge_server.py \
  tests/integration/test_aai_fake_backend.py
```

The integration test drives both WebSocket hops against an in-process fake aai
host, so it needs no API keys and no Node process.

Note: `pytest` lives in the `dev` extra, which `uv sync` skips by default — hence
`--extra dev` (or run `uv sync --all-extras` once). Use `python -m pytest` rather
than `uv run pytest`, which silently resolves to a pytest outside the project
virtualenv.
