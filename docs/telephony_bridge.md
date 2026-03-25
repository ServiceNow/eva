# Telephony Bridge

The telephony bridge adds a benchmark mode for external voice assistants that EVA cannot run locally. In this mode, EVA keeps the existing ElevenLabs user simulator interface and replaces the local Pipecat assistant pipeline with a bridge server that forwards telephony audio to an external transport.

## Architecture

```text
ElevenLabs User Sim
  -> WebSocket (Twilio-style JSON + base64 μ-law 8kHz)
  -> TelephonyBridgeServer
  -> BaseTelephonyTransport
  -> External assistant

External assistant tool webhooks
  -> ToolWebhookService
  -> ToolExecutor
  -> Scenario DB
```

The implementation is split into three pieces:

- `src/eva/assistant/tool_webhook.py`
  Exposes EVA tools over `POST /tools/{call_id}/{tool_name}` and routes requests to a per-conversation `ToolExecutor`.
- `src/eva/assistant/telephony_bridge.py`
  Accepts the same WebSocket audio protocol as `AssistantServer`, records audio artifacts, and generates transcript and audit outputs.
- `src/eva/orchestrator/worker.py` and `src/eva/orchestrator/runner.py`
  Start the webhook service, select the telephony bridge when `TelephonyBridgeConfig` is used, and register or unregister conversation state.

## Configuration

Use `TelephonyBridgeConfig` under `model`:

```yaml
model:
  sip_uri: "sip:assistant@example.com"
  webhook_port: 8888
  webhook_base_url: "https://eva-benchmark.ngrok-free.app"
  stt: "deepgram"
  stt_params:
    api_key: "${DEEPGRAM_API_KEY}"
    model: "nova-2"
```

Fields:

- `sip_uri`
  External assistant SIP URI.
- `webhook_port`
  Local port for the tool webhook FastAPI service.
- `webhook_base_url`
  Public base URL that the external assistant uses for tool webhooks.
- `stt` and `stt_params`
  Optional post-call transcription settings for `transcript.jsonl`. The current implementation supports Deepgram prerecorded transcription.

## Webhook Routing

The webhook service exposes:

- `GET /health`
- `POST /tools/{call_id}/{tool_name}`

Request bodies can be either:

```json
{"confirmation_number": "ABC123"}
```

or:

```json
{"function_params": {"confirmation_number": "ABC123"}}
```

The response is the raw JSON returned by EVA's `ToolExecutor`.

The current bridge registers the EVA conversation ID as `call_id`. A future transport that exposes provider-native call IDs should map those IDs into the same webhook registry.

## Outputs

The telephony bridge writes the same core benchmark artifacts expected by the rest of EVA:

- `audio_user.wav`
- `audio_assistant.wav`
- `audio_mixed.wav`
- `transcript.jsonl`
- `audit_log.json`
- `initial_scenario_db.json`
- `final_scenario_db.json`

Audio received on the WebSocket or transport side is converted from μ-law 8kHz to 24kHz 16-bit PCM for recording. Transcript generation is post-call and segment-based.

## Transport Status

`TelephonyBridgeServer` is implemented against `BaseTelephonyTransport`.

- `SIPTransport` currently exists as a scaffold.
- It preserves the extension point for a real outbound SIP implementation without hard-coding a provider-specific dependency into EVA.
- The server, webhook lifecycle, recordings, transcript generation, and tests all work with an injected concrete transport.

If you want end-to-end external assistant calls today, implement a `BaseTelephonyTransport` subclass that can:

1. Start the outbound session.
2. Accept 8kHz μ-law audio from the bridge via `send_audio()`.
3. Emit 8kHz μ-law audio from the assistant back into the bridge.
4. Stop the external session cleanly.
