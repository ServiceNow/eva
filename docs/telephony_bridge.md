# Telephony Bridge

Benchmark any voice assistant as a black box using EVA's telephony bridge. The bridge connects EVA's ElevenLabs user simulator to an external assistant via Telnyx Call Control, while routing the assistant's tool calls through EVA's deterministic scenario database.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│  EVA Runner                                                           │
│                                                                       │
│  ┌──────────────┐     ┌──────────────────────┐     ┌───────────────┐ │
│  │ ElevenLabs   │ WS  │  Telephony Bridge     │ API │ Tool Webhook  │ │
│  │ User Sim     │◄───►│  (audio routing)      │     │ Service       │ │
│  │              │     │                       │     │ :8888         │ │
│  └──────────────┘     └──────────┬───────────┘     └───────┬───────┘ │
│                                  │                         │         │
│                    Call Control   │ Media Stream            │ ngrok   │
│                    API + WSS     │ (L16/16kHz)             │         │
└──────────────────────────────────┼─────────────────────────┼─────────┘
                                   │                         │
                          ┌────────▼─────────┐     ┌────────▼────────┐
                          │  Telnyx Platform  │     │  External       │
                          │  (Call Control)   │────►│  Assistant      │
                          └──────────────────┘     └─────────────────┘
```

## How It Works

1. **Call Control transport** places an outbound call via Telnyx Call Control API to the assistant's SIP URI
2. Audio flows bidirectionally through Telnyx's **media streaming** WebSocket (L16 codec, 16kHz)
3. The assistant's **tool calls** hit the tool webhook service via ngrok, which executes them against EVA's deterministic scenario database
4. EVA records transcripts, audio, tool calls, and metrics for evaluation

## Prerequisites

- Telnyx account with:
  - A **Call Control Application** (routes call webhooks)
  - A **phone number** assigned to that application (for caller ID)
  - An **API key** (v2)
- A **stable ngrok domain** (free tier includes one static domain)
- ElevenLabs API key + Conversational AI agent(s) for user simulation

## Configuration

All configuration is via environment variables:

```bash
# Required: Telnyx Call Control
export EVA_MODEL__SIP_URI="sip:assistant@sip.telnyx.com"   # or +1NXXNXXXXXX
export EVA_MODEL__TELNYX_API_KEY="KEY..."
export EVA_MODEL__CALL_CONTROL_APP_ID="1234567890"
export EVA_MODEL__CALL_CONTROL_FROM="+12125551234"

# Required: Tool webhooks via ngrok
export EVA_MODEL__WEBHOOK_BASE_URL="https://your-domain.ngrok-free.dev"

# Required: ElevenLabs user simulator
export ELEVENLABS_API_KEY="..."
export ELEVENLABS_USER_AGENT_ID_USER_PERSONA_1="agent_..."
export ELEVENLABS_USER_AGENT_ID_USER_PERSONA_2="agent_..."

# Required: LLM judge (for metrics)
export EVA_MODEL_LIST='[{"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o", "api_key": "sk-..."}}]'

# Optional: STT for transcript generation
export EVA_MODEL__STT="deepgram"
export EVA_MODEL__STT_PARAMS='{"model": "nova-3"}'
```

## Running

```bash
# Start ngrok (in a separate terminal)
ngrok http 8888 --url your-domain.ngrok-free.dev

# Run a single scenario
python -m eva --record-ids 1.1.2

# Run all scenarios
python -m eva
```

## Audio Format

- **Media streaming**: L16 (16-bit PCM, 16kHz, mono) — matches ElevenLabs native rate, no codec conversion needed
- **Recordings**: PCM 16-bit WAV at 24kHz (EVA standard)
- The bridge handles sample rate conversion between the transport (16kHz) and EVA's recording format (24kHz)

## Tool Webhooks

External assistants call tools by hitting webhook URLs. The tool webhook service:

1. Receives POST requests at `https://<ngrok-domain>/tools/{call_control_id}/{tool_name}`
2. Routes to the correct conversation's `ToolExecutor`
3. Executes against EVA's deterministic scenario database
4. Returns the result as JSON

Configure your assistant's tools with URL template:
```
https://your-domain.ngrok-free.dev/tools/{{eva_call_id}}/{tool_name}
```

The `{{eva_call_id}}` is a [customer-defined dynamic variable](https://developers.telnyx.com/docs/inference/ai-assistants/dynamic-variables#customer-defined-dynamic-variables) that EVA injects via the `X-Eva-Call-Id` custom SIP header when placing the call. The assistant resolves it at runtime to route tool calls back to the correct conversation. Because EVA generates this ID before dialing, routing is deterministic even with concurrent benchmark calls — no reliance on platform-assigned call IDs that differ between legs.

## Latency Measurement

Response latency is measured **client-side** at the bridge, matching how Pipecat's `UserBotLatencyObserver` works:

```
User speech ends (last audio chunk received from ElevenLabs)
         ↓ gap = response latency
Assistant speech starts (first audio chunk received from Telnyx media stream)
```

This includes the full round-trip: network latency between the bridge and Telnyx, media streaming overhead, plus the assistant's own processing time (STT → LLM → TTS). The result is directly comparable to Pipecat-based benchmark latencies.

Written to `response_latencies.json` in the same format EVA's metrics processor expects.

## Post-Call Enrichment

After each call ends, the bridge fetches the full conversation history from the Telnyx Conversations API using the call's `call_control_id`. This enriches the local audit log with:

- **User and assistant speech turns** — the local audit log only captures tool calls; the API provides the full transcript
- **Tool call/response pairs** — complete with function arguments and results

This enables EVA's metrics processor to score task completion and conversation quality even though the assistant ran as an opaque black box.

## Stable ngrok Domain (Required)

Ephemeral ngrok URLs change on every restart, causing silent webhook mismatches. EVA rejects them and requires a stable domain:

1. Go to https://dashboard.ngrok.com/domains
2. Claim a free static domain (e.g., `your-name.ngrok-free.dev`)
3. Start ngrok: `ngrok http 8888 --url your-name.ngrok-free.dev`
