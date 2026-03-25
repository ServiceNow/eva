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
│                    API + WSS     │ (PCMU/8kHz)             │         │
└──────────────────────────────────┼─────────────────────────┼─────────┘
                                   │                         │
                          ┌────────▼─────────┐     ┌────────▼────────┐
                          │  Telnyx Platform  │     │  External       │
                          │  (Call Control)   │────►│  Assistant      │
                          └──────────────────┘     └─────────────────┘
```

## How It Works

1. **Call Control transport** places an outbound call via Telnyx Call Control API to the assistant's SIP URI
2. Audio flows bidirectionally through Telnyx's **media streaming** WebSocket (PCMU codec, 8kHz μ-law)
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

- **Media streaming**: PCMU (μ-law, 8kHz, mono) — zero codec conversion on the wire
- **Recordings**: PCM 16-bit WAV at 24kHz (EVA standard)
- The bridge handles sample rate conversion between the transport (8kHz) and EVA's recording format (24kHz)

## Tool Webhooks

External assistants call tools by hitting webhook URLs. The tool webhook service:

1. Receives POST requests at `https://<ngrok-domain>/tools/{call_control_id}/{tool_name}`
2. Routes to the correct conversation's `ToolExecutor`
3. Executes against EVA's deterministic scenario database
4. Returns the result as JSON

Configure your assistant's tools with URL template:
```
https://your-domain.ngrok-free.dev/tools/{{call_control_id}}/{tool_name}
```

The `{{call_control_id}}` dynamic variable is resolved by Telnyx at runtime.

## Stable ngrok Domain (Required)

Ephemeral ngrok URLs change on every restart, causing silent webhook mismatches. EVA rejects them and requires a stable domain:

1. Go to https://dashboard.ngrok.com/domains
2. Claim a free static domain (e.g., `your-name.ngrok-free.dev`)
3. Start ngrok: `ngrok http 8888 --url your-name.ngrok-free.dev`
