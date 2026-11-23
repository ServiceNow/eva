# Apriel Realtime

The Apriel adapter connects EVA's simulated caller to an existing Apriel Realtime
service through the OpenAI-compatible Realtime API. EVA runs the scenario tools
locally and returns their results to Apriel. A running endpoint and its credentials
are required; this integration does not deploy the model service.

Select the adapter with both of these settings:

```bash
export EVA_FRAMEWORK=openai_realtime
export EVA_MODEL__S2S=apriel-realtime
unset EVA_MODEL__LLM EVA_MODEL__AUDIO_LLM
```

`EVA_MODEL__S2S` selects the adapter. The `model` inside `EVA_MODEL__S2S_PARAMS`
selects the model served by the endpoint; obtain that identifier and the supported
session settings from the endpoint owner. Other framework/provider combinations
continue to select their existing adapters.

## Endpoint and authentication

Start with your endpoint's base URLs and model identifier:

```bash
export EVA_MODEL__S2S_PARAMS='{
  "model": "<served-model-id>",
  "base_url": "https://<endpoint-host>/v1",
  "websocket_base_url": "wss://<endpoint-host>/v1"
}'
export APRIEL_REALTIME_API_KEY="<realtime-api-key>"
# Set only when the endpoint sits behind a gateway that requires its own bearer token:
# export APRIEL_BACKEND_AUTH_TOKEN="<backend-auth-token>"
```

The SDK appends `/realtime` to the WebSocket base URL. Set the endpoint explicitly;
otherwise the SDK can fall back to its OpenAI endpoint.

Map the credentials into EVA's model parameters before starting EVA:

```bash
EVA_MODEL__S2S_PARAMS="$(python3 - <<'PY'
import json
import os

params = json.loads(os.environ["EVA_MODEL__S2S_PARAMS"])
key = os.environ["APRIEL_REALTIME_API_KEY"]
backend_token = os.environ.get("APRIEL_BACKEND_AUTH_TOKEN")
params["api_key"] = backend_token or key
params.pop("apriel_api_key", None)
if backend_token:
    params["apriel_api_key"] = key
print(json.dumps(params))
PY
)"
export EVA_MODEL__S2S_PARAMS
```

When `APRIEL_BACKEND_AUTH_TOKEN` is set, it becomes the bearer credential and the realtime
key is sent in the WebSocket subprotocol. Otherwise the realtime key is the bearer
credential.

The adapter reads `api_key` and optional `apriel_api_key` from
`EVA_MODEL__S2S_PARAMS`; it does not read the two `APRIEL_*` environment variables
directly. EVA redacts those parameter keys when saving run configuration. Keep
credentials out of committed files.

Continue using the normal EVA run configuration for the scenario, user simulator,
and judges. Their credentials and settings are separate from the Apriel endpoint.

## Session settings

Add supported options to `EVA_MODEL__S2S_PARAMS` before mapping credentials:

- `voice`, `vad_settings`, and `reasoning_effort` configure the session. No voice
  is sent unless `voice` is set.
- `enable_thinking` and `pre_tool_speech` are JSON booleans forwarded when supplied.
- `apriel` contains provider-specific extensions, such as
  `{"tts_input_streaming": "token"}`. Confirm supported values with the endpoint owner.
- `transcription_model` defaults to `whisper-1`; set it to `null` to disable input
  transcription if that is required by the endpoint.

`EVA_MODEL__PARALLEL_TOOL_CALLS`, when set, is also forwarded to the session.
Live compatibility of these settings must be validated against the target endpoint.

## Local tests

```bash
uv run pytest \
  tests/unit/assistant/test_apriel_realtime_server.py \
  tests/unit/assistant/test_openai_realtime_server.py \
  tests/unit/assistant/test_grok_voice_server.py \
  tests/unit/orchestrator/test_apriel_dispatch.py \
  tests/unit/orchestrator/test_framework_dispatch.py
```

These tests use mocked connections and do not establish live endpoint compatibility.
