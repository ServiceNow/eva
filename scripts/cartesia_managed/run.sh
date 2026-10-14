#!/usr/bin/env bash
# Run one EVA domain against the agents made by provision_agents.py.
#
#   scripts/cartesia_managed/run.sh <domain> [extra eva args...]
#
# CARTESIA_API_KEY comes from the environment or .env; everything else is read from .env as for any EVA run.
#
# Our ElevenLabs simulator agents used Eric for male callers and Hope (tnSpp4vdxKPjI9w0GnoV) for female
# callers, since ElevenLabs no longer allows Natalee Champlin on new agents.
set -euo pipefail

DOMAIN="$1"
shift

EVA_MODEL__S2S_PARAMS="$(uv run python - <<'EOF'
import json
import os

from dotenv import dotenv_values

key = os.environ.get("CARTESIA_API_KEY") or dotenv_values(".env").get("CARTESIA_API_KEY")
if not key:
    raise SystemExit("Set CARTESIA_API_KEY in the environment or .env")
state = json.load(open("output/cartesia_managed_agents.json"))
print(json.dumps({
    "api_key": key,
    "model": f"managed-{state['model']}",
    "alias": f"{state['model']}+cartesia-managed-agents",
    "agent_ids": {f"agent_{domain}": agent_id for domain, agent_id in state["agents"].items()},
}))
EOF
)"
export EVA_MODEL__S2S_PARAMS
export EVA_FRAMEWORK=cartesia_managed
export EVA_MODEL__LLM=""
# Managed Agents owns these components; exclude unused local settings from run artifacts.
export EVA_MODEL__STT=""
export EVA_MODEL__STT_PARAMS=null
export EVA_MODEL__TTS=""
export EVA_MODEL__TTS_PARAMS=null
export EVA_MODEL__AUDIO_LLM=""
export EVA_MODEL__AUDIO_LLM_PARAMS=null
export EVA_MODEL__S2S=cartesia_managed
# Pre-tool speech is configured per tool on the agent.
export EVA_MODEL__PRE_TOOL_SPEECH=off
export EVA_DOMAIN="$DOMAIN"

exec uv run eva --domain "$DOMAIN" "$@"
