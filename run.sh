#!/usr/bin/env bash
set -euo pipefail

# Load base configuration from .env
set -a
source "$(dirname "$0")/.env"
set +a

# --- Run-specific overrides (not in .env or intentionally different) ---

# Specific record IDs to run (comma-separated); empty = run all
export EVA_RECORD_IDS=

# Number of trials per record
export EVA_NUM_TRIALS=3

# Conversation timeout (seconds) — override .env default of 360
export EVA_CONVERSATION_TIMEOUT_SECONDS=360

# Log level — override .env default of INFO
export LOG_LEVEL=DEBUG

# Metrics to run — override .env default (empty)
export EVA_METRICS=all

# Audio LLM
export EVA_MODEL__AUDIO_LLM=vllm
export EVA_MODEL__AUDIO_LLM_PARAMS='{
    "url": "<URL_HERE>",
    "api_key": "dummy_token",
    "model": "<MODEL_NAME_HERE>",
    "temperature": 0.6,
    "max_tokens": 8192
}'

# For debugging LLM requests/responses, set to true to dump them to output folder (one file per request)
export EVA_DUMP_LLM_REQUESTS=false                                                                                                      

# --- Run ---
cd "$(dirname "$0")"
python -m eva.cli
