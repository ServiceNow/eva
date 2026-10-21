"""Create or update one Cartesia managed agent per EVA domain, with EVA's tools as client tools.

The agent passes EVA's system prompt and greeting through untouched: they are sent with each call as
the dynamic variables system_prompt and initial_message. Settings that cannot be changed per call are
set here from EVA's own options, and the assistant server refuses to run against an agent whose
settings do not match the run config:

    --language                -> language.primary              (EVA_LANGUAGE, default en; fr-CA uses fr)
    --pre-tool-speech         -> each tool's pre_tool_speech  (EVA_MODEL__PRE_TOOL_SPEECH, default auto)
    --turn-end-fallback-time  -> turn.inactivity_check_in_secs (EVA_TURN_END_FALLBACK_TIME, default unset)

--voice sets the agent's voice. It is not checked: like any agent setting that doesn't affect the
benchmark, it is identified by the run's assistant_agent_id.

Model fallback is disabled, so only the chosen model answers.

Usage:
    python scripts/provision_cartesia_agents.py [--model gpt-5.5] [--language en] [--voice VOICE_ID]
        [--pre-tool-speech auto|off] [--turn-end-fallback-time SECONDS] [--domains airline itsm medical_hr]

Reads CARTESIA_API_KEY from the environment or .env. Tool and agent IDs are kept in
output/cartesia_agents.json so a rerun updates the same agents. Prints each domain's
EVA_MODEL__S2S_PARAMS (without the api_key) to use for the run.
"""

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

from dotenv import dotenv_values

from eva.models.agents import AgentConfig
from eva.models.config import LANGUAGE_DISPLAY_NAMES, ModelConfig, base_language
from eva.utils import cartesia_agents

DOMAINS = ("airline", "itsm", "medical_hr")
DEFAULT_VOICE_ID = "f786b574-daa5-4673-aa0c-cbe3e8534c02"  # Katie
STATE = Path("output/cartesia_agents.json")


def tool_body(fn: dict, pre_tool_speech: str) -> dict:
    return {
        "type": "client",
        "name": fn["name"],
        "description": fn["description"],
        "parameters": fn["parameters"],
        "expects_response": True,
        "response_timeout_secs": 20,
        "pre_tool_speech": pre_tool_speech,
        "execution_mode": "immediate",
    }


def agent_config(
    model: str, tool_ids: list[str], turn_end_fallback_time: int | None, language: str, voice_id: str
) -> dict:
    return {
        "instructions": cartesia_agents.SYSTEM_PROMPT_VARIABLE,
        "prompt_optimization": False,  # use EVA's system prompt as is
        "initial_message": cartesia_agents.INITIAL_MESSAGE_VARIABLE,
        "speculative_generation": True,
        "model": {"id": model, "fallback_strategy": None},
        "language": {"primary": base_language(language)},
        "turn": {"inactivity_check_in_secs": turn_end_fallback_time},
        "audio": {
            "input": {"noise_suppression": "off", "asr_model": "auto"},
            "output": {"voice_id": voice_id},
        },
        "tools": [{"id": t} for t in tool_ids],
    }


def check_model(request, model: str) -> None:
    path = "/v1/agents/models"
    while True:
        page = request("GET", path)
        if any(m.get("id") == model for m in page["data"]):
            return
        if not page.get("has_more"):
            raise SystemExit(f"{model} is not in GET /v1/agents/models")
        path = f"/v1/agents/models?starting_after={page['next_page']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gpt-5.5", help="Any model id from GET /v1/agents/models")
    parser.add_argument(
        "--language",
        choices=[lang.value for lang in LANGUAGE_DISPLAY_NAMES],
        default="en",
        help="Must match EVA_LANGUAGE for the run",
    )
    parser.add_argument("--voice", default=DEFAULT_VOICE_ID, help="Cartesia voice ID for the agent (default: Katie)")
    parser.add_argument(
        "--pre-tool-speech",
        choices=["auto", "off"],
        default=ModelConfig.model_fields["pre_tool_speech"].default,
        help="Must match EVA_MODEL__PRE_TOOL_SPEECH for the run",
    )
    parser.add_argument(
        "--turn-end-fallback-time",
        type=int,
        default=None,
        help="Must match EVA_TURN_END_FALLBACK_TIME for the run (5-60 seconds; unset disables check-ins)",
    )
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    args = parser.parse_args()
    if args.turn_end_fallback_time is not None and not 5 <= args.turn_end_fallback_time <= 60:
        raise SystemExit("Cartesia supports --turn-end-fallback-time between 5 and 60 seconds")

    key = os.environ.get("CARTESIA_API_KEY") or dotenv_values(".env").get("CARTESIA_API_KEY")
    if not key:
        raise SystemExit("Set CARTESIA_API_KEY in the environment or .env")
    request = partial(cartesia_agents.request, api_key=key)
    check_model(request, args.model)

    state = json.loads(STATE.read_text()) if STATE.exists() else {"tools": {}, "agents": {}}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    for domain in args.domains:
        # Tools keep EVA's order, the order the model sees them in.
        tool_ids = []
        for tool in AgentConfig.from_yaml(f"configs/agents/{domain}_agent.yaml").build_tools_for_agent():
            fn = tool["function"]
            key = f"{domain}/{fn['name']}"
            body = tool_body(fn, args.pre_tool_speech)
            if key in state["tools"]:
                patch = {k: v for k, v in body.items() if k != "type"}  # a tool's type cannot change
                request("PATCH", f"/v1/agents/tools/{state['tools'][key]}", body=patch)
            else:
                state["tools"][key] = request("POST", "/v1/agents/tools", body=body)["id"]
                STATE.write_text(json.dumps(state, indent=2))
            tool_ids.append(state["tools"][key])

        name = f"eva-{domain.replace('_', '-')}-{args.model.replace('.', '').replace('-', '')}"
        payload = {
            "name": name,
            "description": "EVA-Bench agent",
            "config": agent_config(args.model, tool_ids, args.turn_end_fallback_time, args.language, args.voice),
        }
        if domain in state["agents"]:
            request("PATCH", f"/v1/agents/{state['agents'][domain]}", body=payload)
        else:
            state["agents"][domain] = request("POST", "/v1/agents", body=payload)["id"]
        STATE.write_text(json.dumps(state, indent=2))
        print(f"{domain}: {state['agents'][domain]} with {len(tool_ids)} tools", file=sys.stderr)
        print(f"{domain}: " + json.dumps({"model": args.model, "assistant_agent_id": state["agents"][domain]}))


if __name__ == "__main__":
    main()
