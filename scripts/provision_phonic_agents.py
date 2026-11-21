"""Create or update one Phonic agent per EVA domain, with EVA's tools as WebSocket tools.

The agent passes EVA's system prompt and greeting through untouched: they are sent with each session
as the template variables instructions and initial_message. Each EVA tool is registered as a
``custom_websocket`` tool (executed by the assistant server over the STS socket) and attached to the
agent.

Usage:
    python scripts/provision_phonic_agents.py [--model phonic-1] [--voice sabrina] [--language en]
        [--domains airline itsm medical_hr] [--backchannel]

Deploys all domains by default, with assistant backchannel cues off unless --backchannel is passed.

Reads PHONIC_API_KEY from the environment or .env. Agent IDs are kept in output/phonic_agents.json.
Agents are keyed by domain and language, so a rerun updates the same agent for that domain and language
while other languages keep their own agents. Tool names are unique per Phonic organization, so tools are
looked up by name (updated if present, created otherwise) and attached to the domain's agents. Prints each domain's
EVA_MODEL__S2S_PARAMS (without the api_key) to use for the run.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from phonic import AsyncPhonic
from phonic.core.api_error import ApiError

from eva.models.agents import AgentConfig
from eva.models.config import LANGUAGE_DISPLAY_NAMES

DOMAINS = ("airline", "itsm", "medical_hr")
STATE = Path("output/phonic_agents.json")
TOOL_CALL_OUTPUT_TIMEOUT_MS = 30000
# Phonic reserves system_* names for its own system variables (e.g. {{system_time}}).
SYSTEM_PROMPT_VARIABLE = "{{instructions}}"
INITIAL_MESSAGE_VARIABLE = "{{initial_message}}"


def tool_body(fn: dict) -> dict:
    return {
        "name": fn["name"],
        "description": fn["description"],
        "execution_mode": "sync",
        "parameters": fn["parameters"],
        "tool_call_output_timeout_ms": TOOL_CALL_OUTPUT_TIMEOUT_MS,
    }


async def upsert_tool(client: AsyncPhonic, fn: dict) -> None:
    body = tool_body(fn)
    try:
        await client.tools.get(fn["name"])
    except ApiError as e:
        if e.status_code != 404:
            raise
        await client.tools.create(type="custom_websocket", **body)
    else:
        await client.tools.update(fn["name"], **body)  # a tool's type cannot change


def agent_config(model: str, tool_names: list[str], language: str, voice_id: str, backchannel: bool) -> dict:
    return {
        "enable_assistant_backchannel": backchannel,
        "model": model,
        "voice_id": voice_id,
        "default_language": language,
        "system_prompt": SYSTEM_PROMPT_VARIABLE,
        "welcome_message": INITIAL_MESSAGE_VARIABLE,
        "generate_welcome_message": False,
        "template_variables": {"instructions": {"default_value": ""}, "initial_message": {"default_value": ""}},
        "tools": tool_names,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="phonic-1", help="Phonic model id (phonic-1 or phonic-0.5)")
    parser.add_argument("--voice", default="sabrina", help="Phonic voice_id for the agent")
    parser.add_argument(
        "--language",
        choices=[lang.value for lang in LANGUAGE_DISPLAY_NAMES],
        default="en",
        help="Base language code without a region tag (e.g. fr, not fr-CA); must match EVA_LANGUAGE for the run",
    )
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    parser.add_argument("--backchannel", action="store_true", help="Enable assistant backchannel cues (off by default)")
    args = parser.parse_args()
    if "-" in args.language:
        raise SystemExit(
            f"--language takes a base language code without a region tag; use {args.language.split('-')[0]}"
        )

    key = os.environ.get("PHONIC_API_KEY") or dotenv_values(".env").get("PHONIC_API_KEY")
    if not key:
        raise SystemExit("Set PHONIC_API_KEY in the environment or .env")
    client = AsyncPhonic(api_key=key)

    state = json.loads(STATE.read_text()) if STATE.exists() else {"agents": {}}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    for domain in args.domains:
        # Tools keep EVA's order, the order the model sees them in.
        tool_names = []
        for tool in AgentConfig.from_yaml(f"configs/agents/{domain}_agent.yaml").build_tools_for_agent():
            fn = tool["function"]
            await upsert_tool(client, fn)
            tool_names.append(fn["name"])

        agent_key = f"{domain}/{args.language}"
        name = f"eva-{domain.replace('_', '-')}-{args.language.lower()}-{args.model.replace('.', '').replace('-', '')}"
        config = agent_config(args.model, tool_names, args.language, args.voice, args.backchannel)
        if agent_key in state["agents"]:
            await client.agents.update(state["agents"][agent_key]["id"], name=name, **config)
        else:
            state["agents"][agent_key] = {"id": (await client.agents.create(name=name, **config)).id, "name": name}
        state["agents"][agent_key]["name"] = name
        STATE.write_text(json.dumps(state, indent=2))
        print(f"{agent_key}: {state['agents'][agent_key]['id']} with {len(tool_names)} tools", file=sys.stderr)
        print(f"{agent_key}: " + json.dumps({"model": args.model, "agent": name, "voice": args.voice}))


if __name__ == "__main__":
    asyncio.run(main())
