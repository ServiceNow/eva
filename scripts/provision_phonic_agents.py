"""Create or update one Phonic agent per EVA domain, with EVA's tools as WebSocket tools.

The agent passes EVA's system prompt and greeting through untouched: they are sent with each session
as the template variables instructions and initial_message. Each EVA tool is registered as a
``custom_websocket`` tool (executed by the assistant server over the STS socket) and attached to the
agent.

Usage:
    python scripts/provision_phonic_agents.py [--model phonic-1.1] [--voice sabrina] [--language en]
        [--domains airline itsm medical_hr] [--backchannel]

Deploys all domains by default, with assistant backchannel cues off unless --backchannel is passed.
Timing settings mirror the run config: --turn-end-fallback-time (EVA_TURN_END_FALLBACK_TIME) sets
"Respond after" (no_input_poke_sec), and the silence limit (no_input_end_conversation_sec) is the user simulator's INACTIVITY_TIMEOUT_SECONDS.
The assistant server sends the same values with every session, so the run matches the saved agent.

Reads PHONIC_API_KEY from the environment or .env. Agent IDs are kept in output/phonic_agents.json.
Agents are keyed by domain and language, so a rerun updates the same agent for that domain and language
while other languages keep their own agents. Tool names are unique per Phonic organization, so tools are
looked up by name (updated if present, created otherwise) and attached to the domain's agents. A name shared
by several domains gets the domain's prefix from PHONIC_TOOL_PREFIXES (airline keeps plain names), in the tool
name and wherever a description mentions it; the assistant server strips the prefix before execution. Read tools may run while the user is speaking. Prints each domain's
EVA_MODEL__S2S_PARAMS (without the api_key) to use for the run.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from phonic import AsyncPhonic
from phonic.core.api_error import ApiError

from eva.assistant.phonic_server import PHONIC_TOOL_PREFIXES
from eva.models.agents import AgentConfig
from eva.models.config import LANGUAGE_DISPLAY_NAMES, RunConfig
from eva.user_simulator.cascade.simulator import INACTIVITY_TIMEOUT_SECONDS

DOMAINS = ("airline", "itsm", "medical_hr")
STATE = Path("output/phonic_agents.json")
TOOL_CALL_OUTPUT_TIMEOUT_MS = 30000
# Phonic reserves system_* names for its own system variables (e.g. {{system_time}}).
SYSTEM_PROMPT_VARIABLE = "{{instructions}}"
INITIAL_MESSAGE_VARIABLE = "{{initial_message}}"


def shared_tool_names() -> set[str]:
    """Tool names used by more than one domain, across all domains whatever --domains selects."""
    seen: set[str] = set()
    shared: set[str] = set()
    for domain in DOMAINS:
        for tool in AgentConfig.from_yaml(f"configs/agents/{domain}_agent.yaml").tools:
            (shared if tool.function_name in seen else seen).add(tool.function_name)
    return shared


def tool_renames(domain: str, agent: AgentConfig, shared: set[str]) -> dict[str, str]:
    """EVA tool name -> Phonic tool name for the domain's own shared tools."""
    prefix = PHONIC_TOOL_PREFIXES.get(domain, "")
    if not prefix:
        return {}
    return {tool.function_name: prefix + tool.function_name for tool in agent.tools if tool.function_name in shared}


def rename_in_descriptions(value: Any, renames: dict[str, str]) -> Any:
    """Rewrite whole-word tool names in every description string (enum values and keys stay as is)."""
    if not renames:
        return value
    if isinstance(value, list):
        return [rename_in_descriptions(v, renames) for v in value]
    if not isinstance(value, dict):
        return value
    # \b treats "_" as a word character, so i_transfer_to_agent and transfer_to_agent_x don't match.
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, renames)) + r")\b")
    out = {}
    for k, v in value.items():
        if k == "description" and isinstance(v, str):
            out[k] = pattern.sub(lambda m: renames[m[1]], v)
        else:
            out[k] = rename_in_descriptions(v, renames)
    return out


def tool_body(fn: dict, renames: dict[str, str], read_only: bool) -> dict:
    fn = rename_in_descriptions(fn, renames)
    return {
        "name": renames.get(fn["name"], fn["name"]),
        "description": fn["description"],
        "execution_mode": "sync",
        "parameters": fn["parameters"],
        "tool_call_output_timeout_ms": TOOL_CALL_OUTPUT_TIMEOUT_MS,
        "can_run_during_user_speech": read_only,
    }


async def upsert_tool(client: AsyncPhonic, body: dict) -> None:
    try:
        await client.tools.get(body["name"])
    except ApiError as e:
        if e.status_code != 404:
            raise
        await client.tools.create(type="custom_websocket", **body)
    else:
        await client.tools.update(body["name"], **body)  # a tool's type cannot change


def agent_config(
    model: str,
    tool_names: list[str],
    language: str,
    voice_id: str,
    backchannel: bool,
    turn_end_fallback_time: int | None,
) -> dict:
    return {
        "no_input_poke_sec": turn_end_fallback_time,
        "no_input_end_conversation_sec": int(INACTIVITY_TIMEOUT_SECONDS),
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
    parser.add_argument("--model", default="phonic-1.1", help="Phonic model id (phonic-1.1, phonic-1 or phonic-0.5)")
    parser.add_argument("--voice", default="sabrina", help="Phonic voice_id for the agent")
    parser.add_argument(
        "--language",
        choices=[lang.value for lang in LANGUAGE_DISPLAY_NAMES],
        default="en",
        help="Base language code without a region tag (e.g. fr, not fr-CA); must match EVA_LANGUAGE for the run",
    )
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    parser.add_argument(
        "--turn-end-fallback-time",
        type=int,
        default=RunConfig.model_fields["turn_end_fallback_time"].default,
        help="Must match EVA_TURN_END_FALLBACK_TIME for the run -> no_input_poke_sec (unset disables the poke)",
    )
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
    shared = shared_tool_names()
    for domain in args.domains:
        # Tools keep EVA's order, the order the model sees them in.
        tool_names = []
        agent = AgentConfig.from_yaml(f"configs/agents/{domain}_agent.yaml")
        renames = tool_renames(domain, agent, shared)
        for tool, schema in zip(agent.tools, agent.build_tools_for_agent() or [], strict=True):
            body = tool_body(schema["function"], renames, read_only=tool.tool_type == "read")
            await upsert_tool(client, body)
            tool_names.append(body["name"])

        agent_key = f"{domain}/{args.language}"
        name = f"eva-{domain.replace('_', '-')}-{args.language.lower()}-{args.model.replace('.', '').replace('-', '')}"
        config = agent_config(
            args.model,
            tool_names,
            args.language,
            args.voice,
            args.backchannel,
            args.turn_end_fallback_time,
        )
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
