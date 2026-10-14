"""Create or update one Cartesia managed agent per EVA domain, with EVA's tools as client tools.

EVA's system prompt and greeting are sent with each call as the dynamic variables system_prompt and
initial_message. Ids are kept in output/cartesia_managed_agents.json, which run.sh reads, so a rerun
updates the same agents.

    uv run python scripts/cartesia_managed/provision_agents.py [--model gpt-5.5]

Reads CARTESIA_API_KEY from the environment or .env.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

from eva.models.agents import AgentConfig

API = "https://api.cartesia.ai"
CARTESIA_VERSION = "2026-08-14"
DOMAINS = ("airline", "itsm", "medical_hr")
VOICE_ID = "f786b574-daa5-4673-aa0c-cbe3e8534c02"  # Katie
STATE = Path("output/cartesia_managed_agents.json")


def request(method: str, path: str, body: dict | None = None) -> dict:
    key = os.environ.get("CARTESIA_API_KEY") or dotenv_values(".env").get("CARTESIA_API_KEY")
    if not key:
        raise SystemExit("Set CARTESIA_API_KEY in the environment or .env")
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"X-API-Key": key, "Cartesia-Version": CARTESIA_VERSION, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} {method} {path}: {e.read().decode(errors='replace')[:1500]}") from None


def tool_body(fn: dict) -> dict:
    return {
        "type": "client",
        "name": fn["name"],
        "description": fn["description"],
        "parameters": fn["parameters"],
        "expects_response": True,
        "response_timeout_secs": 20,
        "pre_tool_speech": "force",
        "execution_mode": "immediate",
    }


def agent_config(model: str, tool_ids: list[str]) -> dict:
    return {
        "instructions": "{{system_prompt}}",
        "prompt_optimization": False,  # use EVA's system prompt as is
        "initial_message": "{{initial_message}}",
        "speculative_generation": True,
        "model": {"id": model, "fallback_strategy": {"type": "auto"}},
        "language": {"primary": "en"},
        "turn": {"inactivity_check_in_secs": 15},
        "audio": {
            "input": {"noise_suppression": "off", "asr_model": "auto"},
            "output": {"voice_id": VOICE_ID},
        },
        "tools": [{"id": t} for t in tool_ids],
    }


def check_model(model: str) -> None:
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
    args = parser.parse_args()
    check_model(args.model)

    state = json.loads(STATE.read_text()) if STATE.exists() else {"tools": {}, "agents": {}}
    state["model"] = args.model
    STATE.parent.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        # Tools keep EVA's order, the order the model sees them in.
        tool_ids = []
        for tool in AgentConfig.from_yaml(f"configs/agents/{domain}_agent.yaml").build_tools_for_agent():
            fn = tool["function"]
            key = f"{domain}/{fn['name']}"
            if key in state["tools"]:
                patch = {k: v for k, v in tool_body(fn).items() if k != "type"}  # a tool's type cannot change
                request("PATCH", f"/v1/agents/tools/{state['tools'][key]}", patch)
            else:
                state["tools"][key] = request("POST", "/v1/agents/tools", tool_body(fn))["id"]
                STATE.write_text(json.dumps(state, indent=2))
            tool_ids.append(state["tools"][key])

        name = f"eva-{domain.replace('_', '-')}-{args.model.replace('.', '').replace('-', '')}"
        payload = {"name": name, "description": "EVA-Bench agent", "config": agent_config(args.model, tool_ids)}
        if domain in state["agents"]:
            request("PATCH", f"/v1/agents/{state['agents'][domain]}", payload)
        else:
            state["agents"][domain] = request("POST", "/v1/agents", payload)["id"]
        STATE.write_text(json.dumps(state, indent=2))
        print(f"{domain}: {state['agents'][domain]} with {len(tool_ids)} tools", file=sys.stderr)


if __name__ == "__main__":
    main()
