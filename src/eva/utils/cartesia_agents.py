"""Cartesia Managed Agents REST helpers, shared by provisioning, config and the assistant server.

A managed agent's settings live on Cartesia's side and cannot be overridden per call, so EVA sets
them when the agent is provisioned (scripts/provision_cartesia_agents.py) and checks them against
the run config before a conversation starts (``check_agent_matches_config``).
"""

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_REST_URL = "https://api.cartesia.ai"
DEFAULT_CARTESIA_VERSION = "2026-08-14"

# The agent passes EVA's prompt and greeting through untouched; see provision_cartesia_agents.py.
SYSTEM_PROMPT_VARIABLE = "{{system_prompt}}"
INITIAL_MESSAGE_VARIABLE = "{{initial_message}}"


def rest_url_from_params(s2s_params: dict[str, Any]) -> str:
    """REST base URL matching the WebSocket ``base_url`` in s2s_params."""
    base = s2s_params.get("base_url", "wss://api.cartesia.ai").rstrip("/")
    return base.replace("wss://", "https://", 1).replace("ws://", "http://", 1)


def request(
    method: str,
    path: str,
    api_key: str,
    body: dict | None = None,
    base_url: str = DEFAULT_REST_URL,
    cartesia_version: str = DEFAULT_CARTESIA_VERSION,
) -> dict:
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"X-API-Key": api_key, "Cartesia-Version": cartesia_version, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {method} {path}: {e.read().decode(errors='replace')[:1500]}") from None


_agent_cache: dict[str, dict] = {}


def fetch_agent(agent_id: str, s2s_params: dict[str, Any]) -> dict:
    """Fetch an agent with its tools resolved (``config.tools`` as full tool objects). Cached per agent ID."""
    if agent_id in _agent_cache:
        return _agent_cache[agent_id]
    kwargs = {
        "api_key": s2s_params["api_key"],
        "base_url": rest_url_from_params(s2s_params),
        "cartesia_version": s2s_params.get("cartesia_version", DEFAULT_CARTESIA_VERSION),
    }
    agent = request("GET", f"/v1/agents/{agent_id}", **kwargs)
    agent["config"]["tools"] = [
        request("GET", f"/v1/agents/tools/{t['id']}", **kwargs) for t in agent["config"].get("tools", [])
    ]
    _agent_cache[agent_id] = agent
    return agent


def check_agent_matches_config(
    agent: dict, model: str, pre_tool_speech: str, turn_end_fallback_time: int | None, tool_names: list[str]
) -> None:
    """Raise if the provisioned agent would not run the benchmark as EVA's config describes."""
    config = agent["config"]
    problems = []
    if config.get("instructions") != SYSTEM_PROMPT_VARIABLE or config.get("prompt_optimization"):
        problems.append("instructions must be exactly '{{system_prompt}}' with prompt_optimization off")
    if config.get("initial_message") != INITIAL_MESSAGE_VARIABLE:
        problems.append("initial_message must be exactly '{{initial_message}}'")
    if config["model"]["id"] != model:
        problems.append(f"model is {config['model']['id']!r}, but s2s_params.model is {model!r}")
    if config["model"].get("fallback_strategy") is not None:
        problems.append("model.fallback_strategy must be null, so only the reported model answers")
    check_in = config.get("turn", {}).get("inactivity_check_in_secs")
    if check_in != turn_end_fallback_time:
        problems.append(
            f"turn.inactivity_check_in_secs is {check_in}, but turn_end_fallback_time is {turn_end_fallback_time}"
        )
    tools = config.get("tools", [])
    if [t["name"] for t in tools] != tool_names:
        problems.append(f"tools {[t['name'] for t in tools]} do not match the EVA agent's tools {tool_names}")
    wrong_speech = [t["name"] for t in tools if t.get("pre_tool_speech") != pre_tool_speech]
    if wrong_speech:
        problems.append(f"pre_tool_speech must be '{pre_tool_speech}' on tools {wrong_speech}")
    if problems:
        raise ValueError(
            f"Cartesia agent {agent['id']} does not match the EVA run config; re-run "
            "scripts/provision_cartesia_agents.py with matching settings:\n  - " + "\n  - ".join(problems)
        )
