"""Krisp Managed Voice Agent REST helpers for config checks and the assistant server.

An agent's settings live on the platform and cannot be overridden per call, so Krisp provisions each
agent from EVA's prompt and tools, and EVA checks it against the run config before a conversation starts
(``check_agent_matches_config``).
"""

from __future__ import annotations

import email.utils
import json
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from eva.models.agents import AgentConfig
from eva.models.config import base_language
from eva.utils.prompt_manager import PromptManager

CLOCK_TOOL_NAME = "get_current_datetime"
CLOCK_TOOL = {
    "type": "function",
    "function": {
        "name": CLOCK_TOOL_NAME,
        "description": "Return the current date and time for this call.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
# Fills "Today is {datetime}.", the only prompt line that differs from EVA's.
CLOCK_DATETIME = (
    f"the date returned by {CLOCK_TOOL_NAME}; call it before stating, comparing or calculating any date or time"
)
# With call_date=session the platform fills this from the date sent at session start.
SESSION_DATETIME = "{{vars.datetime}}"
CALL_DATE_MODES = ("clock", "session")
PREAMBLE_MODES = ("on", "off")
SESSION_ID_HEADER = "X-Session-Id"
USER_AGENT = "Mozilla/5.0 (compatible; eva-bench)"


def call_date_mode(params: dict[str, Any]) -> str:
    """How the agent learns the scenario date: the session-start variable (default) or the clock tool."""
    mode = params.get("call_date") or "session"
    if mode not in CALL_DATE_MODES:
        raise ValueError(f"s2s_params.call_date must be one of {CALL_DATE_MODES}, got {mode!r}")
    return mode


def preamble_mode(params: dict[str, Any]) -> str:
    """Whether the platform speaks its templated lead-ins while a tool runs; on by default."""
    mode = params.get("preamble") or "on"
    if mode not in PREAMBLE_MODES:
        raise ValueError(f"s2s_params.preamble must be one of {PREAMBLE_MODES}, got {mode!r}")
    return mode


def system_prompt(agent: AgentConfig, pre_tool_speech: str, call_date: str = "clock") -> str:
    """EVA's realtime system prompt; the date comes from the clock tool or the session variable."""
    pm = PromptManager()
    prompt = pm.get_prompt(
        "realtime_agent.system_prompt",
        agent_personality=agent.description,
        agent_instructions=agent.instructions,
        datetime=SESSION_DATETIME if call_date == "session" else CLOCK_DATETIME,
    )
    if pre_tool_speech == "auto":
        prompt += "\n\n" + pm.get_prompt("agent.pre_tool_speech")
    return prompt


def eva_tools(agent: AgentConfig, call_date: str = "clock") -> list[dict]:
    """The tools the Krisp agent exposes, in EVA's order, plus the clock tool in clock mode."""
    tools = list(agent.build_tools_for_agent() or [])
    return tools if call_date == "session" else [*tools, CLOCK_TOOL]


class KrispApiError(RuntimeError):
    """A Krisp Managed Voice Agent API call that failed with an HTTP status."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _retry_after_s(value: str | None) -> float:
    """Seconds from a Retry-After header in either form RFC 9110 allows; 0 when absent or unreadable."""
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, when.timestamp() - time.time())


def request(method: str, path: str, params: dict[str, Any], body: dict | None = None, retries: int = 6) -> Any:
    """Call the platform API (``params``: ``api_url`` + ``api_key``) and unwrap its response envelope."""
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            f"{params['api_url'].rstrip('/')}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {params['api_key']}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                payload = json.loads(raw) if raw else {}
                return payload.get("data", payload) if isinstance(payload, dict) else payload
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code == 429 and attempt < retries:
                time.sleep(_retry_after_s(e.headers.get("Retry-After")) or 0.5 * 2**attempt)
                continue
            raise KrispApiError(e.code, f"HTTP {e.code} {method} {path}: {detail}") from None
    raise AssertionError("unreachable")


def media_stream_url(ws_url: str) -> str:
    """The Twilio media-stream endpoint on the host the session exchange returned."""
    parsed = urlparse(ws_url)
    return f"{parsed.scheme}://{parsed.netloc}/ws"


def mint_voice_session(agent_id: str, s2s_params: dict[str, Any], call_datetime: str | None = None) -> dict[str, str]:
    """Open a voice session; returns ``session_id``, ``session_token`` and the media-stream URL."""
    body: dict[str, Any] = {"agent_id": agent_id, "modality": "voice"}
    if call_datetime is not None:
        body["variables"] = {"datetime": call_datetime}
    data = request("POST", "/access-tokens/exchange/dashboard", s2s_params, body)
    if call_datetime is not None and "datetime" not in (data.get("accepted_variables") or []):
        raise RuntimeError("the platform did not accept the call date; it predates per-call variables")
    token = data.get("session_token")
    if not token or not data.get("session_id"):
        raise RuntimeError(f"session exchange returned no token or session id: {sorted(data)}")
    ws = s2s_params.get("media_stream_url") or media_stream_url(data["ws_url"])
    return {"session_id": data["session_id"], "session_token": token, "media_stream_url": ws}


def fetch_session_history(session_id: str, s2s_params: dict[str, Any], attempts: int = 8) -> list[dict]:
    """The session's turn history once flushed; only a not-yet-flushed session is retried, never auth errors."""
    history: list[dict] = []
    for attempt in range(attempts):
        try:
            data = request("GET", f"/sessions/{session_id}", s2s_params)
        except KrispApiError as e:
            if e.code != 404 and e.code < 500:
                raise
        else:
            history = data.get("history") or []
            if history and (data.get("ended_at") or attempt == attempts - 1):
                break
        time.sleep(1.5)
    if not history:
        raise RuntimeError(f"session {session_id} has no history after {attempts} attempts")
    return sorted(history, key=lambda e: e.get("sequence") or 0)


def unreached_tool_calls(history: list[dict]) -> list[str]:
    """Tool calls that never got the gateway's answer (no response, or a tunnel 5xx), as ``name: error``."""
    unreached = []
    for entry in history:
        for call in entry.get("tool_calls") or []:
            try:
                summary = json.loads(call.get("result_summary") or "{}")
            except (TypeError, ValueError):
                continue
            if (
                isinstance(summary, dict)
                and summary.get("proxy_outcome", "success") != "success"
                and (summary.get("upstream_status") is None or summary.get("upstream_status", 0) >= 500)
            ):
                body = summary.get("body")
                error = body.get("error") if isinstance(body, dict) else body
                unreached.append(f"{call.get('tool_name')}: {error or summary['proxy_outcome']}")
    return unreached


_agent_cache: dict[str, dict] = {}


def fetch_agent(agent_id: str, s2s_params: dict[str, Any]) -> dict:
    """The agent's published config with tools resolved by id. Cached per agent ID."""
    if agent_id in _agent_cache:
        return _agent_cache[agent_id]
    agent = request("GET", f"/agents/{agent_id}", s2s_params)
    branches = agent.get("branches") or []
    if not branches:
        raise RuntimeError(f"agent {agent_id} has no branch")
    main = next((b for b in branches if b.get("is_default")), branches[0])
    branch = request("GET", f"/agents/{agent_id}/branches/{main['id']}", s2s_params)
    pac = branch.get("head_pac") or {}
    if not pac.get("published_at"):
        raise RuntimeError(f"agent {agent_id} is not published")
    published = {
        "id": agent_id,
        "config": pac.get("published_draft_config") or {},
        "tools": [request("GET", f"/tools/{t}", s2s_params) for t in pac.get("tool_refs") or []],
    }
    _agent_cache[agent_id] = published
    return published


def _greeting(config: dict, language: str) -> str | None:
    greeting = config.get("greeting")
    if not isinstance(greeting, dict):
        return greeting
    return greeting.get(language) or greeting.get(base_language(language))


def check_agent_matches_config(
    agent: dict,
    *,
    model: str,
    language: str,
    system_prompt: str,
    initial_message: str,
    tool_names: list[str],
    preamble: str,
) -> None:
    """Raise if the published agent would not run the benchmark as EVA's config describes."""
    config = agent["config"]
    problems = []
    if config.get("system_prompt") != system_prompt:
        problems.append("system_prompt is not EVA's prompt for this domain")
    if _greeting(config, language) != initial_message:
        problems.append(
            f"greeting is {_greeting(config, language)!r}, but EVA's initial message is {initial_message!r}"
        )
    agent_model = (config.get("llm_config") or {}).get("model")
    if agent_model != model:
        problems.append(f"llm_config.model is {agent_model!r}, but s2s_params.model is {model!r}")
    languages = config.get("languages_config") or {}
    if base_language(language) not in (languages.get("supported") or []):
        problems.append(f"languages_config.supported {languages.get('supported')} lacks {base_language(language)!r}")
    names = [t.get("name") for t in agent["tools"]]
    if sorted(names) != sorted(tool_names):
        problems.append(f"tools {sorted(names)} do not match the EVA agent's tools {sorted(tool_names)}")
    want = preamble == "on"
    wrong = sorted(
        t.get("name")
        for t in agent["tools"]
        if t.get("name") != CLOCK_TOOL_NAME
        and bool(((t.get("config_json") or {}).get("filler") or {}).get("enabled")) != want
    )
    if wrong:
        problems.append(
            f"lead-ins are {'off' if want else 'on'} on tools {wrong}, but s2s_params.preamble is {preamble!r}"
        )
    if problems:
        raise ValueError(
            f"Krisp agent {agent['id']} does not match the EVA run config; ask Krisp "
            "to re-provision it with matching settings:\n  - " + "\n  - ".join(problems)
        )
