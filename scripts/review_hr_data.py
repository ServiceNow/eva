"""Streamlit app for reviewing HR domain benchmark data."""

import difflib
import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "data" / "medical_hr_dataset.jsonl"
SCENARIOS_DIR = ROOT / "data" / "medical_hr_scenarios"
AGENT_YAML_PATH = ROOT / "configs" / "agents" / "medical_hr_agent.yaml"
FEEDBACK_DIR = ROOT / "hr_review_feedback"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="HR Data Review", layout="wide")

TOOL_TYPE_COLORS = {
    "auth": "#FF9800",   # amber
    "read": "#4CAF50",   # green
    "write": "#F44336",  # red
}

TOOL_TYPE_LABELS = {
    "auth": "AUTH",
    "read": "READ",
    "write": "WRITE",
}

YES_NO_UNCLEAR = ["", "Yes", "No", "Unclear"]
YES_NO_NA = ["", "Yes", "No", "NA"]
YES_NO = ["", "Yes", "No"]


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_records() -> list[dict]:
    records = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return sorted(records, key=lambda r: r["id"])


@st.cache_data
def load_agent_config() -> tuple[list[dict], str, dict[str, str]]:
    with open(AGENT_YAML_PATH) as f:
        config = yaml.safe_load(f)
    tools = config.get("tools", [])
    instructions = config.get("instructions", "")
    tool_type_map = {t["name"]: t.get("tool_type", "read") for t in tools}
    return tools, instructions, tool_type_map


@st.cache_data
def load_initial_scenario(record_id: str) -> dict:
    path = SCENARIOS_DIR / f"{record_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_feedback(record_id: str) -> dict | None:
    path = FEEDBACK_DIR / f"{record_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_feedback(record_id: str, feedback: dict):
    FEEDBACK_DIR.mkdir(exist_ok=True)
    feedback["record_id"] = record_id
    feedback["last_updated"] = datetime.now().isoformat()
    path = FEEDBACK_DIR / f"{record_id}.json"
    with open(path, "w") as f:
        json.dump(feedback, f, indent=2)


# ── Trace helpers ────────────────────────────────────────────────────────────
def extract_write_tool_calls(
    trace: list[dict], tool_type_map: dict[str, str],
) -> list[dict]:
    """Extract write tool_call events from the expected trace."""
    calls = []
    for msg in trace:
        if msg.get("event_type") == "tool_call":
            name = msg.get("tool_name", "unknown")
            if tool_type_map.get(name) == "write":
                calls.append({
                    "name": name,
                    "params": msg.get("params", {}),
                })
    return calls


def render_trace(trace: list[dict], tool_type_map: dict[str, str]):
    """Render expected trace with color-coded tool calls."""
    for msg in trace:
        event = msg.get("event_type", "")

        if event == "user_utterance":
            with st.chat_message("user"):
                st.markdown(msg.get("utterance", ""))

        elif event == "agent_utterance":
            with st.chat_message("assistant"):
                st.markdown(msg.get("utterance", ""))

        elif event == "tool_call":
            name = msg.get("tool_name", "unknown")
            tool_type = tool_type_map.get(name, "read")
            color = TOOL_TYPE_COLORS.get(tool_type, "#999")
            label = TOOL_TYPE_LABELS.get(tool_type, "?")
            params = msg.get("params", {})
            with st.chat_message("assistant", avatar=":material/build:"):
                st.markdown(
                    f'<span style="background:{color};color:white;padding:2px 8px;'
                    f'border-radius:4px;font-size:0.75em;font-weight:bold">{label}</span> '
                    f'**`{name}`**',
                    unsafe_allow_html=True,
                )
                if params:
                    st.code(json.dumps(params, indent=2), language="json")

        elif event == "tool_response":
            name = msg.get("tool_name", "unknown")
            tool_type = tool_type_map.get(name, "read")
            color = TOOL_TYPE_COLORS.get(tool_type, "#999")
            label = TOOL_TYPE_LABELS.get(tool_type, "?")
            status = msg.get("status", "")
            response = msg.get("response", {})
            with st.chat_message("assistant", avatar=":material/output:"):
                st.markdown(
                    f'<span style="background:{color};color:white;padding:2px 8px;'
                    f'border-radius:4px;font-size:0.75em;font-weight:bold">{label}</span> '
                    f'`{name}` — **{status}**',
                    unsafe_allow_html=True,
                )
                if response:
                    st.json(response)


# ── Load data ────────────────────────────────────────────────────────────────
records = load_records()
ids = [r["id"] for r in records]
id_set = set(ids)
record_by_id = {r["id"]: r for r in records}
tools, instructions, tool_type_map = load_agent_config()

# ── Determine current record from query params ──────────────────────────────
params = st.query_params
current_id = params.get("record_id", ids[0])
if current_id not in id_set:
    current_id = ids[0]
current_idx = ids.index(current_id)

# ── Detect record change and pre-populate feedback state ─────────────────────
if st.session_state.get("_prev_record_id") != current_id:
    st.session_state["_prev_record_id"] = current_id
    existing = load_feedback(current_id)
    if existing:
        ug = existing.get("user_goal", {})
        st.session_state["q_reflects"] = ug.get("reflects_context", "")
        st.session_state["q_realistic"] = ug.get("is_realistic", "")
        st.session_state["q_complete"] = ug.get("is_complete", "")
        st.session_state["q_raw_info"] = ug.get("raw_info_present", "")
        st.session_state["user_goal_comments"] = ug.get("comments", "")

        gt = existing.get("ground_truth_trace", {})
        for i, tc in enumerate(gt.get("write_tool_calls", [])):
            st.session_state[f"wtc_{current_id}_{i}_grounded"] = tc.get("inputs_grounded", "")
            st.session_state[f"wtc_{current_id}_{i}_policy"] = tc.get("policy_consistent", "")
        st.session_state["q_unwanted_mods"] = gt.get("unwanted_modifications", "")
        st.session_state["q_missing_mods"] = gt.get("missing_modifications", "")
        st.session_state["q_alt_path"] = gt.get("alternative_path_exists", "")
        st.session_state["trace_comments"] = gt.get("comments", "")

        df = existing.get("diff", {})
        st.session_state["q_mods_sensible"] = df.get("modifications_sensible", "")
        st.session_state["diff_comments"] = df.get("comments", "")

        gen = existing.get("general", {})
        st.session_state["general_comments"] = gen.get("comments", "")
    else:
        for key in [
            "q_reflects", "q_realistic", "q_complete", "q_raw_info",
            "user_goal_comments", "q_unwanted_mods", "q_missing_mods",
            "q_alt_path", "trace_comments", "q_mods_sensible",
            "diff_comments", "general_comments",
        ]:
            st.session_state[key] = ""

record = record_by_id[current_id]
goal = record.get("user_goal", {})
dt = goal.get("decision_tree", {})
ground_truth = record.get("ground_truth", {})
expected_trace = ground_truth.get("expected_trace", {})
trace = expected_trace.get("trace", None) if expected_trace else None
expected_db = ground_truth.get("expected_scenario_db", {})
initial_db = load_initial_scenario(current_id)

# Extract tool calls from trace (if it exists) for the review form
write_tool_calls = extract_write_tool_calls(trace, tool_type_map) if trace else []

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Navigation + Review Questions
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Navigation")

    # Record selector
    selected = st.selectbox("Record", ids, index=current_idx)
    if selected != current_id:
        st.query_params["record_id"] = selected
        st.rerun()

    # Prev / Next
    c_prev, c_next = st.columns(2)
    with c_prev:
        if st.button("< Prev", disabled=current_idx == 0, use_container_width=True):
            st.query_params["record_id"] = ids[current_idx - 1]
            st.rerun()
    with c_next:
        if st.button("Next >", disabled=current_idx == len(ids) - 1, use_container_width=True):
            st.query_params["record_id"] = ids[current_idx + 1]
            st.rerun()

    st.divider()

    # ── Review Questions ─────────────────────────────────────────────────────
    st.header("Review Questions")

    # -- User Goal --
    st.markdown("#### User Goal")
    q_reflects = st.selectbox(
        "Does it reflect intended scenario context?",
        YES_NO_UNCLEAR, key="q_reflects",
        help="We are trying to test a specific scenario that is described by the scenario context. We just want to check that the user goal is aligned with that scenario. Intents that are not meant to be satisfiable should be in nice to have, whereas intents that are satisfiable should be in must have. Adversarial intents should always be in nice to have.",
    )
    q_realistic = st.selectbox(
        "Is it sufficiently realistic — could a caller reasonably ask this over the phone?",
        YES_NO_UNCLEAR, key="q_realistic",
        help="Just a quick check that the user goal is sufficiently realistic to include in this dataset (i.e. is topical, sounds reasonable).",
    )
    q_complete = st.selectbox(
        "Is it complete/deterministic?",
        YES_NO_UNCLEAR, key="q_complete",
        help="Does this user goal cover all directions the agent might go in? Is there enough information on how to respond to different scenarios, are the resolution and failure conditions sufficiently clear and distinct from each other, etc. You may need to read the trace and check the expected flow to understand this one.",
    )
    q_raw_info = st.selectbox(
        "Is all raw info present? (codes, names, etc.)",
        YES_NO_UNCLEAR, key="q_raw_info",
        help="Does the user info contain all the required raw information that the caller would need to do the flow? You may need to read the trace and check the expected flow to understand this one.",
    )
    user_goal_comments = st.text_area(
        "User goal comments",
        key="user_goal_comments",
        height=80,
        help="Any comments, questions, concerns, etc that you have about this user goal or user goals in general.",
    )

    # -- Ground Truth Trace --
    st.markdown("---")
    st.markdown("#### Ground Truth Trace")

    if write_tool_calls:
        st.markdown("**Per write tool call:**")
        for i, tc in enumerate(write_tool_calls):
            st.markdown(f'`{tc["name"]}`')
            st.selectbox(
                "Inputs grounded?", YES_NO_NA,
                key=f"wtc_{current_id}_{i}_grounded",
                help="Can this tool call's inputs be inferred from previous tool call output, user info, or policies?",
            )
            st.selectbox(
                "Consistent with policies?", YES_NO_NA,
                key=f"wtc_{current_id}_{i}_policy",
                help="Is this tool call consistent with the agent policies?",
            )
    elif trace is None:
        st.info("Trace not available yet — per-tool-call questions will appear when trace data is added.")
    else:
        st.info("No write tool calls found in this trace.")

    st.markdown("**Overall trace:**")
    q_unwanted_mods = st.selectbox(
        "Modification tools that shouldn't have happened?",
        YES_NO, key="q_unwanted_mods",
        help="Are there any modification/write tools in the trace that should not have happened (they violate policies, aren't required for this flow, etc)?",
    )
    q_missing_mods = st.selectbox(
        "Missing modification tools?",
        YES_NO, key="q_missing_mods",
        help="Are there modification tools we expect to see in this flow that are missing? For example maybe a missing notification tool that's in the expected flow sequence, etc.",
    )
    q_alt_path = st.selectbox(
        "Another way to reach a different end DB state (following policies)?",
        YES_NO_UNCLEAR, key="q_alt_path",
        help="Is there a different sequence of modification tools or different parameters that could be used to still arrive at a correct end outcome? If so this is a problem because we need there to only be 1 correct answer.",
    )
    trace_comments = st.text_area(
        "Trace comments", key="trace_comments", height=80,
        help="Any comments you have about the trace.",
    )

    # -- Diff --
    st.markdown("---")
    st.markdown("#### Diff")
    q_mods_sensible = st.selectbox(
        "Do all modifications make sense given tool calls?",
        YES_NO_UNCLEAR, key="q_mods_sensible",
        help="This shows you the comparison between the initial database and the expected database after the correct modification tool calls have happened. Check whether all the changes in the diff are expected given the tool call sequence.",
    )
    diff_comments = st.text_area(
        "Diff comments", key="diff_comments", height=80,
        help="Any comments you have about the diff. If you see changes that don't make sense given the tool sequence, please flag them here.",
    )

    # -- General --
    st.markdown("---")
    st.markdown("#### General")
    general_comments = st.text_area(
        "Any other comments or issues",
        key="general_comments", height=100,
        help="Any other comments or concerns about this record that don't fit into the sections above.",
    )

    # -- Save --
    st.markdown("---")
    if st.button("Save Feedback", type="primary", use_container_width=True):
        feedback = {
            "user_goal": {
                "reflects_context": st.session_state.get("q_reflects", ""),
                "is_realistic": st.session_state.get("q_realistic", ""),
                "is_complete": st.session_state.get("q_complete", ""),
                "raw_info_present": st.session_state.get("q_raw_info", ""),
                "comments": st.session_state.get("user_goal_comments", ""),
            },
            "ground_truth_trace": {
                "write_tool_calls": [
                    {
                        "tool_name": tc["name"],
                        "inputs_grounded": st.session_state.get(f"wtc_{current_id}_{i}_grounded", ""),
                        "policy_consistent": st.session_state.get(f"wtc_{current_id}_{i}_policy", ""),
                    }
                    for i, tc in enumerate(write_tool_calls)
                ],
                "unwanted_modifications": st.session_state.get("q_unwanted_mods", ""),
                "missing_modifications": st.session_state.get("q_missing_mods", ""),
                "alternative_path_exists": st.session_state.get("q_alt_path", ""),
                "comments": st.session_state.get("trace_comments", ""),
            },
            "diff": {
                "modifications_sensible": st.session_state.get("q_mods_sensible", ""),
                "comments": st.session_state.get("diff_comments", ""),
            },
            "general": {
                "comments": st.session_state.get("general_comments", ""),
            },
        }
        save_feedback(current_id, feedback)
        st.success(f"Saved feedback for {current_id}")

    # Show status if feedback exists
    existing_fb = load_feedback(current_id)
    if existing_fb:
        st.caption(f"Last saved: {existing_fb.get('last_updated', '—')}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA: Data display
# ══════════════════════════════════════════════════════════════════════════════

st.title(f"Record {current_id}")
st.info(record.get("scenario_context", "No scenario context available."))

# ── Reference expander ───────────────────────────────────────────────────────
def _render_tools(tool_list: list[dict]):
    for tool in tool_list:
        name = tool["name"]
        st.markdown(f'**`{name}`** — {tool.get("description", "")}')
        req = tool.get("required_parameters", [])
        opt = tool.get("optional_parameters", [])
        if req or opt:
            params_md = ""
            for p in req:
                params_md += f"- **`{p['name']}`** ({p['type']}): {p['description']}\n"
            for p in opt:
                params_md += f"- *`{p['name']}`* ({p['type']}, optional): {p['description']}\n"
            st.markdown(params_md)
        st.divider()

tools_by_type: dict[str, list[dict]] = {"auth": [], "read": [], "write": []}
for tool in tools:
    tt = tool.get("tool_type", "read")
    tools_by_type.setdefault(tt, []).append(tool)

_policy_sections: list[tuple[str, str]] = []
_section_pattern = re.compile(r"^###\s+(.+)$", re.MULTILINE)
_matches = list(_section_pattern.finditer(instructions))
for idx, m in enumerate(_matches):
    title = m.group(1).strip()
    start = m.end()
    end = _matches[idx + 1].start() if idx + 1 < len(_matches) else len(instructions)
    body = instructions[start:end].strip().lstrip("-").strip()
    _policy_sections.append((title, body))
_preamble = instructions[: _matches[0].start()].strip() if _matches else ""

with st.expander("Reference: Tool Schemas & Agent Policies", expanded=False):
    tab_tools, tab_policies = st.tabs(["Tool Schemas", "Agent Policies"])
    with tab_tools:
        for tt, label in [
            ("auth", "Auth Tools"),
            ("read", "Read Tools"),
            ("write", "Write Tools"),
        ]:
            group = tools_by_type.get(tt, [])
            with st.expander(f"{label} ({len(group)})", expanded=False):
                _render_tools(group)
    with tab_policies:
        if _preamble:
            with st.expander("General", expanded=False):
                st.markdown(_preamble)
        for title, body in _policy_sections:
            with st.expander(title, expanded=False):
                st.markdown(body)

# ── User Goal ────────────────────────────────────────────────────────────────
with st.expander("User Goal", expanded=True):
    st.markdown("##### High-level Goal")
    st.info(goal.get("high_level_user_goal", "—"))

    st.markdown("##### Starting Utterance")
    st.code(goal.get("starting_utterance", "—"), language=None)

    # Decision tree
    if dt.get("must_have_criteria"):
        st.markdown("##### Must-Have Criteria")
        for i, item in enumerate(dt["must_have_criteria"], 1):
            st.markdown(f"{i}. {item}")

    if dt.get("nice_to_have_criteria"):
        st.markdown("##### Nice-to-Have Criteria")
        for item in dt["nice_to_have_criteria"]:
            st.markdown(f"- {item}")

    if dt.get("negotiation_behavior"):
        st.markdown("##### Negotiation Behavior")
        for i, item in enumerate(dt["negotiation_behavior"], 1):
            st.markdown(f"{i}. {item}")

    st.markdown("##### Resolution Condition")
    st.success(dt.get("resolution_condition", "—"))

    st.markdown("##### Failure Condition")
    st.error(dt.get("failure_condition", "—"))

    if dt.get("escalation_behavior"):
        st.markdown("##### Escalation Behavior")
        st.warning(dt["escalation_behavior"])

    if dt.get("edge_cases"):
        st.markdown("##### Edge Cases")
        for item in dt["edge_cases"]:
            st.markdown(f"- {item}")

    # Information required
    info = goal.get("information_required", {})
    if info:
        st.markdown("##### Information Required")
        rows = []
        for k, v in info.items():
            if isinstance(v, (dict, list)):
                v = json.dumps(v, default=str)
            rows.append(f"| **{k}** | `{v}` |")
        st.markdown("| Field | Value |\n|---|---|\n" + "\n".join(rows))


# ── Ground Truth Trace ───────────────────────────────────────────────────────
with st.expander("Ground Truth Trace", expanded=True):
    if trace is None:
        st.warning(
            "Trace data not yet available for this record. "
            "This section will populate when ground truth traces are added."
        )
    else:
        render_trace(trace, tool_type_map)

# ── Scenario DB Diff ─────────────────────────────────────────────────────────
with st.expander("Scenario DB Diff (Initial vs Expected Final)", expanded=True):
    if not initial_db:
        st.warning("Initial scenario DB not found.")
    elif not expected_db:
        st.warning("Expected scenario DB not found in ground truth.")
    else:
        initial_json = json.dumps(initial_db, indent=2, sort_keys=True)
        expected_json = json.dumps(expected_db, indent=2, sort_keys=True)
        diff_lines = list(difflib.unified_diff(
            initial_json.splitlines(),
            expected_json.splitlines(),
            fromfile="Initial DB",
            tofile="Expected DB",
            lineterm="",
        ))
        if not diff_lines:
            st.success("No differences between initial and expected scenario DB.")
        else:
            # Build GitHub-style unified diff HTML
            rows = []
            for line in diff_lines:
                escaped = (
                    line.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                if line.startswith("@@"):
                    rows.append(
                        f'<div style="background:#1e3a5f;color:#58a6ff;'
                        f'padding:4px 12px;margin-top:8px;border-radius:3px;'
                        f'font-weight:600">{escaped}</div>'
                    )
                elif line.startswith("+++") or line.startswith("---"):
                    rows.append(
                        f'<div style="color:#8b949e;padding:2px 12px;'
                        f'font-weight:700">{escaped}</div>'
                    )
                elif line.startswith("+"):
                    rows.append(
                        f'<div style="background:#0d2818;color:#56d364;'
                        f'padding:1px 12px;border-left:3px solid #2ea043">'
                        f'{escaped}</div>'
                    )
                elif line.startswith("-"):
                    rows.append(
                        f'<div style="background:#2d1115;color:#f85149;'
                        f'padding:1px 12px;border-left:3px solid #da3633">'
                        f'{escaped}</div>'
                    )
                else:
                    rows.append(
                        f'<div style="color:#c9d1d9;padding:1px 12px">'
                        f'{escaped}</div>'
                    )
            body = "\n".join(rows)
            html = (
                f'<div style="font-family:ui-monospace,SFMono-Regular,'
                f"'SF Mono',Menlo,Consolas,monospace;font-size:13px;"
                f'line-height:1.6;background:#0d1117;border:1px solid #30363d;'
                f'border-radius:6px;padding:8px 0;overflow-x:auto">'
                f'{body}</div>'
            )
            n_lines = len(diff_lines)
            height = min(max(300, n_lines * 24), 800)
            components.html(html, height=height, scrolling=True)
