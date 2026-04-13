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
TOOLS_MODULE_PATH = ROOT / "src" / "eva" / "assistant" / "tools" / "medical_hr_tools.py"
FEEDBACK_DIR = ROOT / "hr_review_feedback"
ASSIGNMENTS_PATH = ROOT / "configs" / "hr_assignments.yaml"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="HR Data Review", layout="wide")

TOOL_TYPE_COLORS = {
    "auth": "#FF9800",  # amber
    "read": "#4CAF50",  # green
    "write": "#F44336",  # red
    "system": "#9E9E9E",  # gray
}

TOOL_TYPE_LABELS = {
    "auth": "AUTH",
    "read": "READ",
    "write": "WRITE",
    "system": "SYSTEM",
}

YES_NO = ["Yes", "No"]


def get_display_style(tool_type: str) -> tuple[str, str]:
    """Return (color, label) for badge display. Auth tools display as write."""
    if tool_type == "auth":
        return TOOL_TYPE_COLORS["write"], TOOL_TYPE_LABELS["write"]
    return TOOL_TYPE_COLORS.get(tool_type, "#999"), TOOL_TYPE_LABELS.get(tool_type, "?")


# ── Question definitions ─────────────────────────────────────────────────────
QUESTIONS = [
    {
        "id": "user_goal",
        "label": "User Goal",
        "fields": [
            {
                "key": "q_reflects",
                "question": "Does it reflect intended scenario context? *",
                "options": YES_NO,
                "help": "We are trying to test a specific scenario that is described by the scenario context. We just want to check that the user goal is aligned with that scenario. Intents that are not meant to be satisfiable should be in nice to have, whereas intents that are satisfiable should be in must have. Adversarial intents should always be in nice to have.",
            },
            {
                "key": "q_realistic",
                "question": "Is it sufficiently realistic — could a caller reasonably ask this over the phone? *",
                "options": YES_NO,
                "help": "Just a quick check that the user goal is sufficiently realistic to include in this dataset (i.e. is topical, sounds reasonable).",
            },
            {
                "key": "q_complete",
                "question": "Is it complete/deterministic? *",
                "options": YES_NO,
                "help": "Does this user goal cover all directions the agent might go in? Is there enough information on how to respond to different scenarios, are the resolution and failure conditions sufficiently clear and distinct from each other, etc. You may need to read the trace and check the expected flow to understand this one.",
            },
            {
                "key": "q_raw_info",
                "question": "Is all raw info present? (codes, names, etc.) *",
                "options": YES_NO,
                "help": "Does the user info contain all the required raw information that the caller would need to do the flow? You may need to read the trace and check the expected flow to understand this one.",
            },
            {
                "key": "q_dates_make_sense",
                "question": "Do dates in the user goal make sense given the current date/time? *",
                "options": ["Yes", "No", "N/A"],
                "help": "If any dates are mentioned in the user goal (appointment dates, shift dates, birth dates of dependents/spouses, etc.), do they make sense relative to the current date and time? Select N/A if no dates are present.",
            },
            {
                "key": "q_issues_from_seed",
                "question": "If the user goal has issues, do they come from the seed data? *",
                "options": ["Yes", "No", "N/A"],
                "help": "For example, if certain pieces of information in 'information required' don't make sense, are those same pieces of information present (and incorrect) in the seed data? Select N/A if the user goal has no issues.",
            },
        ],
        "comment_key": "user_goal_comments",
        "comment_help": "Any comments, questions, concerns, etc that you have about this user goal or user goals in general.",
    },
    {
        "id": "trace",
        "label": "Trace",
        "has_tool_calls": True,
        "fields": [
            {
                "key": "q_unwanted_mods",
                "question": "Modification tools that shouldn't have happened? *",
                "options": YES_NO,
                "help": "Are there any modification/write tools in the trace that should not have happened (they violate policies, aren't required for this flow, etc)?",
            },
            {
                "key": "q_missing_mods",
                "question": "Missing modification tools? *",
                "options": YES_NO,
                "help": "Are there modification tools we expect to see in this flow that are missing? For example maybe a missing notification tool that's in the expected flow sequence, etc.",
            },
            {
                "key": "q_alt_path",
                "question": "Another way to reach a different end DB state (following policies)? *",
                "options": YES_NO,
                "help": "Is there a different sequence of modification tools or different parameters that could be used to still arrive at a correct end outcome? If so this is a problem because we need there to only be 1 correct answer.",
            },
        ],
        "comment_key": "trace_comments",
        "comment_help": "Any comments you have about the trace.",
    },
    {
        "id": "diff",
        "label": "Diff",
        "fields": [],
        "comment_key": "diff_comments",
        "comment_help": "Any comments you have about the diff. If you see changes that don't make sense given the tool sequence, please flag them here.",
    },
    {
        "id": "general",
        "label": "General",
        "fields": [],
        "comment_key": "general_comments",
        "comment_help": "Any other comments or concerns about this record that don't fit into the sections above.",
    },
]


# ── Data loaders ─────────────────────────────────────────────────────────────


def _id_sort_key(rid: str) -> tuple:
    """Numeric-aware sort key for record IDs like '2.1', 'A3', 'D10.3'."""
    match = re.match(r"^([A-Z]*)(\d.*)$", rid)
    if not match:
        return (1, rid, 0, 0)
    prefix = match.group(1)
    parts = tuple(int(x) for x in match.group(2).split("."))
    prefix_key = (0, "") if not prefix else (1, prefix)
    return prefix_key + parts


@st.cache_data
def load_records(mtime: float = 0) -> list[dict]:
    records = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return sorted(records, key=lambda r: _id_sort_key(r["id"]))


@st.cache_data
def load_agent_config(mtime: float = 0) -> tuple[list[dict], str, dict[str, str]]:
    with open(AGENT_YAML_PATH) as f:
        config = yaml.safe_load(f)
    tools = config.get("tools", [])
    instructions = config.get("instructions", "")
    tool_type_map = {t["name"]: t.get("tool_type", "read") for t in tools}
    return tools, instructions, tool_type_map


@st.cache_data
def load_flow_sequences(mtime: float = 0) -> list[dict]:
    """Parse flow sequences from medical_hr_tools.py docstring."""
    with open(TOOLS_MODULE_PATH) as f:
        content = f.read()
    match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if not match:
        return []
    docstring = match.group(1)
    flows = []
    blocks = re.split(r"\n\s*\n", docstring)
    flow_header = re.compile(r"Flow\s+(\d+)\s+[–-]\s+(.+?):\s*\n(.+)", re.DOTALL)
    for block in blocks:
        m = flow_header.search(block.strip())
        if not m:
            continue
        number = int(m.group(1))
        name = m.group(2).strip()
        tool_text = re.sub(r"\s*\n\s*", " ", m.group(3).strip())
        tool_names = [t.strip() for t in re.split(r"\s*→\s*", tool_text)]
        parsed_tools = []
        for t in tool_names:
            repeat_match = re.match(r"(.+?)\s*\(×N\)", t)
            if repeat_match:
                parsed_tools.append({"name": repeat_match.group(1), "repeat": True})
            else:
                parsed_tools.append({"name": t, "repeat": False})
        flows.append({"number": number, "name": name, "tools": parsed_tools})
    return sorted(flows, key=lambda f: f["number"])


@st.cache_data
def load_initial_scenario(record_id: str, mtime: float = 0) -> dict:
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


def load_assignments() -> dict[str, list[str]]:
    if not ASSIGNMENTS_PATH.exists():
        return {}
    with open(ASSIGNMENTS_PATH) as f:
        config = yaml.safe_load(f) or {}
    return config.get("assignments", {}) or {}


# ── Trace helpers ────────────────────────────────────────────────────────────
def extract_review_tool_calls(
    trace: list[dict],
    tool_type_map: dict[str, str],
) -> list[dict]:
    """Extract auth and write tool_call events from the expected trace."""
    calls = []
    for msg in trace:
        if msg.get("event_type") == "tool_call":
            name = msg.get("tool_name", "unknown")
            if tool_type_map.get(name) in ("auth", "write"):
                calls.append(
                    {
                        "name": name,
                        "tool_type": tool_type_map.get(name, "write"),
                        "params": msg.get("params", {}),
                    }
                )
    return calls


def render_trace(trace: list[dict], tool_type_map: dict[str, str]):
    """Render expected trace with styled divs (no chat_message to avoid auto-scroll)."""
    for msg in trace:
        event = msg.get("event_type", "")

        if event == "user_utterance":
            st.markdown(
                f'<div style="background:#1a2733;border-left:3px solid #2196F3;'
                f'padding:8px 12px;margin:4px 0;border-radius:4px">'
                f'<strong style="color:#64B5F6">User:</strong> '
                f"{msg.get('utterance', '')}</div>",
                unsafe_allow_html=True,
            )

        elif event == "agent_utterance":
            st.markdown(
                f'<div style="background:#1a1a2e;border-left:3px solid #9C27B0;'
                f'padding:8px 12px;margin:4px 0;border-radius:4px">'
                f'<strong style="color:#CE93D8">Agent:</strong> '
                f"{msg.get('utterance', '')}</div>",
                unsafe_allow_html=True,
            )

        elif event == "tool_call":
            name = msg.get("tool_name", "unknown")
            tool_type = tool_type_map.get(name, "read")
            color, label = get_display_style(tool_type)
            params = msg.get("params", {})
            badge = (
                f'<span style="background:{color};color:white;padding:2px 8px;'
                f'border-radius:4px;font-size:0.75em;font-weight:bold">{label}</span>'
            )
            st.markdown(
                f'<div style="background:#1c1c1c;border-left:3px solid {color};'
                f'padding:8px 12px;margin:4px 0;border-radius:4px">'
                f"{badge} <code>{name}</code></div>",
                unsafe_allow_html=True,
            )
            if params:
                st.code(json.dumps(params, indent=2), language="json")

        elif event == "tool_response":
            name = msg.get("tool_name", "unknown")
            tool_type = tool_type_map.get(name, "read")
            color, label = get_display_style(tool_type)
            status = msg.get("status", "")
            response = msg.get("response", {})
            badge = (
                f'<span style="background:{color};color:white;padding:2px 8px;'
                f'border-radius:4px;font-size:0.75em;font-weight:bold">{label}</span>'
            )
            st.markdown(
                f'<div style="background:#111;border-left:3px solid #666;'
                f'padding:8px 12px;margin:4px 0;border-radius:4px">'
                f"{badge} <code>{name}</code> — <strong>{status}</strong></div>",
                unsafe_allow_html=True,
            )
            if response:
                st.json(response)


# ── Load data ────────────────────────────────────────────────────────────────
records = load_records(DATASET_PATH.stat().st_mtime)
all_ids = [r["id"] for r in records]
tools, instructions, tool_type_map = load_agent_config(AGENT_YAML_PATH.stat().st_mtime)
flow_sequences = load_flow_sequences(TOOLS_MODULE_PATH.stat().st_mtime)
assignments = load_assignments()
labeler_names = sorted(assignments.keys())

# ── Labeler filter (rendered in nav bar below) ───────────────────────────────
labeler_options = ["All"] + labeler_names
selected_labeler = st.session_state.get("labeler_filter", "All")

if selected_labeler != "All" and selected_labeler in assignments:
    assigned_ids = set(assignments[selected_labeler])
    filtered_records = [r for r in records if r["id"] in assigned_ids]
    if not filtered_records:
        st.sidebar.warning(f"No records assigned to {selected_labeler}")
        filtered_records = records
else:
    filtered_records = records

ids = [r["id"] for r in filtered_records]
id_set = set(ids)
record_by_id = {r["id"]: r for r in filtered_records}

# ── Determine current record from query params ──────────────────────────────
params = st.query_params
current_id = params.get("record_id", ids[0])
if current_id not in id_set:
    current_id = ids[0]
current_idx = ids.index(current_id)

# ── Detect record change and pre-populate review-question feedback state ──────
if st.session_state.get("_prev_record_id") != current_id:
    st.session_state["_prev_record_id"] = current_id
    existing = load_feedback(current_id)
    if existing:
        ug = existing.get("user_goal", {})
        st.session_state["q_reflects"] = ug.get("reflects_context", "")
        st.session_state["q_realistic"] = ug.get("is_realistic", "")
        st.session_state["q_complete"] = ug.get("is_complete", "")
        st.session_state["q_raw_info"] = ug.get("raw_info_present", "")
        st.session_state["q_dates_make_sense"] = ug.get("dates_make_sense", "")
        st.session_state["q_issues_from_seed"] = ug.get("issues_from_seed", "")
        st.session_state["user_goal_comments"] = ug.get("comments", "")

        gt = existing.get("ground_truth_trace", {})
        for i, tc in enumerate(gt.get("review_tool_calls", [])):
            st.session_state[f"wtc_{current_id}_{i}_grounded"] = tc.get("inputs_grounded", "")
            st.session_state[f"wtc_{current_id}_{i}_policy"] = tc.get("policy_consistent", "")
        st.session_state["q_unwanted_mods"] = gt.get("unwanted_modifications", "")
        st.session_state["q_missing_mods"] = gt.get("missing_modifications", "")
        st.session_state["q_alt_path"] = gt.get("alternative_path_exists", "")
        st.session_state["trace_comments"] = gt.get("comments", "")

        df = existing.get("diff", {})
        st.session_state["diff_comments"] = df.get("comments", "")

        gen = existing.get("general", {})
        st.session_state["general_comments"] = gen.get("comments", "")

        usab = existing.get("acceptability", {})
        st.session_state["acceptability_as_is"] = usab.get("can_use_as_is", "")
        st.session_state["acceptability_with_edits"] = usab.get("can_submit_with_edits", "")
        st.session_state["acceptability_needs_change"] = usab.get("needs_change_notes", "")
    else:
        for key in [
            "q_reflects",
            "q_realistic",
            "q_complete",
            "q_raw_info",
            "q_dates_make_sense",
            "q_issues_from_seed",
            "user_goal_comments",
            "q_unwanted_mods",
            "q_missing_mods",
            "q_alt_path",
            "trace_comments",
            "diff_comments",
            "general_comments",
            "acceptability_as_is",
            "acceptability_with_edits",
            "acceptability_needs_change",
        ]:
            st.session_state[key] = ""

record = record_by_id[current_id]
goal = record.get("user_goal", {})
dt = goal.get("decision_tree", {})
ground_truth = record.get("ground_truth", {})
expected_trace = ground_truth.get("expected_trace", {})
trace = expected_trace.get("trace", None) if expected_trace else None
expected_db = ground_truth.get("expected_scenario_db", {})
_scenario_path = SCENARIOS_DIR / f"{current_id}.json"
initial_db = load_initial_scenario(current_id, _scenario_path.stat().st_mtime if _scenario_path.exists() else 0)

# ── Per-record edit-field initialization (run once per record per session) ────
# Uses record-scoped keys so each record has independent state; value= sets the
# initial content the first time the record is visited, then session state takes over.
_eid = current_id  # shorthand for key suffix
if f"edit_high_level_goal_{_eid}" not in st.session_state:
    _fb = load_feedback(current_id) or {}
    _eug = _fb.get("edited_user_goal", {})
    _edt = _eug.get("decision_tree", dt)
    st.session_state[f"edit_high_level_goal_{_eid}"] = _eug.get("high_level_user_goal", goal.get("high_level_user_goal", ""))
    st.session_state[f"edit_starting_utterance_{_eid}"] = _eug.get("starting_utterance", goal.get("starting_utterance", ""))
    for _fkey, _items in [
        (f"edit_must_have_{_eid}", _edt.get("must_have_criteria", dt.get("must_have_criteria", []))),
        (f"edit_nice_to_have_{_eid}", _edt.get("nice_to_have_criteria", dt.get("nice_to_have_criteria", []))),
        (f"edit_negotiation_{_eid}", _edt.get("negotiation_behavior", dt.get("negotiation_behavior", []))),
        (f"edit_edge_cases_{_eid}", _edt.get("edge_cases", dt.get("edge_cases", []))),
    ]:
        st.session_state[f"{_fkey}_count"] = len(_items)
        for _i, _item in enumerate(_items):
            st.session_state[f"{_fkey}_{_i}"] = _item
    st.session_state[f"edit_resolution_{_eid}"] = _edt.get("resolution_condition", dt.get("resolution_condition", ""))
    st.session_state[f"edit_failure_{_eid}"] = _edt.get("failure_condition", dt.get("failure_condition", ""))
    st.session_state[f"edit_escalation_{_eid}"] = _edt.get("escalation_behavior", dt.get("escalation_behavior", ""))
    _info = _eug.get("information_required", goal.get("information_required", {}))
    st.session_state[f"edit_info_required_{_eid}"] = json.dumps(_info, indent=2) if _info else "{}"

# Extract tool calls from trace (if it exists) for the review form
review_tool_calls = extract_review_tool_calls(trace, tool_type_map) if trace else []

# ══════════════════════════════════════════════════════════════════════════════
# HEADER: Title + Navigation
# ══════════════════════════════════════════════════════════════════════════════

nav_left, nav_mid, nav_right, nav_labeler, nav_height = st.columns([1, 3, 1, 2, 1])
with nav_left:
    if st.button("< Prev", disabled=current_idx == 0, use_container_width=True):
        st.query_params["record_id"] = ids[current_idx - 1]
        st.rerun()
with nav_mid:
    selected = st.selectbox("Record", ids, index=current_idx, label_visibility="collapsed")
    if selected != current_id:
        st.query_params["record_id"] = selected
        st.rerun()
with nav_right:
    if st.button("Next >", disabled=current_idx == len(ids) - 1, use_container_width=True):
        st.query_params["record_id"] = ids[current_idx + 1]
        st.rerun()
with nav_labeler:
    labeler_idx = labeler_options.index(selected_labeler) if selected_labeler in labeler_options else 0
    new_labeler = st.selectbox(
        "Labeler", labeler_options, index=labeler_idx, key="labeler_filter", label_visibility="collapsed"
    )
    if new_labeler != selected_labeler:
        st.rerun()
with nav_height:
    q_height = st.slider("Height", 200, 800, 400, 50, label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# REVIEW QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════

q_labels = [q["label"] for q in QUESTIONS] + ["Acceptability"]
st.markdown("**Review Questions** — questions marked \\* are required")
with st.container(height=q_height):
    q_tabs = st.tabs(q_labels)

# ── Helper: build feedback dict and validate required fields ─────────────────
def _build_and_save_feedback(save_key: str):
    required_checks = [
        ("q_reflects", "Does it reflect intended scenario context?"),
        ("q_realistic", "Is it sufficiently realistic?"),
        ("q_complete", "Is it complete/deterministic?"),
        ("q_raw_info", "Is all raw info present?"),
        ("q_dates_make_sense", "Do dates make sense?"),
        ("q_issues_from_seed", "If the user goal has issues, do they come from the seed data?"),
        ("q_unwanted_mods", "Modification tools that shouldn't have happened?"),
        ("q_missing_mods", "Missing modification tools?"),
        ("q_alt_path", "Another way to reach different end DB state?"),
        ("usability_as_is", "Can this sample be used as is?"),
    ]
    for i, tc in enumerate(review_tool_calls):
        required_checks.append((f"wtc_{current_id}_{i}_grounded", f"Inputs grounded? ({tc['name']})"))
        required_checks.append((f"wtc_{current_id}_{i}_policy", f"Consistent with policies? ({tc['name']})"))

    missing = [label for key, label in required_checks if not st.session_state.get(key)]
    if st.session_state.get("usability_as_is") == "No" and not st.session_state.get("usability_with_edits"):
        missing.append("Can it be submitted with your edits?")

    if missing:
        st.error("Please answer all required questions before saving:\n- " + "\n- ".join(missing))
        return

    try:
        _info_edited = json.loads(st.session_state.get(f"edit_info_required_{current_id}", "{}") or "{}")
    except json.JSONDecodeError:
        _info_edited = {}

    feedback = {
        "user_goal": {
            "reflects_context": st.session_state.get("q_reflects", ""),
            "is_realistic": st.session_state.get("q_realistic", ""),
            "is_complete": st.session_state.get("q_complete", ""),
            "raw_info_present": st.session_state.get("q_raw_info", ""),
            "dates_make_sense": st.session_state.get("q_dates_make_sense", ""),
            "issues_from_seed": st.session_state.get("q_issues_from_seed", ""),
            "comments": st.session_state.get("user_goal_comments", ""),
        },
        "edited_user_goal": {
            "high_level_user_goal": st.session_state.get(f"edit_high_level_goal_{current_id}", ""),
            "starting_utterance": st.session_state.get(f"edit_starting_utterance_{current_id}", ""),
            "decision_tree": {
                "must_have_criteria": _get_list_values(f"edit_must_have_{current_id}"),
                "nice_to_have_criteria": _get_list_values(f"edit_nice_to_have_{current_id}"),
                "negotiation_behavior": _get_list_values(f"edit_negotiation_{current_id}"),
                "resolution_condition": st.session_state.get(f"edit_resolution_{current_id}", ""),
                "failure_condition": st.session_state.get(f"edit_failure_{current_id}", ""),
                "escalation_behavior": st.session_state.get(f"edit_escalation_{current_id}", ""),
                "edge_cases": _get_list_values(f"edit_edge_cases_{current_id}"),
            },
            "information_required": _info_edited,
        },
        "ground_truth_trace": {
            "review_tool_calls": [
                {
                    "tool_name": tc["name"],
                    "tool_type": tc.get("tool_type", "write"),
                    "inputs_grounded": st.session_state.get(f"wtc_{current_id}_{i}_grounded", ""),
                    "policy_consistent": st.session_state.get(f"wtc_{current_id}_{i}_policy", ""),
                }
                for i, tc in enumerate(review_tool_calls)
            ],
            "unwanted_modifications": st.session_state.get("q_unwanted_mods", ""),
            "missing_modifications": st.session_state.get("q_missing_mods", ""),
            "alternative_path_exists": st.session_state.get("q_alt_path", ""),
            "comments": st.session_state.get("trace_comments", ""),
        },
        "diff": {
            "comments": st.session_state.get("diff_comments", ""),
        },
        "general": {
            "comments": st.session_state.get("general_comments", ""),
        },
        "acceptability": {
            "can_use_as_is": st.session_state.get("acceptability_as_is", ""),
            "can_submit_with_edits": st.session_state.get("acceptability_with_edits", ""),
            "needs_change_notes": st.session_state.get("acceptability_needs_change", ""),
        },
    }
    save_feedback(current_id, feedback)
    st.success(f"Saved feedback for {current_id}")


for q_section, tab in zip(QUESTIONS, q_tabs[:-1]):
    with tab:
        fields = q_section["fields"]
        # Lay out radio questions in a grid (2 per row)
        if fields:
            for row_start in range(0, len(fields), 2):
                row_fields = fields[row_start : row_start + 2]
                cols = st.columns(len(row_fields))
                for col, field in zip(cols, row_fields):
                    with col:
                        current_val = st.session_state.get(field["key"], "")
                        idx = field["options"].index(current_val) if current_val in field["options"] else None
                        st.radio(
                            field["question"],
                            field["options"],
                            index=idx,
                            key=field["key"],
                            help=field["help"],
                            horizontal=True,
                        )

        # Per auth/write tool call questions (only for trace section)
        if q_section.get("has_tool_calls"):
            if review_tool_calls:
                st.markdown("**Per write tool call:**")
                # Lay out tool calls in a grid (2 per row)
                for row_start in range(0, len(review_tool_calls), 2):
                    row_tcs = review_tool_calls[row_start : row_start + 2]
                    cols = st.columns(len(row_tcs))
                    for col, (i, tc) in zip(
                        cols, enumerate(review_tool_calls[row_start : row_start + 2], start=row_start)
                    ):
                        with col:
                            tt = tc.get("tool_type", "write")
                            color, label = get_display_style(tt)
                            st.markdown(
                                f'{i + 1}) <span style="background:{color};color:white;padding:2px 6px;'
                                f'border-radius:3px;font-size:0.7em">{label}</span> '
                                f"`{tc['name']}`",
                                unsafe_allow_html=True,
                            )
                            _gk = f"wtc_{current_id}_{i}_grounded"
                            _gv = st.session_state.get(_gk, "")
                            st.radio(
                                "Inputs grounded? *",
                                YES_NO,
                                index=YES_NO.index(_gv) if _gv in YES_NO else None,
                                key=_gk,
                                help="Can this tool call's inputs be inferred from previous tool call output, user info, or policies?",
                                horizontal=True,
                            )
                            _pk = f"wtc_{current_id}_{i}_policy"
                            _pv = st.session_state.get(_pk, "")
                            st.radio(
                                "Consistent with policies? *",
                                YES_NO,
                                index=YES_NO.index(_pv) if _pv in YES_NO else None,
                                key=_pk,
                                help="Is this tool call consistent with the agent policies?",
                                horizontal=True,
                            )
            elif trace is None:
                st.info("Trace not available yet — per-tool-call questions will appear when trace data is added.")
            else:
                st.info("No auth/write tool calls found in this trace.")

        # Comment box
        st.text_area(
            "Comments",
            key=q_section["comment_key"],
            height=max(80, q_height - 200),
            help=q_section["comment_help"],
        )

        # Save button in every tab for convenience
        if st.button("Save Feedback", type="primary", key=f"save_{q_section['id']}"):
            _build_and_save_feedback(f"save_{q_section['id']}")

# ── acceptability tab ─────────────────────────────────────────────────────────────
with q_tabs[-1]:
    _as_is_val = st.session_state.get("usability_as_is", "")
    st.radio(
        "Can this sample be used as is? *",
        YES_NO,
        index=YES_NO.index(_as_is_val) if _as_is_val in YES_NO else None,
        key="usability_as_is",
        help="Criteria: the user goal is sufficient AND the modification tools are exactly what we expect.",
        horizontal=True,
    )

    if st.session_state.get("usability_as_is") == "No":
        _with_edits_val = st.session_state.get("usability_with_edits", "")
        st.radio(
            "Can it be submitted with your edits? *",
            YES_NO,
            index=YES_NO.index(_with_edits_val) if _with_edits_val in YES_NO else None,
            key="usability_with_edits",
            help="After applying your edits to the user goal, would this sample be usable?",
            horizontal=True,
        )

        if st.session_state.get("usability_with_edits") == "No":
            st.text_area(
                "What would need to change?",
                key="usability_needs_change",
                height=max(80, q_height - 250),
                help="Describe what fundamental changes would be needed to make this sample usable.",
            )

    if st.button("Save Feedback", type="primary", key="save_usability"):
        _build_and_save_feedback("save_usability")


# ══════════════════════════════════════════════════════════════════════════════
# DATA DISPLAY: User Goal (left) | Trace (right), Diff below
# ══════════════════════════════════════════════════════════════════════════════


# ── Reference expander ───────────────────────────────────────────────────────
def _render_tools(tool_list: list[dict]):
    for tool in tool_list:
        name = tool["name"]
        st.markdown(f"**`{name}`** — {tool.get('description', '')}")
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

with st.expander("Reference: Tool Schemas, Flows & Agent Policies", expanded=False):
    tab_tools, tab_flows, tab_policies = st.tabs(["Tool Schemas", "Flows", "Agent Policies"])
    with tab_tools:
        for tt, label in [
            ("auth", "Auth Tools"),
            ("read", "Read Tools"),
            ("write", "Write Tools"),
            ("system", "System Tools"),
        ]:
            group = tools_by_type.get(tt, [])
            with st.expander(f"{label} ({len(group)})", expanded=False):
                _render_tools(sorted(group, key=lambda t: t["name"]))
    with tab_flows:
        for flow in flow_sequences:
            with st.expander(f"Flow {flow['number']} — {flow['name']}", expanded=False):
                parts = []
                for tool in flow["tools"]:
                    tt = tool_type_map.get(tool["name"], "read")
                    color, label = get_display_style(tt)
                    badge = (
                        f'<span style="background:{color};color:white;padding:2px 6px;'
                        f'border-radius:3px;font-size:0.7em;font-weight:bold">{label}</span>'
                    )
                    name_str = f"<code>{tool['name']}</code>"
                    if tool["repeat"]:
                        name_str += " (xN)"
                    parts.append(f"{badge} {name_str}")
                st.markdown(" &rarr; ".join(parts), unsafe_allow_html=True)
    _policy_flow_map = {
        "Credentialing and Licenses": "Flow 1",
        "Shift Scheduling and Swaps": "Flow 2",
        "Malpractice Coverage": "Flow 3",
        "Onboarding": "Flow 4",
        "DEA Registration": "Flow 5",
        "Leave of Absence (FMLA)": "Flow 6",
        "Payroll Corrections": "Flow 7",
        "Clinical Privileges": "Flow 8",
        "On-Call Registration": "Flow 9",
        "I-9 Work Authorization Verification": "Flow 10",
        "Visa and Immigration": "Flow 11",
        "PTO Request": "Flow 12",
    }
    with tab_policies:
        if _preamble:
            with st.expander("General", expanded=False):
                st.markdown(_preamble)
        _general_sections = [(t, b) for t, b in _policy_sections if t not in _policy_flow_map]
        _flow_sections = [(t, b) for t, b in _policy_sections if t in _policy_flow_map]
        _flow_sections.sort(key=lambda x: int(_policy_flow_map[x[0]].split()[1]))
        for title, body in _general_sections + _flow_sections:
            flow_label = _policy_flow_map.get(title, "")
            display_title = f"{title} ({flow_label})" if flow_label else title
            with st.expander(display_title, expanded=False):
                st.markdown(body)

_seed_data = record.get("seed_data", None)
if _seed_data is not None:
    with st.expander("Seed Data", expanded=False):
        if isinstance(_seed_data, dict):
            st.json(_seed_data)
        else:
            st.write(_seed_data)

with st.expander("Scenario Context", expanded=False):
    _sc = record.get("scenario_context")
    if isinstance(_sc, dict):
        st.json(_sc)
    elif _sc:
        st.info(_sc)
    else:
        st.caption("No scenario context available.")
_raw_dt = record.get("current_date_time", "—")
_dt_display = _raw_dt
try:
    _dt_match = re.match(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s*(.*)", _raw_dt)
    if _dt_match:
        _parsed = datetime.strptime(_dt_match.group(1), "%Y-%m-%d")
        _spelled = _parsed.strftime("%B %d, %Y").replace(" 0", " ")
        _dt_display = f"{_dt_match.group(1)} ({_spelled}) {_dt_match.group(2)} {_dt_match.group(3)}".strip()
except ValueError:
    pass
st.markdown(
    f'<span style="font-size:1.15em"><strong>Current date/time:</strong> <code>{_dt_display}</code></span>',
    unsafe_allow_html=True,
)

# ── Side-by-side: User Goal | Trace ──────────────────────────────────────────
col_goal, col_trace = st.columns(2)

def _auto_height(text: str, min_height: int = 68, chars_per_line: int = 65) -> int:
    """Estimate the pixel height needed to display text without scrolling."""
    if not text:
        return min_height
    lines = text.split("\n")
    total_rows = sum(max(1, -(-len(line) // chars_per_line)) for line in lines)
    return max(min_height, total_rows * 22 + 40)


def _render_list_field(label: str, base_key: str):
    """Render a list field as individually editable text areas, one per item."""
    st.markdown(f"**{label}**")
    count = st.session_state.get(f"{base_key}_count", 0)
    for i in range(count):
        item_key = f"{base_key}_{i}"
        if item_key not in st.session_state:
            st.session_state[item_key] = ""
        text = st.session_state.get(item_key, "")
        st.text_area(f"item {i + 1}", key=item_key, height=_auto_height(text), label_visibility="collapsed")


def _get_list_values(base_key: str) -> list[str]:
    """Read back edited list items from session state."""
    count = st.session_state.get(f"{base_key}_count", 0)
    return [v for i in range(count) if (v := st.session_state.get(f"{base_key}_{i}", "").strip())]


def _render_diff_html(diff_lines: list[str]):
    """Render unified diff lines as colored HTML (reused for user goal and scenario DB diffs)."""
    rows = []
    for line in diff_lines:
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if line.startswith("@@"):
            rows.append(
                f'<div style="background:#1e3a5f;color:#58a6ff;'
                f"padding:4px 12px;margin-top:8px;border-radius:3px;"
                f'font-weight:600">{escaped}</div>'
            )
        elif line.startswith("+++") or line.startswith("---"):
            rows.append(f'<div style="color:#8b949e;padding:2px 12px;font-weight:700">{escaped}</div>')
        elif line.startswith("+"):
            rows.append(
                f'<div style="background:#0d2818;color:#56d364;'
                f'padding:1px 12px;border-left:3px solid #2ea043">'
                f"{escaped}</div>"
            )
        elif line.startswith("-"):
            rows.append(
                f'<div style="background:#2d1115;color:#f85149;'
                f'padding:1px 12px;border-left:3px solid #da3633">'
                f"{escaped}</div>"
            )
        else:
            rows.append(f'<div style="color:#c9d1d9;padding:1px 12px">{escaped}</div>')
    body = "\n".join(rows)
    n_lines = len(diff_lines)
    height = min(max(200, n_lines * 22), 600)
    html = (
        f'<div style="font-family:ui-monospace,SFMono-Regular,'
        f"'SF Mono',Menlo,Consolas,monospace;font-size:12px;"
        f"line-height:1.5;background:#0d1117;border:1px solid #30363d;"
        f'border-radius:6px;padding:8px 0;overflow-x:auto">'
        f"{body}</div>"
    )
    components.html(html, height=height, scrolling=True)


# Estimate total pixel height of the user goal column so the trace container matches
_goal_height = 60  # title + top padding
_goal_height += _auto_height(st.session_state.get(f"edit_high_level_goal_{_eid}", "")) + 48
_goal_height += _auto_height(st.session_state.get(f"edit_starting_utterance_{_eid}", "")) + 48
for _lkey in [f"edit_must_have_{_eid}", f"edit_nice_to_have_{_eid}", f"edit_negotiation_{_eid}", f"edit_edge_cases_{_eid}"]:
    _cnt = st.session_state.get(f"{_lkey}_count", 0)
    _goal_height += 32  # bold label
    for _li in range(_cnt):
        _goal_height += _auto_height(st.session_state.get(f"{_lkey}_{_li}", "")) + 16
_goal_height += _auto_height(st.session_state.get(f"edit_resolution_{_eid}", "")) + 48
_goal_height += _auto_height(st.session_state.get(f"edit_failure_{_eid}", "")) + 48
_goal_height += _auto_height(st.session_state.get(f"edit_escalation_{_eid}", "")) + 48
_goal_height += _auto_height(st.session_state.get(f"edit_info_required_{_eid}", ""), min_height=100) + 48

with col_goal:
    st.markdown("##### User Goal")
    st.text_area("High-level Goal", key=f"edit_high_level_goal_{_eid}",
                 height=_auto_height(st.session_state.get(f"edit_high_level_goal_{_eid}", "")))
    st.text_area("Starting Utterance", key=f"edit_starting_utterance_{_eid}",
                 height=_auto_height(st.session_state.get(f"edit_starting_utterance_{_eid}", "")))
    _render_list_field("Must-Have Criteria", f"edit_must_have_{_eid}")
    _render_list_field("Nice-to-Have Criteria", f"edit_nice_to_have_{_eid}")
    _render_list_field("Negotiation Behavior", f"edit_negotiation_{_eid}")
    st.text_area("Resolution Condition", key=f"edit_resolution_{_eid}",
                 height=_auto_height(st.session_state.get(f"edit_resolution_{_eid}", "")))
    st.text_area("Failure Condition", key=f"edit_failure_{_eid}",
                 height=_auto_height(st.session_state.get(f"edit_failure_{_eid}", "")))
    st.text_area("Escalation Behavior", key=f"edit_escalation_{_eid}",
                 height=_auto_height(st.session_state.get(f"edit_escalation_{_eid}", "")))
    _render_list_field("Edge Cases", f"edit_edge_cases_{_eid}")
    _info_text = st.session_state.get(f"edit_info_required_{_eid}", "")
    st.text_area("Information Required (JSON)", key=f"edit_info_required_{_eid}",
                 height=_auto_height(_info_text, min_height=100))

    # ── Diff: original vs edited ──────────────────────────────────────────────
    try:
        _info_for_diff = json.loads(st.session_state.get(f"edit_info_required_{_eid}", "{}") or "{}")
    except json.JSONDecodeError:
        _info_for_diff = {}

    _original_goal_dict = {
        "high_level_user_goal": goal.get("high_level_user_goal", ""),
        "starting_utterance": goal.get("starting_utterance", ""),
        "decision_tree": {
            "must_have_criteria": dt.get("must_have_criteria", []),
            "nice_to_have_criteria": dt.get("nice_to_have_criteria", []),
            "negotiation_behavior": dt.get("negotiation_behavior", []),
            "resolution_condition": dt.get("resolution_condition", ""),
            "failure_condition": dt.get("failure_condition", ""),
            "escalation_behavior": dt.get("escalation_behavior", ""),
            "edge_cases": dt.get("edge_cases", []),
        },
        "information_required": goal.get("information_required", {}),
    }
    _edited_goal_dict = {
        "high_level_user_goal": st.session_state.get(f"edit_high_level_goal_{_eid}", ""),
        "starting_utterance": st.session_state.get(f"edit_starting_utterance_{_eid}", ""),
        "decision_tree": {
            "must_have_criteria": _get_list_values(f"edit_must_have_{_eid}"),
            "nice_to_have_criteria": _get_list_values(f"edit_nice_to_have_{_eid}"),
            "negotiation_behavior": _get_list_values(f"edit_negotiation_{_eid}"),
            "resolution_condition": st.session_state.get(f"edit_resolution_{_eid}", ""),
            "failure_condition": st.session_state.get(f"edit_failure_{_eid}", ""),
            "escalation_behavior": st.session_state.get(f"edit_escalation_{_eid}", ""),
            "edge_cases": _get_list_values(f"edit_edge_cases_{_eid}"),
        },
        "information_required": _info_for_diff,
    }
    _orig_json = json.dumps(_original_goal_dict, indent=2, sort_keys=True)
    _edit_json = json.dumps(_edited_goal_dict, indent=2, sort_keys=True)
    _goal_diff = list(
        difflib.unified_diff(
            _orig_json.splitlines(),
            _edit_json.splitlines(),
            fromfile="Original",
            tofile="Edited",
            lineterm="",
        )
    )
    if _goal_diff:
        st.markdown("**Edits (diff)**")
        _render_diff_html(_goal_diff)

with col_trace:
    st.markdown("##### Ground Truth Trace")
    with st.container(height=_goal_height, border=False):
        if trace is None:
            st.warning(
                "Trace data not yet available for this record. "
                "This section will populate when ground truth traces are added."
            )
        else:
            render_trace(trace, tool_type_map)

# ── Diff (full width below) ─────────────────────────────────────────────────
with st.expander("Scenario DB Diff (Initial vs Expected Final)", expanded=True):
    if not initial_db:
        st.warning("Initial scenario DB not found.")
    elif not expected_db:
        st.warning("Expected scenario DB not found in ground truth.")
    else:
        initial_json = json.dumps(initial_db, indent=2, sort_keys=True)
        expected_json = json.dumps(expected_db, indent=2, sort_keys=True)
        diff_lines = list(
            difflib.unified_diff(
                initial_json.splitlines(),
                expected_json.splitlines(),
                fromfile="Initial DB",
                tofile="Expected DB",
                lineterm="",
            )
        )
        if not diff_lines:
            st.success("No differences between initial and expected scenario DB.")
        else:
            _render_diff_html(diff_lines)
