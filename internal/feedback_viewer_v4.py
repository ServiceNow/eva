#!/usr/bin/env python3
"""Feedback Viewer v4 - Read-Only Feedback Browser

Streamlit app for viewing submitted review feedback alongside the same data
presentation used in data_review_tool_v4.py.

Usage:
    streamlit run scripts/feedback_viewer_v4.py
"""

import difflib
import json
import os
from pathlib import Path
from typing import Any, Optional

import streamlit as st
import yaml

from eva.models.record import EvaluationRecord
from eva.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_JSONL_PATH = "data/airline_full_2_19.jsonl"
DEFAULT_SCENARIOS_DIR = "data/airline_scenarios_2_19"
DEFAULT_AGENT_CONFIG_PATH = "configs/agents/airline_agent.yaml"

if os.environ.get("TOOLKIT"):
    DEFAULT_FEEDBACK_DIR = "/mnt/voice_agent/review_feedback_v4"
else:
    DEFAULT_FEEDBACK_DIR = "review_feedback_v4"


# ============================================================================
# Helper Functions - Data Loading
# ============================================================================


def load_dataset_and_scenarios(jsonl_path: Path, scenarios_dir: Path) -> tuple[dict[str, EvaluationRecord], list[str]]:
    """Load JSONL records and find matching scenario databases."""
    try:
        records = EvaluationRecord.load_dataset(jsonl_path)
        records_dict = {record.id: record for record in records}
        logger.info(f"Loaded {len(records)} records from {jsonl_path}")

        available_records = []
        for record in records:
            scenario_path = scenarios_dir / f"{record.id}.json"
            if scenario_path.exists():
                available_records.append(record.id)
            else:
                logger.warning(f"Scenario DB not found: {record.id}")

        logger.info(f"Found {len(available_records)} records with scenario databases")
        return records_dict, sorted(available_records)

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        st.error(f"Failed to load dataset: {e}")
        return {}, []


def load_scenario_db(scenarios_dir: Path, record_id: str) -> Optional[dict]:
    """Load initial scenario database for a record."""
    scenario_path = scenarios_dir / f"{record_id}.json"

    if not scenario_path.exists():
        logger.warning(f"Scenario DB not found: {scenario_path}")
        return None

    try:
        with open(scenario_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load scenario DB: {e}", exc_info=True)
        return None


def load_agent_config(config_path: Path) -> Optional[dict]:
    """Load agent configuration YAML."""
    if not config_path.exists():
        logger.warning(f"Agent config not found: {config_path}")
        return None

    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load agent config: {e}", exc_info=True)
        return None


def build_tool_type_map(agent_config: dict) -> dict[str, str]:
    """Build {tool_name: tool_type} map from agent config tools."""
    tool_type_map = {}
    for tool in agent_config.get("tools", []):
        name = tool.get("name") or tool.get("id", "")
        tool_type = tool.get("tool_type", "read")
        tool_type_map[name] = tool_type
    return tool_type_map


# ============================================================================
# Helper Functions - Feedback Loading
# ============================================================================


def load_feedback(feedback_dir: Path, record_id: str) -> Optional[dict]:
    """Load existing feedback if it exists."""
    feedback_path = feedback_dir / f"{record_id}.json"

    if not feedback_path.exists():
        return None

    try:
        with open(feedback_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load feedback: {e}")
        return None


def load_all_feedback(feedback_dir: Path) -> dict[str, dict]:
    """Load all feedback files, keyed by record_id."""
    all_feedback = {}
    if not feedback_dir.exists():
        return all_feedback
    for path in feedback_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            record_id = path.stem
            all_feedback[record_id] = data
        except Exception as e:
            logger.error(f"Failed to load feedback file {path}: {e}")
    return all_feedback


def has_any_issue(feedback: dict) -> bool:
    """Return True if feedback contains any 'No' answer for standard questions.

    Also returns True for 'Yes' on diff_has_nonsensical_mods.
    """
    standard_no_keys = [
        "user_goal_reflects_inspiration",
        "user_goal_makes_sense",
        "user_goal_complete_definitive",
        "scenario_db_reflects_goal",
        "write_tools_correct",
        "trace_single_valid_option",
        "user_simulator_follows_goal",
        "diff_reflects_operations",
        "overall_record_valid",
    ]
    for key in standard_no_keys:
        if feedback.get(key) == "No":
            return True
    if feedback.get("diff_has_nonsensical_mods") == "Yes":
        return True
    return False


# ============================================================================
# Helper Functions - Diff Generation
# ============================================================================


def generate_unified_diff(initial_db: dict, expected_db: dict) -> str:
    """Generate unified diff between initial and expected scenario DBs."""
    initial_json = json.dumps(initial_db, indent=2, sort_keys=True)
    expected_json = json.dumps(expected_db, indent=2, sort_keys=True)

    initial_lines = initial_json.splitlines(keepends=True)
    expected_lines = expected_json.splitlines(keepends=True)

    diff = difflib.unified_diff(
        initial_lines,
        expected_lines,
        fromfile="Initial Scenario DB",
        tofile="Expected Scenario DB",
        lineterm="",
        n=999999,
    )

    diff_text = "".join(diff)
    return diff_text if diff_text else "No changes expected between initial and expected databases."


# ============================================================================
# UI Components - Main Content
# ============================================================================


def render_header(record: EvaluationRecord):
    """Render record header with ID, category, and date."""
    st.header(f"Record: {record.id}")

    if record.category:
        st.caption(f"Category: {record.category}")

    current_date = getattr(record, "current_date_time", None) or getattr(record, "current_date", None)
    if current_date:
        st.caption(f"Scenario Date/Time: {current_date}")


def render_scenario_inspiration_section(record: EvaluationRecord):
    """Render scenario inspiration (subflow_in_depth) section."""
    with st.expander("Scenario Inspiration", expanded=True):
        st.info("This is the seed idea used to generate this scenario. It is not used in evaluation.")
        if hasattr(record, "subflow_in_depth") and record.subflow_in_depth:
            st.json(record.subflow_in_depth)
        else:
            st.warning("No scenario inspiration available")


def render_user_goal_section(record: EvaluationRecord):
    """Render user goal section."""
    with st.expander("User Goal", expanded=True):
        if isinstance(record.user_goal, dict):
            if "high_level_user_goal" in record.user_goal:
                st.info(record.user_goal["high_level_user_goal"])
            st.json(record.user_goal)
        elif isinstance(record.user_goal, str):
            try:
                user_goal_data = json.loads(record.user_goal)
                if "high_level_user_goal" in user_goal_data:
                    st.info(user_goal_data["high_level_user_goal"])
                st.json(user_goal_data)
            except json.JSONDecodeError:
                st.write(record.user_goal)
        else:
            st.write(record.user_goal)


def render_starting_scenario_db_section(initial_db: Optional[dict]):
    """Render starting scenario database section."""
    with st.expander("Starting Scenario Database", expanded=False):
        if initial_db:
            st.json(initial_db)
        else:
            st.warning("Initial scenario database not found")


def render_expected_trace_section(record: EvaluationRecord, tool_type_map: dict[str, str]):
    """Render expected trace section with write/read tool highlighting."""
    with st.expander("Expected Trace", expanded=True):
        if not (
            record.ground_truth
            and hasattr(record.ground_truth, "expected_trace")
            and record.ground_truth.expected_trace
        ):
            st.info("No expected trace available")
            return

        trace_data = record.ground_truth.expected_trace

        # Extract the list of entries — handle known structures
        if isinstance(trace_data, list):
            entries = trace_data
        elif isinstance(trace_data, dict):
            entries = trace_data.get("trace", trace_data.get("entries", trace_data.get("steps", None)))
            if entries is None:
                st.json(trace_data)
                return
        else:
            st.json(trace_data)
            return

        if not isinstance(entries, list):
            st.json(trace_data)
            return

        # Pair tool_call entries with their following tool_response
        response_map: dict[int, dict] = {}
        for i, entry in enumerate(entries):
            if isinstance(entry, dict) and entry.get("event_type") == "tool_response":
                # Attach to previous tool_call
                for j in range(i - 1, -1, -1):
                    if isinstance(entries[j], dict) and entries[j].get("event_type") == "tool_call":
                        response_map[j] = entry
                        break

        step_num = 0
        for i, entry in enumerate(entries):
            if isinstance(entry, dict) and entry.get("event_type") == "tool_response":
                continue  # Rendered alongside its tool_call
            step_num += 1
            _render_trace_entry(step_num, entry, tool_type_map, response_map.get(i))


def _render_trace_entry(
    step: int,
    entry: Any,
    tool_type_map: dict[str, str],
    paired_response: Optional[dict] = None,
):
    """Render a single trace entry with appropriate formatting."""
    if isinstance(entry, str):
        st.markdown(f"**{step}.** {entry}")
        return

    if not isinstance(entry, dict):
        st.write(f"**{step}.** {entry}")
        return

    event_type = entry.get("event_type", entry.get("type", ""))
    tool_name = entry.get("tool_name", entry.get("name", ""))

    if event_type == "tool_call":
        tool_type = tool_type_map.get(tool_name, "read")

        if tool_name == "transfer_to_agent":
            st.markdown(f"**{step}.** :red[**SYSTEM TOOL: {tool_name}**]")
        elif tool_type == "write":
            st.markdown(f"**{step}.** :orange[**WRITE TOOL: {tool_name}**]")
        else:
            st.markdown(f"**{step}.** :blue[**read tool: {tool_name}**]")

        # Show tool parameters
        params = entry.get("params", entry.get("parameters", entry.get("arguments", {})))
        if params:
            st.code(json.dumps(params, indent=2), language="json")

        # Show paired tool_response in nested expander
        if paired_response is not None:
            response = paired_response.get("response", paired_response)
            with st.expander(f"Response for {tool_name}", expanded=False):
                if isinstance(response, (dict, list)):
                    st.json(response)
                else:
                    st.write(response)

    elif event_type in ("user_utterance", "user_message", "user_input", "user"):
        content = entry.get("utterance", entry.get("content", entry.get("message", entry.get("text", ""))))
        with st.chat_message("user"):
            st.markdown(content)

    elif event_type in ("agent_utterance", "assistant_message", "assistant_output", "assistant"):
        content = entry.get("utterance", entry.get("content", entry.get("message", entry.get("text", ""))))
        with st.chat_message("assistant"):
            st.markdown(content)

    else:
        st.markdown(f"**{step}.**")
        st.json(entry)


def render_scenario_db_diff_section(record: EvaluationRecord, initial_db: Optional[dict]):
    """Render scenario database diff section."""
    with st.expander("Scenario Database Diff", expanded=False):
        st.caption("Shows diff between initial and final database after the tool calls in the trace occurred")

        if not initial_db:
            st.warning("Initial scenario database not found")
            return

        if (
            record.ground_truth
            and hasattr(record.ground_truth, "expected_scenario_db")
            and record.ground_truth.expected_scenario_db
        ):
            expected_db = record.ground_truth.expected_scenario_db
            diff_text = generate_unified_diff(initial_db, expected_db)
            st.code(diff_text, language="diff")
        else:
            st.info("No expected scenario database available")


def render_information_section(agent_config: Optional[dict], tool_type_map: dict[str, str]):
    """Render information section with agent instructions and tool configs."""
    st.subheader("Information")

    if not agent_config:
        st.warning("Agent configuration not loaded")
        return

    # Agent instructions expander
    instructions = agent_config.get("instructions", "")
    with st.expander("Agent Instructions", expanded=False):
        if instructions:
            st.text(instructions)
        else:
            st.info("No agent instructions available")

    # Single tools expander with all tools
    tools = agent_config.get("tools", [])
    with st.expander(f"Tools ({len(tools)})", expanded=False):
        for tool in tools:
            tool_name = tool.get("name") or tool.get("id", "unknown")
            tool_type = tool_type_map.get(tool_name, "read")
            description = tool.get("description", "")

            # Tool header with colored badge
            if tool_name == "transfer_to_agent":
                st.markdown(f":red[**{tool_name}**] `system`")
            elif tool_type == "write":
                st.markdown(f":orange[**{tool_name}**] `{tool_type}`")
            else:
                st.markdown(f":blue[**{tool_name}**] `{tool_type}`")

            if description:
                st.caption(description)

            # Parameters table
            req_params = tool.get("required_parameters", [])
            opt_params = tool.get("optional_parameters", [])
            if req_params or opt_params:
                rows = []
                for p in req_params:
                    rows.append(
                        {
                            "Parameter": p.get("name", ""),
                            "Type": p.get("type", ""),
                            "Required": "yes",
                            "Description": p.get("description", ""),
                        }
                    )
                for p in opt_params:
                    rows.append(
                        {
                            "Parameter": p.get("name", ""),
                            "Type": p.get("type", ""),
                            "Required": "no",
                            "Description": p.get("description", ""),
                        }
                    )
                st.dataframe(rows, use_container_width=True, hide_index=True)

            st.divider()


# ============================================================================
# UI Components - Feedback Display (Read-Only)
# ============================================================================


def _answer_badge(key: str, feedback: dict, invert: bool = False) -> str:
    """Return colored markdown badge for a feedback answer.

    Args:
        key: The feedback dict key.
        feedback: The feedback dict.
        invert: If True, 'Yes' is red and 'No' is green (for negative questions
                like diff_has_nonsensical_mods).
    """
    val = feedback.get(key)
    if val is None:
        return ":gray[—]"
    if invert:
        return ":red[Yes]" if val == "Yes" else ":green[No]"
    return ":green[Yes]" if val == "Yes" else ":red[No]"


def _show_question(label: str, badge: str, explanation: Optional[str] = None):
    """Render a single question row with badge and optional explanation."""
    st.sidebar.markdown(f"**{label}** {badge}")
    if explanation and explanation.strip():
        st.sidebar.warning(explanation)


def render_feedback_display(record_id: str, feedback: Optional[dict]):
    """Render read-only feedback panel in the sidebar."""
    st.sidebar.header("Feedback")

    if not feedback:
        st.sidebar.warning("No feedback submitted for this record.")
        return

    # Overall validity badge — shown prominently at the top
    overall = feedback.get("overall_record_valid")
    if overall == "Yes":
        overall_badge = ":green[**Valid**]"
    elif overall == "No":
        overall_badge = ":red[**Invalid**]"
    else:
        overall_badge = ":gray[**Not reviewed**]"

    st.sidebar.markdown(f"**Overall:** {overall_badge}")

    last_updated = feedback.get("last_updated", "Unknown")
    st.sidebar.caption(f"Last updated: {last_updated}")

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # Section 1: User Goal
    # ------------------------------------------------------------------
    st.sidebar.subheader("1. User Goal")

    _show_question(
        "Reflects scenario inspiration?",
        _answer_badge("user_goal_reflects_inspiration", feedback),
        feedback.get("user_goal_reflects_inspiration_explanation"),
    )
    _show_question(
        "Makes sense as a logical scenario?",
        _answer_badge("user_goal_makes_sense", feedback),
        feedback.get("user_goal_makes_sense_explanation"),
    )
    _show_question(
        "Complete and definitive?",
        _answer_badge("user_goal_complete_definitive", feedback),
        feedback.get("user_goal_complete_definitive_explanation"),
    )

    comments = feedback.get("user_goal_comments", "")
    if comments and comments.strip():
        st.sidebar.info(f"**Comments:** {comments}")

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # Section 2: Initial Scenario Database
    # ------------------------------------------------------------------
    st.sidebar.subheader("2. Initial Scenario Database")

    _show_question(
        "DB reflects goal (one correct option)?",
        _answer_badge("scenario_db_reflects_goal", feedback),
        feedback.get("scenario_db_reflects_goal_explanation"),
    )

    comments = feedback.get("scenario_db_comments", "")
    if comments and comments.strip():
        st.sidebar.info(f"**Comments:** {comments}")

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # Section 3: Expected Trace
    # ------------------------------------------------------------------
    st.sidebar.subheader("3. Expected Trace")

    _show_question(
        "All write tools correct?",
        _answer_badge("write_tools_correct", feedback),
        feedback.get("write_tools_correct_explanation"),
    )
    _show_question(
        "Only valid way to complete the task?",
        _answer_badge("trace_single_valid_option", feedback),
        feedback.get("trace_single_valid_option_explanation"),
    )
    _show_question(
        "User simulator follows goal?",
        _answer_badge("user_simulator_follows_goal", feedback),
        feedback.get("user_simulator_follows_goal_explanation"),
    )

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # Section 4: Scenario Diff
    # ------------------------------------------------------------------
    st.sidebar.subheader("4. Scenario Diff")

    _show_question(
        "Accurately reflects agent's operations?",
        _answer_badge("diff_reflects_operations", feedback),
        feedback.get("diff_reflects_operations_explanation"),
    )
    # Inverted: "Yes" means there ARE nonsensical mods (bad)
    _show_question(
        "Has nonsensical modifications?",
        _answer_badge("diff_has_nonsensical_mods", feedback, invert=True),
        feedback.get("diff_has_nonsensical_mods_explanation"),
    )

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # Section 5: Overall Validity
    # ------------------------------------------------------------------
    st.sidebar.subheader("5. Overall Validity")

    _show_question(
        "Record valid for final dataset?",
        _answer_badge("overall_record_valid", feedback),
        feedback.get("overall_invalid_explanation"),
    )


# ============================================================================
# Main App
# ============================================================================


def main():
    st.set_page_config(
        page_title="Feedback Viewer v4",
        page_icon="🔍",
        layout="wide",
    )

    st.title("Feedback Viewer v4")

    # Hardcoded paths
    jsonl_path = Path(DEFAULT_JSONL_PATH)
    scenarios_dir = Path(DEFAULT_SCENARIOS_DIR)
    feedback_dir = Path(DEFAULT_FEEDBACK_DIR)
    agent_config_path = Path(DEFAULT_AGENT_CONFIG_PATH)

    # Initialize session state
    if "record_index" not in st.session_state:
        st.session_state.record_index = 0
    if "filter_mode" not in st.session_state:
        st.session_state.filter_mode = "all"
    if "records_dict" not in st.session_state:
        st.session_state.records_dict = {}
    if "available_records" not in st.session_state:
        st.session_state.available_records = []

    # Load dataset once
    if not st.session_state.records_dict:
        with st.spinner("Loading dataset and scenarios..."):
            records_dict, available_records = load_dataset_and_scenarios(jsonl_path, scenarios_dir)
            st.session_state.records_dict = records_dict
            st.session_state.available_records = available_records

    if not st.session_state.available_records:
        st.error("No valid records found. Please check your dataset path and scenarios directory.")
        return

    records_dict = st.session_state.records_dict
    all_available_records = st.session_state.available_records

    # Load agent config
    agent_config = load_agent_config(agent_config_path)
    tool_type_map = build_tool_type_map(agent_config) if agent_config else {}

    # Load all feedback for filtering
    all_feedback = load_all_feedback(feedback_dir)

    # Restore record from query params on first load
    if "query_param_loaded" not in st.session_state:
        st.session_state.query_param_loaded = True
        query_params = st.query_params
        if "record" in query_params:
            qp_record = query_params["record"]
            if qp_record in all_available_records:
                st.session_state.record_index = all_available_records.index(qp_record)

    # ========================================================================
    # Sidebar: Navigation Section
    # ========================================================================

    st.sidebar.header("Navigation")

    # Filter selector
    filter_options = {
        "all": "All records",
        "reviewed": "Reviewed only",
        "has_issues": "Has issues (any No)",
    }

    def _on_filter_change():
        st.session_state.filter_mode = st.session_state.filter_selector
        st.session_state.record_index = 0

    st.sidebar.selectbox(
        "Filter:",
        options=list(filter_options.keys()),
        format_func=lambda k: filter_options[k],
        index=list(filter_options.keys()).index(st.session_state.filter_mode),
        key="filter_selector",
        on_change=_on_filter_change,
    )

    # Compute filtered list
    filter_mode = st.session_state.filter_mode
    if filter_mode == "reviewed":
        filtered_records = [r for r in all_available_records if r in all_feedback]
    elif filter_mode == "has_issues":
        filtered_records = [r for r in all_available_records if r in all_feedback and has_any_issue(all_feedback[r])]
    else:
        filtered_records = all_available_records

    if not filtered_records:
        st.sidebar.warning("No records match the current filter.")
        st.warning("No records match the current filter.")
        return

    # Clamp index to valid range
    if st.session_state.record_index >= len(filtered_records):
        st.session_state.record_index = len(filtered_records) - 1
    if st.session_state.record_index < 0:
        st.session_state.record_index = 0

    reviewed_count = sum(1 for r in filtered_records if r in all_feedback)

    def _on_selectbox_change():
        selected = st.session_state.record_selector
        st.session_state.record_index = filtered_records.index(selected)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Previous", use_container_width=True, disabled=(st.session_state.record_index == 0)):
            st.session_state.record_index -= 1
            st.session_state.record_selector = filtered_records[st.session_state.record_index]
            st.rerun()

    with col2:
        if st.button(
            "Next", use_container_width=True, disabled=(st.session_state.record_index == len(filtered_records) - 1)
        ):
            st.session_state.record_index += 1
            st.session_state.record_selector = filtered_records[st.session_state.record_index]
            st.rerun()

    st.sidebar.progress((st.session_state.record_index + 1) / len(filtered_records))
    st.sidebar.caption(
        f"Record {st.session_state.record_index + 1} of {len(filtered_records)} ({reviewed_count} reviewed)"
    )

    st.sidebar.selectbox(
        "Select Record:",
        filtered_records,
        index=st.session_state.record_index,
        key="record_selector",
        on_change=_on_selectbox_change,
    )

    record_id = filtered_records[st.session_state.record_index]
    eval_record = records_dict[record_id]

    # Sync query params
    st.query_params["record"] = record_id

    st.sidebar.divider()

    # ========================================================================
    # Sidebar: Feedback Display
    # ========================================================================

    feedback = all_feedback.get(record_id)
    render_feedback_display(record_id, feedback)

    # ========================================================================
    # Main Content Area
    # ========================================================================

    # 1. Header
    render_header(eval_record)

    # 2. Scenario Inspiration
    render_scenario_inspiration_section(eval_record)

    # 3. User Goal
    render_user_goal_section(eval_record)

    # 4. Starting Scenario Database
    initial_db = load_scenario_db(scenarios_dir, record_id)
    render_starting_scenario_db_section(initial_db)

    # 5. Expected Trace (with write/read tool highlighting)
    render_expected_trace_section(eval_record, tool_type_map)

    # 6. Scenario DB Diff
    render_scenario_db_diff_section(eval_record, initial_db)

    # 7. Divider
    st.divider()

    # 8. Information section
    render_information_section(agent_config, tool_type_map)


if __name__ == "__main__":
    main()
