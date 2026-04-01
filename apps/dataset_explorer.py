#!/usr/bin/env python3
"""Streamlit app to explore the EVA airline dataset and scenarios.

Usage:
    streamlit run apps/dataset_explorer.py
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "airline_dataset.jsonl"
SCENARIOS_DIR = DATA_DIR / "airline_scenarios"
AGENT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "agents" / "airline_agent.yaml"

SCENARIO_GROUPS = {
    "1": "Voluntary Changes",
    "2": "Irregular Operations (IRROPS)",
    "3": "Missed Flights",
    "4": "Same-Day Changes & Standby",
    "5": "Cancellations & Refunds",
    "6": "Sold-Out & Alternatives",
    "7": "Edge Cases & Policy Disputes",
}

TOOL_TYPE_COLORS = {
    "read": "#06b6d4",  # cyan
    "write": "#a855f7",  # purple
    "system": "#f59e0b",  # amber
}


def _get_tool_type(tool: dict) -> str:
    """Get tool type from tool config, falling back to name-based heuristic."""
    if "tool_type" in tool:
        return tool["tool_type"]
    name = tool.get("name", tool.get("id", ""))
    if name.startswith("get_") or name.startswith("search_"):
        return "read"
    if name == "transfer_to_agent":
        return "system"
    return "write"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f]


@st.cache_data
def load_agent_config() -> dict:
    with open(AGENT_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_scenario_db(record_id: str) -> dict | None:
    path = SCENARIOS_DIR / f"{record_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_group(record_id: str) -> str:
    prefix = record_id.split(".")[0]
    return SCENARIO_GROUPS.get(prefix, f"Group {prefix}")


def get_routes(record: dict) -> list[tuple[str, str]]:
    edb = record["ground_truth"]["expected_scenario_db"]
    return [(journey.get("origin", ""), journey.get("destination", "")) for journey in edb.get("journeys", {}).values()]


def get_airports(record: dict) -> set[str]:
    airports = set()
    for o, d in get_routes(record):
        airports.add(o)
        airports.add(d)
    return airports


def build_overview_df(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        edb = r["ground_truth"]["expected_scenario_db"]
        journeys = edb.get("journeys", {})
        reservations = edb.get("reservations", {})
        disruptions = edb.get("disruptions", {})

        # Get primary route from first reservation's first booking
        origin = dest = ""
        fare_class = ""
        for res in reservations.values():
            for bk in res.get("bookings", []):
                jid = bk.get("journey_id", "")
                if jid in journeys:
                    origin = journeys[jid].get("origin", "")
                    dest = journeys[jid].get("destination", "")
                fare_class = bk.get("fare_class", "")
                break
            break

        has_disruption = len(disruptions) > 0
        n_journeys = len(journeys)
        n_passengers = sum(len(res.get("passengers", [])) for res in reservations.values())

        rows.append(
            {
                "ID": r["id"],
                "Group": get_group(r["id"]),
                "Origin": origin,
                "Destination": dest,
                "Route": f"{origin} → {dest}" if origin else "",
                "Fare Class": fare_class.replace("_", " ").title(),
                "Passengers": n_passengers,
                "Journeys": n_journeys,
                "Disruption": has_disruption,
                "Persona": r["user_config"]["user_persona_id"],
                "Name": r["user_config"]["name"],
                "Goal": r["user_goal"]["high_level_user_goal"][:120],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------


def render_tool_badge(tool_type: str) -> str:
    color = TOOL_TYPE_COLORS.get(tool_type, "#6b7280")
    return (
        f'<span style="background-color:{color};color:white;padding:2px 8px;'
        f"border-radius:12px;font-size:0.75rem;font-weight:600;"
        f'text-transform:uppercase;">{tool_type}</span>'
    )


def render_tools_panel(agent_config: dict):
    # Config may be a single agent dict or have an "agents" list
    if "agents" in agent_config:
        agent = agent_config["agents"][0]
    elif "tools" in agent_config:
        agent = agent_config
    else:
        st.warning("No agent found in config.")
        return
    tools = agent.get("tools", [])

    st.markdown("### Agent Tools")
    st.markdown(f"**{agent.get('name', 'Agent')}** — {len(tools)} tools available")

    # Group by type
    by_type: dict[str, list] = defaultdict(list)
    for t in tools:
        ttype = _get_tool_type(t)
        by_type[ttype].append(t)

    for ttype in ["read", "write", "system"]:
        if ttype not in by_type:
            continue
        type_tools = by_type[ttype]
        st.markdown(
            f"{render_tool_badge(ttype)} **{len(type_tools)} tools**",
            unsafe_allow_html=True,
        )
        for t in type_tools:
            tname = t.get("name", t.get("id", ""))
            with st.expander(f"`{tname}`"):
                st.markdown(t.get("description", "*No description*"))
                req = t.get("required_parameters", [])
                opt = t.get("optional_parameters", [])
                if req:
                    st.markdown("**Required parameters:**")
                    for p in req:
                        pname = p if isinstance(p, str) else p.get("name", "")
                        pdesc = "" if isinstance(p, str) else p.get("description", "")
                        ptype = "" if isinstance(p, str) else p.get("type", "")
                        extra = f" *({ptype})*" if ptype else ""
                        desc_text = f" — {pdesc}" if pdesc else ""
                        st.markdown(f"- `{pname}`{extra}{desc_text}")
                if opt:
                    st.markdown("**Optional parameters:**")
                    for p in opt:
                        pname = p if isinstance(p, str) else p.get("name", "")
                        pdesc = "" if isinstance(p, str) else p.get("description", "")
                        ptype = "" if isinstance(p, str) else p.get("type", "")
                        extra = f" *({ptype})*" if ptype else ""
                        desc_text = f" — {pdesc}" if pdesc else ""
                        st.markdown(f"- `{pname}`{extra}{desc_text}")


def render_decision_tree(dt: dict):
    must = dt.get("must_have_criteria", [])
    nice = dt.get("nice_to_have_criteria", [])
    negotiation = dt.get("negotiation_behavior", [])
    resolution = dt.get("resolution_condition", "")
    failure = dt.get("failure_condition", "")
    escalation = dt.get("escalation_behavior", "")
    edges = dt.get("edge_cases", [])

    if must:
        st.markdown("**Must-Have Criteria**")
        for item in must:
            st.markdown(f"- :white_check_mark: {item}")

    if nice:
        st.markdown("**Nice-to-Have**")
        for item in nice:
            st.markdown(f"- :star: {item}")

    col1, col2 = st.columns(2)
    with col1:
        if resolution:
            st.success(f"**Resolution:** {resolution}")
    with col2:
        if failure:
            st.error(f"**Failure:** {failure}")

    if negotiation:
        with st.expander("Negotiation Behavior"):
            for item in negotiation:
                st.markdown(f"- {item}")

    if escalation:
        with st.expander("Escalation Policy"):
            st.markdown(escalation)

    if edges:
        with st.expander("Edge Cases"):
            for item in edges:
                st.markdown(f"- :warning: {item}")


def render_scenario_db(record_id: str, scenario_db: dict):
    st.markdown("### Scenario Database")

    reservations = scenario_db.get("reservations", {})
    journeys = scenario_db.get("journeys", {})
    disruptions = scenario_db.get("disruptions", {})
    travel_credits = scenario_db.get("travel_credits", {})
    vouchers = scenario_db.get("meal_vouchers", {})
    refunds = scenario_db.get("refunds", {})

    # Reservations
    if reservations:
        st.markdown(f"#### Reservations ({len(reservations)})")
        for conf, res in reservations.items():
            status_color = {
                "confirmed": "green",
                "cancelled": "red",
                "changed": "orange",
            }.get(res.get("status", ""), "gray")
            status = res.get("status", "unknown")

            with st.expander(f"Booking {conf} — {status.upper()}"):
                st.markdown(
                    f"**Status:** :{status_color}[{status}] &nbsp; "
                    f"**Fare type:** {res.get('fare_type', 'N/A')} &nbsp; "
                    f"**Booked:** {res.get('booking_date', 'N/A')}"
                )

                # Passengers
                pax = res.get("passengers", [])
                if pax:
                    pax_df = pd.DataFrame(pax)
                    display_cols = [
                        c
                        for c in [
                            "first_name",
                            "last_name",
                            "elite_status",
                            "seat_preference",
                            "meal_preference",
                        ]
                        if c in pax_df.columns
                    ]
                    st.dataframe(pax_df[display_cols], hide_index=True)

                # Bookings / segments
                for bk in res.get("bookings", []):
                    jid = bk.get("journey_id", "")
                    j = journeys.get(jid, {})
                    route = f"{j.get('origin', '?')} → {j.get('destination', '?')}"
                    st.markdown(
                        f"**{route}** &nbsp; | &nbsp; "
                        f"Fare: `{bk.get('fare_class', '')}` &nbsp; "
                        f"Paid: **${bk.get('fare_paid', 0):.0f}** &nbsp; "
                        f"Status: {bk.get('status', '')}"
                    )
                    segs = bk.get("segments", [])
                    if segs:
                        seg_df = pd.DataFrame(segs)
                        show = [
                            c
                            for c in [
                                "flight_number",
                                "date",
                                "seat",
                                "bags_checked",
                                "fare_paid",
                            ]
                            if c in seg_df.columns
                        ]
                        st.dataframe(seg_df[show], hide_index=True)

                # Ancillaries
                anc = res.get("ancillaries", {})
                if anc and any(v for v in anc.values()):
                    st.markdown(
                        f"**Ancillaries:** bags fee ${anc.get('bags_fee', 0):.0f}, "
                        f"seat selection ${anc.get('seat_selection_fee', 0):.0f}"
                    )

    # Journeys
    if journeys:
        st.markdown(f"#### Flights ({len(journeys)})")
        for j in journeys.values():
            status = j.get("status", "scheduled")
            icon = {"scheduled": ":airplane:", "cancelled": ":x:", "delayed": ":warning:"}.get(status, ":airplane:")
            route = f"{j.get('origin', '')} → {j.get('destination', '')}"
            with st.expander(f"{icon} {route} — {j.get('date', '')} ({status})"):
                st.markdown(
                    f"**Duration:** {j.get('total_duration_minutes', 0)} min &nbsp; "
                    f"**Stops:** {j.get('num_stops', 0)} &nbsp; "
                    f"**Bookable:** {'Yes' if j.get('bookable') else 'No'}"
                )

                # Fares
                fares = j.get("fares", {})
                if fares:
                    fare_items = {k.replace("_", " ").title(): f"${v:.0f}" if v else "—" for k, v in fares.items()}
                    st.markdown(" &nbsp;|&nbsp; ".join(f"**{k}:** {v}" for k, v in fare_items.items()))

                # Segments
                for seg in j.get("segments", []):
                    st.markdown("---")
                    seg_route = f"{seg.get('origin', '')} → {seg.get('destination', '')}"
                    st.markdown(
                        f"**{seg.get('flight_number', '')}** {seg_route} &nbsp; "
                        f"Dep: {seg.get('scheduled_departure', '')} → "
                        f"Arr: {seg.get('scheduled_arrival', '')} &nbsp; "
                        f"({seg.get('duration_minutes', 0)} min) &nbsp; "
                        f"Aircraft: {seg.get('aircraft_type', '')}"
                    )
                    seg_status = seg.get("status", "scheduled")
                    if seg_status != "scheduled":
                        reason = seg.get("delay_reason") or seg.get("cancellation_reason") or ""
                        delay = seg.get("delay_minutes")
                        extra = f" ({delay} min)" if delay else ""
                        st.warning(f"**{seg_status.upper()}**{extra} — {reason}")

                    # Seat availability
                    avail = seg.get("available_seats", {})
                    if avail:
                        st.markdown(
                            "**Available seats:** "
                            + " &nbsp;|&nbsp; ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in avail.items())
                        )

    # Disruptions
    if disruptions:
        st.markdown(f"#### Disruptions ({len(disruptions)})")
        for did, d in disruptions.items():
            st.error(f"**{did}** — {json.dumps(d, indent=2)}")

    # Credits, vouchers, refunds
    extras = [
        ("Travel Credits", travel_credits),
        ("Meal Vouchers", vouchers),
        ("Refunds", refunds),
    ]
    for label, data in extras:
        if data:
            st.markdown(f"#### {label} ({len(data)})")
            st.json(data)


def render_record_detail(record: dict):
    rid = record["id"]
    group = get_group(rid)
    user_cfg = record["user_config"]
    user_goal = record["user_goal"]
    dt = user_goal.get("decision_tree", {})
    info = user_goal.get("information_required", {})
    subflow = record.get("subflow_in_depth", {})

    # Header
    st.markdown(f"## Scenario {rid}")
    st.markdown(f"**{group}** &nbsp; | &nbsp; {record.get('current_date_time', '')}")

    # User persona card
    col1, col2 = st.columns([1, 3])
    with col1:
        st.html(
            f"<div style='text-align:center;padding:1rem;'>"
            f"<div style='width:64px;height:64px;border-radius:50%;background:#6366f1;"
            f"display:inline-flex;align-items:center;justify-content:center;"
            f"color:white;font-size:1.5rem;font-weight:bold;'>"
            f"{user_cfg['name'][0]}</div>"
            f"<div style='margin-top:0.5rem;font-weight:600;'>{user_cfg['name']}</div>"
            f"<div style='color:#888;font-size:0.85rem;'>Persona {user_cfg['user_persona_id']}</div>"
            f"</div>"
        )
    with col2:
        st.markdown("**User Goal**")
        st.info(user_goal["high_level_user_goal"])
        st.html(f"<div style='color:#888;font-size:0.9rem;padding:0.5rem 0;'><em>{user_cfg['user_persona']}</em></div>")

    # Starting utterance
    st.markdown("**Starting Utterance**")
    st.markdown(f'> :speech_balloon: *"{user_goal.get("starting_utterance", "")}"*')

    # Decision tree
    st.markdown("### Decision Tree")
    render_decision_tree(dt)

    # Information required
    if info:
        with st.expander("Information Required"):
            for k, v in info.items():
                display_v = v if isinstance(v, str) else json.dumps(v, indent=2)
                st.markdown(f"- **{k}:** {display_v}")

    # Expected flow
    if record.get("expected_flow"):
        with st.expander("Expected Flow"):
            st.markdown(record["expected_flow"])

    # Subflow details
    if subflow:
        with st.expander("Scenario Context & User Priorities"):
            if subflow.get("scenario_context"):
                st.markdown(subflow["scenario_context"])
            if subflow.get("user_priorities"):
                st.markdown("---")
                st.markdown(subflow["user_priorities"])

    # Scenario database
    scenario_db = get_scenario_db(rid)
    if scenario_db:
        render_scenario_db(rid, scenario_db)
    else:
        # Fall back to ground truth
        edb = record["ground_truth"]["expected_scenario_db"]
        render_scenario_db(rid, edb)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview(records: list[dict]):
    st.markdown("## Dataset Overview")

    df = build_overview_df(records)

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenarios", len(records))
    c2.metric("Groups", len(df["Group"].unique()))
    c3.metric("Unique Routes", len(df[["Origin", "Destination"]].drop_duplicates()))
    c4.metric("Airports", len(set(df["Origin"].tolist() + df["Destination"].tolist()) - {""}))

    st.markdown("---")

    # Distribution charts
    col1, col2 = st.columns(2)

    with col1:
        group_counts = df["Group"].value_counts().reset_index()
        group_counts.columns = ["Group", "Count"]
        fig = px.bar(
            group_counts,
            x="Group",
            y="Count",
            color="Group",
            title="Scenarios by Category",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-30, height=350)
        st.plotly_chart(fig)

    with col2:
        fare_counts = df["Fare Class"].value_counts().reset_index()
        fare_counts.columns = ["Fare Class", "Count"]
        fig = px.pie(
            fare_counts,
            names="Fare Class",
            values="Count",
            title="Fare Class Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig)

    col3, col4 = st.columns(2)
    with col3:
        disruption_counts = df["Disruption"].value_counts().reset_index()
        disruption_counts.columns = ["Has Disruption", "Count"]
        disruption_counts["Has Disruption"] = disruption_counts["Has Disruption"].map(
            {True: "With Disruption", False: "No Disruption"}
        )
        fig = px.pie(
            disruption_counts,
            names="Has Disruption",
            values="Count",
            title="Disruption Scenarios",
            color_discrete_sequence=["#f87171", "#6ee7b7"],
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig)

    with col4:
        # Top routes
        route_counts = df[df["Route"] != ""]["Route"].value_counts().head(10).reset_index()
        route_counts.columns = ["Route", "Count"]
        fig = px.bar(
            route_counts,
            y="Route",
            x="Count",
            orientation="h",
            title="Top 10 Routes",
            color_discrete_sequence=["#818cf8"],
        )
        fig.update_layout(height=350, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig)

    # Data table
    st.markdown("### All Scenarios")
    st.dataframe(
        df[["ID", "Group", "Route", "Fare Class", "Passengers", "Disruption", "Name", "Goal"]],
        hide_index=True,
        height=400,
    )


def page_explore(records: list[dict]):
    st.markdown("## Explore Scenarios")

    df = build_overview_df(records)

    # Filters sidebar
    with st.sidebar:
        st.markdown("### Filters")

        # Group filter
        groups = sorted(df["Group"].unique())
        selected_groups = st.multiselect("Category", groups, default=groups)

        # Airport filter
        all_airports = sorted(set(df["Origin"].tolist() + df["Destination"].tolist()) - {""})
        selected_airports = st.multiselect("Airports (origin or destination)", all_airports)

        # Fare class filter
        fares = sorted(df["Fare Class"].unique())
        selected_fares = st.multiselect("Fare Class", fares, default=fares)

        # Disruption filter
        disruption_filter = st.radio("Disruptions", ["All", "With Disruption", "No Disruption"])

        # Persona filter
        persona_filter = st.radio("Persona", ["All", "Persona 1", "Persona 2"])

    # Apply filters
    mask = df["Group"].isin(selected_groups) & df["Fare Class"].isin(selected_fares)
    if selected_airports:
        mask &= df["Origin"].isin(selected_airports) | df["Destination"].isin(selected_airports)
    if disruption_filter == "With Disruption":
        mask &= df["Disruption"]
    elif disruption_filter == "No Disruption":
        mask &= ~df["Disruption"]
    if persona_filter == "Persona 1":
        mask &= df["Persona"] == 1
    elif persona_filter == "Persona 2":
        mask &= df["Persona"] == 2

    filtered = df[mask]
    st.markdown(f"**{len(filtered)}** scenarios match filters")

    # Record selector
    if filtered.empty:
        st.warning("No scenarios match the current filters.")
        return

    record_ids = filtered["ID"].tolist()
    selected_id = st.selectbox(
        "Select scenario",
        record_ids,
        format_func=lambda x: f"{x} — {filtered[filtered['ID'] == x]['Goal'].values[0]}",
    )

    # Find and render the selected record
    record = next(r for r in records if r["id"] == selected_id)
    st.markdown("---")
    render_record_detail(record)


def page_tools(agent_config: dict):
    st.markdown("## Agent Configuration & Tools")
    if "agents" in agent_config:
        agent = agent_config["agents"][0]
    elif "tools" in agent_config:
        agent = agent_config
    else:
        st.warning("No agent found.")
        return

    # Agent info
    st.markdown(f"### {agent.get('name', 'Agent')}")
    st.markdown(agent.get("description", ""))

    with st.expander("Agent Role"):
        st.markdown(agent.get("role", ""))

    with st.expander("Agent Instructions"):
        st.markdown(agent.get("instructions", ""))

    st.markdown("---")
    render_tools_panel(agent_config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="EVA Dataset Explorer",
        page_icon=":airplane:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.html(
        "<h1 style='margin-bottom:0;'>EVA Dataset Explorer</h1>"
        "<p style='color:#888;margin-top:0;'>Airline benchmark scenarios & tools</p>"
    )

    records = load_dataset()
    agent_config = load_agent_config()

    tab1, tab2, tab3 = st.tabs(["Overview", "Explore Scenarios", "Agent & Tools"])

    with tab1:
        page_overview(records)
    with tab2:
        page_explore(records)
    with tab3:
        page_tools(agent_config)


if __name__ == "__main__":
    main()
