"""Streamlit app for editing EVA's .env file with a friendly UI.

Run with:

    streamlit run apps/config_editor.py

The app reads .env.example to discover variables and their metadata
(widget type, options, ranges, tooltips, conditions) from annotation
prefixes (#i, #d, #e, #r, #g, #x, #v).  .env is read on startup to
prefill values and written on save.
"""

from __future__ import annotations

import hashlib
import html as html_module
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as st_components
from config_io import (
    AnnotatedVar,
    ParsedEnvExample,
    compute_disabled,
    load_env,
    parse_env_example,
    serialize_env,
)
from config_schema import (
    GROUP_API_CONFIGS,
    GROUP_DEPLOYMENTS,
    GROUP_MISC,
    GROUP_PERTURBATIONS,
    GROUP_RUNTIME,
    GROUPS,
    MUTEX_RADIOS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
ENV_PATH = REPO_ROOT / ".env"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def _coerce(widget: str, raw: str) -> Any:
    if not raw:
        return _empty_for(widget)
    raw = raw.strip()
    try:
        if widget == "bool":
            return raw.lower() in ("true", "1", "yes", "on")
        if widget == "int":
            return int(raw)
        if widget == "float":
            return float(raw)
        if widget == "csv_list":
            return [x.strip() for x in raw.split(",") if x.strip()]
        if widget in ("json_object", "json_deployment_list"):
            s = raw.strip()
            if s.startswith("'") and s.endswith("'"):
                s = s[1:-1]
            return json.loads(s)
    except Exception:
        return _empty_for(widget)
    return raw


def _empty_for(widget: str) -> Any:
    if widget == "bool":
        return False
    if widget in ("int", "float"):
        return None
    if widget in ("csv_list", "json_deployment_list"):
        return []
    if widget == "json_object":
        return {}
    return ""


def _detect_pipeline_mode(env: dict[str, str]) -> str:
    if env.get("EVA_MODEL__S2S"):
        return "S2S"
    if env.get("EVA_MODEL__AUDIO_LLM"):
        return "AudioLLM"
    return "LLM"


def _detect_perturbation_mode(env: dict[str, str]) -> str:
    if env.get("EVA_PERTURBATION__ACCENT"):
        return "Accent"
    if env.get("EVA_PERTURBATION__BEHAVIOR"):
        return "Behavior"
    return "None"


def _init_state() -> None:
    if "initialized" in st.session_state:
        return
    parsed = parse_env_example(ENV_EXAMPLE_PATH)
    st.session_state.parsed = parsed
    existing = load_env(ENV_PATH)
    values: dict[str, Any] = {}
    for var in parsed.vars:
        raw = existing.get(var.name)
        if raw is None and var.is_active:
            raw = var.example_value.strip().strip("'\"")
        values[var.name] = _coerce(var.widget, raw or "")
    for name, raw in existing.items():
        if name not in {v.name for v in parsed.vars}:
            values[name] = raw
    st.session_state.field_values = values
    st.session_state.loaded_keys = set(existing.keys())
    st.session_state.pipeline_mode = _detect_pipeline_mode(existing)
    st.session_state.perturbation_mode = _detect_perturbation_mode(existing)
    # Initialise all mutex radio states
    for mx in MUTEX_RADIOS:
        if mx.state_key not in st.session_state:
            st.session_state[mx.state_key] = st.session_state.get(mx.state_key, mx.default)
    st.session_state.initialized = True


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------


def _is_visible_av(var: AnnotatedVar) -> bool:
    """Return True when all #x conditions for this var are satisfied.

    Comma-separated values in a single condition are treated as OR
    (e.g. `#x pipeline_mode=LLM,AudioLLM`).
    """
    for cond_key, cond_val in var.conditions:
        actual = st.session_state.get(cond_key)
        if actual is None:
            actual = st.session_state.get("field_values", {}).get(cond_key)
        allowed = {v.strip() for v in cond_val.split(",") if v.strip()}
        if actual not in allowed:
            return False
    return True


# ---------------------------------------------------------------------------
# Widget renderers
# ---------------------------------------------------------------------------


def _render_annotated_var(var: AnnotatedVar) -> None:
    if not _is_visible_av(var):
        return
    values = st.session_state.field_values
    current = values.get(var.name)
    help_text = var.info or None

    if var.widget in ("string", "path"):
        values[var.name] = st.text_input(var.name, value=current or "", help=help_text, key=f"w_{var.name}")
    elif var.widget == "secret":
        values[var.name] = st.text_input(
            var.name, value=current or "", help=help_text, type="password", key=f"w_{var.name}"
        )
    elif var.widget == "bool":
        values[var.name] = st.checkbox(var.name, value=bool(current), help=help_text, key=f"w_{var.name}")
    elif var.widget == "int":
        rng = var.range
        v = current if isinstance(current, int) else (int(rng[0]) if rng else 0)
        values[var.name] = st.number_input(
            var.name,
            value=v,
            min_value=int(rng[0]) if rng else None,
            max_value=int(rng[1]) if rng and len(rng) > 1 else None,
            step=int(rng[2]) if rng and len(rng) > 2 else 1,
            help=help_text,
            key=f"w_{var.name}",
        )
    elif var.widget == "float":
        rng = var.range
        v = float(current) if isinstance(current, (int, float)) else (float(rng[0]) if rng else 0.0)
        values[var.name] = st.number_input(
            var.name,
            value=float(v),
            min_value=float(rng[0]) if rng else None,
            max_value=float(rng[1]) if rng and len(rng) > 1 else None,
            step=float(rng[2]) if rng and len(rng) > 2 else 0.1,
            help=help_text,
            key=f"w_{var.name}",
        )
    elif var.widget == "enum":
        options = _enum_options_for(var)
        display = ["(unset)"] + options
        idx = (options.index(current) + 1) if current in options else 0
        choice = st.selectbox(var.name, display, index=idx, help=help_text, key=f"w_{var.name}")
        values[var.name] = "" if choice == "(unset)" else choice
    elif var.widget == "multi_enum":
        choices = current if isinstance(current, list) else []
        values[var.name] = st.multiselect(var.name, var.options, default=choices, help=help_text, key=f"w_{var.name}")
    elif var.widget == "csv_list":
        as_text = ",".join(current) if isinstance(current, list) else (current or "")
        text = st.text_input(var.name, value=as_text, help=help_text, key=f"w_{var.name}")
        values[var.name] = [x.strip() for x in text.split(",") if x.strip()]
    elif var.widget == "json_object":
        _render_json_object(var.name, var.info, current or {})
    elif var.widget == "json_deployment_list":
        _render_deployment_list(var.name, var.info, current or [])


def _enum_options_for(var: AnnotatedVar) -> list[str]:
    if var.name == "EVA_MODEL__LLM":
        deployments = st.session_state.field_values.get("EVA_MODEL_LIST") or []
        return sorted({d.get("model_name", "") for d in deployments if isinstance(d, dict)} - {""})
    return var.options


def _render_json_object(name: str, info: str, current: dict) -> None:
    st.markdown(f"**{name}**" + (f" — {info}" if info else ""))

    # Both widgets are keyed by a hash of the current value so they always
    # re-initialize from field_values after any write + rerun.
    val_hash = hashlib.md5(json.dumps(current, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]

    rows = [{"key": k, "value": _scalar_to_str(v)} for k, v in current.items()] or [{"key": "", "value": ""}]
    edited = st.data_editor(
        rows,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "key": st.column_config.TextColumn("key", required=False),
            "value": st.column_config.TextColumn("value", required=False),
        },
        key=f"_de_{name}_{val_hash}",
    )
    parsed_from_table: dict[str, Any] = {
        (r.get("key") or "").strip(): _str_to_scalar(r.get("value")) for r in edited if (r.get("key") or "").strip()
    }

    if json.dumps(parsed_from_table, sort_keys=True, ensure_ascii=False) != json.dumps(
        current, sort_keys=True, ensure_ascii=False
    ):
        st.session_state.field_values[name] = parsed_from_table
        st.rerun()

    with st.expander("Raw JSON", expanded=False):
        text = st.text_area(
            "Edit as JSON",
            value=json.dumps(current, indent=2, ensure_ascii=False) if current else "",
            key=f"_rawtxt_{name}_{val_hash}",
            height=140,
        )

    if text.strip():
        try:
            parsed_kv = json.loads(text)
            if json.dumps(parsed_kv, sort_keys=True, ensure_ascii=False) != json.dumps(
                current, sort_keys=True, ensure_ascii=False
            ):
                st.session_state.field_values[name] = parsed_kv
                st.rerun()
        except json.JSONDecodeError as e:
            st.warning(f"Invalid JSON: {e}")

    st.session_state.field_values[name] = current


def _scalar_to_str(v: Any) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    return str(v)


def _str_to_scalar(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s == "":
        return ""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.startswith(("{", "[")):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s
    try:
        return int(s) if "." not in s else float(s)
    except ValueError:
        return s


def _render_deployment_list(name: str, info: str, current: list) -> None:
    st.markdown(f"**{name}**" + (f" — {info}" if info else ""))
    deployments: list[dict] = [d for d in current if isinstance(d, dict)]

    st.caption("All deployments — add / remove rows here, then select one below to edit its params.")
    summary_rows = [
        {"model_name": d.get("model_name", ""), "provider/model": (d.get("litellm_params") or {}).get("model", "")}
        for d in deployments
    ] or [{"model_name": "", "provider/model": ""}]

    edited_summary = st.data_editor(
        summary_rows,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "model_name": st.column_config.TextColumn("model_name (alias)", required=False),
            "provider/model": st.column_config.TextColumn("provider/model (litellm_params.model)", required=False),
        },
        key=f"de_summary_{name}",
    )

    old_by_name = {d.get("model_name", ""): d for d in deployments}
    merged: list[dict] = []
    for idx, row in enumerate(edited_summary):
        rname = (row.get("model_name") or "").strip()
        if not rname:
            continue
        base = dict(
            old_by_name.get(rname) or old_by_name.get(list(old_by_name)[idx] if idx < len(old_by_name) else "") or {}
        )
        base["model_name"] = rname
        lp = dict(base.get("litellm_params") or {})
        pm = (row.get("provider/model") or "").strip()
        if pm:
            lp["model"] = pm
        base["litellm_params"] = lp
        merged.append(base)
    deployments = merged

    model_names = [d.get("model_name", "") for d in deployments if d.get("model_name")]
    if not model_names:
        st.session_state.field_values[name] = deployments
        return

    sel_key = f"_depl_sel_{name}"
    prev = st.session_state.get(sel_key)
    default_idx = model_names.index(prev) if prev in model_names else 0
    selected = st.selectbox("Edit deployment", options=model_names, index=default_idx, key=sel_key)

    sel_idx = model_names.index(selected)
    depl = deployments[sel_idx]

    st.markdown("**litellm_params**")
    lp = depl.get("litellm_params") or {}
    lp_rows = [{"key": k, "value": _scalar_to_str(v)} for k, v in lp.items()] or [{"key": "", "value": ""}]
    edited_lp = st.data_editor(
        lp_rows,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "key": st.column_config.TextColumn("key", required=False),
            "value": st.column_config.TextColumn("value", required=False),
        },
        key=f"de_lp_{name}_{selected}",
    )
    new_lp: dict[str, Any] = {
        (r.get("key") or "").strip(): _str_to_scalar(r.get("value")) for r in edited_lp if (r.get("key") or "").strip()
    }
    depl["litellm_params"] = new_lp

    extra_fields = {k: v for k, v in depl.items() if k not in ("model_name", "litellm_params")}
    if extra_fields or st.checkbox("Add extra top-level fields", key=f"_extra_chk_{name}_{selected}"):
        st.markdown("**Extra top-level fields** (e.g. `model_info`)")
        extra_rows = [{"key": k, "value": _scalar_to_str(v)} for k, v in extra_fields.items()] or [
            {"key": "", "value": ""}
        ]
        edited_extra = st.data_editor(
            extra_rows,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "key": st.column_config.TextColumn("key", required=False),
                "value": st.column_config.TextColumn("value", required=False),
            },
            key=f"de_extra_{name}_{selected}",
        )
        edited_extra_keys = {(r.get("key") or "").strip() for r in edited_extra if (r.get("key") or "").strip()}
        for k in list(extra_fields):
            if k not in edited_extra_keys:
                depl.pop(k, None)
        for r in edited_extra:
            k = (r.get("key") or "").strip()
            if k:
                depl[k] = _str_to_scalar(r.get("value"))

    deployments[sel_idx] = depl
    st.session_state.field_values[name] = deployments


# ---------------------------------------------------------------------------
# Group / tab rendering
# ---------------------------------------------------------------------------


def _auto_group_for(name: str) -> str | None:
    """Auto-assign group for vars not in the parsed template."""
    if "KEY" in name or "URL" in name:
        return GROUP_API_CONFIGS
    if "_USER" in name and "DEFAULT" not in name:
        return GROUP_PERTURBATIONS
    if name.startswith("EVA_"):
        return GROUP_RUNTIME
    return None


def _render_unmapped_var(name: str) -> None:
    values = st.session_state.field_values
    v = values.get(name, "")
    if not isinstance(v, str):
        v = json.dumps(v, ensure_ascii=False) if v else ""
    widget_type = "password" if "KEY" in name else "default"
    values[name] = st.text_input(name, value=v, key=f"w_{name}", type=widget_type)


def _render_add_var_widget(context: str) -> None:
    st.divider()
    st.markdown("**Add a new variable**")
    counter_key = f"_add_var_counter_{context}"
    if counter_key not in st.session_state:
        st.session_state[counter_key] = 0
    input_key = f"_add_var_input_{context}_{st.session_state[counter_key]}"
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        new_name = st.text_input(
            "Variable name",
            key=input_key,
            label_visibility="collapsed",
            placeholder="e.g. MY_API_KEY",
        )
    with col_btn:
        if st.button("Add", key=f"_add_var_btn_{context}", width="stretch"):
            name = new_name.strip().upper()
            if not name:
                st.warning("Please enter a variable name.")
            elif name.startswith("EVA_"):
                st.error(
                    "`EVA_*` variables are managed via `apps/config_schema.py`. Add it there to get a proper widget."
                )
            elif name in st.session_state.field_values or name in {v.name for v in st.session_state.parsed.vars}:
                st.warning(f"`{name}` already exists.")
            else:
                st.session_state.field_values[name] = ""
                st.session_state[counter_key] += 1
                st.rerun()
    st.caption(
        "Variables containing **KEY** or **URL** are placed under *API Configs*. "
        "Everything else stays here. `EVA_*` variables cannot be added here — "
        "add them to `.env.example` instead."
    )


def _render_group(group: str) -> None:
    parsed: ParsedEnvExample = st.session_state.parsed

    # Render mutex radio buttons for this group
    for mx in MUTEX_RADIOS:
        if mx.group == group:
            options = mx.options
            current = st.session_state.get(mx.state_key, mx.default)
            idx = options.index(current) if current in options else 0
            st.session_state[mx.state_key] = st.radio(
                mx.label,
                options=options,
                index=idx,
                horizontal=True,
                help=mx.help,
                key=f"radio_{mx.state_key}",
            )
            st.divider()

    # Template vars for this group
    group_vars = [v for v in parsed.vars if v.group == group]

    # Auto-routed unmapped vars (from loaded .env, not in template)
    all_known = set(parsed.by_name)
    auto_names = [n for n in st.session_state.field_values if n not in all_known and _auto_group_for(n) == group]

    if group == GROUP_API_CONFIGS:
        # Sort alphabetically so KEYs and URLs cluster
        schema_map = {v.name: v for v in group_vars}
        for name in sorted(set(schema_map) | set(auto_names)):
            if name in schema_map:
                _render_annotated_var(schema_map[name])
            else:
                _render_unmapped_var(name)
        _render_add_var_widget("api")
    else:
        for var in group_vars:
            _render_annotated_var(var)
        for name in auto_names:
            _render_unmapped_var(name)

    # Cross-field validation for deployments tab
    if group == GROUP_DEPLOYMENTS:
        deployments = st.session_state.field_values.get("EVA_MODEL_LIST") or []
        chosen = st.session_state.field_values.get("EVA_MODEL__LLM")
        names = {d.get("model_name") for d in deployments if isinstance(d, dict)}
        if chosen and chosen not in names:
            st.error(
                f"EVA_MODEL__LLM = `{chosen}` does not match any deployment in EVA_MODEL_LIST. "
                "Add it above or pick a different alias."
            )


def _render_misc_tab(parsed: ParsedEnvExample) -> None:
    known = set(parsed.by_name)
    truly_misc = [n for n in st.session_state.field_values if n not in known and _auto_group_for(n) is None]
    # Also add template vars with no group assignment
    for var in parsed.vars:
        if var.group is None and var.name not in list(truly_misc):
            truly_misc.append(var.name)

    if not truly_misc:
        st.info("No unmapped variables. 🎉")
    else:
        st.warning(
            f"Found {len(truly_misc)} variable(s) not covered by the template. "
            "Add them to `.env.example` for proper widgets."
        )
        for name in truly_misc:
            _render_unmapped_var(name)
    _render_add_var_widget("misc")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _is_meaningful(name: str, value: Any) -> bool:
    if name in st.session_state.get("loaded_keys", set()):
        return True
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    if isinstance(value, bool) and value is False:
        return False
    if isinstance(value, (list, dict)) and len(value) == 0:
        return False
    if isinstance(value, (int, float)) and value == 0:
        return False
    return True


def _build_serialized() -> str:
    values = {k: v for k, v in st.session_state.field_values.items() if _is_meaningful(k, v)}
    parsed: ParsedEnvExample = st.session_state.parsed
    known = set(parsed.by_name)
    # csv_list → comma-separated string for serializer
    for var in parsed.vars:
        if var.widget == "csv_list" and isinstance(values.get(var.name), list):
            values[var.name] = ",".join(values[var.name])
    # Collect current mode state for condition evaluation
    mode_state: dict[str, str] = {}
    for mx in MUTEX_RADIOS:
        mode_state[mx.state_key] = st.session_state.get(mx.state_key, mx.default)
    mode_state.update({k: str(v) for k, v in values.items() if isinstance(v, str)})
    disabled = compute_disabled(parsed, **mode_state)
    # Split extras by auto-routing: inline into their parent section or fall through to Misc
    extras = {k: v for k, v in values.items() if k not in known}
    api_extras = {k: v for k, v in extras.items() if _auto_group_for(k) == GROUP_API_CONFIGS}
    runtime_extras = {k: v for k, v in extras.items() if _auto_group_for(k) == GROUP_RUNTIME}
    section_extras: dict[str, dict] = {}
    if api_extras:
        section_extras[GROUP_API_CONFIGS] = dict(sorted(api_extras.items()))
    if runtime_extras:
        section_extras[GROUP_RUNTIME] = dict(sorted(runtime_extras.items()))
    # vars with no auto-route → auto-collected into Misc by serialize_env
    serializer_values = {k: v for k, v in values.items() if k in known or _auto_group_for(k) is None}
    return serialize_env(serializer_values, parsed, disabled=disabled, section_extras=section_extras or None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="EVA Config Editor", layout="wide", page_icon="⚙️")
    _init_state()

    st.markdown(
        """
        <style>
        div[role="tablist"] {
            overflow-x: auto !important;
            scrollbar-width: thin;
            scrollbar-color: #888 transparent;
            padding-bottom: 2px;
        }
        div[role="tablist"]::-webkit-scrollbar { height: 4px; }
        div[role="tablist"]::-webkit-scrollbar-thumb { background: #888; border-radius: 2px; }
        div[role="tablist"]::-webkit-scrollbar-track { background: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("EVA Config Editor")
    if not ENV_PATH.exists():
        st.info(
            f"No `.env` file found at `{ENV_PATH.relative_to(REPO_ROOT)}`. "
            "Fill in your values below and click **Save to .env** to create it.",
            icon="ℹ️",
        )
    st.caption(
        f"Reading variable set from `{ENV_EXAMPLE_PATH.relative_to(REPO_ROOT)}`. "
        + (
            f"Loaded existing values from `{ENV_PATH.relative_to(REPO_ROOT)}`."
            if ENV_PATH.exists()
            else "Defaults seeded from `.env.example`."
        )
    )

    left, right = st.columns([2, 1], gap="large")

    with left:
        tabs = st.tabs(GROUPS + [GROUP_MISC])
        for tab, group in zip(tabs[:-1], GROUPS):
            with tab:
                _render_group(group)
        with tabs[-1]:
            _render_misc_tab(st.session_state.parsed)

    with right:
        st.subheader("Preview & Save")
        text = _build_serialized()
        st.download_button(
            "⬇️ Download .env",
            data=text,
            file_name=".env",
            mime="text/plain",
            width="stretch",
        )
        data_attr = html_module.escape(json.dumps(text, ensure_ascii=False), quote=True)
        st_components.html(
            f"""
            <button data-content="{data_attr}"
                onclick="navigator.clipboard.writeText(JSON.parse(this.dataset.content)).then(()=>{{
                    this.textContent='✅ Copied!';
                    setTimeout(()=>this.textContent='📋 Copy to clipboard',1500);
                }})"
                style="width:100%;padding:0.4rem 0.8rem;font-size:0.875rem;
                    border:1px solid #d1d5db;border-radius:0.375rem;background:#fff;
                    cursor:pointer;font-family:inherit;">
              📋 Copy to clipboard
            </button>
            """,
            height=42,
        )
        if st.button("💾 Save to .env", width="stretch", type="primary"):
            ENV_PATH.write_text(text)
            st.success(f"Wrote {ENV_PATH}")
        if st.button("👁️ View preview", width="stretch"):
            _show_preview(text)


@st.dialog("Preview .env", width="large")
def _show_preview(text: str) -> None:
    st.code(text, language="ini")


if __name__ == "__main__":
    main()
