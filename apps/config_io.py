"""Parsing and serialization helpers for the EVA config editor.

Annotation prefix scheme for .env.example:

  # <text>     True comment — ignored by editor, preserved verbatim.
  #i <text>    Info/tooltip text for the following variable.
  #d <type>    Widget datatype: secret|bool|int|float|string|path|enum|
               multi_enum|csv_list|json_object|json_deployment_list
  #e <opts>    Comma-separated enum options for enum/multi_enum.
  #r <range>   Numeric range: min,max  or  min,max,step
  #g <group>   Override tab/group assignment for this variable.
  #x <cond>    Visibility condition VAR=value (AND semantics; multiple lines ok).
  #v <var=val> Inactive variable definition (off by default, fully configurable).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class AnnotatedVar:
    name: str
    is_active: bool  # False = declared with #v
    example_value: str  # raw default from file
    widget: str  # from #d or inferred
    info: str  # from #i lines (joined)
    options: list[str]  # from #e
    range: tuple[float, ...] | None  # (min, max[, step]) from #r
    group: str | None  # from #g or section header
    conditions: list[tuple[str, str]]  # from #x lines (AND semantics)
    line_start: int
    line_end: int


@dataclass
class ParsedEnvExample:
    lines: list[str]
    vars: list[AnnotatedVar]
    by_name: dict[str, AnnotatedVar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.by_name:
            self.by_name = {v.name: v for v in self.vars}

    # ── back-compat shim so old tests still compile ───────────────────────
    @property
    def specs(self) -> list[AnnotatedVar]:
        return self.vars


def _is_section_rule(line: str) -> bool:
    s = line.strip()
    return bool(re.match(r"^\s*#\s*={3,}\s*$", s))


def _consume_quoted_continuation(lines: list[str], start_idx: int, value_head: str) -> int:
    """If value_head opens an unterminated single/double-quoted string, scan forward."""
    stripped = value_head.strip()
    if not stripped:
        return start_idx
    quote = stripped[0]
    if quote not in ("'", '"'):
        return start_idx
    rest = stripped[1:]
    if quote in rest:
        return start_idx
    for j in range(start_idx + 1, len(lines)):
        if quote in lines[j]:
            return j
    return len(lines) - 1


def _infer_widget(name: str, value: str) -> str:
    """Best-effort widget type from variable name and example value."""
    n = name.upper()
    v = value.strip().lower()
    if any(x in n for x in ("KEY", "SECRET", "TOKEN", "PASSWORD")):
        return "secret"
    if "CREDENTIALS" in n or n.endswith("_PATH") or n.endswith("_DIR"):
        return "path"
    if v in ("true", "false"):
        return "bool"
    raw = v.strip("'\"")
    if raw.startswith("["):
        return "json_deployment_list" if "model_name" in raw else "json_object"
    if raw.startswith("{"):
        return "json_object"
    try:
        int(raw)
        return "int"
    except ValueError:
        pass
    try:
        float(raw)
        return "float"
    except ValueError:
        pass
    return "string"


def parse_env_example(path: str | Path) -> ParsedEnvExample:
    """Parse a .env.example file that uses the annotation prefix scheme."""
    text = Path(path).read_text()
    raw_lines = text.splitlines(keepends=False)

    vars_list: list[AnnotatedVar] = []
    seen: set[str] = set()
    current_section: str | None = None

    ann_info: list[str] = []
    ann_widget: str | None = None
    ann_options: list[str] = []
    ann_range: tuple[float, ...] | None = None
    ann_group: str | None = None
    ann_conditions: list[tuple[str, str]] = []

    def reset_ann() -> None:
        nonlocal ann_info, ann_widget, ann_options, ann_range, ann_group, ann_conditions
        ann_info = []
        ann_widget = None
        ann_options = []
        ann_range = None
        ann_group = None
        ann_conditions = []

    def emit_var(name: str, is_active: bool, value_head: str, line_start: int) -> int:
        end_idx = _consume_quoted_continuation(raw_lines, line_start, value_head)
        raw_value = (
            "\n".join([value_head, *raw_lines[line_start + 1 : end_idx + 1]]) if end_idx > line_start else value_head
        )
        widget = ann_widget or _infer_widget(name, raw_value)
        vars_list.append(
            AnnotatedVar(
                name=name,
                is_active=is_active,
                example_value=raw_value,
                widget=widget,
                info=" ".join(ann_info),
                options=list(ann_options),
                range=ann_range,
                group=ann_group or current_section,
                conditions=list(ann_conditions),
                line_start=line_start,
                line_end=end_idx,
            )
        )
        seen.add(name)
        reset_ann()
        return end_idx

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        stripped = line.strip()

        # Section header block (# ===...=== / # Title / # ===...===)
        if _is_section_rule(line):
            if i + 1 < len(raw_lines):
                inner = raw_lines[i + 1].lstrip("#").strip()
                if inner and not _is_section_rule(raw_lines[i + 1]):
                    current_section = inner
            reset_ann()
            j = i + 1
            while j < len(raw_lines) and not _is_section_rule(raw_lines[j]):
                j += 1
            i = j + 1 if j < len(raw_lines) else j
            continue

        # Annotation lines — accumulate until next variable or reset
        if stripped.startswith("#i "):
            ann_info.append(stripped[3:].strip())
            i += 1
            continue
        if stripped.startswith("#d "):
            ann_widget = stripped[3:].strip()
            i += 1
            continue
        if stripped.startswith("#e "):
            ann_options = [o.strip() for o in stripped[3:].split(",") if o.strip()]
            i += 1
            continue
        if stripped.startswith("#r "):
            parts = [p.strip() for p in stripped[3:].split(",")]
            try:
                ann_range = tuple(float(p) for p in parts[:3])  # type: ignore[assignment]
            except ValueError:
                pass
            i += 1
            continue
        if stripped.startswith("#g "):
            ann_group = stripped[3:].strip()
            i += 1
            continue
        if stripped.startswith("#x "):
            cond = stripped[3:].strip()
            if "=" in cond:
                k, _, v = cond.partition("=")
                ann_conditions.append((k.strip(), v.strip()))
            i += 1
            continue

        # Inactive variable: #v NAME=value
        if stripped.startswith("#v "):
            rest = stripped[3:].strip()
            if "=" in rest:
                name, _, value_head = rest.partition("=")
                name = name.strip()
                if _NAME_RE.match(name) and name not in seen:
                    end_idx = emit_var(name, False, value_head, i)
                    i = end_idx + 1
                    continue
            reset_ann()
            i += 1
            continue

        # Active variable: NAME=value  (no leading #)
        if not stripped.startswith("#") and "=" in stripped:
            name, _, value_head = stripped.partition("=")
            name = name.strip()
            if _NAME_RE.match(name) and name not in seen:
                end_idx = emit_var(name, True, value_head, i)
                i = end_idx + 1
                continue

        # True comment or blank — reset annotation accumulator
        reset_ann()
        i += 1

    return ParsedEnvExample(lines=raw_lines, vars=vars_list)


@dataclass
class LoadedEnv:
    """Result of parsing a .env file, split by active/inactive status."""

    active: dict[str, str]  # NAME=value lines
    inactive: dict[str, str]  # #v NAME=value lines (saved-but-disabled)

    @property
    def all_values(self) -> dict[str, str]:
        """Merged dict; active values take priority over inactive for the same name."""
        return {**self.inactive, **self.active}

    @property
    def active_names(self) -> set[str]:
        return set(self.active)

    @property
    def inactive_names(self) -> set[str]:
        return set(self.inactive) - set(self.active)


def load_env(path: str | Path) -> LoadedEnv:
    """Parse a .env file into active and inactive value dicts.

    Active lines  (NAME=value)    → LoadedEnv.active
    Inactive lines (#v NAME=value) → LoadedEnv.inactive
    Values have surrounding quotes stripped.
    """
    p = Path(path)
    if not p.exists():
        return LoadedEnv(active={}, inactive={})
    active: dict[str, str] = {}
    inactive: dict[str, str] = {}
    i = 0
    lines = p.read_text().splitlines(keepends=False)
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Active variable
        if not stripped.startswith("#") and "=" in stripped:
            name, _, value_head = stripped.partition("=")
            name = name.strip()
            if _NAME_RE.match(name):
                end_idx = _consume_quoted_continuation(lines, i, value_head)
                raw = "\n".join([value_head, *lines[i + 1 : end_idx + 1]]) if end_idx > i else value_head
                active[name] = _unquote(raw.strip())
                i = end_idx + 1
                continue

        # Inactive variable (#v NAME=value) — only load if it carries a real value
        if stripped.startswith("#v ") and "=" in stripped:
            rest = stripped[3:].strip()
            name, _, value_head = rest.partition("=")
            name = name.strip()
            if _NAME_RE.match(name) and value_head.strip():
                end_idx = _consume_quoted_continuation(lines, i, value_head)
                raw = "\n".join([value_head, *lines[i + 1 : end_idx + 1]]) if end_idx > i else value_head
                inactive.setdefault(name, _unquote(raw.strip()))
                i = end_idx + 1
                continue

        i += 1

    return LoadedEnv(active=active, inactive=inactive)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        return f"'{json.dumps(value, ensure_ascii=False)}'"
    s = str(value)
    if not s:
        return ""
    if any(c in s for c in (" ", "\t", "#", "'", '"', "$", "\n")):
        if "'" not in s:
            return f"'{s}'"
        return json.dumps(s, ensure_ascii=False)
    return s


def _has_value(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and v == "":
        return False
    if isinstance(v, (list, dict)) and len(v) == 0:
        return False
    return True


def serialize_env(
    values: dict[str, Any],
    parsed: ParsedEnvExample,
    disabled: set[str] | None = None,
    section_extras: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Produce a .env text using parsed as the structural template.

    - Variables in values with a user-set entry → emitted as NAME=value (active).
    - Variables in disabled with a value → emitted as #v NAME=value (inactive, value preserved).
    - Everything else → original line(s) from the template verbatim.
    - section_extras: {section_title: {name: value}} injected inline at the end of each
      named section (just before the next section header starts).
    - Any values not in the template and not in section_extras are auto-appended as Misc.
    """
    disabled = disabled or set()
    section_extras = section_extras or {}
    out: list[str] = []
    handled: set[str] = set()
    var_by_start = {v.line_start: v for v in parsed.vars}
    current_section: str | None = None

    def _flush_extras(section: str | None) -> None:
        if not section or section not in section_extras:
            return
        for name, val in section_extras[section].items():
            if _has_value(val):
                out.append(f"{name}={_format_value(val)}")

    i = 0
    while i < len(parsed.lines):
        line = parsed.lines[i]

        # Detect the opening rule of a new section (rule whose next line is the title)
        if _is_section_rule(line):
            next_line = parsed.lines[i + 1] if i + 1 < len(parsed.lines) else ""
            next_content = next_line.lstrip("#").strip()
            if next_content and not _is_section_rule(next_line):
                # Flush extras for the section we're leaving before writing the new header
                _flush_extras(current_section)
                current_section = next_content

        if i in var_by_start:
            var = var_by_start[i]
            user_value = values.get(var.name)
            if var.name in disabled:
                if _has_value(user_value):
                    out.append(f"#v {var.name}={_format_value(user_value)}")
                else:
                    out.append(f"#v {var.name}={var.example_value.strip()}")
            elif _has_value(user_value):
                out.append(f"{var.name}={_format_value(user_value)}")
            else:
                out.extend(parsed.lines[var.line_start : var.line_end + 1])
            handled.add(var.name)
            i = var.line_end + 1
            continue

        out.append(line)
        i += 1

    # Flush extras for the final section
    _flush_extras(current_section)

    # Auto-collect any values not in the template into a Misc section
    extras = [name for name in values if name not in handled and _has_value(values[name])]
    if extras:
        out.append("")
        out.append("# ==============================================")
        out.append("# Misc / Unmapped (added by config editor)")
        out.append("# ==============================================")
        for name in extras:
            out.append(f"{name}={_format_value(values[name])}")

    return "\n".join(out) + "\n"


def compute_disabled(parsed: ParsedEnvExample, **state_values: str) -> set[str]:
    """Return names of vars whose #x conditions are not all satisfied.

    Pass mode keys as kwargs, e.g. compute_disabled(parsed, pipeline_mode="LLM").
    """
    disabled: set[str] = set()
    for var in parsed.vars:
        for cond_key, cond_val in var.conditions:
            allowed = {v.strip() for v in cond_val.split(",") if v.strip()}
            if state_values.get(cond_key, "") not in allowed:
                disabled.add(var.name)
                break
    return disabled
