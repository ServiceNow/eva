# Paper figures with CIs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate clean-condition LaTeX tables, accuracy-vs-experience scatter PDFs, and a Pareto-frontier JSON with bootstrap confidence intervals, all driven by `results_pooled.csv` produced by the existing stats pipeline on this branch.

**Architecture:** Five new modules under `analysis/eva-bench-stats/` — `paper_config.py` (palettes, display names, model ordering), `paper_tables.py` (LaTeX writer), `paper_plots.py` (scatter + error bars), `paper_frontier.py` (JSON writer), and `run_paper.py` (entry point). All artifacts land in `output_processed/eva-bench-stats/CIs/paper/`. No edits to upstream pipeline files.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-05-02-paper-figures-with-cis-design.md`

---

## File map

Create:
- `analysis/eva-bench-stats/paper_config.py` — palettes, display-name maps, config loader, model ordering helper
- `analysis/eva-bench-stats/paper_tables.py` — cell formatter, shading, LaTeX writers
- `analysis/eva-bench-stats/paper_plots.py` — scatter + error-bar plot
- `analysis/eva-bench-stats/paper_frontier.py` — frontier filter + JSON writer
- `analysis/eva-bench-stats/run_paper.py` — entry point
- `tests/unit/eva_bench_stats/test_paper_tables.py`
- `tests/unit/eva_bench_stats/test_paper_plots.py`
- `tests/unit/eva_bench_stats/test_paper_frontier.py`
- `tests/integration/test_run_paper.py`

Modify:
- `local/eva-bench-stats/CIs_config.yaml` — add `paper:` section with metric groupings (Task 1)

Reference (read only, do not edit):
- `output_processed/eva-bench-stats/CIs/stats/results_pooled.csv` (columns: `model_label, metric, domain, n, point_estimate, ci_lower, ci_upper`; rows where `domain == "pooled"` are the inputs)
- `local/eva-bench-stats/CIs_config.yaml` for `models:` and `metrics:`

---

## Task 1: paper_config — constants and loader

**Files:**
- Create: `analysis/eva-bench-stats/paper_config.py`
- Modify: `local/eva-bench-stats/CIs_config.yaml`
- Test: `tests/unit/eva_bench_stats/test_paper_config.py`

- [ ] **Step 1: Add `paper:` section to CIs_config.yaml**

Append to `local/eva-bench-stats/CIs_config.yaml`:

```yaml
paper:
  output_dir: output_processed/eva-bench-stats/CIs/paper
  # Metric names are resolved against results_pooled.csv. pass_at_k / pass_power_k
  # placeholders will fill in once trial data is re-pulled with those metrics.
  accuracy:
    aggregate:
      pass_at_1:    EVA-A_pass
      pass_at_k:    EVA-A_pass_at_k
      pass_power_k: EVA-A_pass_power_k
    submetrics:
      task_completion:       Task Completion
      faithfulness:          Faithfulness
      agent_speech_fidelity: Agent Speech Fidelity
  experience:
    aggregate:
      pass_at_1:    EVA-X_pass
      pass_at_k:    EVA-X_pass_at_k
      pass_power_k: EVA-X_pass_power_k
    submetrics:
      turn_taking:             Turn-Taking
      conciseness:             Conciseness
      conversation_progression: Conv. Progression
  scatter:
    pass_at_1:    {x: EVA-A_pass,         y: EVA-X_pass}
    pass_at_k:    {x: EVA-A_pass_at_k,    y: EVA-X_pass_at_k}
    pass_power_k: {x: EVA-A_pass_power_k, y: EVA-X_pass_power_k}
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/eva_bench_stats/test_paper_config.py`:

```python
from pathlib import Path

import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "analysis" / "eva-bench-stats"))
from paper_config import PaperConfig, load_paper_config, ARCH_ORDER


def test_load_paper_config_parses_sections(tmp_path: Path) -> None:
    cfg_path = tmp_path / "CIs_config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "models": {
            "Sys A": {"alias": "alias-a", "type": "cascade"},
            "Sys B": {"alias": "alias-b", "type": "s2s"},
        },
        "paper": {
            "output_dir": "out/paper",
            "accuracy": {
                "aggregate": {"pass_at_1": "EVA-A_pass"},
                "submetrics": {"task_completion": "Task Completion"},
            },
            "experience": {
                "aggregate": {"pass_at_1": "EVA-X_pass"},
                "submetrics": {"turn_taking": "Turn-Taking"},
            },
            "scatter": {"pass_at_1": {"x": "EVA-A_pass", "y": "EVA-X_pass"}},
        },
    }))
    cfg = load_paper_config(cfg_path)
    assert isinstance(cfg, PaperConfig)
    assert cfg.output_dir == "out/paper"
    assert cfg.accuracy_aggregate == {"pass_at_1": "EVA-A_pass"}
    assert cfg.accuracy_submetrics == {"task_completion": "Task Completion"}
    assert cfg.scatter == {"pass_at_1": {"x": "EVA-A_pass", "y": "EVA-X_pass"}}
    assert cfg.models["Sys A"].arch == "cascade"


def test_arch_order_constant() -> None:
    assert ARCH_ORDER == ("cascade", "hybrid", "s2s")
```

- [ ] **Step 3: Run the test (expect failure)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_config.py -v
```
Expected: ImportError / ModuleNotFoundError on `paper_config`.

- [ ] **Step 4: Implement paper_config.py**

```python
"""Configuration loader and shared constants for the paper-figure generators."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ARCH_ORDER: tuple[str, str, str] = ("cascade", "hybrid", "s2s")

# Heatmap palettes (HTML hex, light → dark) — must match the existing tables.
ACCENT_PALETTE = [
    ("acc1", "edeaf4"), ("acc2", "d9d2e6"), ("acc3", "bfb3d4"),
    ("acc4", "9d8dbb"), ("acc5", "7a679f"), ("acc6", "584981"),
    ("acc7", "3b3060"),
]
TEAL_PALETTE = [
    ("tel1", "b8dede"), ("tel2", "90cece"), ("tel3", "62b8b8"),
    ("tel4", "3a9e9e"), ("tel5", "1e8484"), ("tel6", "0f6b6b"),
    ("tel7", "075656"),
]
PINK_PALETTE = [
    ("pnk1", "fde4ec"), ("pnk2", "fac4d4"), ("pnk3", "f59ab5"),
    ("pnk4", "ed6f95"), ("pnk5", "db4577"), ("pnk6", "b82d5c"),
    ("pnk7", "8c1f44"),
]
# Cells in the bottom 3 palette steps use black text; top 4 use white text.
LIGHT_TEXT_THRESHOLD_INDEX = 3


@dataclass(frozen=True)
class ModelEntry:
    label: str
    alias: str
    arch: str  # one of ARCH_ORDER


@dataclass(frozen=True)
class PaperConfig:
    output_dir: str
    accuracy_aggregate: dict[str, str]
    accuracy_submetrics: dict[str, str]
    experience_aggregate: dict[str, str]
    experience_submetrics: dict[str, str]
    scatter: dict[str, dict[str, str]]
    models: dict[str, ModelEntry]


def load_paper_config(config_path: Path) -> PaperConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    paper = raw["paper"]
    models_raw = raw.get("models") or {}
    models = {
        label: ModelEntry(
            label=label,
            alias=spec["alias"],
            arch=spec.get("type", "cascade"),
        )
        for label, spec in models_raw.items()
    }
    return PaperConfig(
        output_dir=paper["output_dir"],
        accuracy_aggregate=dict(paper["accuracy"]["aggregate"]),
        accuracy_submetrics=dict(paper["accuracy"]["submetrics"]),
        experience_aggregate=dict(paper["experience"]["aggregate"]),
        experience_submetrics=dict(paper["experience"]["submetrics"]),
        scatter={k: dict(v) for k, v in paper["scatter"].items()},
        models=models,
    )


def sort_models(models: dict[str, ModelEntry]) -> list[ModelEntry]:
    """Sorted by (arch order, label). Used for stable row order in tables and scatter."""
    arch_index = {a: i for i, a in enumerate(ARCH_ORDER)}
    return sorted(models.values(), key=lambda m: (arch_index.get(m.arch, 99), m.label.lower()))
```

- [ ] **Step 5: Run the test (expect pass)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_config.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add analysis/eva-bench-stats/paper_config.py \
        tests/unit/eva_bench_stats/test_paper_config.py \
        local/eva-bench-stats/CIs_config.yaml
git commit -m "feat(eva-bench-stats): add paper_config with palettes and loader"
```

---

## Task 2: paper_tables — cell formatter

**Files:**
- Create: `analysis/eva-bench-stats/paper_tables.py`
- Test: `tests/unit/eva_bench_stats/test_paper_tables.py`

- [ ] **Step 1: Write the failing test for `format_cell`**

Create `tests/unit/eva_bench_stats/test_paper_tables.py`:

```python
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "analysis" / "eva-bench-stats"))
from paper_tables import format_cell, shade_index


def test_format_cell_symmetric() -> None:
    assert format_cell(0.428, 0.393, 0.463) == "0.428 $\\pm$0.035"


def test_format_cell_asymmetric_uses_max_halfwidth() -> None:
    # point - lo = 0.080, hi - point = 0.020 → max = 0.080
    assert format_cell(0.500, 0.420, 0.520) == "0.500 $\\pm$0.080"


def test_format_cell_missing() -> None:
    assert format_cell(None, None, None) == "--"
    assert format_cell(float("nan"), float("nan"), float("nan")) == "--"


def test_shade_index_clamps_and_buckets() -> None:
    # min=0.1, max=0.5 → 7 even buckets across range
    assert shade_index(0.1, lo=0.1, hi=0.5, n_steps=7) == 0
    assert shade_index(0.5, lo=0.1, hi=0.5, n_steps=7) == 6
    assert shade_index(0.05, lo=0.1, hi=0.5, n_steps=7) == 0  # clamp low
    assert shade_index(0.6, lo=0.1, hi=0.5, n_steps=7) == 6   # clamp high
    # degenerate range
    assert shade_index(0.3, lo=0.3, hi=0.3, n_steps=7) == 0
```

- [ ] **Step 2: Run the test (expect failure)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_tables.py -v
```
Expected: ImportError on `paper_tables`.

- [ ] **Step 3: Implement `format_cell` and `shade_index`**

Create `analysis/eva-bench-stats/paper_tables.py`:

```python
"""LaTeX writers for the paper's domain-collapsed accuracy and experience tables."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd

from paper_config import (
    ACCENT_PALETTE, TEAL_PALETTE, PINK_PALETTE,
    ARCH_ORDER, LIGHT_TEXT_THRESHOLD_INDEX,
    ModelEntry, PaperConfig, sort_models,
)

MISSING_CELL = "--"


def _is_missing(x: Optional[float]) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def format_cell(point: Optional[float], ci_lower: Optional[float], ci_upper: Optional[float]) -> str:
    """Render a pooled cell as ``"<point> ±<halfwidth>"`` or ``"--"`` if missing.

    halfwidth = max(point - ci_lower, ci_upper - point) so asymmetric
    percentile CIs aren't undersold.
    """
    if _is_missing(point) or _is_missing(ci_lower) or _is_missing(ci_upper):
        return MISSING_CELL
    half = max(point - ci_lower, ci_upper - point)
    return f"{point:.3f} $\\pm${half:.3f}"


def shade_index(value: float, lo: float, hi: float, n_steps: int) -> int:
    """Return the palette bucket index in [0, n_steps-1] for ``value``."""
    if hi <= lo:
        return 0
    frac = (value - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    idx = int(frac * n_steps)
    if idx >= n_steps:
        idx = n_steps - 1
    return idx
```

- [ ] **Step 4: Run the test (expect pass)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_tables.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/eva-bench-stats/paper_tables.py tests/unit/eva_bench_stats/test_paper_tables.py
git commit -m "feat(eva-bench-stats): add cell formatter and shade-index helpers"
```

---

## Task 3: paper_tables — pooled CI lookup helper

**Files:**
- Modify: `analysis/eva-bench-stats/paper_tables.py`
- Test: `tests/unit/eva_bench_stats/test_paper_tables.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/unit/eva_bench_stats/test_paper_tables.py`:

```python
import pandas as pd
from paper_tables import lookup_pooled


def _pooled_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"model_label": "M1", "metric": "EVA-A_pass", "domain": "pooled",
         "point_estimate": 0.40, "ci_lower": 0.35, "ci_upper": 0.45},
        {"model_label": "M1", "metric": "task_completion", "domain": "pooled",
         "point_estimate": 0.60, "ci_lower": 0.55, "ci_upper": 0.65},
        # non-pooled rows must be ignored
        {"model_label": "M1", "metric": "EVA-A_pass", "domain": "itsm",
         "point_estimate": 0.99, "ci_lower": 0.98, "ci_upper": 1.0},
    ])


def test_lookup_pooled_hits() -> None:
    df = _pooled_df()
    point, lo, hi = lookup_pooled(df, "M1", "EVA-A_pass")
    assert (point, lo, hi) == (0.40, 0.35, 0.45)


def test_lookup_pooled_misses_returns_nones() -> None:
    df = _pooled_df()
    assert lookup_pooled(df, "M1", "EVA-A_pass_at_k") == (None, None, None)
    assert lookup_pooled(df, "Ghost", "EVA-A_pass") == (None, None, None)
```

- [ ] **Step 2: Run the test (expect failure)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_tables.py::test_lookup_pooled_hits -v
```
Expected: ImportError on `lookup_pooled`.

- [ ] **Step 3: Add `lookup_pooled` to `paper_tables.py`**

Append to `analysis/eva-bench-stats/paper_tables.py`:

```python
def lookup_pooled(
    pooled_df: pd.DataFrame,
    model_label: str,
    metric: str,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (point, ci_lower, ci_upper) for one (model, metric) pooled row.

    Returns (None, None, None) when the row is absent.
    """
    mask = (
        (pooled_df["domain"] == "pooled")
        & (pooled_df["model_label"] == model_label)
        & (pooled_df["metric"] == metric)
    )
    sub = pooled_df.loc[mask]
    if sub.empty:
        return (None, None, None)
    row = sub.iloc[0]
    return (float(row["point_estimate"]), float(row["ci_lower"]), float(row["ci_upper"]))
```

- [ ] **Step 4: Run the tests (expect pass)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_tables.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/eva-bench-stats/paper_tables.py tests/unit/eva_bench_stats/test_paper_tables.py
git commit -m "feat(eva-bench-stats): add lookup_pooled helper"
```

---

## Task 4: paper_tables — full LaTeX writers

**Files:**
- Modify: `analysis/eva-bench-stats/paper_tables.py`
- Test: `tests/unit/eva_bench_stats/test_paper_tables.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/unit/eva_bench_stats/test_paper_tables.py`:

```python
from paper_config import ModelEntry, PaperConfig
from paper_tables import write_accuracy_table, write_experience_table


def _cfg() -> PaperConfig:
    return PaperConfig(
        output_dir="ignored",
        accuracy_aggregate={
            "pass_at_1": "EVA-A_pass",
            "pass_at_k": "EVA-A_pass_at_k",
            "pass_power_k": "EVA-A_pass_power_k",
        },
        accuracy_submetrics={
            "task_completion": "Task Completion",
            "faithfulness": "Faithfulness",
            "agent_speech_fidelity": "Agent Speech Fidelity",
        },
        experience_aggregate={
            "pass_at_1": "EVA-X_pass",
            "pass_at_k": "EVA-X_pass_at_k",
            "pass_power_k": "EVA-X_pass_power_k",
        },
        experience_submetrics={
            "turn_taking": "Turn-Taking",
            "conciseness": "Conciseness",
            "conversation_progression": "Conv. Progression",
        },
        scatter={},
        models={
            "Sys A": ModelEntry(label="Sys A", alias="a", arch="cascade"),
            "Sys B": ModelEntry(label="Sys B", alias="b", arch="s2s"),
        },
    )


def _pooled_with_pass1_only() -> pd.DataFrame:
    rows = []
    for m, p in [("Sys A", 0.42), ("Sys B", 0.61)]:
        for metric, val in [
            ("EVA-A_pass", p), ("EVA-X_pass", p + 0.05),
            ("task_completion", p), ("faithfulness", p),
            ("agent_speech_fidelity", p), ("turn_taking", p),
            ("conciseness", p), ("conversation_progression", p),
        ]:
            rows.append({
                "model_label": m, "metric": metric, "domain": "pooled",
                "point_estimate": val, "ci_lower": val - 0.04, "ci_upper": val + 0.04,
            })
    return pd.DataFrame(rows)


def test_write_accuracy_table_emits_expected_structure(tmp_path: Path) -> None:
    out = tmp_path / "accuracy_table.tex"
    write_accuracy_table(_pooled_with_pass1_only(), _cfg(), out)
    text = out.read_text()
    assert "\\begin{table}" in text
    assert "EVA-A" in text
    assert "Task Completion" in text
    # pass_at_1 present, pass_at_k missing → "--" appears in pass_at_k cells
    assert "0.420 $\\pm$0.040" in text
    assert "--" in text
    # arch grouping
    assert "Cascade" in text
    assert "S2S" in text


def test_write_experience_table_uses_eva_x(tmp_path: Path) -> None:
    out = tmp_path / "experience_table.tex"
    write_experience_table(_pooled_with_pass1_only(), _cfg(), out)
    text = out.read_text()
    assert "EVA-X" in text
    assert "Turn-Taking" in text
    assert "0.470 $\\pm$0.040" in text  # 0.42 + 0.05
```

- [ ] **Step 2: Run the test (expect failure)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_tables.py::test_write_accuracy_table_emits_expected_structure -v
```
Expected: ImportError on `write_accuracy_table`.

- [ ] **Step 3: Implement the writers**

Append to `analysis/eva-bench-stats/paper_tables.py`:

```python
ARCH_DISPLAY = {"cascade": "Cascade", "hybrid": "Hybrid", "s2s": "S2S"}


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _palette_macros(palette: list[tuple[str, str]], macro_name: str) -> str:
    """Return the \\definecolor + \\<macro_name>{value} command body for one palette."""
    color_defs = "\n".join(f"\\definecolor{{{name}}}{{HTML}}{{{hex_}}}" for name, hex_ in palette)
    n = len(palette)
    branches: list[str] = []
    # Even thresholds across [0, 1]; uses point-estimate buckets normalized externally.
    for i, (name, _) in enumerate(palette):
        text_color = "white" if i >= LIGHT_TEXT_THRESHOLD_INDEX else "black"
        branches.append(f"\\cellcolor{{{name}}}\\textcolor{{{text_color}}}")
    return color_defs


def _shaded_cell(palette: list[tuple[str, str]], idx: int, body: str) -> str:
    name, _ = palette[idx]
    text_color = "white" if idx >= LIGHT_TEXT_THRESHOLD_INDEX else "black"
    return f"\\cellcolor{{{name}}}\\textcolor{{{text_color}}}{{{body}}}"


def _column_shading_bounds(
    pooled_df: pd.DataFrame,
    models: list[ModelEntry],
    metric: str,
) -> tuple[float, float]:
    """Min/max of point estimates for one metric across the configured models.

    Returns (0.0, 0.0) if no values are present so shade_index returns 0.
    """
    vals = []
    for m in models:
        p, _, _ = lookup_pooled(pooled_df, m.label, metric)
        if not _is_missing(p):
            vals.append(p)
    if not vals:
        return (0.0, 0.0)
    return (min(vals), max(vals))


def _write_table(
    pooled_df: pd.DataFrame,
    cfg: PaperConfig,
    out_path: Path,
    aggregate: dict[str, str],
    submetrics: dict[str, str],
    aggregate_palette: list[tuple[str, str]],
    submetric_palette: list[tuple[str, str]],
    aggregate_header: str,
    table_label: str,
    caption: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agg_keys = list(aggregate.keys())          # e.g. pass_at_1, pass_at_k, pass_power_k
    agg_metrics = [aggregate[k] for k in agg_keys]
    sub_keys = list(submetrics.keys())
    sub_display = [submetrics[k] for k in sub_keys]

    # Per-column shading bounds (one set per metric column)
    bounds = {m: _column_shading_bounds(pooled_df, list(cfg.models.values()), m)
              for m in agg_metrics + sub_keys}

    # Color macro definitions for the preamble
    palette_defs = []
    for name, hex_ in aggregate_palette + submetric_palette:
        palette_defs.append(f"\\definecolor{{{name}}}{{HTML}}{{{hex_}}}")

    n_agg = len(agg_keys)
    n_sub = len(sub_keys)
    n_metric_cols = n_agg + n_sub
    col_spec = "ll" + "c" * n_metric_cols

    sorted_models = sort_models(cfg.models)

    # Group into arches preserving sort_models order.
    arch_groups: dict[str, list[ModelEntry]] = {}
    for m in sorted_models:
        arch_groups.setdefault(m.arch, []).append(m)

    lines: list[str] = []
    lines.append("% Auto-generated by analysis/eva-bench-stats/run_paper.py")
    lines.append("% Requires: xcolor, colortbl, booktabs, multirow, array")
    lines.extend(palette_defs)
    lines.append("")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering\\small")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    # Two-row header: aggregate group label spans agg cols; per-submetric labels each get one col.
    spanner = " & " * 2 + f"\\multicolumn{{{n_agg}}}{{c}}{{\\textbf{{{aggregate_header}}}}}"
    spanner += "".join(f" & \\textbf{{{_latex_escape(name)}}}" for name in sub_display)
    lines.append(spanner + " \\\\")
    lines.append(f"\\cmidrule(lr){{3-{2 + n_agg}}}")
    sub_label_row = "\\textbf{Arch.} & \\textbf{System}"
    for k in agg_keys:
        sub_label_row += f" & \\textbf{{{_latex_escape(k.replace('_', '\\_'))}}}"
    for _ in sub_keys:
        sub_label_row += " &"
    lines.append(sub_label_row + " \\\\")
    lines.append("\\midrule")

    for arch in ARCH_ORDER:
        members = arch_groups.get(arch, [])
        if not members:
            continue
        first = True
        for m in members:
            row_cells: list[str] = []
            arch_cell = f"\\multirow{{{len(members)}}}{{*}}{{{ARCH_DISPLAY[arch]}}}" if first else ""
            row_cells.append(arch_cell)
            row_cells.append(_latex_escape(m.label))

            for metric in agg_metrics:
                point, lo, hi = lookup_pooled(pooled_df, m.label, metric)
                body = format_cell(point, lo, hi)
                if body == MISSING_CELL or _is_missing(point):
                    row_cells.append(body)
                else:
                    bmin, bmax = bounds[metric]
                    idx = shade_index(point, bmin, bmax, n_steps=len(aggregate_palette))
                    row_cells.append(_shaded_cell(aggregate_palette, idx, body))
            for sub_key in sub_keys:
                point, lo, hi = lookup_pooled(pooled_df, m.label, sub_key)
                body = format_cell(point, lo, hi)
                if body == MISSING_CELL or _is_missing(point):
                    row_cells.append(body)
                else:
                    bmin, bmax = bounds[sub_key]
                    idx = shade_index(point, bmin, bmax, n_steps=len(submetric_palette))
                    row_cells.append(_shaded_cell(submetric_palette, idx, body))

            lines.append(" & ".join(row_cells) + " \\\\")
            first = False
        lines.append("\\midrule")

    # Replace trailing midrule with bottomrule.
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")

    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\vspace{6pt}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{table_label}}}")
    lines.append("\\end{table}")

    out_path.write_text("\n".join(lines) + "\n")


def write_accuracy_table(pooled_df: pd.DataFrame, cfg: PaperConfig, out_path: Path) -> None:
    _write_table(
        pooled_df, cfg, out_path,
        aggregate=cfg.accuracy_aggregate,
        submetrics=cfg.accuracy_submetrics,
        aggregate_palette=ACCENT_PALETTE,
        submetric_palette=TEAL_PALETTE,
        aggregate_header="EVA-A",
        table_label="tab:accuracy-metrics",
        caption=(
            "Accuracy metrics for all evaluated systems under clean-audio conditions, "
            "pooled equal-weighted across the three EVA domains. Each cell shows the "
            "pooled point estimate $\\pm$ the percentile bootstrap CI half-width "
            "($\\alpha = 0.05$). Cell shading is scaled per metric column from min to "
            "max of the point estimate (darker = higher)."
        ),
    )


def write_experience_table(pooled_df: pd.DataFrame, cfg: PaperConfig, out_path: Path) -> None:
    _write_table(
        pooled_df, cfg, out_path,
        aggregate=cfg.experience_aggregate,
        submetrics=cfg.experience_submetrics,
        aggregate_palette=ACCENT_PALETTE,
        submetric_palette=PINK_PALETTE,
        aggregate_header="EVA-X",
        table_label="tab:experience-metrics",
        caption=(
            "Experience metrics for all evaluated systems under clean-audio conditions, "
            "pooled equal-weighted across the three EVA domains. Each cell shows the "
            "pooled point estimate $\\pm$ the percentile bootstrap CI half-width "
            "($\\alpha = 0.05$). Cell shading is scaled per metric column from min to "
            "max of the point estimate (darker = higher)."
        ),
    )
```

- [ ] **Step 4: Run all paper_tables tests (expect pass)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_tables.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/eva-bench-stats/paper_tables.py tests/unit/eva_bench_stats/test_paper_tables.py
git commit -m "feat(eva-bench-stats): generate accuracy and experience LaTeX tables"
```

---

## Task 5: paper_plots — scatter with CI error bars

**Files:**
- Create: `analysis/eva-bench-stats/paper_plots.py`
- Test: `tests/unit/eva_bench_stats/test_paper_plots.py`

- [ ] **Step 1: Write the failing test for the asymmetric error helper**

Create `tests/unit/eva_bench_stats/test_paper_plots.py`:

```python
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "analysis" / "eva-bench-stats"))
from paper_plots import asymmetric_err, build_scatter_points, write_scatter
from paper_config import ModelEntry, PaperConfig


def test_asymmetric_err_returns_nonneg_pair() -> None:
    lo_arr, hi_arr = asymmetric_err([0.5, 0.2], [0.4, 0.15], [0.55, 0.30])
    np.testing.assert_allclose(lo_arr, [0.10, 0.05])
    np.testing.assert_allclose(hi_arr, [0.05, 0.10])


def test_asymmetric_err_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        asymmetric_err([0.5], [0.6], [0.55])
```

- [ ] **Step 2: Run the test (expect failure)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_plots.py -v
```
Expected: ImportError on `paper_plots`.

- [ ] **Step 3: Implement the helper**

Create `analysis/eva-bench-stats/paper_plots.py`:

```python
"""Accuracy-vs-experience scatter plots with bootstrap CI error bars."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_config import ARCH_ORDER, ModelEntry, PaperConfig, sort_models
from paper_tables import lookup_pooled, _is_missing

ARCH_MARKER = {"cascade": "o", "hybrid": "D", "s2s": "s"}
ARCH_COLOR = {"cascade": "#3a86ff", "hybrid": "#2a9d8f", "s2s": "#7b2cbf"}
ARCH_DISPLAY = {"cascade": "Cascade", "hybrid": "Hybrid (AudioLLM + TTS)", "s2s": "S2S"}


@dataclass(frozen=True)
class ScatterPoint:
    label: str
    arch: str
    x: float
    x_lo: float
    x_hi: float
    y: float
    y_lo: float
    y_hi: float


def asymmetric_err(
    points: list[float],
    los: list[float],
    his: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(point - lo, hi - point)`` for matplotlib's xerr/yerr asymmetric form."""
    p = np.asarray(points, dtype=float)
    lo = np.asarray(los, dtype=float)
    hi = np.asarray(his, dtype=float)
    if np.any(lo > p) or np.any(hi < p):
        raise ValueError("ci_lower must be <= point <= ci_upper for every point")
    return p - lo, hi - p
```

- [ ] **Step 4: Run the helper test (expect pass)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_plots.py::test_asymmetric_err_returns_nonneg_pair tests/unit/eva_bench_stats/test_paper_plots.py::test_asymmetric_err_rejects_inverted_bounds -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit the helper**

```bash
git add analysis/eva-bench-stats/paper_plots.py tests/unit/eva_bench_stats/test_paper_plots.py
git commit -m "feat(eva-bench-stats): add asymmetric error-bar helper"
```

---

## Task 6: paper_plots — point assembly

**Files:**
- Modify: `analysis/eva-bench-stats/paper_plots.py`
- Test: `tests/unit/eva_bench_stats/test_paper_plots.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/unit/eva_bench_stats/test_paper_plots.py`:

```python
def _cfg_two_models() -> PaperConfig:
    return PaperConfig(
        output_dir="ignored",
        accuracy_aggregate={}, accuracy_submetrics={},
        experience_aggregate={}, experience_submetrics={},
        scatter={"pass_at_1": {"x": "EVA-A_pass", "y": "EVA-X_pass"}},
        models={
            "Sys A": ModelEntry(label="Sys A", alias="a", arch="cascade"),
            "Sys B": ModelEntry(label="Sys B", alias="b", arch="s2s"),
        },
    )


def _pooled_full() -> pd.DataFrame:
    rows = []
    for m, ax, ay in [("Sys A", 0.40, 0.50), ("Sys B", 0.62, 0.30)]:
        rows.append({"model_label": m, "metric": "EVA-A_pass", "domain": "pooled",
                     "point_estimate": ax, "ci_lower": ax - 0.04, "ci_upper": ax + 0.05})
        rows.append({"model_label": m, "metric": "EVA-X_pass", "domain": "pooled",
                     "point_estimate": ay, "ci_lower": ay - 0.03, "ci_upper": ay + 0.06})
    return pd.DataFrame(rows)


def test_build_scatter_points_full() -> None:
    pts = build_scatter_points(_pooled_full(), _cfg_two_models(), "EVA-A_pass", "EVA-X_pass")
    labels = {p.label for p in pts}
    assert labels == {"Sys A", "Sys B"}
    a = next(p for p in pts if p.label == "Sys A")
    assert (a.x, a.x_lo, a.x_hi) == (0.40, 0.36, 0.45)
    assert (a.y, a.y_lo, a.y_hi) == (0.50, 0.47, 0.56)


def test_build_scatter_points_drops_missing() -> None:
    df = _pooled_full().drop(_pooled_full().index[0])  # drop Sys A x-row
    pts = build_scatter_points(df, _cfg_two_models(), "EVA-A_pass", "EVA-X_pass")
    assert {p.label for p in pts} == {"Sys B"}
```

- [ ] **Step 2: Run the test (expect failure)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_plots.py::test_build_scatter_points_full -v
```
Expected: ImportError on `build_scatter_points`.

- [ ] **Step 3: Implement `build_scatter_points`**

Append to `analysis/eva-bench-stats/paper_plots.py`:

```python
def build_scatter_points(
    pooled_df: pd.DataFrame,
    cfg: PaperConfig,
    x_metric: str,
    y_metric: str,
) -> list[ScatterPoint]:
    """One ScatterPoint per system that has pooled CIs for both axes."""
    out: list[ScatterPoint] = []
    for m in sort_models(cfg.models):
        x, x_lo, x_hi = lookup_pooled(pooled_df, m.label, x_metric)
        y, y_lo, y_hi = lookup_pooled(pooled_df, m.label, y_metric)
        if any(_is_missing(v) for v in (x, x_lo, x_hi, y, y_lo, y_hi)):
            continue
        out.append(ScatterPoint(
            label=m.label, arch=m.arch,
            x=x, x_lo=x_lo, x_hi=x_hi,
            y=y, y_lo=y_lo, y_hi=y_hi,
        ))
    return out
```

- [ ] **Step 4: Run the tests (expect pass)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_plots.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/eva-bench-stats/paper_plots.py tests/unit/eva_bench_stats/test_paper_plots.py
git commit -m "feat(eva-bench-stats): assemble scatter points from pooled CIs"
```

---

## Task 7: paper_plots — frontier on point estimates

**Files:**
- Modify: `analysis/eva-bench-stats/paper_plots.py`
- Test: `tests/unit/eva_bench_stats/test_paper_plots.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/unit/eva_bench_stats/test_paper_plots.py`:

```python
from paper_plots import pareto_frontier_indices


def test_pareto_frontier_indices_simple() -> None:
    # Points: (x, y). Higher-better on both. Frontier = {(0.6, 0.5), (0.4, 0.7)}.
    pts = [(0.6, 0.5), (0.4, 0.7), (0.3, 0.4), (0.5, 0.55)]
    idx = pareto_frontier_indices(pts)
    assert sorted(idx) == [0, 1, 3]


def test_pareto_frontier_indices_empty() -> None:
    assert pareto_frontier_indices([]) == []
```

- [ ] **Step 2: Run the test (expect failure)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_plots.py::test_pareto_frontier_indices_simple -v
```
Expected: ImportError on `pareto_frontier_indices`.

- [ ] **Step 3: Implement the frontier**

Append to `analysis/eva-bench-stats/paper_plots.py`:

```python
def pareto_frontier_indices(points: list[tuple[float, float]]) -> list[int]:
    """Return indices of points on the upper-right (higher-is-better) Pareto frontier."""
    n = len(points)
    keep: list[int] = []
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            if (xj >= xi and yj >= yi) and (xj > xi or yj > yi):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep
```

- [ ] **Step 4: Run the tests (expect pass)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_plots.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/eva-bench-stats/paper_plots.py tests/unit/eva_bench_stats/test_paper_plots.py
git commit -m "feat(eva-bench-stats): pareto frontier on point estimates"
```

---

## Task 8: paper_plots — `write_scatter` end-to-end

**Files:**
- Modify: `analysis/eva-bench-stats/paper_plots.py`
- Test: `tests/unit/eva_bench_stats/test_paper_plots.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/unit/eva_bench_stats/test_paper_plots.py`:

```python
def test_write_scatter_creates_pdf(tmp_path: Path) -> None:
    out = tmp_path / "scatter.pdf"
    n_drawn = write_scatter(
        _pooled_full(), _cfg_two_models(),
        x_metric="EVA-A_pass", y_metric="EVA-X_pass",
        x_label="Accuracy (EVA-A pass@1)",
        y_label="Experience (EVA-X pass@1)",
        title="Accuracy vs Experience pass@1",
        out_path=out,
    )
    assert n_drawn == 2
    assert out.exists() and out.stat().st_size > 0


def test_write_scatter_returns_zero_when_no_points(tmp_path: Path) -> None:
    out = tmp_path / "skip.pdf"
    df = pd.DataFrame(columns=["model_label", "metric", "domain",
                               "point_estimate", "ci_lower", "ci_upper"])
    n_drawn = write_scatter(
        df, _cfg_two_models(),
        x_metric="EVA-A_pass_at_k", y_metric="EVA-X_pass_at_k",
        x_label="x", y_label="y", title="t", out_path=out,
    )
    assert n_drawn == 0
    assert not out.exists()
```

- [ ] **Step 2: Run the test (expect failure)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_plots.py::test_write_scatter_creates_pdf -v
```
Expected: ImportError on `write_scatter`.

- [ ] **Step 3: Implement `write_scatter`**

Append to `analysis/eva-bench-stats/paper_plots.py`:

```python
def write_scatter(
    pooled_df: pd.DataFrame,
    cfg: PaperConfig,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> int:
    """Write a 1-pt-per-system scatter PDF with asymmetric CI error bars.

    Returns the number of points drawn. If zero, no file is written.
    """
    points = build_scatter_points(pooled_df, cfg, x_metric, y_metric)
    if not points:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Pareto frontier on point estimates
    xy = [(p.x, p.y) for p in points]
    frontier_idx = pareto_frontier_indices(xy)
    if len(frontier_idx) >= 2:
        frontier_pts = sorted([(points[i].x, points[i].y) for i in frontier_idx])
        ax.plot(
            [fx for fx, _ in frontier_pts],
            [fy for _, fy in frontier_pts],
            linestyle="--", color="grey", alpha=0.7, zorder=1,
            label="Pareto frontier",
        )

    seen_archs: set[str] = set()
    for arch in ARCH_ORDER:
        arch_points = [p for p in points if p.arch == arch]
        if not arch_points:
            continue
        seen_archs.add(arch)
        xs = [p.x for p in arch_points]
        ys = [p.y for p in arch_points]
        x_err = asymmetric_err(xs, [p.x_lo for p in arch_points], [p.x_hi for p in arch_points])
        y_err = asymmetric_err(ys, [p.y_lo for p in arch_points], [p.y_hi for p in arch_points])
        ax.errorbar(
            xs, ys, xerr=x_err, yerr=y_err,
            fmt=ARCH_MARKER[arch], color=ARCH_COLOR[arch],
            ecolor=ARCH_COLOR[arch], elinewidth=1.0, capsize=2,
            markersize=8, alpha=0.9, zorder=3,
            label=ARCH_DISPLAY[arch],
        )

    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return len(points)
```

- [ ] **Step 4: Run the tests (expect pass)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_plots.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/eva-bench-stats/paper_plots.py tests/unit/eva_bench_stats/test_paper_plots.py
git commit -m "feat(eva-bench-stats): write_scatter with CI error bars and frontier"
```

---

## Task 9: paper_frontier — JSON writer

**Files:**
- Create: `analysis/eva-bench-stats/paper_frontier.py`
- Test: `tests/unit/eva_bench_stats/test_paper_frontier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/eva_bench_stats/test_paper_frontier.py`:

```python
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "analysis" / "eva-bench-stats"))
from paper_config import ModelEntry, PaperConfig
from paper_frontier import write_frontier_json


def _cfg() -> PaperConfig:
    return PaperConfig(
        output_dir="ignored",
        accuracy_aggregate={}, accuracy_submetrics={},
        experience_aggregate={}, experience_submetrics={},
        scatter={
            "pass_at_1":    {"x": "EVA-A_pass",         "y": "EVA-X_pass"},
            "pass_at_k":    {"x": "EVA-A_pass_at_k",    "y": "EVA-X_pass_at_k"},
            "pass_power_k": {"x": "EVA-A_pass_power_k", "y": "EVA-X_pass_power_k"},
        },
        models={
            "Sys A": ModelEntry(label="Sys A", alias="a", arch="cascade"),
            "Sys B": ModelEntry(label="Sys B", alias="b", arch="s2s"),
            "Sys C": ModelEntry(label="Sys C", alias="c", arch="cascade"),
        },
    )


def _pooled() -> pd.DataFrame:
    rows = []
    # pass@1: Sys A=(0.4,0.5), Sys B=(0.6,0.3), Sys C=(0.3,0.2)
    # Frontier: A and B (both non-dominated); C is dominated by A.
    for m, ax, ay in [("Sys A", 0.4, 0.5), ("Sys B", 0.6, 0.3), ("Sys C", 0.3, 0.2)]:
        rows.append({"model_label": m, "metric": "EVA-A_pass", "domain": "pooled",
                     "point_estimate": ax, "ci_lower": ax - 0.02, "ci_upper": ax + 0.02})
        rows.append({"model_label": m, "metric": "EVA-X_pass", "domain": "pooled",
                     "point_estimate": ay, "ci_lower": ay - 0.03, "ci_upper": ay + 0.03})
    return pd.DataFrame(rows)


def test_write_frontier_json_shape(tmp_path: Path) -> None:
    out = tmp_path / "pareto_frontier.json"
    write_frontier_json(_pooled(), _cfg(), out)
    payload = json.loads(out.read_text())
    assert set(payload.keys()) == {"pass_at_1", "pass_at_k", "pass_power_k"}
    pass1 = payload["pass_at_1"]
    assert {e["system"] for e in pass1} == {"Sys A", "Sys B"}
    a = next(e for e in pass1 if e["system"] == "Sys A")
    assert a["system_type"] == "cascade"
    assert a["eva_a"] == {"point": 0.4, "ci_low": 0.38, "ci_high": 0.42}
    assert a["eva_x"] == {"point": 0.5, "ci_low": 0.47, "ci_high": 0.53}
    # missing variants → empty list
    assert payload["pass_at_k"] == []
    assert payload["pass_power_k"] == []
    # sorted by eva_a.point ascending
    assert [e["eva_a"]["point"] for e in pass1] == sorted(e["eva_a"]["point"] for e in pass1)
```

- [ ] **Step 2: Run the test (expect failure)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_frontier.py -v
```
Expected: ImportError on `paper_frontier`.

- [ ] **Step 3: Implement `paper_frontier.py`**

Create `analysis/eva-bench-stats/paper_frontier.py`:

```python
"""Pareto-frontier JSON writer with bootstrap CIs for the paper."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paper_config import PaperConfig
from paper_plots import build_scatter_points, pareto_frontier_indices


def _round(x: float, places: int = 6) -> float:
    return float(round(x, places))


def write_frontier_json(pooled_df: pd.DataFrame, cfg: PaperConfig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[dict]] = {}
    for variant_key, axes in cfg.scatter.items():
        points = build_scatter_points(pooled_df, cfg, axes["x"], axes["y"])
        if not points:
            payload[variant_key] = []
            continue
        xy = [(p.x, p.y) for p in points]
        frontier = sorted(
            (points[i] for i in pareto_frontier_indices(xy)),
            key=lambda p: p.x,
        )
        payload[variant_key] = [
            {
                "system": p.label,
                "system_type": p.arch,
                "eva_a": {
                    "point":   _round(p.x),
                    "ci_low":  _round(p.x_lo),
                    "ci_high": _round(p.x_hi),
                },
                "eva_x": {
                    "point":   _round(p.y),
                    "ci_low":  _round(p.y_lo),
                    "ci_high": _round(p.y_hi),
                },
            }
            for p in frontier
        ]
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
```

- [ ] **Step 4: Run the tests (expect pass)**

```bash
uv run pytest tests/unit/eva_bench_stats/test_paper_frontier.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/eva-bench-stats/paper_frontier.py tests/unit/eva_bench_stats/test_paper_frontier.py
git commit -m "feat(eva-bench-stats): pareto frontier JSON with CIs"
```

---

## Task 10: run_paper.py — entry point

**Files:**
- Create: `analysis/eva-bench-stats/run_paper.py`

- [ ] **Step 1: Implement the entry point**

Create `analysis/eva-bench-stats/run_paper.py`:

```python
#!/usr/bin/env python3
"""Generate paper-ready LaTeX tables, scatter PDFs, and Pareto-frontier JSON.

Reads results_pooled.csv produced by stats_CIs.py and writes to
output_processed/eva-bench-stats/CIs/paper/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from paper_config import load_paper_config  # noqa: E402
from paper_frontier import write_frontier_json  # noqa: E402
from paper_plots import write_scatter  # noqa: E402
from paper_tables import write_accuracy_table, write_experience_table  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "local" / "eva-bench-stats" / "CIs_config.yaml"
DEFAULT_POOLED = PROJECT_ROOT / "output_processed" / "eva-bench-stats" / "CIs" / "stats" / "results_pooled.csv"

SCATTER_LABELS = {
    "pass_at_1":    {"title": "Accuracy vs Experience pass@1",
                     "x": "Accuracy (EVA-A pass@1)",
                     "y": "Experience (EVA-X pass@1)",
                     "filename": "accuracy_vs_experience_pass_at_1.pdf"},
    "pass_at_k":    {"title": "Accuracy vs Experience pass@k",
                     "x": "Accuracy (EVA-A pass@k)",
                     "y": "Experience (EVA-X pass@k)",
                     "filename": "accuracy_vs_experience_pass_at_k.pdf"},
    "pass_power_k": {"title": "Accuracy vs Experience pass^k",
                     "x": "Accuracy (EVA-A pass^k)",
                     "y": "Experience (EVA-X pass^k)",
                     "filename": "accuracy_vs_experience_pass_power_k.pdf"},
}


def run(config_path: Path, pooled_path: Path) -> None:
    if not pooled_path.exists():
        raise FileNotFoundError(
            f"results_pooled.csv not found at {pooled_path}. Run stats_CIs.py first."
        )
    cfg = load_paper_config(config_path)
    pooled_df = pd.read_csv(pooled_path)
    out_dir = PROJECT_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {out_dir}")

    accuracy_path = out_dir / "accuracy_table.tex"
    write_accuracy_table(pooled_df, cfg, accuracy_path)
    print(f"  wrote {accuracy_path.name}")

    experience_path = out_dir / "experience_table.tex"
    write_experience_table(pooled_df, cfg, experience_path)
    print(f"  wrote {experience_path.name}")

    for variant_key, info in SCATTER_LABELS.items():
        axes = cfg.scatter.get(variant_key)
        if axes is None:
            print(f"  [skip] no scatter config for '{variant_key}'")
            continue
        scatter_path = out_dir / info["filename"]
        n = write_scatter(
            pooled_df, cfg,
            x_metric=axes["x"], y_metric=axes["y"],
            x_label=info["x"], y_label=info["y"],
            title=info["title"], out_path=scatter_path,
        )
        if n == 0:
            print(f"  [skip] {variant_key}: no rows in pooled CSV")
        else:
            print(f"  wrote {scatter_path.name} ({n} systems)")

    frontier_path = out_dir / "pareto_frontier.json"
    write_frontier_json(pooled_df, cfg, frontier_path)
    print(f"  wrote {frontier_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--pooled", type=Path, default=DEFAULT_POOLED)
    args = ap.parse_args()
    run(args.config, args.pooled)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test against real data**

```bash
uv run python analysis/eva-bench-stats/run_paper.py
```
Expected: writes `accuracy_table.tex`, `experience_table.tex`, at least the `pass_at_1` scatter PDF, and `pareto_frontier.json` under `output_processed/eva-bench-stats/CIs/paper/`. Pass@k / pass^k variants will print `[skip]` until the re-pull lands.

- [ ] **Step 3: Visually inspect outputs**

Open `output_processed/eva-bench-stats/CIs/paper/accuracy_vs_experience_pass_at_1.pdf` and `accuracy_table.tex`. Confirm error bars on each scatter point, `±` half-widths in cells, `--` for pass@k columns, arch grouping intact.

- [ ] **Step 4: Commit**

```bash
git add analysis/eva-bench-stats/run_paper.py
git commit -m "feat(eva-bench-stats): run_paper entry point for paper artifacts"
```

---

## Task 11: integration test

**Files:**
- Create: `tests/integration/test_run_paper.py`

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_run_paper.py`:

```python
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "eva-bench-stats"))
import run_paper  # noqa: E402


def _build_pooled_csv(path: Path) -> None:
    rows = []
    metrics = [
        "EVA-A_pass", "EVA-X_pass",
        "task_completion", "faithfulness", "agent_speech_fidelity",
        "turn_taking", "conciseness", "conversation_progression",
    ]
    for m, base in [("Sys A", 0.40), ("Sys B", 0.62), ("Sys C", 0.25)]:
        for metric in metrics:
            rows.append({
                "model_label": m, "metric": metric, "domain": "pooled", "n": "pooled",
                "point_estimate": base,
                "ci_lower": base - 0.04,
                "ci_upper": base + 0.04,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_config(path: Path, output_dir: str) -> None:
    path.write_text(yaml.safe_dump({
        "models": {
            "Sys A": {"alias": "a", "type": "cascade"},
            "Sys B": {"alias": "b", "type": "s2s"},
            "Sys C": {"alias": "c", "type": "hybrid"},
        },
        "paper": {
            "output_dir": output_dir,
            "accuracy": {
                "aggregate": {
                    "pass_at_1": "EVA-A_pass",
                    "pass_at_k": "EVA-A_pass_at_k",
                    "pass_power_k": "EVA-A_pass_power_k",
                },
                "submetrics": {
                    "task_completion": "Task Completion",
                    "faithfulness": "Faithfulness",
                    "agent_speech_fidelity": "Agent Speech Fidelity",
                },
            },
            "experience": {
                "aggregate": {
                    "pass_at_1": "EVA-X_pass",
                    "pass_at_k": "EVA-X_pass_at_k",
                    "pass_power_k": "EVA-X_pass_power_k",
                },
                "submetrics": {
                    "turn_taking": "Turn-Taking",
                    "conciseness": "Conciseness",
                    "conversation_progression": "Conv. Progression",
                },
            },
            "scatter": {
                "pass_at_1": {"x": "EVA-A_pass", "y": "EVA-X_pass"},
                "pass_at_k": {"x": "EVA-A_pass_at_k", "y": "EVA-X_pass_at_k"},
                "pass_power_k": {"x": "EVA-A_pass_power_k", "y": "EVA-X_pass_power_k"},
            },
        },
    }))


def test_run_paper_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "CIs_config.yaml"
    pooled_path = tmp_path / "results_pooled.csv"
    out_rel = "out/paper"
    _build_config(cfg_path, out_rel)
    _build_pooled_csv(pooled_path)
    monkeypatch.setattr(run_paper, "PROJECT_ROOT", tmp_path)

    run_paper.run(cfg_path, pooled_path)

    out_dir = tmp_path / out_rel
    tex_a = (out_dir / "accuracy_table.tex").read_text()
    tex_x = (out_dir / "experience_table.tex").read_text()
    assert "\\begin{table}" in tex_a and "EVA-A" in tex_a and "Cascade" in tex_a
    assert "\\begin{table}" in tex_x and "EVA-X" in tex_x

    pdf = out_dir / "accuracy_vs_experience_pass_at_1.pdf"
    assert pdf.exists() and pdf.stat().st_size > 0
    # pass@k variants are absent from the pooled CSV → no PDF written.
    assert not (out_dir / "accuracy_vs_experience_pass_at_k.pdf").exists()

    payload = json.loads((out_dir / "pareto_frontier.json").read_text())
    assert set(payload.keys()) == {"pass_at_1", "pass_at_k", "pass_power_k"}
    assert payload["pass_at_k"] == []
    assert len(payload["pass_at_1"]) >= 1
```

- [ ] **Step 2: Run the test (expect pass)**

```bash
uv run pytest tests/integration/test_run_paper.py -v
```
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_run_paper.py
git commit -m "test(eva-bench-stats): integration test for run_paper"
```

---

## Self-review

- **Spec coverage**
  - Domain-collapsed accuracy + experience tables with pooled CIs → Task 4
  - `point ±halfwidth` cell format → Task 2
  - Heatmap shading by point estimate → Task 4 (`_column_shading_bounds` + `_shaded_cell`)
  - `pass@k` / `pass^k` placeholder columns → driven by config (Task 1) and `--` rendering in Tasks 2/4
  - Hybrid model present in latest pull → no special-case needed; `lookup_pooled` returns rows when present
  - Scatter with horizontal/vertical asymmetric CI bars → Tasks 5–8
  - Pareto frontier on point estimates → Task 7, used by both Task 8 and Task 9
  - JSON with three variants and CI bounds → Task 9
  - Output dir `output_processed/eva-bench-stats/CIs/paper/` → Task 1 config, Task 10 runner
  - `run_paper.py` entry point → Task 10
  - Error handling: missing pooled CSV (Task 10), missing metrics → `--` in tables, skipped PDFs, `[]` in JSON (Tasks 4, 8, 9)
  - Tests: cell formatter, error-bar conversion, frontier filter, integration → Tasks 2, 5, 7, 11

- **Placeholder scan:** every code step contains complete code; no TBDs.

- **Type consistency:** `lookup_pooled` returns `tuple[Optional[float], Optional[float], Optional[float]]` and is consumed identically in `paper_tables` and `paper_plots`. `ScatterPoint` fields used consistently across `paper_plots` and `paper_frontier`. `PaperConfig` field names match across all consumers.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-02-paper-figures-with-cis.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
