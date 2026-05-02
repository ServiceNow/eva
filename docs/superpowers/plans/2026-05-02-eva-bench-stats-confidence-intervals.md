# eva-bench-stats Confidence Intervals — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Confidence Intervals tab for the `eva-bench-stats` Streamlit app — 95% bootstrapped CIs (per-domain + equal-weighted pooled) on model-level metric scores including pass@1 and EVA composites.

**Architecture:** A new remote-side puller (`local/eva-bench-stats/pull_clean_data.py`) emits a `trial_scores.csv` of clean (non-perturbation) runs. A new analysis chain (`data_CIs.py` → `stats_CIs.py` → `plots_CIs.py`) computes scenario-level means, runs a stability check at 1k vs 2k bootstrap resamples, then produces per-domain and pooled CIs. A shared `stats_utils.py` extracts the bootstrap primitive so `stats_perturbations` reuses it without regression. The Streamlit `CIs_page()` placeholder is replaced with metric-tab forest plots, a paper-ready summary table, and stability/completeness expanders.

**Tech Stack:** Python (uv), numpy, pandas, plotly, streamlit, pyyaml. Existing project conventions: long-form CSVs, `*_config.yaml` per analysis, output under `output_processed/eva-bench-stats/<analysis>/`.

**Spec:** `docs/superpowers/specs/2026-05-02-eva-bench-stats-confidence-intervals-design.md`

**Testing convention:** No unit tests exist for sibling analyses (`stats_perturbations`, `stats_variance`); validation is via smoke runs and regression diffs. We add **light unit tests** for the new pure math utility (`stats_utils.py`) only, because it is shared across analyses. Everything else is validated by smoke run + regression diff against the existing perturbation outputs.

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `analysis/eva-bench-stats/stats_utils.py` | NEW | `bootstrap_resample`, `bootstrap_ci` (shared primitives) |
| `analysis/eva-bench-stats/stats_perturbations.py` | EDIT | Re-import `bootstrap_ci` from `stats_utils` |
| `tests/unit/eva_bench_stats/test_stats_utils.py` | NEW | Unit tests for the shared primitives |
| `local/eva-bench-stats/pull_clean_data.py` | NEW | Remote puller for clean trial scores |
| `local/eva-bench-stats/CIs_config.yaml` | NEW | CI analysis config |
| `analysis/eva-bench-stats/data_CIs.py` | REPLACE | Load + validate + scenario means |
| `analysis/eva-bench-stats/stats_CIs.py` | REPLACE | Stability + per-domain + pooled bootstrap |
| `analysis/eva-bench-stats/plots_CIs.py` | REPLACE | Forest plots + paper summary table |
| `analysis/eva-bench-stats/run_data.py` | EDIT | Add `_CIs_data()` hook |
| `analysis/eva-bench-stats/run_stats.py` | EDIT | Add `_CIs_stats()` hook |
| `analysis/eva-bench-stats/app.py` | EDIT | Replace `CIs_page()` placeholder |

---

## Task 1: Create shared bootstrap primitive in `stats_utils.py`

**Files:**
- Create: `analysis/eva-bench-stats/stats_utils.py`
- Test: `tests/unit/eva_bench_stats/test_stats_utils.py`

- [ ] **Step 1: Create test directory init**

```bash
mkdir -p tests/unit/eva_bench_stats
touch tests/unit/eva_bench_stats/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/unit/eva_bench_stats/test_stats_utils.py`:

```python
"""Unit tests for analysis/eva-bench-stats/stats_utils.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "eva-bench-stats"))

from stats_utils import bootstrap_ci, bootstrap_resample  # noqa: E402


def test_bootstrap_resample_shape_and_determinism():
    values = np.array([0.0, 0.5, 1.0, 0.25, 0.75])
    a = bootstrap_resample(values, n_boot=100, seed=42)
    b = bootstrap_resample(values, n_boot=100, seed=42)
    assert a.shape == (100,)
    np.testing.assert_array_equal(a, b)


def test_bootstrap_resample_different_seeds_differ():
    values = np.array([0.0, 0.5, 1.0])
    a = bootstrap_resample(values, n_boot=100, seed=1)
    b = bootstrap_resample(values, n_boot=100, seed=2)
    assert not np.array_equal(a, b)


def test_bootstrap_resample_constant_input_constant_output():
    values = np.full(10, 0.7)
    boot = bootstrap_resample(values, n_boot=50, seed=0)
    np.testing.assert_allclose(boot, 0.7)


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.1, size=100)
    lo, hi = bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.05)
    assert lo < values.mean() < hi
    assert hi - lo < 0.1


def test_bootstrap_ci_alpha_widens_to_narrows():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.1, size=100)
    lo90, hi90 = bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.10)
    lo95, hi95 = bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.05)
    assert (hi95 - lo95) > (hi90 - lo90)
```

- [ ] **Step 3: Run tests; verify they fail with ImportError**

Run: `uv run pytest tests/unit/eva_bench_stats/test_stats_utils.py -v`
Expected: collection error / ImportError ("No module named 'stats_utils'").

- [ ] **Step 4: Implement `stats_utils.py`**

Create `analysis/eva-bench-stats/stats_utils.py`:

```python
"""Shared statistical primitives for eva-bench-stats analyses.

Pure NumPy. No file I/O, no plotting. Consumed by stats_perturbations and stats_CIs.
"""
from __future__ import annotations

import numpy as np


def bootstrap_resample(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Return shape (n_boot,) array of bootstrap-resample means.

    Resamples `values` with replacement `n_boot` times and returns the mean
    of each resample. Callers needing a percentile CI should use bootstrap_ci.

    Args:
        values: 1-D array of observations (e.g. scenario-level scores).
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        1-D float array of length n_boot containing the resample means.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.zeros(n_boot, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    return values[indices].mean(axis=1)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval on the mean.

    Args:
        values: 1-D array of observations.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.
        alpha: Significance level; CI covers 1 - alpha probability.

    Returns:
        (lower, upper) CI bounds at the alpha/2 and 1-alpha/2 percentiles.
    """
    boot_means = bootstrap_resample(values, n_boot=n_boot, seed=seed)
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper
```

- [ ] **Step 5: Run tests; verify pass**

Run: `uv run pytest tests/unit/eva_bench_stats/test_stats_utils.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add analysis/eva-bench-stats/stats_utils.py tests/unit/eva_bench_stats/__init__.py tests/unit/eva_bench_stats/test_stats_utils.py
git commit -m "feat(eva-bench-stats): add shared bootstrap primitives in stats_utils

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Refactor `stats_perturbations` to consume `stats_utils.bootstrap_ci`

**Files:**
- Modify: `analysis/eva-bench-stats/stats_perturbations.py:84-111`

- [ ] **Step 1: Capture baseline outputs (regression reference)**

```bash
test -d output_processed/eva-bench-stats/perturbations && \
  cp output_processed/eva-bench-stats/perturbations/results_pooled.csv /tmp/before_pooled.csv && \
  cp output_processed/eva-bench-stats/perturbations/results_per_domain.csv /tmp/before_perdomain.csv \
  || echo "no baseline outputs to compare; will skip regression diff"
```

If no baseline exists, run once to create one:
`uv run python analysis/eva-bench-stats/run_stats.py` (only if `output_processed/eva-bench-stats/perturbations/scenario_deltas.csv` exists).

- [ ] **Step 2: Replace local `bootstrap_ci` definition with re-export**

In `analysis/eva-bench-stats/stats_perturbations.py`, delete the existing `bootstrap_ci` function (lines around 84–111) and replace with:

```python
from stats_utils import bootstrap_ci  # noqa: F401  (re-exported for backward compatibility)
```

Make sure the import lives next to the other top-level imports. Do NOT remove `permutation_test`.

- [ ] **Step 3: Run perturbation stats; verify byte-identical output (when baseline exists)**

```bash
uv run python analysis/eva-bench-stats/run_stats.py
diff /tmp/before_pooled.csv output_processed/eva-bench-stats/perturbations/results_pooled.csv
diff /tmp/before_perdomain.csv output_processed/eva-bench-stats/perturbations/results_per_domain.csv
```

Expected: no diff. If diff appears, investigate seed handling — both functions must use `np.random.default_rng(seed)` with identical `(n_boot, n)` shape.

- [ ] **Step 4: Commit**

```bash
git add analysis/eva-bench-stats/stats_perturbations.py
git commit -m "refactor(eva-bench-stats): consume bootstrap_ci from stats_utils

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Create `pull_clean_data.py` (remote puller)

**Files:**
- Create: `local/eva-bench-stats/pull_clean_data.py`

- [ ] **Step 1: Author the puller (sibling of `pull_perturbation_data.py`)**

Create `local/eva-bench-stats/pull_clean_data.py`:

```python
#!/usr/bin/env python3
"""Extract per-trial metric scores from EVA-Bench CLEAN runs for CI analysis.

Walks --runs-dir, keeps runs whose perturbation_category == "clean", dedupes
by latest per (canonical_alias, domain), emits all clean runs found. Manifest
flags runs below --success-threshold but does NOT drop them; downstream
data_CIs.py decides whether to filter.

Outputs to --output-dir (defaults to eva_bench_stats/eva_clean_data_<ts>/):
  trial_scores.csv      — long-form: run_id, system_alias, system_type, domain,
                           perturbation_category, scenario_id, trial, metric, value
  run_manifest.csv      — one row per run: coverage, eligibility, success_rate
  coverage_summary.csv  — per (alias, domain) presence flag
  <output-dir>.zip      — zip of the above
"""
# REMOTE WORKFLOW (snow.core_llm.voice_agent on Toolkit):
#   Push:  eai data push snow.core_llm.voice_agent ./local/eva-bench-stats/pull_clean_data.py:eva_bench_stats/pull_clean_data.py
#   Run:   python3 eva_bench_stats/pull_clean_data.py
#   List:  eai data content ls snow.core_llm.voice_agent --path eva_bench_stats/ --fields name --no-header --format csv
#   Pull:  eai data pull snow.core_llm.voice_agent eva_bench_stats/eva_clean_data_YYYYMMDD_HHMMSS.zip:eva_clean_data_YYYYMMDD_HHMMSS.zip output/eva-bench-stats/
#   Unzip: unzip output/eva-bench-stats/eva_clean_data_YYYYMMDD_HHMMSS.zip -d output/eva-bench-stats/

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

OUTPUT_SUBDIR = Path("eva_bench_stats")

INDIVIDUAL_METRICS = [
    "task_completion",
    "faithfulness",
    "agent_speech_fidelity",
    "conversation_progression",
    "turn_taking",
    "conciseness",
]
COMPOSITE_METRICS = [
    "EVA-A_mean",
    "EVA-X_mean",
    "EVA-overall_mean",
    "EVA-A_pass",
    "EVA-X_pass",
]
ALL_METRICS = INDIVIDUAL_METRICS + COMPOSITE_METRICS

DOMAINS = ["itsm", "medical_hr", "airline"]
RUN_DIR_REGEX = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2}\.\d+)_(.+)$")
PERT_SUFFIX_TOKENS = ("-accent", "-background-noise", "-background_noise", "-both")
ALIAS_REMAP: dict[str, str] = {
    "gemini-live": "gemini-3.1-flash-live-preview",
}
SYSTEM_TYPE_OVERRIDES: dict[str, str] = {
    "ultravox": "hybrid",
    "gemini-3-flash-preview + gemini-3.1-flash-tts-preview": "hybrid",
}
EXPECTED_K_CLEAN = 5


def parse_run_timestamp(folder_name: str) -> datetime | None:
    m = RUN_DIR_REGEX.match(folder_name)
    if not m:
        return None
    date_part, time_part, _ = m.groups()
    try:
        return datetime.strptime(f"{date_part}_{time_part}", "%Y-%m-%d_%H-%M-%S.%f")
    except ValueError:
        return None


def perturbation_category(pert: dict | None) -> str | None:
    if pert is None:
        return "clean"
    accent = bool(pert.get("accent"))
    bg = bool(pert.get("background_noise"))
    other = bool(pert.get("behavior")) or bool(pert.get("connection_degradation"))
    if other or accent or bg:
        return None
    return "clean"


def detect_system_type(model_cfg: dict) -> str:
    if isinstance(model_cfg, dict):
        s2s = model_cfg.get("s2s")
        if s2s is not None and s2s != "elevenlabs":
            return "s2s"
        if model_cfg.get("realtime_model"):
            return "s2s"
    return "cascade"


def derive_alias(pipeline_parts: dict, folder_name: str) -> str:
    if isinstance(pipeline_parts, dict) and pipeline_parts:
        if "s2s" in pipeline_parts and pipeline_parts.get("s2s"):
            return str(pipeline_parts["s2s"])
        components = []
        for key in ("stt", "audio_llm", "llm", "tts"):
            v = pipeline_parts.get(key)
            if v:
                components.append(str(v))
        if components:
            return " + ".join(components)
    m = RUN_DIR_REGEX.match(folder_name)
    suffix = m.group(3) if m else folder_name
    for tok in ("accent", "background_noise", "background-noise", "both"):
        suffix = suffix.replace(f"_{tok}", "").replace(f"-{tok}", "")
    return suffix.rstrip("_-")


def canonical_alias(alias: str) -> str:
    s = alias
    if " + " in s:
        head, _, tail = s.rpartition(" + ")
        for tok in PERT_SUFFIX_TOKENS:
            if tail.endswith(tok):
                tail = tail[: -len(tok)]
                break
        s = f"{head} + {tail}".rstrip("-_")
    else:
        for tok in PERT_SUFFIX_TOKENS:
            if s.endswith(tok):
                s = s[: -len(tok)].rstrip("-_")
                break
    return ALIAS_REMAP.get(s, s)


def discover_runs(runs_dir: Path) -> tuple[list[dict], list[dict]]:
    """Walk runs_dir, return (clean_runs, excluded_records).

    Note: success threshold is NOT applied here — every clean run is included;
    success_rate is recorded for downstream filtering / manifest flagging.
    """
    included: list[dict] = []
    excluded: list[dict] = []

    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir() or not RUN_DIR_REGEX.match(entry.name):
            continue
        cfg_path = entry / "config.json"
        eval_path = entry / "evaluation_summary.json"
        if not cfg_path.exists():
            excluded.append({"run_id": entry.name, "reason": "missing config.json"})
            continue
        try:
            with cfg_path.open() as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            excluded.append({"run_id": entry.name, "reason": f"config parse error: {e}"})
            continue

        domain = cfg.get("domain")
        if domain not in DOMAINS:
            excluded.append({"run_id": entry.name, "reason": f"unknown domain: {domain}"})
            continue

        pert_cat = perturbation_category(cfg.get("perturbation"))
        if pert_cat != "clean":
            excluded.append({"run_id": entry.name, "reason": "not a clean run"})
            continue

        sys_type = detect_system_type(cfg.get("model") or {})
        raw_alias = derive_alias(cfg.get("pipeline_parts") or {}, entry.name)
        canon = canonical_alias(raw_alias)
        if canon in SYSTEM_TYPE_OVERRIDES:
            sys_type = SYSTEM_TYPE_OVERRIDES[canon]
        ts = parse_run_timestamp(entry.name)

        n_records: int | None = None
        n_expected: int | None = None
        success_rate: float | None = None
        if eval_path.exists():
            try:
                with eval_path.open() as f:
                    ev = json.load(f)
                sim = ev.get("simulation", {}) or {}
                n_records = sim.get("successful_records") or 0
                n_expected = sim.get("total_records") or 0
                success_rate = (n_records / n_expected) if n_expected else 0.0
            except (OSError, json.JSONDecodeError):
                pass

        included.append({
            "run_id": entry.name,
            "dir": entry,
            "canonical_alias": canon,
            "system_type": sys_type,
            "domain": domain,
            "perturbation_category": pert_cat,
            "timestamp": ts,
            "n_records_found": n_records,
            "n_expected": n_expected,
            "success_rate": success_rate,
        })
    return included, excluded


def dedupe_latest(runs: list[dict]) -> dict[tuple[str, str], dict]:
    """Keep the best run per (canonical_alias, domain) — clean only, so no pert dimension."""

    def sort_key(r: dict) -> tuple:
        sr = r.get("success_rate")
        return (
            sr is not None,
            sr if sr is not None else -1.0,
            r.get("timestamp") or datetime.min,
        )

    by_key: dict[tuple[str, str], dict] = {}
    for r in runs:
        k = (r["canonical_alias"], r["domain"])
        prev = by_key.get(k)
        if prev is None or sort_key(r) > sort_key(prev):
            by_key[k] = r
    return by_key


def extract_trial_scores(run: dict) -> list[dict]:
    run_dir: Path = run["dir"]
    records_dir = run_dir / "records"
    if not records_dir.exists():
        return []
    base = {
        "run_id": run["run_id"],
        "system_alias": run["canonical_alias"],
        "system_type": run["system_type"],
        "domain": run["domain"],
        "perturbation_category": run["perturbation_category"],
    }
    rows: list[dict] = []
    for sc_dir in sorted(records_dir.iterdir()):
        if not sc_dir.is_dir():
            continue
        scenario_id = sc_dir.name
        for trial_dir in sorted(sc_dir.iterdir()):
            if not trial_dir.is_dir() or not re.fullmatch(r"trial_\d+", trial_dir.name):
                continue
            trial = int(trial_dir.name.split("_")[1])
            metrics_path = trial_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                with metrics_path.open() as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            per_metric = d.get("metrics", {}) or {}
            agg = d.get("aggregate_metrics", {}) or {}

            for metric in INDIVIDUAL_METRICS:
                entry = per_metric.get(metric)
                if not isinstance(entry, dict) or entry.get("error"):
                    continue
                v = entry.get("normalized_score")
                if v is None:
                    continue
                rows.append({**base, "scenario_id": scenario_id, "trial": trial, "metric": metric, "value": float(v)})
            for metric in COMPOSITE_METRICS:
                v = agg.get(metric)
                if v is None:
                    continue
                rows.append({**base, "scenario_id": scenario_id, "trial": trial, "metric": metric, "value": float(v)})
    return rows


TRIAL_SCORES_FIELDS = [
    "run_id", "system_alias", "system_type", "domain", "perturbation_category",
    "scenario_id", "trial", "metric", "value",
]
MANIFEST_FIELDS = [
    "run_id", "canonical_alias", "system_type", "domain",
    "n_records_found", "n_expected", "success_rate", "below_threshold", "included", "reason",
]


def write_trial_scores(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_SCORES_FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_run_manifest(
    path: Path,
    included: list[dict],
    deduped_out: list[dict],
    excluded: list[dict],
    threshold: float,
) -> None:
    rows: list[dict] = []
    for r in included:
        sr = r.get("success_rate")
        below = (sr is not None) and (sr < threshold)
        rows.append({
            "run_id": r["run_id"],
            "canonical_alias": r["canonical_alias"],
            "system_type": r["system_type"],
            "domain": r["domain"],
            "n_records_found": r.get("n_records_found", ""),
            "n_expected": r.get("n_expected", ""),
            "success_rate": f"{sr:.4f}" if sr is not None else "",
            "below_threshold": below,
            "included": True,
            "reason": "below threshold (still included)" if below else "",
        })
    for r in deduped_out:
        rows.append({
            "run_id": r["run_id"],
            "canonical_alias": r.get("canonical_alias", ""),
            "system_type": r.get("system_type", ""),
            "domain": r.get("domain", ""),
            "n_records_found": r.get("n_records_found", ""),
            "n_expected": r.get("n_expected", ""),
            "success_rate": f"{r['success_rate']:.4f}" if r.get("success_rate") is not None else "",
            "below_threshold": False,
            "included": False,
            "reason": "duplicate run (older timestamp; newer kept)",
        })
    for r in excluded:
        rows.append({
            "run_id": r["run_id"],
            "canonical_alias": r.get("canonical_alias", ""),
            "system_type": "",
            "domain": "",
            "n_records_found": "",
            "n_expected": "",
            "success_rate": "",
            "below_threshold": False,
            "included": False,
            "reason": r.get("reason", ""),
        })
    rows.sort(key=lambda r: (r["canonical_alias"].lower(), r["domain"], str(r["run_id"])))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_coverage_summary(path: Path, included: list[dict]) -> None:
    by: dict[tuple[str, str], dict] = {}
    for r in included:
        k = (r["canonical_alias"], r["domain"])
        by[k] = {
            "canonical_alias": r["canonical_alias"],
            "system_type": r["system_type"],
            "domain": r["domain"],
            "clean": True,
        }
    fields = ["canonical_alias", "system_type", "domain", "clean"]
    rows = sorted(by.values(), key=lambda r: (r["canonical_alias"].lower(), r["domain"]))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def zip_folder(folder: Path) -> Path:
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for child in folder.rglob("*"):
            if child.is_file():
                zf.write(child, arcname=child.relative_to(folder.parent))
    return zip_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="/mnt/voice_agent", type=Path)
    p.add_argument("--output-dir", default=None, type=Path)
    p.add_argument(
        "--success-threshold", default=1.0, type=float,
        help="Runs below this success rate are flagged in the manifest but still included.",
    )
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = OUTPUT_SUBDIR / f"eva_clean_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Scanning {args.runs_dir} ...")
    included_raw, excluded = discover_runs(args.runs_dir)
    print(f"  {len(included_raw)} clean runs found, {len(excluded)} excluded (non-clean / unparseable)")

    by_key = dedupe_latest(included_raw)
    included_ids = {r["run_id"] for r in by_key.values()}
    deduped_out = [r for r in included_raw if r["run_id"] not in included_ids]
    if deduped_out:
        print(f"  {len(deduped_out)} older duplicate runs set aside (newer kept)")

    eligible = list(by_key.values())
    by_alias: dict[str, set[str]] = defaultdict(set)
    for r in eligible:
        by_alias[r["canonical_alias"]].add(r["domain"])
    print(f"\nClean coverage ({len(by_alias)} unique aliases):")
    for alias in sorted(by_alias):
        print(f"  {alias}: {', '.join(sorted(by_alias[alias]))}")

    print(f"\nExtracting trial scores from {len(eligible)} runs ...")
    all_rows: list[dict] = []
    for i, run in enumerate(eligible, 1):
        rows = extract_trial_scores(run)
        all_rows.extend(rows)
        print(f"  [{i}/{len(eligible)}] {run['run_id']}: {len(rows)} score rows")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_trial_scores(out_dir / "trial_scores.csv", all_rows)
    write_run_manifest(out_dir / "run_manifest.csv", eligible, deduped_out, excluded, args.success_threshold)
    write_coverage_summary(out_dir / "coverage_summary.csv", eligible)
    zip_path = zip_folder(out_dir)

    print(f"\nOutput:")
    print(f"  trial_scores.csv     — {len(all_rows):,} rows")
    print(f"  run_manifest.csv     — {len(eligible) + len(deduped_out) + len(excluded)} runs")
    print(f"  coverage_summary.csv")
    print(f"  Folder: {out_dir}")
    print(f"  Zip:    {zip_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Local syntax check**

Run: `uv run python -c "import ast; ast.parse(open('local/eva-bench-stats/pull_clean_data.py').read())"`
Expected: no output (success).

- [ ] **Step 3: Commit**

```bash
git add local/eva-bench-stats/pull_clean_data.py
git commit -m "feat(eva-bench-stats): add pull_clean_data.py for clean trial scores

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Author `CIs_config.yaml`

**Files:**
- Create: `local/eva-bench-stats/CIs_config.yaml`

- [ ] **Step 1: Write the config**

Create `local/eva-bench-stats/CIs_config.yaml`:

```yaml
# Confidence Interval analysis config.
# Reads clean trial scores produced by local/eva-bench-stats/pull_clean_data.py.

trial_scores_dir: output/eva-bench-stats     # auto-pick latest "eva_clean_data_*" subfolder
# trial_scores_path: output/eva-bench-stats/eva_clean_data_YYYYMMDD_HHMMSS/trial_scores.csv  # alt
output_dir: output_processed/eva-bench-stats/CIs

random_seed: 42
n_bootstrap: 1000           # may be raised to 2000 from stability check
alpha: 0.05
stability_threshold: 0.002
# stability_check_model: ""   # optional; defaults to first model in models:

expected_domains: [airline, itsm, medical_hr]
expected_scenarios:
  airline: 50
  itsm: 80
  medical_hr: 83
expected_k: 5

metrics:
  - task_completion
  - faithfulness
  - agent_speech_fidelity
  - conversation_progression
  - turn_taking
  - conciseness
  - EVA-A_mean
  - EVA-X_mean
  - EVA-A_pass        # pass@1 for EVA-A
  - EVA-X_pass        # pass@1 for EVA-X

# Fill in models as cleared runs become available. Display label → alias mapping.
models: {}
```

- [ ] **Step 2: Verify YAML parses**

Run: `uv run python -c "import yaml; yaml.safe_load(open('local/eva-bench-stats/CIs_config.yaml'))"`
Expected: no output.

- [ ] **Step 3: Note: this file is in `local/` (gitignored) — do NOT commit it.**

Verify it is ignored:
`git check-ignore local/eva-bench-stats/CIs_config.yaml`
Expected: prints the path (means ignored).

If not ignored, do not commit it; ask the user how to handle (the existing perturbations/variance configs are also gitignored per README).

---

## Task 5: Implement `data_CIs.py`

**Files:**
- Create (replacing placeholder): `analysis/eva-bench-stats/data_CIs.py`

- [ ] **Step 1: Replace placeholder with full implementation**

Overwrite `analysis/eva-bench-stats/data_CIs.py`:

```python
"""Process clean run trial-score data into scenario-level means for CI analysis.

Reads trial_scores.csv produced by local/eva-bench-stats/pull_clean_data.py,
filters to configured (alias, perturbation_category=="clean") combinations,
checks completeness against expected scenario counts and trial counts, and
writes scenario-level means.

Run from project root:
    uv run python analysis/eva-bench-stats/data_CIs.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "eva-bench-stats" / "CIs_config.yaml"


def load_trial_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trial"] = df["trial"].astype(int)
    df["value"] = df["value"].astype(float)
    return df


def check_completeness(
    df: pd.DataFrame,
    alias: str,
    expected_scenarios: dict[str, int],
    expected_k: int,
    metrics: list[str],
) -> tuple[bool, list[dict]]:
    """Validate per-domain coverage for one model.

    Returns (model_complete, per_domain_rows). per_domain_rows has one entry per
    expected domain with keys: alias, domain, n_scenarios, n_expected,
    n_scenarios_with_wrong_trial_count, n_metrics_missing, complete, issues (str).
    """
    rows: list[dict] = []
    model_df = df[df["system_alias"] == alias]
    sentinel = "task_completion" if "task_completion" in metrics else metrics[0]

    for domain, n_expected in expected_scenarios.items():
        d = model_df[(model_df["domain"] == domain) & (model_df["metric"] == sentinel)]
        n_scenarios = d["scenario_id"].nunique()
        if n_scenarios > 0:
            trial_counts = d.groupby("scenario_id")["trial"].nunique()
            bad_trials = int((trial_counts < expected_k).sum())
        else:
            bad_trials = 0

        # Count metric coverage gaps among the configured metrics
        configured_present = (
            model_df[(model_df["domain"] == domain) & (model_df["metric"].isin(metrics))]
            .groupby("metric")["scenario_id"]
            .nunique()
        )
        n_missing_metrics = sum(1 for m in metrics if configured_present.get(m, 0) == 0)

        ok = (n_scenarios == n_expected) and (bad_trials == 0) and (n_missing_metrics == 0)
        issues_parts: list[str] = []
        if n_scenarios != n_expected:
            issues_parts.append(f"{n_scenarios}/{n_expected} scenarios")
        if bad_trials > 0:
            issues_parts.append(f"{bad_trials} scenarios with <{expected_k} trials")
        if n_missing_metrics > 0:
            issues_parts.append(f"{n_missing_metrics} metrics absent")

        rows.append({
            "alias": alias,
            "domain": domain,
            "n_scenarios": n_scenarios,
            "n_expected": n_expected,
            "n_scenarios_with_wrong_trial_count": bad_trials,
            "n_metrics_missing": n_missing_metrics,
            "complete": ok,
            "issues": "; ".join(issues_parts),
        })

    model_complete = all(r["complete"] for r in rows)
    return model_complete, rows


def compute_scenario_means(
    df: pd.DataFrame,
    alias: str,
    metrics: list[str],
) -> pd.DataFrame:
    """Mean over trials per (system_alias, domain, scenario_id, metric).

    For binary pass metrics (EVA-A_pass / EVA-X_pass) this mean is the
    scenario-level pass proportion = pass@1.
    """
    sub = df[(df["system_alias"] == alias) & (df["metric"].isin(metrics))]
    if sub.empty:
        return pd.DataFrame(columns=["system_alias", "domain", "scenario_id", "metric", "scenario_mean"])
    keys = ["system_alias", "domain", "scenario_id", "metric"]
    return sub.groupby(keys, sort=False)["value"].mean().reset_index().rename(columns={"value": "scenario_mean"})


def _resolve_trial_scores_path(config: dict, project_root: Path) -> Path:
    if "trial_scores_dir" in config:
        data_dir = project_root / config["trial_scores_dir"]
        subdirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("eva_clean_data_"))
        if not subdirs:
            raise FileNotFoundError(f"No eva_clean_data_* subdirectories in {data_dir}")
        path = subdirs[-1] / "trial_scores.csv"
        print(f"Auto-selected most recent clean data folder: {subdirs[-1].name}")
        return path
    return project_root / config["trial_scores_path"]


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    trial_scores_path = _resolve_trial_scores_path(config, project_root)
    output_dir = project_root / config["output_dir"]
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[str] = config["metrics"]
    expected_scenarios: dict[str, int] = config["expected_scenarios"]
    expected_k: int = config["expected_k"]

    print(f"Loading trial scores from {trial_scores_path} ...")
    trial_scores = load_trial_scores(trial_scores_path)
    print(f"  {len(trial_scores):,} rows loaded")

    # Restrict to clean only (puller already filters; double-defense here in case path is overridden).
    trial_scores = trial_scores[trial_scores["perturbation_category"] == "clean"]

    all_means: list[pd.DataFrame] = []
    completeness_rows: list[dict] = []

    for model_label, model_cfg in (config.get("models") or {}).items():
        alias: str = model_cfg["alias"]
        complete, dom_rows = check_completeness(
            trial_scores, alias, expected_scenarios, expected_k, metrics,
        )
        for r in dom_rows:
            completeness_rows.append({"model_label": model_label, "model_complete": complete, **r})
        status = "COMPLETE" if complete else "INCOMPLETE"
        print(f"  [{status}] {model_label}")
        if not complete:
            for r in dom_rows:
                if not r["complete"]:
                    print(f"    - {r['domain']}: {r['issues']}")

        if complete:
            means = compute_scenario_means(trial_scores, alias, metrics)
            means.insert(0, "model_label", model_label)
            all_means.append(means)
        else:
            # Still include partial scenario means for the domains that ARE complete.
            partial_alias_df = trial_scores[trial_scores["system_alias"] == alias]
            complete_domains = [r["domain"] for r in dom_rows if r["complete"]]
            if complete_domains:
                partial = partial_alias_df[partial_alias_df["domain"].isin(complete_domains)]
                means = compute_scenario_means(partial, alias, metrics)
                if not means.empty:
                    means.insert(0, "model_label", model_label)
                    all_means.append(means)

    means_df = pd.concat(all_means, ignore_index=True) if all_means else pd.DataFrame(
        columns=["model_label", "system_alias", "domain", "scenario_id", "metric", "scenario_mean"]
    )
    completeness_df = pd.DataFrame(completeness_rows)

    means_path = data_dir / "scenario_means.csv"
    report_path = data_dir / "completeness_report.csv"
    means_df.to_csv(means_path, index=False)
    completeness_df.to_csv(report_path, index=False)

    print(f"\nWrote {len(means_df):,} scenario-mean rows → {means_path}")
    print(f"Wrote completeness report → {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run: `uv run python -c "import ast; ast.parse(open('analysis/eva-bench-stats/data_CIs.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add analysis/eva-bench-stats/data_CIs.py
git commit -m "feat(eva-bench-stats): implement data_CIs scenario-mean pipeline

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Implement `stats_CIs.py`

**Files:**
- Create (replacing placeholder): `analysis/eva-bench-stats/stats_CIs.py`

- [ ] **Step 1: Replace placeholder with full implementation**

Overwrite `analysis/eva-bench-stats/stats_CIs.py`:

```python
"""Bootstrapped confidence intervals on scenario-level means for clean runs.

Pipeline:
  1. stability_check       — bootstrap @ 1000 vs 2000 on a representative model;
                             choose n_boot for full run.
  2. compute_domain_ci     — per (model, metric, domain): point + percentile CI
                             from scenario-level bootstrap.
  3. compute_pooled_ci     — equal-weighted pooled CI: mean of 3 domain points,
                             percentiles of elementwise mean of 3 domain bootstraps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from stats_utils import bootstrap_resample  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "eva-bench-stats" / "CIs_config.yaml"


def _cell_seed(base_seed: int, *parts: str) -> int:
    return (base_seed + hash(":".join(parts))) % (2**31)


def compute_domain_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float, np.ndarray]:
    """(point_estimate, ci_lower, ci_upper, boot_dist) for one model × metric × domain."""
    values = np.asarray(values, dtype=float)
    point = float(values.mean()) if len(values) else float("nan")
    boot = bootstrap_resample(values, n_boot=n_boot, seed=seed)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi, boot


def compute_pooled_ci(
    domain_points: list[float],
    domain_dists: list[np.ndarray],
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Equal-weighted pooled CI.

    Pooled point: mean of the 3 domain point estimates.
    Pooled CI:   percentile of the elementwise mean of the 3 bootstrap distributions.
    """
    point = float(np.mean(domain_points))
    stacked = np.vstack(domain_dists)             # shape (n_domains, n_boot)
    pooled_dist = stacked.mean(axis=0)            # shape (n_boot,)
    lo = float(np.percentile(pooled_dist, 100 * alpha / 2))
    hi = float(np.percentile(pooled_dist, 100 * (1 - alpha / 2)))
    return point, lo, hi


def stability_check(
    scenario_means_df: pd.DataFrame,
    model_label: str,
    metrics: list[str],
    expected_domains: list[str],
    base_seed: int,
    alpha: float,
    threshold: float,
    n_low: int = 1000,
    n_high: int = 2000,
) -> tuple[int, pd.DataFrame]:
    """Run bootstrap @ n_low and n_high for one model, all metrics × all domains.

    Returns (chosen_n_boot, log_df). log_df has one row per metric with
    max |Δ CI bound| across domains and a within_tolerance flag.
    """
    sub = scenario_means_df[scenario_means_df["model_label"] == model_label]
    rows: list[dict] = []
    overall_max = 0.0

    for metric in metrics:
        diffs: list[float] = []
        for domain in expected_domains:
            cell = sub[(sub["metric"] == metric) & (sub["domain"] == domain)]
            if cell.empty:
                continue
            x = cell["scenario_mean"].to_numpy()
            seed = _cell_seed(base_seed, "stability", model_label, metric, domain)
            _, lo_a, hi_a, _ = compute_domain_ci(x, n_boot=n_low, seed=seed, alpha=alpha)
            _, lo_b, hi_b, _ = compute_domain_ci(x, n_boot=n_high, seed=seed, alpha=alpha)
            diffs.append(max(abs(lo_a - lo_b), abs(hi_a - hi_b)))
        max_diff = max(diffs) if diffs else 0.0
        overall_max = max(overall_max, max_diff)
        rows.append({
            "metric": metric,
            "n_boot_ref": n_low,
            "n_boot_test": n_high,
            "max_abs_ci_diff": max_diff,
            "within_tolerance": max_diff <= threshold,
        })

    chosen = n_low if overall_max <= threshold else n_high
    log_df = pd.DataFrame(rows)
    log_df.attrs["overall_max"] = overall_max
    log_df.attrs["chosen_n_boot"] = chosen
    return chosen, log_df


def run_analysis(
    scenario_means_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run stability check + per-domain CIs + pooled CIs.

    Returns (per_domain_df, pooled_df, stability_log_df).
    """
    metrics: list[str] = config["metrics"]
    expected_domains: list[str] = config["expected_domains"]
    base_seed: int = config["random_seed"]
    alpha: float = config["alpha"]
    n_boot_default: int = config["n_bootstrap"]
    threshold: float = config["stability_threshold"]

    # Choose stability check model
    models = scenario_means_df["model_label"].drop_duplicates().tolist()
    if not models:
        empty = pd.DataFrame(columns=["model_label", "metric", "domain", "n", "point_estimate", "ci_lower", "ci_upper"])
        return empty, empty.copy(), pd.DataFrame()
    stability_model = config.get("stability_check_model") or models[0]
    if stability_model not in models:
        print(f"  [stability] configured model '{stability_model}' not in data; falling back to '{models[0]}'")
        stability_model = models[0]

    print(f"  [stability] running 1000 vs 2000 on '{stability_model}' ...")
    chosen_n_boot, stability_log = stability_check(
        scenario_means_df, stability_model, metrics, expected_domains,
        base_seed=base_seed, alpha=alpha, threshold=threshold,
    )
    if chosen_n_boot != n_boot_default:
        print(f"  [stability] max Δ = {stability_log.attrs['overall_max']:.5f} > {threshold} → using n_boot = {chosen_n_boot}")
    else:
        print(f"  [stability] max Δ = {stability_log.attrs['overall_max']:.5f} ≤ {threshold} → using n_boot = {chosen_n_boot}")

    per_domain_rows: list[dict] = []
    pooled_rows: list[dict] = []

    for model_label in models:
        model_df = scenario_means_df[scenario_means_df["model_label"] == model_label]
        for metric in metrics:
            domain_dists: list[np.ndarray] = []
            domain_points: list[float] = []
            domains_present: list[str] = []

            for domain in expected_domains:
                cell = model_df[(model_df["metric"] == metric) & (model_df["domain"] == domain)]
                if cell.empty:
                    continue
                x = cell["scenario_mean"].to_numpy()
                seed = _cell_seed(base_seed, "main", model_label, metric, domain)
                point, lo, hi, boot = compute_domain_ci(x, n_boot=chosen_n_boot, seed=seed, alpha=alpha)
                per_domain_rows.append({
                    "model_label": model_label,
                    "metric": metric,
                    "domain": domain,
                    "n": len(x),
                    "point_estimate": point,
                    "ci_lower": lo,
                    "ci_upper": hi,
                })
                domain_dists.append(boot)
                domain_points.append(point)
                domains_present.append(domain)

            if len(domain_dists) == len(expected_domains):
                p_point, p_lo, p_hi = compute_pooled_ci(domain_points, domain_dists, alpha=alpha)
                pooled_rows.append({
                    "model_label": model_label,
                    "metric": metric,
                    "domain": "pooled",
                    "n": "pooled",
                    "point_estimate": p_point,
                    "ci_lower": p_lo,
                    "ci_upper": p_hi,
                })
            else:
                missing = [d for d in expected_domains if d not in domains_present]
                print(f"  [skip pooled] {model_label} × {metric}: missing {missing}")

    per_domain_df = pd.DataFrame(per_domain_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    return per_domain_df, pooled_df, stability_log


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    output_dir = project_root / config["output_dir"]
    data_dir = output_dir / "data"
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    means_path = data_dir / "scenario_means.csv"
    if not means_path.exists():
        raise FileNotFoundError(f"scenario_means.csv not found at {means_path}. Run data_CIs.py first.")

    print(f"Loading scenario means from {means_path} ...")
    means_df = pd.read_csv(means_path)
    print(f"  {len(means_df):,} rows loaded")

    per_domain_df, pooled_df, stability_log = run_analysis(means_df, config)

    per_domain_df.to_csv(stats_dir / "results_per_domain.csv", index=False)
    pooled_df.to_csv(stats_dir / "results_pooled.csv", index=False)
    stability_log.to_csv(stats_dir / "stability_log.csv", index=False)

    print(f"\nWrote {len(per_domain_df):,} per-domain rows → {stats_dir / 'results_per_domain.csv'}")
    print(f"Wrote {len(pooled_df):,} pooled rows      → {stats_dir / 'results_pooled.csv'}")
    print(f"Wrote stability log                       → {stats_dir / 'stability_log.csv'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run: `uv run python -c "import ast; ast.parse(open('analysis/eva-bench-stats/stats_CIs.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add analysis/eva-bench-stats/stats_CIs.py
git commit -m "feat(eva-bench-stats): implement stats_CIs (stability + per-domain + pooled)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Implement `plots_CIs.py`

**Files:**
- Create (replacing placeholder): `analysis/eva-bench-stats/plots_CIs.py`

- [ ] **Step 1: Replace placeholder with full implementation**

Overwrite `analysis/eva-bench-stats/plots_CIs.py`:

```python
"""Plots and display tables for CI analysis."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PRETTY_METRIC_LABELS: dict[str, str] = {
    "task_completion": "Task Completion",
    "faithfulness": "Faithfulness",
    "agent_speech_fidelity": "Agent Speech Fidelity",
    "conversation_progression": "Conversation Progression",
    "turn_taking": "Turn Taking",
    "conciseness": "Conciseness",
    "EVA-A_mean": "EVA-A (mean)",
    "EVA-X_mean": "EVA-X (mean)",
    "EVA-A_pass": "EVA-A (pass@1)",
    "EVA-X_pass": "EVA-X (pass@1)",
}

DEFAULT_METRIC_ORDER: list[str] = [
    "EVA-A_pass",
    "EVA-X_pass",
    "task_completion",
    "faithfulness",
    "agent_speech_fidelity",
    "conversation_progression",
    "turn_taking",
    "conciseness",
    "EVA-A_mean",
    "EVA-X_mean",
]


def metric_label(metric: str) -> str:
    return PRETTY_METRIC_LABELS.get(metric, metric)


def order_metrics(metrics: list[str]) -> list[str]:
    rank = {m: i for i, m in enumerate(DEFAULT_METRIC_ORDER)}
    return sorted(metrics, key=lambda m: rank.get(m, 1_000))


def forest_plot(
    per_domain_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    metric: str,
    domains: list[str],
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    """Forest plot for one metric: 4 columns (3 domains + pooled), rows = models."""
    facets = list(domains) + ["pooled"]
    fig = make_subplots(rows=1, cols=len(facets), shared_yaxes=True, subplot_titles=facets)

    combined = pd.concat([
        per_domain_df[per_domain_df["metric"] == metric],
        pooled_df[pooled_df["metric"] == metric],
    ], ignore_index=True)

    if combined.empty:
        fig.update_layout(title=f"{metric_label(metric)} (no data)")
        return fig

    models = list(dict.fromkeys(combined["model_label"]))
    color_map = color_map or {}

    for col, facet in enumerate(facets, start=1):
        sub = combined[combined["domain"] == facet]
        for model in models:
            row = sub[sub["model_label"] == model]
            if row.empty:
                continue
            point = float(row["point_estimate"].iloc[0])
            lo = float(row["ci_lower"].iloc[0])
            hi = float(row["ci_upper"].iloc[0])
            color = color_map.get(model, "#3B82F6")
            fig.add_trace(
                go.Scatter(
                    x=[point],
                    y=[model],
                    mode="markers",
                    marker={"color": color, "size": 9},
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": [hi - point],
                        "arrayminus": [point - lo],
                        "color": color,
                        "thickness": 1.5,
                        "width": 4,
                    },
                    name=model,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        title=f"{metric_label(metric)} — 95% bootstrap CI",
        height=max(280, 28 * max(len(models), 4) + 80),
        margin={"l": 200, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "v"},
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def paper_summary_table(
    per_domain_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    metrics: tuple[str, ...] = ("EVA-A_mean", "EVA-X_mean"),
    domains: tuple[str, ...] = ("airline", "itsm", "medical_hr"),
) -> pd.DataFrame:
    """Wide table: one row per (model, metric); columns = each domain + pooled.

    Each cell is "{point:.3f} [{lo:.3f}, {hi:.3f}]".
    """
    parts = []
    rows = pd.concat([per_domain_df, pooled_df], ignore_index=True)
    for metric in metrics:
        for model, mdf in rows[rows["metric"] == metric].groupby("model_label"):
            row: dict[str, str] = {"model_label": model, "metric": metric_label(metric)}
            for facet in list(domains) + ["pooled"]:
                cell = mdf[mdf["domain"] == facet]
                if cell.empty:
                    row[facet] = ""
                    continue
                p = float(cell["point_estimate"].iloc[0])
                lo = float(cell["ci_lower"].iloc[0])
                hi = float(cell["ci_upper"].iloc[0])
                row[facet] = f"{p:.3f} [{lo:.3f}, {hi:.3f}]"
            parts.append(row)
    return pd.DataFrame(parts)
```

- [ ] **Step 2: Syntax check**

Run: `uv run python -c "import ast; ast.parse(open('analysis/eva-bench-stats/plots_CIs.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add analysis/eva-bench-stats/plots_CIs.py
git commit -m "feat(eva-bench-stats): implement plots_CIs forest plot and summary table

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Wire CIs into `run_data.py` and `run_stats.py`

**Files:**
- Modify: `analysis/eva-bench-stats/run_data.py`
- Modify: `analysis/eva-bench-stats/run_stats.py`

- [ ] **Step 1: Read `run_stats.py` to mirror its structure**

```bash
cat analysis/eva-bench-stats/run_stats.py
```

Expected: same shape as `run_data.py` (imports per-analysis `main` functions and calls them in order). Use the same shape for the new hooks.

- [ ] **Step 2: Add `_CIs_data()` to `run_data.py`**

In `analysis/eva-bench-stats/run_data.py`, replace the file with:

```python
"""Run all data preprocessing scripts for eva-bench-stats.

Run from project root:
    uv run python analysis/eva-bench-stats/run_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_perturbations import main as _perturbations_data
from data_variance import main as _variance_data
from data_CIs import main as _CIs_data


def main() -> None:
    print("=== Perturbations: data ===")
    _perturbations_data()
    print()

    print("=== Variance: data ===")
    _variance_data()
    print()

    print("=== CIs: data ===")
    try:
        _CIs_data()
    except FileNotFoundError as e:
        print(f"  [CIs] skipped: {e}")
    print()


if __name__ == "__main__":
    main()
```

The try/except around `_CIs_data()` is intentional: if the user has not yet pulled clean data, the rest of the pipeline must still run.

- [ ] **Step 3: Add `_CIs_stats()` to `run_stats.py`**

Read `run_stats.py` first, then add the analogous block. The new wiring is the same shape — import `from stats_CIs import main as _CIs_stats`, then call it in the `main()` function under a new `=== CIs: stats ===` header, wrapped in `try/except FileNotFoundError` so a missing `scenario_means.csv` does not break the perturbations / variance pipelines.

- [ ] **Step 4: Verify both scripts execute end-to-end (skipping CIs gracefully if no data)**

```bash
uv run python analysis/eva-bench-stats/run_data.py
uv run python analysis/eva-bench-stats/run_stats.py
```

Expected: both exit 0, perturbation and variance outputs unchanged, CIs section either runs or prints "skipped: ...".

- [ ] **Step 5: Commit**

```bash
git add analysis/eva-bench-stats/run_data.py analysis/eva-bench-stats/run_stats.py
git commit -m "feat(eva-bench-stats): wire CIs analysis into run_data and run_stats

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Implement `CIs_page()` in `app.py`

**Files:**
- Modify: `analysis/eva-bench-stats/app.py:441-443` (replace the placeholder `CIs_page()` function)

- [ ] **Step 1: Read the surrounding helpers**

```bash
sed -n '1,140p' analysis/eva-bench-stats/app.py
sed -n '380,445p' analysis/eva-bench-stats/app.py
```

Read enough to know what `_load_config`, `download_button`, and `CONFIG_DIR` are and how the perturbations page renders metric tabs and tables. Mirror those conventions exactly (do not invent helpers).

- [ ] **Step 2: Replace the placeholder body**

Replace the `CIs_page()` function (currently at lines 441–443) with:

```python
def CIs_page():
    import pandas as pd
    import streamlit as st
    from plots_CIs import (
        DEFAULT_METRIC_ORDER,
        forest_plot,
        metric_label,
        order_metrics,
        paper_summary_table,
    )
    from plots_utils import download_button

    st.header("Confidence Intervals")
    st.caption(
        "95% percentile bootstrap CIs over scenario-level means. "
        "Pooled estimates use equal weighting across the 3 domains. "
        "Bootstrap resample count is selected by a 1k-vs-2k stability check (threshold 0.002 on the metric scale)."
    )

    config = _load_config("CIs")
    if config is None:
        st.warning(f"Config not found: `{CONFIG_DIR / 'CIs_config.yaml'}`. Create it to get started.")
        return

    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / config["output_dir"]
    stats_dir = output_dir / "stats"
    data_dir = output_dir / "data"

    per_domain_path = stats_dir / "results_per_domain.csv"
    pooled_path = stats_dir / "results_pooled.csv"
    stability_path = stats_dir / "stability_log.csv"
    completeness_path = data_dir / "completeness_report.csv"

    if not per_domain_path.exists() or not pooled_path.exists():
        st.warning(
            f"CI outputs not found in `{stats_dir}`. "
            f"Run the pipeline (Run pipeline expander in the sidebar) or "
            f"`uv run python analysis/eva-bench-stats/run_stats.py`."
        )
        return

    per_domain_df = pd.read_csv(per_domain_path)
    pooled_df = pd.read_csv(pooled_path)
    stability_df = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    completeness_df = pd.read_csv(completeness_path) if completeness_path.exists() else pd.DataFrame()

    # ── Color map by system_type ──────────────────────────────────────────
    type_colors = {"cascade": "#3B82F6", "s2s": "#CC61B0", "hybrid": "#A855F7"}
    model_to_color: dict[str, str] = {}
    for label, model_cfg in (config.get("models") or {}).items():
        model_to_color[label] = type_colors.get(model_cfg.get("type", "cascade"), "#3B82F6")

    # ── Always-visible expanders ──────────────────────────────────────────
    with st.expander("Stability check", expanded=False):
        if stability_df.empty:
            st.info("No stability log available.")
        else:
            st.dataframe(stability_df, use_container_width=True)
            download_button(stability_df, "stability_log.csv")

    with st.expander("Paper-ready summary (EVA-A & EVA-X means)", expanded=False):
        summary = paper_summary_table(
            per_domain_df, pooled_df,
            metrics=("EVA-A_mean", "EVA-X_mean"),
            domains=tuple(config["expected_domains"]),
        )
        st.dataframe(summary, use_container_width=True)
        download_button(summary, "ci_summary_paper.csv")

    with st.expander("Completeness report", expanded=False):
        if completeness_df.empty:
            st.info("No completeness report available.")
        else:
            st.dataframe(completeness_df, use_container_width=True)
            download_button(completeness_df, "completeness_report.csv")

    # ── Metric tabs ───────────────────────────────────────────────────────
    metrics_present = sorted(set(per_domain_df["metric"]) | set(pooled_df["metric"]))
    metrics_ordered = order_metrics(metrics_present)
    if not metrics_ordered:
        st.info("No metrics to display.")
        return

    tabs = st.tabs([metric_label(m) for m in metrics_ordered])
    for tab, metric in zip(tabs, metrics_ordered):
        with tab:
            st.plotly_chart(
                forest_plot(
                    per_domain_df, pooled_df,
                    metric=metric,
                    domains=config["expected_domains"],
                    color_map=model_to_color,
                ),
                use_container_width=True,
            )
            combined = pd.concat([
                per_domain_df[per_domain_df["metric"] == metric],
                pooled_df[pooled_df["metric"] == metric],
            ], ignore_index=True)[["model_label", "domain", "n", "point_estimate", "ci_lower", "ci_upper"]]
            st.dataframe(combined, use_container_width=True)
            download_button(combined, f"ci_{metric}.csv")
```

If any imports (`Path`, `_load_config`, `CONFIG_DIR`) are not already in scope at the top of `app.py`, add them to the function-local imports rather than reorganizing the module — match the style of the existing `variance_page()` which does function-local imports.

- [ ] **Step 3: Verify the placeholder reference in the README's "Files" table is still accurate**

Skim `analysis/eva-bench-stats/README.md` lines 179–181: those rows describe `data_CIs.py`, `stats_CIs.py`, `plots_CIs.py` as "(placeholder)". Update them in this commit:

Change:
```
| `data_CIs.py` | Process runs for CI analysis (placeholder) |
| `stats_CIs.py` | Cluster bootstrap CI computation (placeholder) |
| `plots_CIs.py` | CI figures and tables (placeholder) |
```

To:
```
| `data_CIs.py` | Process clean trial scores → scenario-level means for CI analysis |
| `stats_CIs.py` | Bootstrap stability check, per-domain, and equal-weighted pooled CIs |
| `plots_CIs.py` | CI forest plots and paper-ready summary table |
```

- [ ] **Step 4: Sanity-check imports / launch the app**

```bash
uv run python -c "import sys; sys.path.insert(0, 'analysis/eva-bench-stats'); import app  # noqa"
```

Expected: no error.

(Manual UI check is in Task 10.)

- [ ] **Step 5: Commit**

```bash
git add analysis/eva-bench-stats/app.py analysis/eva-bench-stats/README.md
git commit -m "feat(eva-bench-stats): implement CIs_page in Streamlit app

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: End-to-end smoke test

**Files:** none modified (validation only)

- [ ] **Step 1: Confirm clean data is on disk**

```bash
ls output/eva-bench-stats/eva_clean_data_* 2>/dev/null | head
```

If empty: the puller needs to be run on the remote per the instructions in `pull_clean_data.py`'s header comment, the zip pulled, and unzipped under `output/eva-bench-stats/`. Stop here and ask the user before continuing.

- [ ] **Step 2: Populate `models:` in `CIs_config.yaml`**

```bash
head -1 output/eva-bench-stats/eva_clean_data_*/trial_scores.csv | head -1
awk -F, 'NR>1 {print $2}' output/eva-bench-stats/eva_clean_data_*/trial_scores.csv | sort -u
```

For each `system_alias` listed, add an entry to `local/eva-bench-stats/CIs_config.yaml` under `models:` with a sensible display label and `type:` (cascade / s2s / hybrid). If unsure for any alias, ask the user.

- [ ] **Step 3: Run the data pipeline**

```bash
uv run python analysis/eva-bench-stats/data_CIs.py
```

Expected output: prints `[COMPLETE]` per model (or `[INCOMPLETE]` with details), writes `output_processed/eva-bench-stats/CIs/data/scenario_means.csv` and `completeness_report.csv`. Verify shapes:

```bash
wc -l output_processed/eva-bench-stats/CIs/data/scenario_means.csv
head -1 output_processed/eva-bench-stats/CIs/data/scenario_means.csv
```

- [ ] **Step 4: Run the stats pipeline**

```bash
uv run python analysis/eva-bench-stats/stats_CIs.py
```

Expected output: stability message, then writes `results_per_domain.csv`, `results_pooled.csv`, `stability_log.csv`. Sanity check:

```bash
head -3 output_processed/eva-bench-stats/CIs/stats/results_per_domain.csv
head -3 output_processed/eva-bench-stats/CIs/stats/results_pooled.csv
cat output_processed/eva-bench-stats/CIs/stats/stability_log.csv
```

For every row, confirm `ci_lower <= point_estimate <= ci_upper` (Python one-liner):
```bash
uv run python -c "
import pandas as pd
for p in ['output_processed/eva-bench-stats/CIs/stats/results_per_domain.csv',
          'output_processed/eva-bench-stats/CIs/stats/results_pooled.csv']:
    df = pd.read_csv(p)
    bad = df[(df.ci_lower > df.point_estimate) | (df.point_estimate > df.ci_upper)]
    print(p, 'bad rows:', len(bad))
    assert bad.empty, bad
"
```
Expected: `bad rows: 0` for both.

- [ ] **Step 5: Launch app and inspect**

```bash
uv run streamlit run analysis/eva-bench-stats/app.py
```

In the browser, navigate to the CIs page and confirm:
- Stability expander shows the chosen `n_boot`.
- Paper-ready summary table renders for `EVA-A_mean` and `EVA-X_mean` with `point [lo, hi]` strings.
- Each metric tab shows a forest plot (4 facets: 3 domains + pooled) and a downloadable table.
- `download_button`s produce CSVs that open cleanly.

Report any UI regressions; fix in a follow-up commit if found.

- [ ] **Step 6: Final commit if any UI fixes were needed**

If everything passed without further changes, no commit is required. Otherwise:

```bash
git add analysis/eva-bench-stats/app.py
git commit -m "fix(eva-bench-stats): UI fixes from CIs smoke test

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** Stability check (Task 6), per-domain bootstrap (Task 6), equal-weighted pooled CI (Task 6), composite re-aggregation via puller-emitted scenario means (Task 3 + 5), pass@1 via averaged binary (Task 5), missing-data graceful skip (Task 5 + 6), paper-ready summary (Task 7 + 9), forest plot (Task 7 + 9), stability reporting (Task 6 + 9). Reuse of existing `bootstrap_ci` is via the extracted `stats_utils` module (Task 1 + 2). All addressed.
- **Type consistency:** `bootstrap_resample`, `compute_domain_ci`, `compute_pooled_ci`, `stability_check`, `run_analysis` signatures and column names (`model_label`, `metric`, `domain`, `n`, `point_estimate`, `ci_lower`, `ci_upper`) are consistent across Tasks 6, 7, and 9.
- **Placeholder-free:** Every code-bearing step shows the full code; commands include expected output; no "TBD" / "TODO" / "similar to" references.
- **Out-of-scope items deferred:** model-vs-model significance testing, HTML report integration, full migration of perturbation analysis to the new `bootstrap_resample` interface — all explicitly noted in the spec.
