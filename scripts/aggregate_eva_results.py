#!/usr/bin/env python3
"""Aggregate EVA-Bench run outputs into paper-ready CSVs and plots.

Walks a runs directory (default /mnt/voice_agent), reads config.json,
evaluation_summary.json, and metrics_summary.json from each run, classifies
runs by (domain, perturbation_category, system_alias), drops runs that fall
below the success-rate threshold, deduplicates by latest timestamp, and emits
a timestamped output folder containing:

- clean/{domain}/{domain}_clean_*.csv|.png
- perturbed/{accent|background_noise|both}/{domain}/{domain}_{cat}_*.csv|.png
- status.json
- a sibling .zip of the whole folder

Per-bucket files (in each leaf directory):
  {prefix}_main.csv               (EVA-A_pass / EVA-X_pass means + counts)
  {prefix}_pass_k.csv             (pass@5 + pass^k for both composites)
  {prefix}_per_metric_accuracy.csv
  {prefix}_per_metric_experience.csv
  {prefix}_per_metric_diagnostic.csv
  {prefix}_accuracy_vs_experience.png  (Pareto-frontier scatter)

Metric categorization is the canonical EVA categorization from
src/eva/metrics/aggregation.py and src/eva/metrics/diagnostic/__init__.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

# ── Metric categorization ─────────────────────────────────────────────
# Sources of truth:
#   src/eva/metrics/aggregation.py:34-83 (EVA_COMPOSITES)
#   src/eva/metrics/diagnostic/__init__.py
ACCURACY_METRICS = ["task_completion", "faithfulness", "agent_speech_fidelity"]
EXPERIENCE_METRICS = ["conversation_progression", "turn_taking", "conciseness"]
DIAGNOSTIC_METRICS = [
    "response_speed",
    "tool_call_validity",
    "speakability",
    "stt_wer",
    "transcription_accuracy_key_entities",
    "conversation_correctly_finished",
    "authentication_success",
]

# ── Bucketing ─────────────────────────────────────────────────────────
DOMAINS = ["itsm", "medical_hr", "airline"]
PERTURBATION_CATEGORIES = ["clean", "accent", "background_noise", "both"]

ALIAS_STRIP_TOKENS = ("accent", "background_noise", "background-noise", "both")
RUN_DIR_REGEX = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2}\.\d+)_(.+)$")
# Expected k for pass-k metrics. Clean runs use k=5, perturbed runs use k=3.
EXPECTED_K_CLEAN = 5
EXPECTED_K_PERTURBED = 3

# When any perturbation is applied, only 90 simulations are run per domain
# regardless of what evaluation_summary.simulation.total_records reports.
PERTURBED_EXPECTED_SIMS = 90


def expected_k_for(perturbation_category: str) -> int:
    return EXPECTED_K_CLEAN if perturbation_category == "clean" else EXPECTED_K_PERTURBED

DEFAULT_RUNS_DIR = "/mnt/voice_agent"
DEFAULT_OUTPUT_DIR = "./aggregated_results"
DEFAULT_SUCCESS_THRESHOLD = 0.95

# Run-id substrings that bypass the success-rate threshold (kept regardless).
# Use sparingly — these are deliberate hardcoded inclusions for runs whose
# accounting is structurally off (e.g., results landed in a subdir).
FORCE_INCLUDE_SUBSTRINGS: list[str] = [
    "fixie-ai",
    "ultravox",
]

# Hardcoded run injections. Each entry loads metrics_summary.json from an exact
# relative path under runs_dir and synthesizes the surrounding parsed-run dict
# from the values listed here. Used for runs whose layout is non-standard or
# missing config.json/evaluation_summary.json. These entries always bypass
# success-rate filtering and dedupe — they are emitted as-is.
HARDCODED_RUN_INJECTIONS: list[dict] = [
    {
        "run_id": "2026-04-27_04-53-11.007976_fixie-ai_ultravox_itsm_clean",
        "metrics_relpath": "2026-04-27_04-53-11.007976_fixie-ai/ultravox/metrics_summary.json",
        "domain": "itsm",
        "perturbation_category": "clean",
        "system_type": "audio_llm",
        "alias": "ultravox-realtime",
        "stt": None,
    },
]

# Component-name normalization rules. Applied to each pipeline_part value
# (stt / llm / tts / audio_llm / s2s) so display aliases stay short and
# consistent across plots/tables. The first matching predicate wins.
def _normalize_component(name: str | None) -> str | None:
    if not name:
        return name
    s = str(name)
    low = s.lower()
    if "cohere" in low:
        return "cohere"
    if "scribe" in low:
        return "scribe-v2.2-rlt"
    if "ultravox" in low or "fixie-ai" in low:
        return "ultravox-realtime"
    if "gemma" in low and "26b" in low:
        return "gemma-4-26b"
    if "gemma" in low and "31b" in low:
        return "gemma-4-31b"
    if "qwen" in low:
        return "qwen3.5-27b"
    if low == "voxtral_4b_tts" or low == "voxtral-4b-tts":
        return "voxtral-4b"
    if low == "gemini-3-flash-preview":
        return "gemini-3-flash"
    if low == "gemini-3.1-flash-tts-preview":
        return "gemini-3.1-flash-tts"
    if low == "gemini-3.1-flash-live-preview":
        return "gemini-3.1-flash-live"
    return s


# Special-case rewrites applied to the entire pipeline_parts dict before
# alias derivation / system-type detection. The elevenlabs s2s product is
# actually a cascade of (scribe-v2.2-realtime + gemini 3 flash + eleven-v3-
# conversational), so we normalize it to its true cascade representation.
def _rewrite_pipeline_parts(pipeline_parts: dict) -> dict:
    if not isinstance(pipeline_parts, dict):
        return pipeline_parts
    s2s = pipeline_parts.get("s2s")
    if s2s == "elevenlabs":
        return {
            "stt": "scribe-v2.2-realtime",
            "llm": "gemini-3-flash",
            "tts": "v3-conv",
        }
    return pipeline_parts


# ── JSON helpers ──────────────────────────────────────────────────────
def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def parse_run_timestamp(folder_name: str) -> datetime | None:
    m = RUN_DIR_REGEX.match(folder_name)
    if not m:
        return None
    date_part, time_part, _ = m.groups()
    # time_part is "HH-MM-SS.microseconds"
    try:
        return datetime.strptime(f"{date_part}_{time_part}", "%Y-%m-%d_%H-%M-%S.%f")
    except ValueError:
        return None


# ── Classification ────────────────────────────────────────────────────
def perturbation_category(pert: dict | None) -> str | None:
    """Return 'clean', 'accent', 'background_noise', 'both', or None.

    None means the run has perturbations outside the four supported categories
    (e.g. behavior or connection_degradation) and should be excluded.
    """
    if pert is None:
        return "clean"
    accent = bool(pert.get("accent"))
    bg = bool(pert.get("background_noise"))
    other = bool(pert.get("behavior")) or bool(pert.get("connection_degradation"))
    if other:
        return None
    if accent and bg:
        return "both"
    if accent:
        return "accent"
    if bg:
        return "background_noise"
    return "clean"


def detect_system_type(model_cfg: dict, pipeline_parts: dict | None = None) -> str:
    """Mirrors src/eva/models/config.py:343-362 get_pipeline_type.

    Returns 'cascade', 'audio_llm', or 's2s'.
    """
    if isinstance(model_cfg, dict):
        s2s = model_cfg.get("s2s")
        if s2s is not None and s2s != "elevenlabs":
            return "s2s"
        if model_cfg.get("realtime_model"):
            return "s2s"
        if model_cfg.get("audio_llm"):
            return "audio_llm"
    if isinstance(pipeline_parts, dict):
        if pipeline_parts.get("s2s"):
            return "s2s"
        if pipeline_parts.get("audio_llm") and not pipeline_parts.get("stt"):
            return "audio_llm"
    return "cascade"


def derive_alias(pipeline_parts: dict, folder_name: str) -> str:
    """Build a display alias from pipeline_parts.

    Cascade: 'stt + llm + tts'
    S2S:     '<s2s_model>'
    Audio-LLM: 'audio_llm + tts'

    Falls back to parsing the folder name if pipeline_parts is malformed,
    in which case perturbation suffixes are stripped.
    """
    if isinstance(pipeline_parts, dict) and pipeline_parts:
        if "s2s" in pipeline_parts and pipeline_parts.get("s2s"):
            return str(_normalize_component(pipeline_parts["s2s"]))
        components = []
        for key in ("stt", "audio_llm", "llm", "tts"):
            v = pipeline_parts.get(key)
            if v:
                components.append(str(_normalize_component(v)))
        if components:
            # Whole-system alias collapses: certain models are referred to by
            # a single name in the paper regardless of the rest of the pipeline.
            if "ultravox-realtime" in components:
                return "ultravox-realtime"
            return " + ".join(components)

    # Fallback: parse folder name and strip perturbation suffix tokens.
    m = RUN_DIR_REGEX.match(folder_name)
    suffix = m.group(3) if m else folder_name
    for tok in ALIAS_STRIP_TOKENS:
        suffix = suffix.replace(f"_{tok}", "").replace(f"-{tok}", "")
    return suffix.rstrip("_-")


# ── Per-run loading + parsing ─────────────────────────────────────────
def load_run(run_dir: Path) -> tuple[dict | None, str | None]:
    """Load and parse a run. Returns (parsed, error_msg).

    parsed is None on failure, with error_msg explaining why.
    """
    name = run_dir.name
    config_path = run_dir / "config.json"
    eval_path = run_dir / "evaluation_summary.json"
    metrics_path = run_dir / "metrics_summary.json"

    # Some runs (notably fixie-ai/ultravox) write their result files into a
    # subdirectory rather than the run root. If any of the three required files
    # is missing at the top level, search one level deep for the first subdir
    # that contains metrics_summary.json and pull the missing files from there.
    if not (config_path.exists() and eval_path.exists() and metrics_path.exists()):
        try:
            for sub in run_dir.iterdir():
                if sub.is_dir() and (sub / "metrics_summary.json").exists():
                    if not metrics_path.exists():
                        metrics_path = sub / "metrics_summary.json"
                    if not config_path.exists() and (sub / "config.json").exists():
                        config_path = sub / "config.json"
                    if not eval_path.exists() and (sub / "evaluation_summary.json").exists():
                        eval_path = sub / "evaluation_summary.json"
                    break
        except OSError:
            pass

    for p, label in [
        (config_path, "config.json"),
        (eval_path, "evaluation_summary.json"),
        (metrics_path, "metrics_summary.json"),
    ]:
        if not p.exists():
            return None, f"missing {label}"

    try:
        config = load_json(config_path)
        evaluation = load_json(eval_path)
        metrics = load_json(metrics_path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"json load failed: {exc}"

    sim = evaluation.get("simulation", {}) or {}
    total = sim.get("total_records") or 0
    successful = sim.get("successful_records") or 0
    rate = (successful / total) if total else 0.0

    domain = config.get("domain")
    pert = config.get("perturbation")
    pert_cat = perturbation_category(pert)
    pipeline_parts = _rewrite_pipeline_parts(config.get("pipeline_parts") or {})
    model_cfg = config.get("model") or {}
    # If the pipeline rewrite produced a cascade-shaped dict, force cascade.
    if isinstance(pipeline_parts, dict) and pipeline_parts.get("stt") and pipeline_parts.get("llm"):
        sys_type = "cascade"
    else:
        sys_type = detect_system_type(model_cfg, pipeline_parts)
    alias = derive_alias(pipeline_parts, name)
    # Whole-system arch overrides: certain models always belong to a specific
    # architecture regardless of how the run config classifies them. Ultravox
    # is an audio-LLM, so even runs whose model.s2s field is set to ultravox
    # should appear in the AudioLLM + TTS bucket.
    if alias == "ultravox-realtime":
        sys_type = "audio_llm"
    stt_model = (
        _normalize_component(pipeline_parts.get("stt"))
        if isinstance(pipeline_parts, dict)
        else None
    )
    timestamp = parse_run_timestamp(name)

    overall = metrics.get("overall_scores", {}) or {}
    eva_a = overall.get("EVA-A_pass") or {}
    eva_x = overall.get("EVA-X_pass") or {}
    pass_k = overall.get("pass_k") or {}

    # Per-metric: only top-level mean/count for the metrics we care about.
    per_metric_raw = metrics.get("per_metric", {}) or {}
    per_metric_summary = {}
    for m_name in ACCURACY_METRICS + EXPERIENCE_METRICS + DIAGNOSTIC_METRICS:
        entry = per_metric_raw.get(m_name) or {}
        per_metric_summary[m_name] = {
            "mean": entry.get("mean"),
            "count": entry.get("count"),
        }

    data_quality = metrics.get("data_quality", {}) or {}
    metric_errors_summary = {
        "records_with_errors": data_quality.get("records_with_errors", 0),
        "metrics_with_errors": data_quality.get("metrics_with_errors", {}) or {},
    }

    parsed = {
        "run_id": name,
        "run_dir": str(run_dir),
        "timestamp": timestamp,
        "domain": domain,
        "perturbation": pert,
        "perturbation_category": pert_cat,
        "system_type": sys_type,
        "alias": alias,
        "stt": stt_model,
        "successful_records": successful,
        "total_records": total,
        "success_rate": rate,
        "main": {
            "eva_a_mean": eva_a.get("mean"),
            "eva_a_count": eva_a.get("count"),
            "eva_x_mean": eva_x.get("mean"),
            "eva_x_count": eva_x.get("count"),
        },
        "pass_k": {
            "eva_a": pass_k.get("EVA-A_pass") or {},
            "eva_x": pass_k.get("EVA-X_pass") or {},
        },
        "per_metric": per_metric_summary,
        "metric_errors": metric_errors_summary,
    }
    return parsed, None


def _load_hardcoded_run(runs_dir: Path, spec: dict) -> dict | None:
    """Build a parsed-run dict from a HARDCODED_RUN_INJECTIONS entry.

    Reads only metrics_summary.json from the explicit path; all other fields
    (domain, perturbation_category, system_type, alias) come from the spec.
    Returns None if the metrics file isn't present.
    """
    metrics_path = runs_dir / spec["metrics_relpath"]
    if not metrics_path.exists():
        return None
    try:
        metrics = load_json(metrics_path)
    except (OSError, json.JSONDecodeError):
        return None

    overall = metrics.get("overall_scores", {}) or {}
    eva_a = overall.get("EVA-A_pass") or {}
    eva_x = overall.get("EVA-X_pass") or {}
    pass_k = overall.get("pass_k") or {}

    per_metric_raw = metrics.get("per_metric", {}) or {}
    per_metric_summary = {}
    for m_name in ACCURACY_METRICS + EXPERIENCE_METRICS + DIAGNOSTIC_METRICS:
        entry = per_metric_raw.get(m_name) or {}
        per_metric_summary[m_name] = {
            "mean": entry.get("mean"),
            "count": entry.get("count"),
        }

    data_quality = metrics.get("data_quality", {}) or {}
    metric_errors_summary = {
        "records_with_errors": data_quality.get("records_with_errors", 0),
        "metrics_with_errors": data_quality.get("metrics_with_errors", {}) or {},
    }

    a_count = eva_a.get("count") or 0
    return {
        "run_id": spec["run_id"],
        "run_dir": str(metrics_path.parent),
        "timestamp": datetime.now(),
        "domain": spec["domain"],
        "perturbation": None if spec["perturbation_category"] == "clean" else {},
        "perturbation_category": spec["perturbation_category"],
        "system_type": spec["system_type"],
        "alias": spec["alias"],
        "stt": spec.get("stt"),
        "successful_records": a_count,
        "total_records": a_count,
        "success_rate": 1.0,
        "expected_total": a_count,
        "effective_success_rate": 1.0,
        "main": {
            "eva_a_mean": eva_a.get("mean"),
            "eva_a_count": eva_a.get("count"),
            "eva_x_mean": eva_x.get("mean"),
            "eva_x_count": eva_x.get("count"),
        },
        "pass_k": {
            "eva_a": pass_k.get("EVA-A_pass") or {},
            "eva_x": pass_k.get("EVA-X_pass") or {},
        },
        "per_metric": per_metric_summary,
        "metric_errors": metric_errors_summary,
    }


# ── Aggregation pipeline ──────────────────────────────────────────────
def aggregate(
        runs_dir: Path,
        success_threshold: float,
) -> tuple[dict, dict]:
    """Walk runs_dir and return (buckets, status).

    buckets: {(domain, perturbation_category): [parsed_run, ...]} with one
    canonical run per (alias, domain, perturbation_category).

    status: skip-reason categorization for status.json.
    """
    status: dict[str, Any] = {
        "skipped": {
            "non_run_directories": [],
            "missing_files": [],
            "low_success_rate": [],
            "duplicate_resolved": [],
            "excluded_other_perturbation": [],
            "non_standard_k_runs": [],
            "unknown_domain": [],
        },
        "metric_errors_per_run": {},
    }

    parsed_runs: list[dict] = []
    discovered = 0

    if not runs_dir.exists():
        raise SystemExit(f"runs-dir does not exist: {runs_dir}")

    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not RUN_DIR_REGEX.match(entry.name):
            status["skipped"]["non_run_directories"].append(entry.name)
            continue
        discovered += 1
        parsed, err = load_run(entry)
        if parsed is None:
            status["skipped"]["missing_files"].append(
                {"run": entry.name, "reason": err}
            )
            continue

        if parsed["perturbation_category"] is None:
            status["skipped"]["excluded_other_perturbation"].append(
                {"run": parsed["run_id"], "perturbation": parsed["perturbation"]}
            )
            continue

        if parsed["domain"] not in DOMAINS:
            status["skipped"]["unknown_domain"].append(
                {"run": parsed["run_id"], "domain": parsed["domain"]}
            )
            continue

        # Perturbed runs always have 90 expected sims regardless of total_records.
        if parsed["perturbation_category"] == "clean":
            expected_total = parsed["total_records"]
        else:
            expected_total = PERTURBED_EXPECTED_SIMS
        rate = (parsed["successful_records"] / expected_total) if expected_total else 0.0
        parsed["expected_total"] = expected_total
        parsed["effective_success_rate"] = rate
        force_include = any(s in parsed["run_id"] for s in FORCE_INCLUDE_SUBSTRINGS)
        if not force_include and (expected_total == 0 or rate < success_threshold):
            status["skipped"]["low_success_rate"].append(
                {
                    "run": parsed["run_id"],
                    "successful": parsed["successful_records"],
                    "expected": expected_total,
                    "reported_total": parsed["total_records"],
                    "rate": round(rate, 4),
                }
            )
            continue

        # Kept-but-imperfect: passed the threshold but still missing some sims.
        if rate < 1.0:
            status.setdefault("imperfect_success_rate", []).append(
                {
                    "run": parsed["run_id"],
                    "domain": parsed["domain"],
                    "perturbation": parsed["perturbation_category"],
                    "alias": parsed["alias"],
                    "successful": parsed["successful_records"],
                    "expected": expected_total,
                    "rate": round(rate, 4),
                }
            )
            print(
                f"  imperfect: {parsed['domain']}/{parsed['perturbation_category']}/{parsed['alias']}"
                f" — {parsed['successful_records']}/{expected_total} = {rate:.3f}"
                f"  ({parsed['run_id']})",
                flush=True,
            )

        # Pass-k presence check: drop pass-k data with the wrong k for this
        # bucket (5 for clean, 3 for perturbed). The rest of the run is kept.
        expected_k = expected_k_for(parsed["perturbation_category"])
        for which in ("eva_a", "eva_x"):
            pk = parsed["pass_k"][which]
            if pk and pk.get("k") not in (None, expected_k):
                status["skipped"]["non_standard_k_runs"].append(
                    {
                        "run": parsed["run_id"],
                        "composite": "EVA-A_pass" if which == "eva_a" else "EVA-X_pass",
                        "expected_k": expected_k,
                        "actual_k": pk.get("k"),
                    }
                )
                parsed["pass_k"][which] = {}

        # Track metric errors for runs we keep.
        me = parsed["metric_errors"]
        if me["records_with_errors"] or me["metrics_with_errors"]:
            status["metric_errors_per_run"][parsed["run_id"]] = me

        parsed_runs.append(parsed)

    # Hardcoded injections — appended after the discovery loop so they always
    # land in the buckets regardless of whether the run dir was traversed.
    status.setdefault("hardcoded_injections", [])
    for spec in HARDCODED_RUN_INJECTIONS:
        injected = _load_hardcoded_run(runs_dir, spec)
        if injected is None:
            status["hardcoded_injections"].append(
                {"run_id": spec["run_id"], "loaded": False, "metrics_relpath": spec["metrics_relpath"]}
            )
            continue
        parsed_runs.append(injected)
        status["hardcoded_injections"].append(
            {"run_id": spec["run_id"], "loaded": True, "metrics_relpath": spec["metrics_relpath"]}
        )
        print(f"  injected: {injected['domain']}/{injected['perturbation_category']}/{injected['alias']}", flush=True)

    status["total_runs_discovered"] = discovered
    status["total_runs_loaded"] = len(parsed_runs)

    # Dedupe by (domain, perturbation_category, alias) — keep latest timestamp.
    by_key: dict[tuple, dict] = {}
    for r in parsed_runs:
        key = (r["domain"], r["perturbation_category"], r["alias"])
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = r
            continue
        prev_ts = prev["timestamp"] or datetime.min
        cur_ts = r["timestamp"] or datetime.min
        if cur_ts > prev_ts:
            kept, discarded = r["run_id"], prev["run_id"]
            by_key[key] = r
        else:
            kept, discarded = prev["run_id"], r["run_id"]
        status["skipped"]["duplicate_resolved"].append(
            {
                "domain": r["domain"],
                "perturbation": r["perturbation_category"],
                "alias": r["alias"],
                "kept": kept,
                "discarded": discarded,
            }
        )
        print(
            f"  duplicate: {r['domain']}/{r['perturbation_category']}/{r['alias']}"
            f" — kept {kept}, discarded {discarded}",
            flush=True,
        )

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (domain, cat, _alias), r in by_key.items():
        buckets[(domain, cat)].append(r)

    # Stable order within each bucket: alias asc, then system_type.
    for k in buckets:
        buckets[k].sort(key=lambda r: (r["alias"].lower(), r["system_type"]))

    status["total_runs_kept"] = sum(len(v) for v in buckets.values())
    return buckets, status


# ── CSV writers ───────────────────────────────────────────────────────
def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return v


def write_main_csv(path: Path, runs: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "system",
                "system_type",
                "EVA-A_pass_mean",
                "EVA-A_pass_count",
                "EVA-X_pass_mean",
                "EVA-X_pass_count",
            ]
        )
        for r in runs:
            m = r["main"]
            w.writerow(
                [
                    r["alias"],
                    r["system_type"],
                    _fmt(m["eva_a_mean"]),
                    _fmt(m["eva_a_count"]),
                    _fmt(m["eva_x_mean"]),
                    _fmt(m["eva_x_count"]),
                ]
            )


def write_pass_k_csv(path: Path, runs: list[dict], k: int) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "system",
                "system_type",
                f"EVA-A_pass@{k}",
                "EVA-A_pass^k",
                "EVA-A_count",
                f"EVA-X_pass@{k}",
                "EVA-X_pass^k",
                "EVA-X_count",
            ]
        )
        for r in runs:
            a = r["pass_k"]["eva_a"] or {}
            x = r["pass_k"]["eva_x"] or {}
            w.writerow(
                [
                    r["alias"],
                    r["system_type"],
                    _fmt(a.get("pass_at_k")),
                    _fmt(a.get("pass_power_k_theoretical")),
                    _fmt(a.get("count")),
                    _fmt(x.get("pass_at_k")),
                    _fmt(x.get("pass_power_k_theoretical")),
                    _fmt(x.get("count")),
                ]
            )


def write_per_metric_csv(path: Path, runs: list[dict], metric_names: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["system", "system_type"]
        for m in metric_names:
            header.append(f"{m}_mean")
            header.append(f"{m}_count")
        w.writerow(header)
        for r in runs:
            row = [r["alias"], r["system_type"]]
            for m in metric_names:
                entry = r["per_metric"].get(m) or {}
                row.append(_fmt(entry.get("mean")))
                row.append(_fmt(entry.get("count")))
            w.writerow(row)


# ── Plot ──────────────────────────────────────────────────────────────
def _pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return upper-right Pareto frontier (both axes higher-is-better),
    sorted by x ascending."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier: list[tuple[float, float]] = []
    best_y = -float("inf")
    for x, y in reversed(sorted_pts):
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    frontier.reverse()
    return frontier


_PERT_DISPLAY = {
    "clean": "Clean",
    "accent": "Accent",
    "background_noise": "Background Noise",
    "both": "Accent + Background Noise",
}
_DOMAIN_DISPLAY = {
    "itsm": "ITSM",
    "medical_hr": "HR",
    "airline": "CSM",
}
# Color + marker per system_type. Distinct hues + shapes so colorblind viewers
# can tell them apart, and so cascade/s2s are visually distinguishable in B&W.
_TYPE_STYLE = {
    "cascade": {"color": "#1f77b4", "marker": "o", "label": "Cascade"},
    "audio_llm": {"color": "#2ca02c", "marker": "^", "label": "Audio-native"},
    "s2s": {"color": "#d62728", "marker": "s", "label": "S2S"},
}
# Paper-style scatter: matches the NeurIPS preamble color palette
# (evablue/evateal/evaviolet) so figures read consistently with the rest of
# the document. Hybrid (audio_llm) gets a diamond marker to distinguish it
# from cascade circles and S2S squares.
_PAPER_TYPE_STYLE = {
    "cascade":   {"color": "#378ADD", "marker": "o", "label": "Cascade"},
    "audio_llm": {"color": "#1D9E75", "marker": "D", "label": "Hybrid (AudioLLM + TTS)"},
    "s2s":       {"color": "#7F77DD", "marker": "s", "label": "S2S"},
}
_ARCH_DISPLAY = {"cascade": "Cascade", "audio_llm": "Hybrid", "s2s": "S2S"}
_ARCH_ORDER = ["cascade", "audio_llm", "s2s"]


def _scatter_plot(
        path: Path,
        title: str,
        points: list[tuple[float, float, str, str]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    # Pareto frontier first so it sits behind the markers.
    xy_only = [(x, y) for x, y, _, _ in points]
    frontier = _pareto_frontier(xy_only)
    if len(frontier) >= 2:
        ax.plot(
            [fx for fx, _ in frontier],
            [fy for _, fy in frontier],
            linestyle="--",
            color="#555555",
            linewidth=1.6,
            alpha=0.7,
            zorder=2,
            label="Pareto frontier",
        )

    # Scatter, one call per system_type so the legend gets one entry per type.
    seen_types: set[str] = set()
    for stype, style in _TYPE_STYLE.items():
        xs = [x for x, _, _, t in points if t == stype]
        ys = [y for _, y, _, t in points if t == stype]
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            s=80,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
            label=style["label"],
            alpha=0.9,
        )
        seen_types.add(stype)

    # Place all labels first, then let adjustText push them apart and draw
    # connector lines back to their markers. Long cascade aliases get wrapped
    # at " + " so they form 2- or 3-line stacks that take less horizontal
    # room and are easier for adjust_text to fit. Perturbation suffixes are
    # stripped from labels because the bucket title already names the
    # perturbation.
    def _strip_pert(label: str) -> str:
        for tok in ALIAS_STRIP_TOKENS:
            if label.endswith(f"-{tok}"):
                return label[: -(len(tok) + 1)].rstrip("-_")
        return label

    def _wrap(label: str) -> str:
        if len(label) > 28 and " + " in label:
            return label.replace(" + ", "\n+ ")
        return label

    texts = []
    for x, y, alias, _ in points:
        texts.append(
            ax.text(
                x,
                y,
                _wrap(_strip_pert(alias)),
                fontsize=8.5,
                color="#222222",
                zorder=5,
                ha="center",
                va="center",
            )
        )
    if texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6, alpha=0.7),
            expand=(1.5, 2.0),
            force_text=(1.6, 2.0),
            force_static=(1.2, 1.6),
            force_explode=(0.6, 0.9),
            max_move=140,
            iter_lim=800,
            prevent_crossings=True,
        )

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.06, 1.04)
    ax.set_xlabel("Accuracy (EVA-A_pass@1)", fontsize=11)
    ax.set_ylabel("Experience (EVA-X_pass@1)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(True, color="#dddddd", linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888888")

    # Legend outside the axes on the right so it never overlaps points.
    if seen_types or len(frontier) >= 2:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=10,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_plot(path: Path, domain: str, cat: str, runs: list[dict]) -> None:
    points: list[tuple[float, float, str, str]] = []
    for r in runs:
        x = r["main"]["eva_a_mean"]
        y = r["main"]["eva_x_mean"]
        if x is None or y is None:
            continue
        points.append((x, y, r["alias"], r["system_type"]))
    title = f"{_DOMAIN_DISPLAY.get(domain, domain)} — {_PERT_DISPLAY.get(cat, cat)}"
    _scatter_plot(path, title, points)


def _paper_scatter_plot(
        path: Path,
        points: list[tuple[float, float, str, str]],
        title: str | None = None,
        show_labels: bool = True,
        x_label: str = "Accuracy (EVA-A pass@1)",
        y_label: str = "Experience (EVA-X pass@1)",
) -> None:
    """Paper-quality scatter for the clean domain-averaged plot.

    Uses serif fonts, the NeurIPS preamble color palette, and saves both PDF
    (vector, for inclusion in the paper) and PNG (preview).
    """
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    }):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.set_aspect("equal", adjustable="box")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        xy_only = [(x, y) for x, y, _, _ in points]
        frontier = _pareto_frontier(xy_only)
        if len(frontier) >= 2:
            ax.plot(
                [fx for fx, _ in frontier],
                [fy for _, fy in frontier],
                linestyle=(0, (4, 3)),
                color="#888888",
                linewidth=1.0,
                alpha=0.55,
                zorder=2,
                label="Pareto frontier",
            )

        seen_types: set[str] = set()
        for stype, style in _PAPER_TYPE_STYLE.items():
            xs = [x for x, _, _, t in points if t == stype]
            ys = [y for _, y, _, t in points if t == stype]
            if not xs:
                continue
            ax.scatter(
                xs, ys,
                color=style["color"],
                marker=style["marker"],
                s=95,
                edgecolor="white",
                linewidth=1.2,
                zorder=4,
                label=style["label"],
                alpha=0.95,
            )
            seen_types.add(stype)

        def _wrap(label: str) -> str:
            if len(label) > 26 and " + " in label:
                return label.replace(" + ", "\n+ ")
            return label

        texts = [
            ax.text(
                x, y, _wrap(alias),
                fontsize=8.0,
                color="#1a1a1a",
                zorder=5, ha="center", va="center",
            )
            for x, y, alias, _ in points
        ] if show_labels else []
        if texts:
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5, alpha=0.7),
                expand=(1.4, 1.8),
                force_text=(1.4, 1.8),
                force_static=(1.0, 1.4),
                max_move=120,
                iter_lim=600,
                prevent_crossings=True,
            )

        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title, pad=10)

        ax.grid(True, color="#e6e6e6", linewidth=0.6, zorder=1)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#666666")
        ax.tick_params(colors="#444444", length=3, width=0.6)

        if seen_types or len(frontier) >= 2:
            leg = ax.legend(
                loc="lower right",
                frameon=True, framealpha=0.95,
                facecolor="white", edgecolor="#cccccc",
                handletextpad=0.5, borderpad=0.5,
            )
            leg.get_frame().set_linewidth(0.6)

        fig.tight_layout()
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def write_domain_averaged_plot(
        out_dir: Path,
        buckets: dict[tuple[str, str], list[dict]],
) -> None:
    """Average EVA-A/EVA-X across the three clean-domain buckets per system."""
    by_system: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (domain, cat), runs in buckets.items():
        if cat != "clean":
            continue
        for r in runs:
            x = r["main"]["eva_a_mean"]
            y = r["main"]["eva_x_mean"]
            if x is None or y is None:
                continue
            by_system[(r["alias"], r["system_type"])].append((x, y))

    points: list[tuple[float, float, str, str]] = []
    for (alias, stype), xys in by_system.items():
        xs = [p[0] for p in xys]
        ys = [p[1] for p in xys]
        points.append((sum(xs) / len(xs), sum(ys) / len(ys), alias, stype))

    out_path = out_dir / "clean" / "_average" / "accuracy_vs_experience_average.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _paper_scatter_plot(out_path, points, title="Accuracy vs Experience pass@1")

    unlabeled_path = out_dir / "clean" / "_average" / "accuracy_vs_experience_average_unlabeled.png"
    _paper_scatter_plot(unlabeled_path, points, title="Accuracy vs Experience pass@1", show_labels=False)

    by_system_pk: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (domain, cat), runs in buckets.items():
        if cat != "clean":
            continue
        for r in runs:
            a = (r.get("pass_k") or {}).get("eva_a") or {}
            x = (r.get("pass_k") or {}).get("eva_x") or {}
            ax_val = a.get("pass_power_k_theoretical")
            ay_val = x.get("pass_power_k_theoretical")
            if ax_val is None or ay_val is None:
                continue
            by_system_pk[(r["alias"], r["system_type"])].append((ax_val, ay_val))

    pk_points: list[tuple[float, float, str, str]] = []
    for (alias, stype), xys in by_system_pk.items():
        xs = [p[0] for p in xys]
        ys = [p[1] for p in xys]
        pk_points.append((sum(xs) / len(xs), sum(ys) / len(ys), alias, stype))

    if pk_points:
        pk_path = out_dir / "clean" / "_average" / "accuracy_vs_experience_average_unlabeled_passk.png"
        _paper_scatter_plot(
            pk_path, pk_points, show_labels=False,
            title="Accuracy vs Experience pass^k",
            x_label="Accuracy (EVA-A pass^k)",
            y_label="Experience (EVA-X pass^k)",
        )

    def _frontier_entries(pts: list[tuple[float, float, str, str]]) -> list[dict]:
        xy = [(x, y) for x, y, _, _ in pts]
        frontier_xy = set((round(x, 12), round(y, 12)) for x, y in _pareto_frontier(xy))
        entries = [
            {"system": alias, "system_type": stype, "eva_a": x, "eva_x": y}
            for x, y, alias, stype in pts
            if (round(x, 12), round(y, 12)) in frontier_xy
        ]
        entries.sort(key=lambda e: e["eva_a"])
        return entries

    frontier_payload = {
        "pass_at_1": _frontier_entries(points),
        "pass_power_k": _frontier_entries(pk_points),
    }
    frontier_path = out_dir / "clean" / "_average" / "pareto_frontier.json"
    frontier_path.write_text(json.dumps(frontier_payload, indent=2) + "\n")


# ── LaTeX output ──────────────────────────────────────────────────────
_TEAL_PALETTE = [
    ("tel1", "b8dede"),
    ("tel2", "90cece"),
    ("tel3", "62b8b8"),
    ("tel4", "3a9e9e"),
    ("tel5", "1e8484"),
    ("tel6", "0f6b6b"),
    ("tel7", "075656"),
]
_PINK_PALETTE = [
    ("pnk1", "fde4ec"),
    ("pnk2", "fac4d4"),
    ("pnk3", "f59ab5"),
    ("pnk4", "ed6f95"),
    ("pnk5", "db4577"),
    ("pnk6", "b82d5c"),
    ("pnk7", "8c1f44"),
]
# Accent palette for the pass@1 (composite) column — subtle slate-indigo so
# the main metric reads distinctly from the per-submetric color groups.
_ACCENT_PALETTE = [
    ("acc1", "edeaf4"),
    ("acc2", "d9d2e6"),
    ("acc3", "bfb3d4"),
    ("acc4", "9d8dbb"),
    ("acc5", "7a679f"),
    ("acc6", "584981"),
    ("acc7", "3b3060"),
]


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("$", "\\$")
    )


def _band_thresholds(vmin: float, vmax: float) -> list[float]:
    """Return 6 ascending thresholds dividing [vmin, vmax] into 7 even bands."""
    if vmax <= vmin:
        # Degenerate: all values equal — push all to top band.
        return [vmin - 1] * 6
    step = (vmax - vmin) / 7.0
    return [vmin + step * i for i in range(1, 7)]


def _cell_macro_def(macro: str, palette: list, thresholds: list[float]) -> str:
    """Generate a \\newcommand{\\<macro>}[1]{...} matching the example shape."""
    lines = [f"\\newcommand{{\\{macro}}}[1]{{%"]
    for i, t in enumerate(thresholds):
        cmd = "\\lightcell" if i < 3 else "\\darkcell"
        name = palette[i][0]
        lines.append(f"  \\ifdim#1pt<{t:.4f}pt {cmd}{{{name}}}{{#1}}\\else")
    final_cmd = "\\darkcell"
    final_name = palette[6][0]
    lines.append(f"                      {final_cmd}{{{final_name}}}{{#1}}" + "\\fi" * len(thresholds) + "}")
    return "\n".join(lines)


def _palette_definecolor(palette: list) -> str:
    return "\n".join(f"\\definecolor{{{name}}}{{HTML}}{{{hexv}}}" for name, hexv in palette)


def _collect_clean_systems(buckets: dict) -> list[tuple[str, str]]:
    """Return ordered (system_type, alias) pairs across clean buckets."""
    seen: set[tuple[str, str]] = set()
    for (domain, cat), runs in buckets.items():
        if cat != "clean":
            continue
        for r in runs:
            seen.add((r["system_type"], r["alias"]))
    return sorted(seen, key=lambda p: (_ARCH_ORDER.index(p[0]) if p[0] in _ARCH_ORDER else 99, p[1].lower()))


def _clean_value_lookup(buckets: dict) -> dict[tuple[str, str], dict]:
    """(alias, domain) -> parsed run (clean only)."""
    out: dict[tuple[str, str], dict] = {}
    for (domain, cat), runs in buckets.items():
        if cat != "clean":
            continue
        for r in runs:
            out[(r["alias"], domain)] = r
    return out


def _format_table(
        title_short: str,
        caption: str,
        label: str,
        composite_key: str,
        composite_macro: str,
        metric_columns: list[tuple[str, str, str]],
        palette: list,
        accent_palette: list,
        buckets: dict,
        domains_present: list[str],
        n_trials: int,
) -> str:
    """Build a LaTeX table.

    metric_columns: list of (metric_name, header_label, macro_prefix).
    composite_key: 'eva_a_mean' or 'eva_x_mean'.
    composite_macro: macro prefix for the composite column.
    """
    lookup = _clean_value_lookup(buckets)
    systems = _collect_clean_systems(buckets)

    # All cell-macro keys: composite + per-metric, paired with the palette
    # used for that column's shading.
    all_macros: list[tuple[str, str, list]] = [
        (composite_macro, composite_key, accent_palette)
    ]
    for m_name, _, prefix in metric_columns:
        all_macros.append((prefix, m_name, palette))

    # Compute thresholds per macro from observed values.
    thresholds_by_macro: dict[str, list[float]] = {}
    for macro, key, _pal in all_macros:
        vals: list[float] = []
        for (alias, domain), r in lookup.items():
            if domain not in domains_present:
                continue
            v = (
                r["main"].get(key)
                if key in ("eva_a_mean", "eva_x_mean")
                else (r["per_metric"].get(key) or {}).get("mean")
            )
            if v is not None:
                vals.append(float(v))
        if vals:
            thresholds_by_macro[macro] = _band_thresholds(min(vals), max(vals))
        else:
            thresholds_by_macro[macro] = [0] * 6

    # Preamble: colors + cell macros.
    preamble_lines = [
        "% Auto-generated by scripts/aggregate_eva_results.py",
        "% Requires: xcolor, colortbl, booktabs, multirow, array",
        _palette_definecolor(palette),
        _palette_definecolor(accent_palette),
        "\\newcommand{\\darkcell}[2]{\\cellcolor{#1}\\textcolor{white}{#2}}",
        "\\newcommand{\\lightcell}[2]{\\cellcolor{#1}\\textcolor{black}{#2}}",
    ]
    for macro, _key, pal in all_macros:
        preamble_lines.append(_cell_macro_def(macro, pal, thresholds_by_macro[macro]))

    # Group systems by arch.
    groups: dict[str, list[str]] = defaultdict(list)
    for stype, alias in systems:
        groups[stype].append(alias)

    n_metric_groups = 1 + len(metric_columns)  # composite + per-metric groups
    n_domains = len(domains_present)
    # Column spec: l l <accent group> | <metric group> | <metric group> | ...
    # Use a thin vertical rule between metric sections as a divider, plus
    # extra horizontal spacing on either side for breathing room.
    sep = "@{\\hskip 8pt}!{\\color{black!25}\\vrule}@{\\hskip 8pt}"
    group_block = "c" * n_domains
    col_spec = "ll@{\\hskip 8pt}" + (sep.join([group_block] * n_metric_groups))

    # Header rows.
    # $\textbf{EVA-A}_{\textbf{pass@1}}$
    header_groups = [("\\textbf{" + title_short + "}$_{\\textbf{pass@1}}$", composite_macro)]
    for _, hdr, prefix in metric_columns:
        header_groups.append((f"\\textbf{{{hdr}}}", prefix))

    multicol_cells = []
    cmidrules = []
    col_idx = 3
    for header_label, _macro in header_groups:
        span = n_domains
        multicol_cells.append(f"\\multicolumn{{{span}}}{{c}}{{{header_label}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + span - 1}}}")
        col_idx += span

    domain_headers = []
    for _ in header_groups:
        for d in domains_present:
            domain_headers.append(f"\\textbf{{{_DOMAIN_DISPLAY.get(d, d)}}}")

    body_lines = []
    body_lines.append("\\toprule")
    body_lines.append("& & " + " & ".join(multicol_cells) + " \\\\")
    body_lines.append(" ".join(cmidrules))
    body_lines.append(
        "\\textbf{Arch.} & \\textbf{System} & " + " & ".join(domain_headers) + " \\\\"
    )
    body_lines.append("\\midrule")

    arch_blocks = []
    for stype in _ARCH_ORDER:
        aliases = groups.get(stype, [])
        if not aliases:
            continue
        n_rows = len(aliases)
        rows = []
        for i, alias in enumerate(aliases):
            cells = []
            for macro, key, _pal in all_macros:
                for d in domains_present:
                    r = lookup.get((alias, d))
                    if r is None:
                        cells.append("--")
                        continue
                    if key in ("eva_a_mean", "eva_x_mean"):
                        v = r["main"].get(key)
                    else:
                        v = (r["per_metric"].get(key) or {}).get("mean")
                    if v is None:
                        cells.append("--")
                    else:
                        cells.append(f"\\{macro}{{{float(v):.3f}}}")
            arch_cell = (
                f"\\multirow{{{n_rows}}}{{*}}{{{_ARCH_DISPLAY[stype]}}}" if i == 0 else ""
            )
            rows.append(f"  {arch_cell} & {_latex_escape(alias)} & " + " & ".join(cells) + " \\\\")
        arch_blocks.append("\n".join(rows))
    body_lines.append("\n\\midrule\n".join(arch_blocks))
    body_lines.append("\\bottomrule")

    table_env = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\n".join(body_lines),
        "\\end{tabular}%",
        "}",
        "\\vspace{6pt}",
        f"\\caption{{{caption.replace('{N}', str(n_trials))}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ]

    return "\n".join(preamble_lines) + "\n\n" + "\n".join(table_env) + "\n"


def write_latex_tables(out_dir: Path, buckets: dict) -> None:
    domains_present = [d for d in DOMAINS if any((d, "clean") == k for k in buckets if buckets[k])]
    if not domains_present:
        return
    n_trials = expected_k_for("clean")
    latex_dir = out_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    accuracy_caption = (
        "Accuracy metrics---EVA-A\\textsubscript{pass@1}, Task Completion, Faithfulness, and "
        "Agent Speech Fidelity---for all evaluated systems under clean-audio "
        "conditions, across the EVA domains over {N} trials. Cell shading is "
        "scaled independently per metric column from the observed minimum to "
        "maximum (darker~=~higher). All scores normalized 0 to 1.0, higher is better."
    )
    experience_caption = (
        "Experience metrics---EVA-X\\textsubscript{pass@1}, Turn-Taking, Conciseness, and "
        "Conversation Progression---for all evaluated systems under clean-audio "
        "conditions, across the EVA domains over {N} trials. Cell shading is "
        "scaled independently per metric column from the observed minimum to "
        "maximum (darker~=~higher). All scores normalized 0 to 1.0, higher is better."
    )

    accuracy_tex = _format_table(
        title_short="EVA-A",
        caption=accuracy_caption,
        label="tab:accuracy-metrics",
        composite_key="eva_a_mean",
        composite_macro="EAcomp",
        metric_columns=[
            ("task_completion", "Task Completion", "TCcell"),
            ("faithfulness", "Faithfulness", "FAcell"),
            ("agent_speech_fidelity", "Agent Speech Fidelity", "SFcell"),
        ],
        palette=_TEAL_PALETTE,
        accent_palette=_ACCENT_PALETTE,
        buckets=buckets,
        domains_present=domains_present,
        n_trials=n_trials,
    )
    experience_tex = _format_table(
        title_short="EVA-X",
        caption=experience_caption,
        label="tab:experience-metrics",
        composite_key="eva_x_mean",
        composite_macro="EXcomp",
        metric_columns=[
            ("turn_taking", "Turn-Taking", "TTcell"),
            ("conciseness", "Conciseness", "COcell"),
            ("conversation_progression", "Conv.\\ Progression", "CPcell"),
        ],
        palette=_PINK_PALETTE,
        accent_palette=_ACCENT_PALETTE,
        buckets=buckets,
        domains_present=domains_present,
        n_trials=n_trials,
    )

    (latex_dir / "accuracy_table.tex").write_text(accuracy_tex)
    (latex_dir / "experience_table.tex").write_text(experience_tex)


# The transcription chart uses the `barfill`/`bardraw` and `barfill2`/
# `bardraw2` colors that are already defined in the NeurIPS preamble — one
# fill per domain, single color across all perturbations within a domain.
_PERT_ORDER_CHART = ["clean", "accent", "background_noise", "both"]
_PERT_SHORT = {
    "clean": "clean",
    "accent": "accent",
    "background_noise": "bg-noise",
    "both": "accent+bg",
}
# Domains shown in the transcription chart (ITSM excluded — too few STTs).
_TRANSCRIPTION_DOMAINS = ["medical_hr"]

# Per-model base colors. Cycled through alphabetically by STT name. Picked to
# be distinct, paper-readable, and harmonious. Each model's perturbations are
# rendered as progressively lighter mixes of the base via `\<name>!<pct>!white`.
_STT_BASE_PALETTE = [
    ("sttA", "1f9e89"),  # teal
    ("sttB", "3d6cb9"),  # blue
    ("sttC", "8b5cb8"),  # violet
    ("sttD", "d0826b"),  # coral
    ("sttE", "c2a74e"),  # gold
    ("sttF", "b5527c"),  # rose
    ("sttG", "5e8c5b"),  # sage
    ("sttH", "9c6b4a"),  # bronze
]

# Tint ramp: clean = full saturation, perturbations get progressively lighter
# (mixed with white). Keeps the same hue so a model's bars read as a group.
_PERT_TINT_PCT = {
    "clean": 100,
    "accent": 70,
    "background_noise": 50,
    "both": 32,
}


def write_latex_transcription_chart(out_dir: Path, buckets: dict) -> None:
    """Vertical grouped bar chart of transcription_accuracy_key_entities for
    the Medical HR domain. One xtick per STT model; up to four touching bars
    per group (clean, accent, bg-noise, accent+bg) drawn as progressively
    lighter shades of a per-model base color."""
    # domain -> perturbation -> stt -> [values]
    per_domain: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for (domain, cat), runs in buckets.items():
        if domain not in _TRANSCRIPTION_DOMAINS or cat not in _PERT_ORDER_CHART:
            continue
        for r in runs:
            stt = r.get("stt")
            if not stt:
                continue
            v = (r["per_metric"].get("transcription_accuracy_key_entities") or {}).get("mean")
            if v is None:
                continue
            per_domain[domain][cat][stt].append(float(v))

    if "medical_hr" not in per_domain:
        return

    latex_dir = out_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    pert_to_stt = per_domain["medical_hr"]
    all_stts: set[str] = set()
    for cat, by_stt in pert_to_stt.items():
        all_stts.update(by_stt.keys())
    avg: dict[str, dict[str, float]] = {
        cat: {stt: sum(vs) / len(vs) for stt, vs in by_stt.items()}
        for cat, by_stt in pert_to_stt.items()
    }

    # Order STTs by clean score asc; reverse so best is on the right.
    sort_pert = "clean" if "clean" in avg else next(iter(avg.keys()))
    stts_xaxis = sorted(
        all_stts, key=lambda s: avg.get(sort_pert, {}).get(s, -1)
    )[::-1]

    perts_present = [
        c for c in _PERT_ORDER_CHART if any(stt in avg.get(c, {}) for stt in stts_xaxis)
    ]
    if not perts_present or not stts_xaxis:
        return

    # Assign each STT a base color (cycling through the palette).
    stt_color: dict[str, str] = {}
    color_defs: list[str] = []
    for i, stt in enumerate(stts_xaxis):
        name, hexv = _STT_BASE_PALETTE[i % len(_STT_BASE_PALETTE)]
        stt_color[stt] = name
        color_defs.append(f"\\definecolor{{{name}}}{{HTML}}{{{hexv}}}")

    symbolic_coords = ", ".join(_latex_escape(s) for s in stts_xaxis)

    # One \addplot per perturbation, but each bar's fill is the model's base
    # color mixed with white at the perturbation-specific tint percentage.
    # pgfplots resolves `\<color>!<pct>!white` per coordinate via point meta,
    # but the simpler cross-cutting approach is one addplot per (stt, pert)
    # using `bar shift` to place each bar within its group. We emit one
    # \addplot per perturbation and rely on `forget plot` siblings + per-
    # coordinate fill via `nodes near coords`-free customization.
    #
    # Cleaner: emit one \addplot PER (stt, pert) pair. Each plot has a single
    # coordinate, exact fill, and explicit `bar shift` to position it next
    # to its siblings within the group.
    bar_w = 5.5  # pt
    n_perts = len(perts_present)
    # Centered offsets: for n_perts bars touching, shifts are
    # (-(n-1)/2, -(n-3)/2, …, (n-1)/2) * bar_width.
    offsets = [(i - (n_perts - 1) / 2.0) * bar_w for i in range(n_perts)]

    addplots: list[str] = []
    for cat_idx, cat in enumerate(perts_present):
        shift = offsets[cat_idx]
        tint = _PERT_TINT_PCT[cat]
        cat_avg = avg.get(cat, {})
        for stt in stts_xaxis:
            v = cat_avg.get(stt)
            if v is None:
                continue
            base = stt_color[stt]
            fill_expr = base if tint == 100 else f"{base}!{tint}!white"
            addplots.append(
                f"\\addplot+[ybar, bar width={bar_w}pt, bar shift={shift:.2f}pt, "
                f"fill={fill_expr}, draw={base}!80!black, line width=0.3pt, forget plot] "
                f"coordinates {{({_latex_escape(stt)},{v:.4f})}};"
            )

    # Manual legend swatches for the perturbation tint ramp. We use
    # \addlegendimage rather than dummy \addplot coordinates because the
    # axis uses symbolic x coords and any dummy coordinate would have to
    # be one of the existing STT names.
    ref_color = stt_color[stts_xaxis[0]]
    legend_entries = []
    for cat in perts_present:
        tint = _PERT_TINT_PCT[cat]
        fill_expr = ref_color if tint == 100 else f"{ref_color}!{tint}!white"
        label = _PERT_DISPLAY[cat]
        legend_entries.append(
            f"\\addlegendimage{{ybar, bar width=6pt, fill={fill_expr}, "
            f"draw={ref_color}!80!black, line width=0.3pt, area legend}}"
            f"\\addlegendentry{{{label}}}"
        )

    chart_width = max(7.0, 2.2 * len(stts_xaxis) + 1.0)
    chart_height = 5.0

    tikz = (
            "\\begin{tikzpicture}\n"
            "\\begin{axis}[\n"
            f"  width={chart_width:.2f}cm,\n"
            f"  height={chart_height:.2f}cm,\n"
            "  ymin=0, ymax=1.0,\n"
            "  ylabel={\\footnotesize Key entity transcription accuracy},\n"
            "  ylabel style={font=\\footnotesize, yshift=-2pt},\n"
            "  ytick={0,0.2,0.4,0.6,0.8,1.0},\n"
            "  yticklabel style={font=\\scriptsize},\n"
            f"  symbolic x coords={{{symbolic_coords}}},\n"
            "  xtick=data,\n"
            "  xticklabel style={font=\\footnotesize, yshift=-1pt},\n"
            "  enlarge x limits=0.12,\n"
            "  axis x line*=bottom,\n"
            "  axis y line*=left,\n"
            "  axis line style={gray!55, line width=0.5pt},\n"
            "  ymajorgrids=true,\n"
            "  grid style={solid, gray!15, line width=0.4pt},\n"
            "  tick align=outside,\n"
            "  tick style={gray!55},\n"
            "  clip=false,\n"
            "  legend style={\n"
            "    font=\\scriptsize, draw=gray!40, line width=0.3pt,\n"
            "    fill=white, fill opacity=0.95, text opacity=1,\n"
            "    at={(0.5,-0.18)}, anchor=north, legend columns=-1,\n"
            "    /tikz/every even column/.append style={column sep=6pt},\n"
            "  },\n"
            "]\n"
            + "\n".join(addplots)
            + "\n% Legend swatches (single-coordinate dummy bars off-axis)\n"
            + "\n".join(legend_entries)
            + "\n\\end{axis}\n\\end{tikzpicture}"
    )

    figure = (
            "% Auto-generated by scripts/aggregate_eva_results.py\n"
            "% Requires: pgfplots, xcolor.\n"
            + "\n".join(color_defs)
            + "\n\n\\begin{figure}[h]\n\\centering\n"
            + tikz
            + "\n\\caption{Key entity transcription accuracy by STT model on the "
              "Medical HR domain. Each model is shown in its own hue; within a "
              "group, lighter shades correspond to the perturbation conditions "
              "(clean, accent, background noise, and accent + background noise). "
              "Scores are averaged across all systems sharing the same STT.}\n"
              "\\label{fig:key-entity-transcription}\n"
              "\\end{figure}\n"
    )
    (latex_dir / "transcription_chart.tex").write_text(figure)


# ── Output orchestration ──────────────────────────────────────────────
def _bucket_dir(out_dir: Path, domain: str, cat: str) -> Path:
    """Resolve the subdirectory for a bucket.

    clean/{domain}/                        for cat == "clean"
    perturbed/{cat}/{domain}/              for cat in {accent, background_noise, both}
    """
    if cat == "clean":
        return out_dir / "clean" / domain
    return out_dir / "perturbed" / cat / domain


def write_outputs(
        out_dir: Path,
        buckets: dict[tuple[str, str], list[dict]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (domain, cat), runs in sorted(buckets.items()):
        if not runs:
            continue
        bucket_dir = _bucket_dir(out_dir, domain, cat)
        bucket_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{domain}_{cat}"
        write_main_csv(bucket_dir / f"{prefix}_main.csv", runs)
        write_pass_k_csv(bucket_dir / f"{prefix}_pass_k.csv", runs, expected_k_for(cat))
        write_per_metric_csv(
            bucket_dir / f"{prefix}_per_metric_accuracy.csv", runs, ACCURACY_METRICS
        )
        write_per_metric_csv(
            bucket_dir / f"{prefix}_per_metric_experience.csv", runs, EXPERIENCE_METRICS
        )
        write_per_metric_csv(
            bucket_dir / f"{prefix}_per_metric_diagnostic.csv", runs, DIAGNOSTIC_METRICS
        )
        write_plot(
            bucket_dir / f"{prefix}_accuracy_vs_experience.png", domain, cat, runs
        )

    write_domain_averaged_plot(out_dir, buckets)
    write_latex_tables(out_dir, buckets)
    write_latex_transcription_chart(out_dir, buckets)


def write_status(out_dir: Path, status: dict, runs_dir: Path) -> None:
    status_full = {
        "generated_at": datetime.now().isoformat(),
        "runs_dir": str(runs_dir),
        **status,
    }
    with (out_dir / "status.json").open("w") as f:
        json.dump(status_full, f, indent=2, default=str)


def zip_folder(folder: Path) -> Path:
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for child in folder.rglob("*"):
            if child.is_file():
                zf.write(child, arcname=child.relative_to(folder.parent))
    return zip_path


# ── CLI ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument(
        "--success-threshold", default=DEFAULT_SUCCESS_THRESHOLD, type=float
    )
    args = parser.parse_args()

    buckets, status = aggregate(args.runs_dir, args.success_threshold)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = args.output_dir / f"eva_results_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_outputs(out_dir, buckets)
    write_status(out_dir, status, args.runs_dir)
    zip_path = zip_folder(out_dir)

    populated = sum(1 for v in buckets.values() if v)
    total_systems = sum(len(v) for v in buckets.values())
    print(f"Discovered runs:     {status['total_runs_discovered']}")
    print(f"Kept runs:           {status['total_runs_kept']}")
    print(f"Duplicates resolved: {len(status['skipped']['duplicate_resolved'])}")
    print(f"Imperfect rate:      {len(status.get('imperfect_success_rate', []))}")
    print(f"Buckets written:     {populated}")
    print(f"Total systems:       {total_systems}")
    print(f"Output folder:       {out_dir}")
    print(f"Zip:                 {zip_path}")


if __name__ == "__main__":
    main()
