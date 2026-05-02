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
