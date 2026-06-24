#!/usr/bin/env python3
"""Write perturbation_delta + metric_values blocks into leaderboardStats.json from results_*.csv.

Design / contract:
  - ADDITIVE ONLY. Touches only the metrics listed in the config, on systems that
    already exist in the leaderboard JSON. Never writes `clean` or system metadata —
    those belong to the leaderboard (clean) pipeline, which owns each system's row.
  - Run this AFTER the leaderboard pipeline. A new system must already have a row
    (added by that pipeline) before its perturbation data can be attached here; if a
    results system has no matching row, this script aborts rather than inventing one.
  - DOES NOT PRUNE. Metrics/systems present in the JSON but absent from the current
    results are left untouched (by design, for now).
  - Idempotent: a block is written only when it is new or differs from what's there.

All deployment specifics (paths, which metrics, label→name overrides) live in a local
(gitignored) config so this script carries no data/host information. Default is
DRY-RUN (writes <json>.regen + prints a verify diff); pass --write to apply.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "local/perturbations/regenerate_perturbation_blocks.yaml"
DOMAINS = ["airline", "itsm", "medical_hr"]


def _f(x):
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else float(x)


def _bool(x):
    return str(x).strip().lower() in ("true", "1", "1.0")


def _delta_node(r):
    return {
        "point": _f(r["observed_mean_delta"]),
        "ci_lower": _f(r["ci_lower"]),
        "ci_upper": _f(r["ci_upper"]),
        "corrected_p": _f(r["corrected_p"]),
        "raw_p": _f(r["raw_p"]),
        "reject": _bool(r["reject"]),
    }


def _ci_node(r):
    return {"point": _f(r["point"]), "ci_lower": _f(r["ci_lower"]), "ci_upper": _f(r["ci_upper"]), "n": int(_f(r["n"]))}


def _build(pooled_csv, perdom_csv, cond_col, node_fn):
    """-> {model_label: {metric: {condition: {pooled: node, per_domain: {domain: node}}}}}"""
    out: dict = {}
    for r in pd.read_csv(pooled_csv).to_dict("records"):
        out.setdefault(r["model_label"], {}).setdefault(r["metric"], {}).setdefault(r[cond_col], {})["pooled"] = (
            node_fn(r)
        )
    for r in pd.read_csv(perdom_csv).to_dict("records"):
        cell = out.setdefault(r["model_label"], {}).setdefault(r["metric"], {}).setdefault(r[cond_col], {})
        cell.setdefault("per_domain", {})[r["domain"]] = node_fn(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="local (gitignored) config YAML")
    ap.add_argument("--write", action="store_true", help="overwrite the JSON (default: dry-run to <json>.regen)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    json_path = ROOT / cfg["leaderboard_json"]
    results = ROOT / cfg["results_dir"]
    delta_metrics = cfg["delta_metrics"]
    value_metrics = cfg.get("value_metrics", [])
    overrides = cfg.get("label_overrides", {})

    delta = _build(
        results / "results_pooled.csv", results / "results_per_domain.csv", "perturbation_condition", _delta_node
    )
    values = _build(
        results / "results_metricvalues_pooled.csv",
        results / "results_metricvalues_per_domain.csv",
        "condition",
        _ci_node,
    )

    doc = json.loads(json_path.read_text())
    before = copy.deepcopy(doc)
    by_name = {s["name"]: s for s in doc["systems"]}

    # Fail loudly if a results system we'd write has no leaderboard row yet
    # (the clean pipeline must add it first — see contract above).
    unmatched = sorted(
        overrides.get(label, label)
        for label in set(delta) | set(values)
        if any(m in delta.get(label, {}) for m in delta_metrics)
        or any(m in values.get(label, {}) for m in value_metrics)
        if overrides.get(label, label) not in by_name
    )
    if unmatched:
        print(
            "ERROR — results systems missing from the leaderboard JSON (run the leaderboard "
            "pipeline first so their rows exist):",
            file=sys.stderr,
        )
        for n in unmatched:
            print(f"   {n}", file=sys.stderr)
        sys.exit(1)

    report = []  # (system, block, metric, kind, max|Δpoint|)
    for label in sorted(set(delta) | set(values)):
        sys_obj = by_name.get(overrides.get(label, label))
        if sys_obj is None:
            continue
        name = sys_obj["name"]
        for block_key, src, metrics in (
            ("perturbation_delta", delta, delta_metrics),
            ("metric_values", values, value_metrics),
        ):
            for m in metrics:
                new = src.get(label, {}).get(m)
                if new is None:
                    continue
                block = sys_obj.setdefault(block_key, {})
                old = block.get(m)
                if old == new:
                    continue  # idempotent: only write new/changed
                block[m] = new
                kind = "add" if old is None else "change"
                report.append((name, block_key, m, kind, _maxdiff(old, new, "point")))

    violations = _verify_scope(before, doc, delta_metrics, value_metrics)

    print(f"\n{json_path.relative_to(ROOT)}: {len({r[0] for r in report})} systems, {len(report)} block writes")
    for name in sorted({r[0] for r in report}):
        print(f"  {name}")
        for _n, blk, m, kind, md in [r for r in report if r[0] == name]:
            extra = f"  (max |Δpoint|={md:.2e})" if md is not None else ""
            print(f"      {blk:18} {kind:6} {m}{extra}")
    if not report:
        print("  (nothing to write — JSON already matches results)")

    print("\nScope check (clean / out-of-scope metrics / other systems must be unchanged):")
    if violations:
        for v in violations:
            print(f"   !! {v}")
        print("ABORTING — out-of-scope change detected; not writing.")
        sys.exit(1)
    print("   OK")

    out_path = json_path if args.write else json_path.with_suffix(json_path.suffix + ".regen")
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"\n{'WROTE' if args.write else 'DRY-RUN wrote'} {out_path.relative_to(ROOT)}"
        + ("" if args.write else "  (re-run with --write to apply)")
    )


def _maxdiff(old, new, key):
    """Max |Δ| in `key` between two condition→node blocks (0.0 if old is a pure add)."""
    if old is None:
        return None
    diffs = []
    for cond, node in new.items():
        on = old.get(cond) or {}
        for bucket in ("pooled",):
            a, b = (node.get(bucket) or {}).get(key), (on.get(bucket) or {}).get(key)
            if a is not None and b is not None:
                diffs.append(abs(a - b))
        for d, dn in (node.get("per_domain") or {}).items():
            ob = (on.get("per_domain") or {}).get(d) or {}
            a, b = dn.get(key), ob.get(key)
            if a is not None and b is not None:
                diffs.append(abs(a - b))
    return max(diffs) if diffs else 0.0


def _verify_scope(before, after, delta_metrics, value_metrics):
    """Confirm the write touched ONLY the configured metrics' blocks.

    Checks: same system set, unchanged `clean`/metadata, and unchanged out-of-scope metrics.
    """
    v = []
    a_by = {s["name"]: s for s in after["systems"]}
    b_by = {s["name"]: s for s in before["systems"]}
    if set(a_by) != set(b_by):
        v.append(f"system set changed: +{set(a_by) - set(b_by)} -{set(b_by) - set(a_by)}")
    scoped = {"perturbation_delta": set(delta_metrics), "metric_values": set(value_metrics)}
    for name, b in b_by.items():
        a = a_by.get(name, {})
        for key in set(b) | set(a):
            if b.get(key) == a.get(key):
                continue
            allowed = scoped.get(key)
            if allowed is None:
                v.append(f"{name}: out-of-scope key changed: {key}")
                continue
            for m in set(b.get(key) or {}) | set(a.get(key) or {}):
                if m not in allowed and (b.get(key) or {}).get(m) != (a.get(key) or {}).get(m):
                    v.append(f"{name}: out-of-scope {key} metric changed: {m}")
    return v


if __name__ == "__main__":
    main()
