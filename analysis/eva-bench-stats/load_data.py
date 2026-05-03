"""Shared utilities for loading raw EVA-Bench run directories into DataFrames."""

import json
import re
from pathlib import Path

import pandas as pd

_SCORES_COLS = ["run_id", "record_id", "trial", "metric", "normalized_score"]
_AGG_COLS = ["run_id", "record_id", "trial", "EVA-A_pass", "EVA-X_pass", "EVA-A_mean", "EVA-X_mean", "EVA-overall_mean"]


def _iter_trial_metrics(run_dir: Path):
    """Yield (record_id, trial, metrics_data) for every trial metrics.json in a run dir."""
    records_dir = run_dir / "records"
    if not records_dir.is_dir():
        return
    for record_dir in sorted(records_dir.iterdir()):
        if not record_dir.is_dir():
            continue
        record_id = record_dir.name
        for trial_dir in sorted(record_dir.iterdir()):
            if not trial_dir.is_dir() or not re.fullmatch(r"trial_\d+", trial_dir.name):
                continue
            trial = int(trial_dir.name.split("_")[1])
            metrics_path = trial_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            yield record_id, trial, json.loads(metrics_path.read_text())


def load_scores(run_dir: Path) -> pd.DataFrame:
    """Load per-(scenario, trial, metric) normalized scores from a run directory.

    Returns DataFrame with columns: run_id, record_id, trial, metric, normalized_score.
    Rows where the metric has an error are excluded.
    """
    run_id = run_dir.name
    rows = []
    for record_id, trial, data in _iter_trial_metrics(run_dir):
        for metric_name, metric_data in data.get("metrics", {}).items():
            if metric_data.get("error"):
                continue
            score = metric_data.get("normalized_score")
            if score is None:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "record_id": record_id,
                    "trial": trial,
                    "metric": metric_name,
                    "normalized_score": float(score),
                }
            )
    return pd.DataFrame(rows, columns=_SCORES_COLS) if rows else pd.DataFrame(columns=_SCORES_COLS)


def load_aggregate_scores(run_dir: Path) -> pd.DataFrame:
    """Load per-(scenario, trial) composite scores from a run directory.

    Returns DataFrame with columns: run_id, record_id, trial,
    EVA-A_pass, EVA-X_pass, EVA-A_mean, EVA-X_mean, EVA-overall_mean.
    Rows with missing aggregate_metrics are excluded.
    """
    run_id = run_dir.name
    rows = []
    for record_id, trial, data in _iter_trial_metrics(run_dir):
        agg = data.get("aggregate_metrics", {})
        if not agg:
            continue
        rows.append({"run_id": run_id, "record_id": record_id, "trial": trial, **agg})
    return pd.DataFrame(rows, columns=_AGG_COLS) if rows else pd.DataFrame(columns=_AGG_COLS)


_ITER_SCORES_COLS = [
    "run_id",
    "run_label",
    "domain",
    "record_id",
    "trial",
    "iteration",
    "metric",
    "normalized_score",
]
_ITER_AGG_COLS = [
    "run_id",
    "run_label",
    "domain",
    "record_id",
    "trial",
    "iteration",
    "EVA-A_pass",
    "EVA-X_pass",
    "EVA-overall_pass",
    "EVA-A_mean",
    "EVA-X_mean",
    "EVA-overall_mean",
]


def _iter_iterations(run_dir: Path):
    """Yield (iteration, record_id, trial, metrics_data) for every trial in every iter_N dir."""
    if not run_dir.is_dir():
        return
    for iter_dir in sorted(run_dir.iterdir()):
        if not iter_dir.is_dir() or not re.fullmatch(r"iter_\d+", iter_dir.name):
            continue
        iteration = int(iter_dir.name.split("_")[1])
        for record_id, trial, data in _iter_trial_metrics(iter_dir):
            yield iteration, record_id, trial, data


def load_scores_iter(run_id: str, run_label: str, archive_root: Path, domain: str | None = None) -> pd.DataFrame:
    """Load per-(iteration, record, trial, metric) scores from an iteration-archive run.

    Archive layout: <archive_root>/<run_id>/iter_<N>/records/<record_id>/trial_<N>/metrics.json

    Returns DataFrame with columns:
        run_id, run_label, record_id, trial, iteration, metric, normalized_score
    """
    rows = []
    run_dir = archive_root / run_id
    for iteration, record_id, trial, data in _iter_iterations(run_dir):
        for metric_name, metric_data in data.get("metrics", {}).items():
            if metric_data.get("error"):
                continue
            score = metric_data.get("normalized_score")
            if score is None:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "run_label": run_label,
                    "domain": domain,
                    "record_id": record_id,
                    "trial": trial,
                    "iteration": iteration,
                    "metric": metric_name,
                    "normalized_score": float(score),
                }
            )
    return pd.DataFrame(rows, columns=_ITER_SCORES_COLS) if rows else pd.DataFrame(columns=_ITER_SCORES_COLS)


def load_aggregate_scores_iter(
    run_id: str, run_label: str, archive_root: Path, domain: str | None = None
) -> pd.DataFrame:
    """Load per-(iteration, record, trial) composite scores from an iteration-archive run.

    Returns DataFrame with columns:
        run_id, run_label, record_id, trial, iteration,
        EVA-A_pass, EVA-X_pass, EVA-overall_pass, EVA-A_mean, EVA-X_mean, EVA-overall_mean
    """
    rows = []
    run_dir = archive_root / run_id
    for iteration, record_id, trial, data in _iter_iterations(run_dir):
        agg = data.get("aggregate_metrics", {})
        if not agg:
            continue
        rows.append(
            {
                "run_id": run_id,
                "run_label": run_label,
                "domain": domain,
                "record_id": record_id,
                "trial": trial,
                "iteration": iteration,
                **agg,
            }
        )
    return (
        pd.DataFrame(rows).reindex(columns=_ITER_AGG_COLS, fill_value=None)
        if rows
        else pd.DataFrame(columns=_ITER_AGG_COLS)
    )
