# apps/judge_variance_analysis/load_data.py
"""Load archived judge variance iteration data into analysis-ready dataframes."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "output" / "judge_variance_analysis"
CSV_DIR = PROJECT_ROOT / "local" / "judge_variance_analysis" / "data"

# To register your runs, create local/judge_variance_analysis/runs_config.py with:
#
# RUNS: list[str] = ["your_run_id", ...]
#
# RUN_LABELS: dict[str, str] = {
#     "your_run_id": "Human-readable label",
# }
# RUN_METADATA: dict[str, dict] = {
#     "your_run_id": {"type": "cascade", "stt": "...", "llm": "...", "tts": "..."},
#     # or for S2S: {"type": "s2s", "s2s": "model-name", "voice": "voice-name"}
# }


def _load_runs_config() -> tuple[dict, dict]:
    """Load RUN_LABELS and RUN_METADATA from local/judge_variance_analysis/runs_config.py, or ({}, {}) if absent."""
    config_path = PROJECT_ROOT / "local" / "judge_variance_analysis" / "runs_config.py"
    if not config_path.exists():
        return {}, {}
    spec = importlib.util.spec_from_file_location("_runs_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "RUN_LABELS", {}), getattr(module, "RUN_METADATA", {})


RUN_LABELS, RUN_METADATA = _load_runs_config()


def _iter_metrics_files(archive_dir: Path):
    """Yield (run_id, iteration, record_id, trial, metrics_data) for every metrics.json."""
    if not archive_dir.is_dir():
        return
    for run_dir in sorted(archive_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        for iter_dir in sorted(run_dir.iterdir()):
            if not iter_dir.is_dir() or not iter_dir.name.startswith("iter_"):
                continue
            try:
                iteration = int(iter_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            records_dir = iter_dir / "records"
            if not records_dir.exists():
                continue
            for record_dir in sorted(records_dir.iterdir()):
                record_id = record_dir.name
                for trial_dir in sorted(record_dir.iterdir()):
                    if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                        continue
                    try:
                        trial = int(trial_dir.name.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    metrics_path = trial_dir / "metrics.json"
                    if not metrics_path.exists():
                        continue
                    data = json.loads(metrics_path.read_text())
                    yield run_id, iteration, record_id, trial, data


def load_scores(archive_dir: Path = ARCHIVE_DIR) -> pd.DataFrame:
    """Load per-(run, iteration, record, trial, metric) normalized scores.

    Returns a flat dataframe with columns:
        run_id, run_label, record_id, trial, iteration, metric, normalized_score
    Rows with errors are excluded.
    """
    rows = []
    for run_id, iteration, record_id, trial, data in _iter_metrics_files(archive_dir):
        for metric_name, metric_data in data.get("metrics", {}).items():
            if metric_data.get("error"):
                continue
            score = metric_data.get("normalized_score")
            if score is None:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "run_label": RUN_LABELS.get(run_id, run_id),
                    "record_id": record_id,
                    "trial": trial,
                    "iteration": iteration,
                    "metric": metric_name,
                    "normalized_score": float(score),
                }
            )
    return pd.DataFrame(rows)


def load_aggregate_scores(archive_dir: Path = ARCHIVE_DIR) -> pd.DataFrame:
    """Load per-(run, iteration, record, trial) composite (aggregate) scores.

    Returns a flat dataframe with columns:
        run_id, run_label, record_id, trial, iteration,
        EVA-A_pass, EVA-X_pass, EVA-overall_pass, EVA-A_mean, EVA-X_mean, EVA-overall_mean
    """
    rows = []
    for run_id, iteration, record_id, trial, data in _iter_metrics_files(archive_dir):
        agg = data.get("aggregate_metrics", {})
        if not agg:
            continue
        row = {
            "run_id": run_id,
            "run_label": RUN_LABELS.get(run_id, run_id),
            "record_id": record_id,
            "trial": trial,
            "iteration": iteration,
        }
        row.update(agg)
        rows.append(row)
    return pd.DataFrame(rows)


def get_run_metadata() -> dict[str, dict]:
    """Return run metadata dict keyed by run_id."""
    return RUN_METADATA


def _latest_csv(prefix: str) -> Path | None:
    matches = sorted(CSV_DIR.glob(f"{prefix}_*.csv"))
    return matches[-1] if matches else None


def load_scores_from_csv() -> pd.DataFrame:
    path = _latest_csv("scores")
    if path is None:
        raise FileNotFoundError(f"No scores CSV found in {CSV_DIR}")
    return pd.read_csv(path)


def load_aggregate_scores_from_csv() -> pd.DataFrame:
    path = _latest_csv("aggregate_scores")
    if path is None:
        raise FileNotFoundError(f"No aggregate scores CSV found in {CSV_DIR}")
    return pd.read_csv(path)


def latest_csv_timestamp() -> str | None:
    """Return the timestamp from the most recent scores CSV filename, or None."""
    path = _latest_csv("scores")
    if path is None:
        return None
    parts = path.stem.split("_", 1)
    return parts[1] if len(parts) > 1 else None


def available_sources() -> list[str]:
    """Return available data sources in preference order: 'csv' and/or 'json'."""
    sources = []
    if _latest_csv("scores") is not None and _latest_csv("aggregate_scores") is not None:
        sources.append("csv")
    if ARCHIVE_DIR.is_dir() and any(ARCHIVE_DIR.iterdir()):
        sources.append("json")
    return sources
