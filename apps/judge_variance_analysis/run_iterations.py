# apps/judge_variance_analysis/run_iterations.py
"""Run --force-rerun-metrics N times per run and archive results.

Usage (run from project root):
    uv run python apps/judge_variance_analysis/run_iterations.py

Configure which runs to process in local/judge_variance_analysis/runs_config.py:
    RUNS: list[str] = ["your_run_id", ...]

Edit the constants below to change iteration count, metrics, and skip behavior.

Recovery: if the script is interrupted, re-run it. Iterations that were fully
archived are skipped automatically (SKIP_EXISTING = True). Set SKIP_EXISTING = False
to force a full re-run. The log file at output/judge_variance_analysis/run.log
records all progress with timestamps.
"""

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Configure iteration settings here ─────────────────────────────────────────
NUM_ITERATIONS = 3
METRICS = [
    "faithfulness",
    "agent_speech_fidelity",
    "conversation_progression",
    "conciseness",
    "transcription_accuracy_key_entities",  # cascade only; silently skipped for audio-native
]
SKIP_EXISTING = True  # Set False to overwrite already-archived iterations
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
ARCHIVE_DIR = OUTPUT_DIR / "judge_variance_analysis"


def _load_runs() -> list[str]:
    """Load RUNS from local/judge_variance_analysis/runs_config.py, or [] if absent."""
    config_path = PROJECT_ROOT / "local" / "judge_variance_analysis" / "runs_config.py"
    if not config_path.exists():
        return []
    spec = importlib.util.spec_from_file_location("_runs_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "RUNS", [])


RUNS = _load_runs()


def _setup_logging() -> logging.Logger:
    """Configure logging to stdout and a persistent log file."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ARCHIVE_DIR / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


def _is_iteration_done(run_id: str, iteration: int) -> bool:
    """Return True if this iteration has already been fully archived."""
    records_dir = ARCHIVE_DIR / run_id / f"iter_{iteration}" / "records"
    if not records_dir.exists():
        return False
    files = list(records_dir.rglob("metrics.json"))
    return len(files) > 0


def _run_metrics(run_id: str, log: logging.Logger) -> bool:
    """Call --force-rerun-metrics for a single run. Returns True on success."""
    cmd = [
        "uv",
        "run",
        "python",
        "main.py",
        "--run-id",
        run_id,
        "--force-rerun-metrics",
        "--metrics",
        ",".join(METRICS),
    ]
    log.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=os.environ.copy())
    if result.returncode != 0:
        log.error(f"  FAILED: command exited with code {result.returncode} for run {run_id}")
        return False
    return True


def _archive_iteration(run_id: str, iteration: int, log: logging.Logger) -> int:
    """Copy metrics.json files into the archive. Returns number of files copied."""
    src_records = OUTPUT_DIR / run_id / "records"
    dst_iter = ARCHIVE_DIR / run_id / f"iter_{iteration}"

    if dst_iter.exists():
        shutil.rmtree(dst_iter)
    dst_iter.mkdir(parents=True)

    src_summary = OUTPUT_DIR / run_id / "metrics_summary.json"
    if src_summary.exists():
        shutil.copy2(src_summary, dst_iter / "metrics_summary.json")
    else:
        log.warning(f"  metrics_summary.json not found for {run_id}")

    n_copied = 0
    for metrics_json in src_records.rglob("metrics.json"):
        relative = metrics_json.relative_to(src_records)
        dst = dst_iter / "records" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_json, dst)
        n_copied += 1

    if n_copied == 0:
        log.warning(f"  No metrics.json files found under {src_records}")
    else:
        log.info(f"  Archived iter {iteration} — {n_copied} files → {dst_iter.relative_to(PROJECT_ROOT)}")
    return n_copied


def main() -> None:
    log = _setup_logging()

    if not RUNS:
        log.error(
            "No runs configured. Create local/judge_variance_analysis/runs_config.py with RUNS = ['your_run_id', ...]"
        )
        return

    log.info("=" * 60)
    log.info("Judge variance study — starting")
    log.info(f"  Runs:          {RUNS}")
    log.info(f"  Iterations:    {NUM_ITERATIONS}")
    log.info(f"  Metrics:       {METRICS}")
    log.info(f"  Archive:       {ARCHIVE_DIR.relative_to(PROJECT_ROOT)}")
    log.info(f"  Skip existing: {SKIP_EXISTING}")
    log.info("=" * 60)

    completed: list[tuple[str, int]] = []
    skipped: list[tuple[str, int]] = []
    failed: list[tuple[str, int]] = []

    for run_id in RUNS:
        run_dir = OUTPUT_DIR / run_id
        if not run_dir.exists():
            log.warning(f"SKIP {run_id} — run directory not found")
            continue

        log.info(f"Run: {run_id}")
        for i in range(1, NUM_ITERATIONS + 1):
            if SKIP_EXISTING and _is_iteration_done(run_id, i):
                log.info(f"  Iteration {i}/{NUM_ITERATIONS} — already archived, skipping")
                skipped.append((run_id, i))
                continue

            log.info(f"  Iteration {i}/{NUM_ITERATIONS} — starting")
            success = _run_metrics(run_id, log)

            if not success:
                log.error(f"  Iteration {i}/{NUM_ITERATIONS} — metrics run failed, skipping archive")
                failed.append((run_id, i))
                continue

            n_files = _archive_iteration(run_id, i, log)
            if n_files == 0:
                log.error(f"  Iteration {i}/{NUM_ITERATIONS} — archive empty, marking as failed")
                failed.append((run_id, i))
            else:
                log.info(f"  Iteration {i}/{NUM_ITERATIONS} — done")
                completed.append((run_id, i))

        log.info(f"Run {run_id} complete.")

    log.info("=" * 60)
    log.info("Study complete.")
    log.info(f"  Completed: {len(completed)} iterations")
    log.info(f"  Skipped:   {len(skipped)} iterations (already archived)")
    log.info(f"  Failed:    {len(failed)} iterations")
    if failed:
        log.warning("  Failed iterations (re-run to retry):")
        for run_id, i in failed:
            log.warning(f"    {run_id}  iter {i}")
    log.info(f"  Log saved: {ARCHIVE_DIR / 'run.log'}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
