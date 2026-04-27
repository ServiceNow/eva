# Judge Variance Analysis

Interactive tool for quantifying the **relative importance of different sources of variance**
in EVA metric scores. Understanding these sources answers: how much of the spread we observe
across benchmark runs is real signal vs. noise?

## Sources of Variance

| Source | Description | How isolated |
|---|---|---|
| **Judge variance** | LLM judge produces different outputs when re-evaluating the same conversation (non-deterministic text generation) | Std dev across N iterations on identical data |
| **Trajectory variance** | Genuine differences in conversation trajectories across simulation trials | Std dev across M trials, averaged over iterations to remove judge noise |
| **Scenario variance (ICC)** | How well metric scores differentiate between scenarios vs. within-scenario noise | Intraclass correlation coefficient |

## Metrics Analyzed

The 6 LLM-judge metrics in EVA (this study adds no new metrics):

| Metric | Judge type | Notes |
|---|---|---|
| `faithfulness` | Text | |
| `agent_speech_fidelity` | Audio | |
| `conversation_progression` | Text | |
| `turn_taking` | Text | |
| `conciseness` | Text | |
| `transcription_accuracy_key_entities` | Text | Cascade only; silently skipped for S2S runs |

Deterministic/code-based metrics are excluded — they produce identical results each time.

## Study Design

- **N iterations** per run: re-run `--force-rerun-metrics` N times on the same existing
  conversation data (same audio, transcripts, tool calls). Isolates judge stochasticity.
- **M trials** per scenario: each scenario has M simulation trials with different conversation
  trajectories. Isolates real behavioral variance after averaging out judge noise.
- **ICC**: intraclass correlation at the scenario level — what fraction of total score variance
  is attributable to scenario identity (signal) vs. residual noise.

## Archive Structure

Data is collected into `output/judge_variance_analysis/` (gitignored):

```
output/judge_variance_analysis/
  <run_id>/
    iter_1/
      metrics_summary.json
      records/<record_id>/trial_<n>/metrics.json
    iter_2/ ...
    iter_3/ ...
  run.log
```

Only `metrics.json` files (not audio) are archived to keep storage small.

## Setup

**1. Register your runs**

Create `local/judge_variance_analysis/runs_config.py` (gitignored):

```python
RUNS: list[str] = ["your_run_id", ...]  # run IDs from output/

RUN_LABELS: dict[str, str] = {
    "your_run_id": "Human-readable label",
}

RUN_METADATA: dict[str, dict] = {
    "your_run_id": {"type": "cascade", "stt": "...", "llm": "...", "tts": "..."},
    # or for S2S: {"type": "s2s", "s2s": "model-name", "voice": "voice-name"}
}
```

**2. Collect iterations**

Edit `NUM_ITERATIONS`, `METRICS`, and `SKIP_EXISTING` at the top of `run_iterations.py`
if needed, then run:

```bash
uv run python apps/judge_variance_analysis/run_iterations.py
```

This can take a long time. Use `caffeinate -i` on macOS to prevent sleep:

```bash
caffeinate -i uv run python apps/judge_variance_analysis/run_iterations.py
```

The script is resumable: re-run it and already-archived iterations are skipped
(`SKIP_EXISTING = True`).

## Launching the App

```bash
uv run streamlit run apps/judge_variance_analysis/app.py --server.port 8502 --theme.primaryColor "#6b7280"
```

The app reads from `output/judge_variance_analysis/` and `local/judge_variance_analysis/data/`
(optional CSV cache). Use the data source selector in the sidebar to switch between them.

## Extending

**Add a new run:** register in `local/judge_variance_analysis/runs_config.py` and add the
run ID to `RUNS`. All analysis functions operate on the flat dataframe from `load_scores()`
and pick up new runs automatically.

**Add a new analysis:** add functions to `analysis.py`, wire them into new tabs in `app.py`.
