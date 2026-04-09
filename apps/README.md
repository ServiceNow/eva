# EVA Apps

Streamlit applications for exploring EVA results.

## Analysis App

Interactive dashboard for visualizing and comparing results.

### Usage

```bash
streamlit run apps/analysis.py
```

By default, the app looks for runs in the `output/` directory. You can change this in the sidebar or by setting the `EVA_OUTPUT_DIR` environment variable:

```bash
EVA_OUTPUT_DIR=path/to/results streamlit run apps/analysis.py
```

### Views

**Cross-Run Comparison** — Compare aggregate metrics across multiple runs. Filter by model, provider, and pipeline type. Includes an EVA scatter plot (accuracy vs. experience) and per-metric bar charts.

![Cross-Run Comparison view](images/cross_run_comparison.png)

**Run Overview** — Drill into a single run: per-category metric breakdowns, score distributions, and a full records table with per-metric scores.

![Run Overview view](images/run_overview.png)

**Record Detail** — Deep-dive into individual conversation records:
- Audio playback (mixed recording)
- Transcript with color-coded speaker turns
- Metric scores with explanations
- Conversation trace: tool calls, LLM calls, and audit log entries with a timeline view
- Database state diff (expected vs. actual)
- User goal, persona, and ground truth from the evaluation record

![Record Detail view](images/analysis_record_detail.png)

### Sidebar Navigation

1. **Output Directory** — Path to the directory containing run folders
2. **View** — Switch between the three views above
3. **Run Selection** — Pick a run (with metadata summary)
4. **Record Selection** — Pick a record within the selected run
5. **Trial Selection** — If a record has multiple trials, pick one
