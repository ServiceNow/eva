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

---

## Audio Analysis Tab

The **Audio Analysis** tab in the Record Detail view renders an interactive Plotly figure built from the audio files and timestamp logs of a single trial. It is implemented in `apps/audio_plots.py`.

### Subplots

| Row | Content | Always shown |
|-----|---------|--------------|
| 1 | Mixed audio waveform, colour-coded by speaker | Yes |
| 2 | Mixed audio spectrogram | Optional (checkbox) |
| 3 | ElevenLabs audio waveform, colour-coded by speaker | Yes |
| 4 | ElevenLabs audio spectrogram | Optional (checkbox) |
| 5 | Speaker Turn Timeline with per-turn durations and pause markers | Yes |

Toggle spectrograms on or off using the checkboxes above the chart. Results are cached per trial so switching between records is fast after the first load.

### Colour Coding

| Colour | Meaning |
|--------|---------|
| Blue | User speaker turn |
| Orange | Assistant speaker turn |
| Gray (semi-transparent line) | Silence — audio not covered by any speaker turn |
| Gray shaded box | Pause — gap between consecutive speaker turns |

Colours are chosen for visibility in both Streamlit light and dark mode.

### Hover Tooltips

Hovering over any waveform sample shows the **transcript text** for the active speaker turn, along with the turn start/end time and duration. Hovering over a pause region shows the pause duration and the from/to speakers. The timeline row shows the same transcript text when hovering over each bar.

### Silence vs. Pause

- **Pause** — derived from speaker turn event logs. The gap between one speaker's audio end event and the next speaker's audio start event: `pause = turns[i+1].start − turns[i].end`. Only recorded when `> 0`.
- **Silence** — derived from the waveform timeline. Any portion of the audio not covered by a speaker turn event (including audio before the first turn or after the last turn).

A Pause always coincides with a Silence region, but Silence can be wider (e.g. leading/trailing audio with no events).
