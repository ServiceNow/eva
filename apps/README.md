# EVA Apps

Streamlit applications for exploring and configuring EVA.

## Config Editor

Interactive UI for building and editing `.env` configuration files without hand-editing JSON or looking up variable names.

### Usage

```bash
streamlit run apps/config_editor.py
```

The app reads `.env.example` for the full variable set and loads existing values from `.env` if present. Each variable's widget type, enum options, ranges, and tooltips are declared directly in `.env.example` using annotation prefixes. Use the **Preview** button to inspect the generated file before saving, or **Download** to export it without writing to disk.

### `.env.example` Annotation Scheme

The editor is driven entirely from annotated comments in `.env.example` — there is no separate schema file. Each annotation prefix applies to the **immediately following** variable definition (active or inactive). Annotation order doesn't matter, but the block must be contiguous: any blank line or `# ` true-comment between annotations resets the accumulator.

| Prefix | Name | Purpose |
|---|---|---|
| `# ` | True comment | Human-readable prose. Preserved verbatim, never parsed as metadata. |
| `#i ` | Info | Tooltip text shown next to the widget. Multiple `#i` lines join with spaces. |
| `#d ` | Datatype | Widget type — see table below. If omitted, inferred from name + value. |
| `#e ` | Enum options | Comma-separated valid values for `enum` / `multi_enum`. |
| `#r ` | Range | Numeric `min,max` or `min,max,step` for `int` / `float`. |
| `#g ` | Group | Override tab assignment (otherwise inherited from section header). |
| `#x ` | Condition | `VAR=value` — only render when that var equals that value. Comma-separated values are OR (`#x pipeline_mode=LLM,AudioLLM`). Multiple `#x` lines = AND. |
| `#v ` | Inactive var | `#v VARNAME=value` — a variable definition that ships off by default but is fully configurable. |

#### Widget types (`#d`)

| Type | Renders as |
|---|---|
| `string` | `st.text_input` |
| `secret` | `st.text_input(type="password")` |
| `bool` | `st.checkbox` |
| `int` | `st.number_input` (integers, range from `#r`) |
| `float` | `st.number_input` (floats, range from `#r`) |
| `enum` | `st.selectbox` (options from `#e`) |
| `multi_enum` | `st.multiselect` (options from `#e`) |
| `csv_list` | `st.text_input` split/joined on comma |
| `path` | `st.text_input` with existence hint |
| `json_object` | Key/value table + raw JSON expander |
| `json_deployment_list` | Special-cased deployment-card editor for `EVA_MODEL_LIST` |

#### Widget inference (when `#d` is omitted)

- Name contains `KEY`, `SECRET`, `TOKEN`, or `PASSWORD` → `secret`
- Name contains `CREDENTIALS` or ends with `_PATH` / `_DIR` → `path`
- Value is `true` / `false` → `bool`
- Value parses as an integer → `int`, as a float → `float`
- Value looks like a JSON array containing `model_name` → `json_deployment_list`
- Value looks like a JSON array or object → `json_object`
- Otherwise → `string`

#### Section headers

Top-level groups are declared by a 3-line header block. Variables that follow inherit the group name until the next header.

```bash
# ==============================================
# Voice Pipeline
# ==============================================
```

The section title must match one of the tab name constants in [`config_schema.py`](config_schema.py) (`API Configs`, `Voice Pipeline`, `LiteLLM Deployments`, `Framework & Runtime`, `Turn Detection & VAD`, `User Config`, `Debug & Logging`).

#### Variable states

```bash
# Just a note — ignored entirely.

#i Maximum parallel conversations.
#d int
#r 1,100,1
EVA_MAX_CONCURRENT_CONVERSATIONS=5        # active — written to .env

#i Domain for dataset/agent paths.
#d enum
#e airline,itsm,medical_hr
#v EVA_DOMAIN=airline                     # inactive — user can enable in UI

#i French accent agent ID.
#d secret
#x perturbation_mode=Accent
#x EVA_PERTURBATION__ACCENT=french
#v EVA_FRENCH_ACCENT_USER_F=              # only renders when both conditions hold
```

#### Conditions and modes

`#x` conditions can reference either:
- Another env variable's value (e.g. `#x EVA_PERTURBATION__ACCENT=french`)
- A UI-only state key managed by a mutex radio button (e.g. `#x pipeline_mode=LLM`)

Mutex radio buttons are declared in [`config_schema.py`](config_schema.py) via `MUTEX_RADIOS`. Each radio writes to a session-state key (`pipeline_mode`, `perturbation_mode`) that `#x` conditions can match against.

#### Serialization rules

When the user saves `.env`:

| In `.env.example` | User sets a value | Disabled by mutex / `#x` | Output |
|---|---|---|---|
| Active (`VAR=…`) | yes | no | `VAR=value` |
| Active (`VAR=…`) | no | no | original line verbatim |
| Active (`VAR=…`) | any | yes | `#v VAR=value` (or example value) |
| Inactive (`#v VAR=…`) | yes | no | `VAR=value` (activated) |
| Inactive (`#v VAR=…`) | no | any | `#v VAR=…` verbatim |
| Not in template, in user's loaded `.env` | — | — | appended in matching tab section (KEY/URL → API Configs, `EVA_*` → Framework & Runtime, otherwise Misc) |

Round-tripping is lossless: `serialize_env({}, parse_env_example(...))` reproduces the original file byte-for-byte.

#### Implementation

- [`config_io.py`](config_io.py) — `parse_env_example`, `load_env`, `serialize_env`, `compute_disabled`. Pure functions, no Streamlit dependency.
- [`config_schema.py`](config_schema.py) — group constants, tab ordering, mutex radio definitions. Everything else lives in `.env.example`.
- [`config_editor.py`](config_editor.py) — Streamlit UI that dispatches on `AnnotatedVar.widget`.

---

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

| Row | Content | Shown when |
|-----|---------|------------|
| 1 | Mixed audio waveform, colour-coded by speaker turn | Always |
| 2 | Mixed audio spectrogram | "Show Mixed Audio Spectrogram" checkbox is on |
| 3 | ElevenLabs audio waveform, colour-coded by speaker turn | `elevenlabs_audio_recording.mp3` exists in the record directory |
| 4 | ElevenLabs audio spectrogram | EL recording exists **and** "Show ElevenLabs Spectrogram" checkbox is on |
| 5 | Speaker Turn Timeline with per-turn durations and pause markers | Always |

When `elevenlabs_audio_recording.mp3` is not found, rows 3 and 4 are hidden and an info message is shown instead. Spectrogram checkboxes appear above the chart only for the recordings that are available. Results are cached per trial so switching between records is fast after the first load.

### Waveform Rendering

Each waveform subplot is drawn in two layers:

1. **Speaker segments** — drawn in colour for each active turn window. Clicking a legend item (User or Assistant) hides all traces for that speaker.
2. **Pause bands** — semi-transparent gray rectangles over speaker-transition gaps, linked to the **Pause** legend item so they can be toggled on/off.

### Colour Coding

| Colour | Meaning |
|--------|---------|
| Blue | User speaker turn |
| Orange-red | Assistant speaker turn |
| Gray shaded band | Pause — speaker-transition gap (user→assistant or assistant→user) |

Colours are chosen for visibility in both Streamlit light and dark mode. Clicking a legend item (User, Assistant, Pause) toggles that category across all subplots simultaneously.

### Hover Tooltips

Hovering over any waveform sample or timeline bar shows:
- Turn ID, speaker, start/end time, and duration
- Transcript text (heard and intended where available)
- Response latency in ms for user turns (time from user's last segment end to assistant's first segment start)

Hovering over a pause band shows the pause duration and the from/to speakers.

### Pause Definition

Pauses are computed consistently with `turn_taking.py`:

- Only **speaker-transition gaps** count as pauses: a gap between a user segment end and the next assistant segment start, or vice versa.
- Same-speaker consecutive segments (e.g. two user audio sessions back to back) are not marked as pauses.
- Formula: `pause_duration = next_speaker.segments[0].start − current_speaker.segments[-1].end`
- Only gaps `> 1 ms` are shown.

### Turn Data Source

Turn timestamps, transcripts, and response latencies are loaded in priority order:

1. **`metrics.json` context** (primary) — uses the same `MetricContext` fields (`audio_timestamps_user_turns`, `audio_timestamps_assistant_turns`, `transcribed_*_turns`) that `turn_taking.py` operates on. Latency is computed as `asst.segments[0].start − user.segments[-1].end` per matching turn ID.
2. **`user_simulator_events.jsonl`** (fallback) — used when `metrics.json` is absent or contains no timestamp data. One entry per completed `audio_start`/`audio_end` session; latency computed by temporal proximity. Also resolves legacy `elevenlabs_events.jsonl` from older runs.

### Spectrogram Details

Spectrograms are computed at a 4 kHz intermediate sample rate (via `librosa.resample`) to preserve speech content up to 2 kHz (Nyquist) while keeping heatmap size bounded (~60–250 K cells for typical 5–90 s recordings). The time axis starts at `t = 0` to align with the waveform.
