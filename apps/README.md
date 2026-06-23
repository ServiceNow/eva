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

Interactive Streamlit dashboard for exploring and comparing EVA benchmark results. Provides cross-run comparison, run-level overviews, and deep-dive per-record analysis with rich visualizations.

### Usage

```bash
streamlit run apps/analysis.py
```

By default, the app looks for runs in the `output/` directory. Override via the sidebar or environment variable:

```bash
EVA_OUTPUT_DIR=path/to/results streamlit run apps/analysis.py
```

### Views

The app has three main pages:

#### 1. Cross-Run Comparison

Compare metrics across all selected runs with filtering and aggregation options.

- **Filters**: Pipeline Type (Cascade, Speech-to-Speech, Audio-Native), Provider (OpenAI, Gemini, etc.), System/Model
- **EVA Scatter Plot**: Visualizes EVA-A (Accuracy) vs EVA-X (Experience) with:
  - Multiple view modes: pass@1, pass@k, pass^k, Mean
  - Pareto frontier overlay (shows non-dominated points)
  - Color-coded by pipeline type
  - Hover for detailed model and component information
- **Bar Charts**: Grouped metrics by model for Accuracy and Experience categories
- **Summary Tables**:
  - EVA composites (pass@1, Mean)
  - Accuracy metrics (task_completion, faithfulness, etc.)
  - Experience metrics (agent_speech_fidelity, turn_taking, etc.)
  - Diagnostic & validation metrics
- **Metric by Domain Pivot**: See how metrics vary across domains for each system
- **Per-Sample Heatmap**: Visual grid showing per-record scores across systems (swap axes as needed)
- **Error Summary**: Track conversation failures and metric computation errors

**Options**:
- Complete runs only: Hide runs with failed records
- Average across domains: Collapse same-system runs across different datasets
- Show sub-metrics: Include granular breakdowns (e.g., latency percentiles)

![Cross-Run Comparison view](images/cross_run_comparison.png)

#### 2. Run Overview

Aggregate metrics and per-record table for a single run.

- **Model Details** (top): Shows LLM, STT, TTS, S2S, or Audio-Native models
- **Aggregate Metrics**: Horizontal bar chart with mean ± std dev, min, max for each metric
  - Sorted by category and metric name
  - Color-coded score ranges
  - Error bars show ±1 standard deviation
- **Errors**: Breakdown of conversation failures and metric errors by metric type
- **Per-Record Metrics Table**:
  - EVA composites, then metrics sorted by category
  - Color-coded cells (green ≥0.8, orange 0.5–0.8, red <0.5)
  - 🔍 link to navigate to Record Detail for each row
  - Download table as CSV

**Options**:
- Show failed attempts: Include records marked as retries/failed
- Show sub-metrics: Include granular metric breakdowns

![Run Overview view](images/run_overview.png)

#### 3. Record Detail

Deep-dive into a single conversation record with multiple perspectives.

- **Status & Summary**: Completed/failed status, duration, turn count
- **Audio Player**: Mixed stereo audio and (if available) ElevenLabs recording
- **Expandable Sections**: User goal, ground truth (expected scenario database)

**5 Tabs**:

##### Conversation Trace

Chat-like interface with rich context and inline metrics.

- **Metrics Overview** (top): EVA composites, then metric buttons grouped by category
  - Click any metric to see its per-turn breakdown and full details
  - Color-coded by score: 🟢 ≥0.8, 🟡 0.4–0.8, 🔴 <0.4
  - Selected metric shown with detailed explanation/breakdown
- **Conversation Trace** (main): Messages with per-turn context
  - **Assistant messages** (blue): Shows intended text + transcribed (STT) variant if different
  - **User messages** (purple): Shows intended (TTS) text + transcribed (STT) variant if different
  - **WER display**: Word Error Rate with substitutions/deletions/insertions breakdown
  - **Per-turn metric badges**: Inline colored pills (hover for explanation)
  - **Tool calls & responses**: Expandable code blocks with JSON
  - Right-side column shows metric explanations when hovering over messages

##### Transcript

Structured table view of the conversation.

- Columns: timestamp (formatted), role (user/assistant), content
- Filter by speaker using the role column
- Searchable via browser find (Cmd+F / Ctrl+F)
- Useful for long conversations where you need to search or copy text

##### Metrics Detail

All computed metrics organized by category with deep details.

- **EVA Composites** (top): Cards showing pass@1 and Mean scores for EVA-A/X
- **Metrics by Category**: Accuracy, Experience, Conversation Quality, Diagnostic, Validation
  - Each metric in an expandable section with:
    - **Score**: Raw score value (e.g., 3 for 1–3 rating)
    - **Normalized**: 0–1 scale version
    - **Dimensions** (if applicable): Multi-dimensional metrics show per-dimension ratings with ⚠ flags for lowest-scoring dimensions
    - **Details**: Judge explanation or summary
    - **View Judge Prompt**: Expandable section showing exact LLM prompt (for per-turn metrics: multiple prompts)
    - **Additional Details**: Raw JSON data (per-turn ratings, error details, etc.)

##### Processed Data

Raw processed variables from the metrics computation pipeline.

- **Conversation Trace**: Full turn-by-turn trace with types (message, tool_call, tool_response)
- **Tool Parameters**: Extracted tool call parameters as JSON
- **Tool Responses**: Extracted tool responses as JSON
- **Transcripts**: Assistant and user transcribed text by turn
- **TTS Text**: Intended assistant and user text (before TTS) by turn
- **Statistics**: Turn counts, conversation finished flag
- **Agent Instructions**: System prompt and instructions given to the agent

Use this tab for debugging data pipeline issues.

##### Turn Taking Analysis (Audio Analysis)

Interactive Plotly-based audio visualization showing speaker turns, timing, and pauses. Built from audio files and timestamp logs using the same data that powers the `turn_taking.py` metric.

**Subplots**:
| Row | Content | Shown when |
|-----|---------|------------|
| 1 | Mixed audio waveform, color-coded by speaker | Always |
| 2 | Mixed audio spectrogram | "Show Mixed Audio Spectrogram" checkbox is on |
| 3 | ElevenLabs audio waveform | `elevenlabs_audio_recording.mp3` exists |
| 4 | ElevenLabs audio spectrogram | EL recording exists AND "Show ElevenLabs Spectrogram" is on |
| 5 | Speaker Turn Timeline with durations and pauses | Always |

**Waveform Rendering**:
- **Speaker segments**: Color-coded by turn (click legend to hide speaker)
- **Pause bands**: Semi-transparent gray rectangles marking speaker-transition gaps
- Click legend items (User, Assistant, Pause) to toggle across all subplots

**Color Coding**:
| Color | Meaning |
|-------|---------|
| Blue | User speaker turn |
| Orange-red | Assistant speaker turn |
| Gray shaded band | Pause (gap between turns) |

**Hover Tooltips**:
- Turn ID, speaker, start/end time, duration
- Transcript text (transcribed and intended if available)
- Response latency in ms for user turns

**Pause Definition** (consistent with `turn_taking.py`):
- Only speaker-transition gaps count (user→assistant or assistant→user)
- Same-speaker consecutive segments are not marked as pauses
- Formula: `pause_duration = next_speaker.segments[0].start − current_speaker.segments[-1].end`
- Only gaps > 1 ms are shown

**Data Sources** (in priority order):
1. **`metrics.json` context** (primary): Uses `audio_timestamps_user_turns`, `audio_timestamps_assistant_turns`, `transcribed_*_turns` fields
2. **`user_simulator_events.jsonl`** (fallback) — used when `metrics.json` is absent or contains no timestamp data. One entry per completed `audio_start`/`audio_end` session; latency computed by temporal proximity. Also resolves legacy `elevenlabs_events.jsonl` from older runs.

**Spectrogram Details**:
- 4 kHz intermediate sample rate (via `librosa.resample`)
- Preserves speech content up to 2 kHz (Nyquist limit)
- Time axis starts at `t = 0` to align with waveform
- Results cached per trial for fast switching

![Record Detail view](images/analysis_record_detail.png)

### Sidebar Navigation

1. **Output Directories**: Specify one or more paths containing run folders
2. **Run Selection**: Pick a run (shows model/domain metadata)
3. **Record Selection**: Pick a record within the selected run
4. **Trial Selection**: If a record has multiple trials, pick one

### Common Tasks

**Find a metric value**: Run Overview → locate record in table → click 🔍 → Record Detail → Metrics Detail tab → expand metric

**Understand metric failure**: Cross-Run Comparison → Errors section → Run Overview → Errors table → Record Detail → Metrics Detail → check Error field

**Compare agent's intended vs transcribed speech**: Record Detail → Conversation Trace tab → look for indented text ("transcribed (STT):" or "intended (TTS):")

**Analyze per-turn metrics**: Record Detail → Conversation Trace → click metric button → see per-turn breakdown with reasons and scores

**Download results as CSV**: Run Overview or Cross-Run Comparison → scroll to bottom of table → Download CSV button

### Color Coding

- **Score cells**: Green ≥0.8, Orange 0.5–0.8, Red <0.5
- **Metric buttons**: 🟢 ≥0.8, 🟡 0.4–0.8, 🔴 <0.4
- **Turn Taking cards**: Color bar on left matches score color
- **Conversation speakers**: Blue (assistant), Purple (user)

### Tips & Tricks

- **Hover everywhere**: Hover over chart points, metric cells, badges, and timeline bars for detailed information
- **Browser find**: Use Cmd+F / Ctrl+F in the Transcript tab to search the conversation
- **Pareto frontier**: The dashed line in the EVA scatter plot shows non-dominated runs—useful for identifying "best" approaches
- **Swap heatmap axes**: See samples as rows and systems as columns (or vice versa)
- **Sub-metrics**: Enable "Show sub-metrics" for granular breakdowns (e.g., latency percentiles)
