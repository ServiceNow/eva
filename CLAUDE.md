# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end Voice Agent Benchmark (EVA) is an end-to-end evaluation framework for voice assistants. It runs bot-to-bot conversations (assistant vs. user simulator) at scale, recording audio/transcripts/tool calls and computing metrics for effectiveness, efficiency, and humanness.

## Code Style Guidelines

### CRITICAL: Import Placement
**ALWAYS place ALL imports at the top of Python files.**

- ✅ **DO:** Place all imports at the top of the file, immediately after the module docstring
- ❌ **DO NOT:** Place imports inside functions, methods, or conditional blocks
- ❌ **DO NOT:** Use lazy imports or defer imports to later in the file

This rule applies to ALL Python files without exception. Even utility imports or imports that are only used in one function must be at the top.

**Example:**
```python
"""Module docstring."""

# All imports here
from typing import Any
from pathlib import Path

from eva.metrics.base import CodeMetric
from eva.utils.logging import get_logger


def my_function():
    # ❌ NEVER import here
    pass
```

### CRITICAL: JSON Parsing for LLM Responses
**ALWAYS use `extract_and_load_json` from `eva.utils.json_utils` for parsing LLM responses.**

- ✅ **DO:** Use `extract_and_load_json(response_text)` for all judge metric and audio metric LLM responses
- ❌ **DO NOT:** Use `json.loads()` for parsing LLM-generated JSON
- ✅ **DO:** Use `json.loads()` only for parsing known-format files (JSONL, config files, etc.)

**Why?** LLM responses may contain:
- JSON embedded in surrounding text (e.g., "Here's the result: {...}")
- Unescaped backslashes or quotes
- Formatting inconsistencies

The `extract_and_load_json` function handles all these cases robustly by:
- Extracting JSON from surrounding text
- Fixing unescaped backslashes and quotes
- Being more lenient with formatting

**Example:**
```python
"""Judge metric example."""

from eva.metrics.base import TextJudgeMetric
from eva.utils.json_utils import extract_and_load_json  # ✅ Import this


class MyJudgeMetric(TextJudgeMetric):
    async def compute(self, context):
        response_text = await self.llm_client.generate_text(messages)

        # ✅ DO: Use extract_and_load_json for LLM responses
        parsed = extract_and_load_json(response_text)

        # ❌ DO NOT: Use json.loads for LLM responses
        # parsed = json.loads(response_text)  # This will fail if LLM adds text

        rating = parsed.get("rating")
        return MetricScore(...)
```

**When to use `json.loads()` instead:**
- Parsing lines from JSONL files
- Loading configuration files
- Parsing structured log entries
- Any case where the JSON format is guaranteed and controlled

## Common Commands

### Review Feedback Tools

```bash
# Analyze all review feedback and generate comprehensive summary
python scripts/analyze_review_feedback.py \
    --run-dir flight_booking_review/20260127_215803 \
    --output-dir feedback_analysis \
    --llm-model gpt-5.2

# Run without LLM theme extraction (quantitative analysis only)
python scripts/analyze_review_feedback.py \
    --run-dir flight_booking_review/20260127_215803 \
    --output-dir feedback_analysis \
    --skip-llm

# Launch feedback viewer (read-only Streamlit app)
streamlit run scripts/feedback_viewer_app.py
```

**Review Feedback Analysis:**
- Analyzes all review_feedback.json files in a run directory
- Generates quantitative statistics (counts, percentages)
- Uses LLM (GPT-5.2) to extract common themes from free-text comments
- Produces JSON and Markdown reports with actionable insights
- Includes high-level reviewer feedback integration

**Feedback Viewer App:**
- Read-only Streamlit app for viewing conversation records with feedback
- Deep-dive focus with rich context (transcript, audio, metrics, feedback)
- Feedback-centric display with color-coded status badges
- Supports both output/ and flight_booking_review/ directories
- Navigation: Previous/Next buttons, jump to record, filtering
- Filters: Has feedback, user simulator issues, tool mocking issues, premature endings

### Setup and Installation
```bash
# Install the package in editable mode
pip install -e .

# Install with dev dependencies (testing, linting, type checking)
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with required API keys (ELEVENLABS_API_KEY, OPENAI_API_KEY)
```

### Running Benchmarks
```bash
# Run with domain-based conventions (easiest)
EVA_DOMAIN=airline python main.py
# Automatically uses:
#   data/airline_dataset.jsonl
#   configs/agents/airline_agent.yaml
#   data/airline_scenarios/

# Run with CLI overrides
python main.py \
    --max-concurrent 20 \
    --llm-model gpt-4o

# Debug mode - run only 1 record
python main.py --debug

# Run with specific metrics
python main.py \
    --metrics task_completion,faithfulness,stt_wer

# Re-run metrics on an existing run (validates + reruns failures)
python main.py \
    --run-id <existing_run_id> \
    --metrics task_completion,conciseness

# Outputs to: output/<run_id>/
```

**Metrics Parallelization:**
Concurrency is managed by the LiteLLM Router via per-deployment `max_parallel_requests` and `rpm`/`tpm` limits configured in `EVA_MODEL_LIST`. The metrics runner launches all records concurrently via `asyncio.gather()`, and the Router automatically bounds concurrent API calls per deployment.

**Troubleshooting SSL Errors with Gemini/Vertex AI:**
If you see SSL/OAuth errors when running audio metrics with Gemini (e.g., `SSLError: EOF occurred in violation of protocol`), this is caused by too many concurrent authentication requests to Google's OAuth endpoint. The system will automatically retry with exponential backoff. To reduce these errors:
- Lower `max_parallel_requests` for the Vertex AI deployment in `EVA_MODEL_LIST`
- Run fewer audio metrics concurrently
- The errors are transient and will succeed after retries

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=eva

# Run specific test file
pytest tests/unit/test_orchestrator.py

# Run specific test function
pytest tests/unit/test_orchestrator.py::test_function_name

# Run tests with verbose output
pytest tests/ -v
```

### Code Quality
```bash
# Lint and format code
pre-commit run --all-files

# Format code with ruff (included in pre-commit)
ruff check --fix
ruff format

# Type check with mypy
mypy src/eva

# Run ruff without fixing
ruff check
ruff format --check
```

### Docker
```bash
# Build Docker image
docker-compose build

# Run benchmark in Docker
docker-compose run --rm benchmark main.py
```

## Architecture

### Core Components and Data Flow

The system follows a **hierarchical orchestration pattern** with decoupled components:

```
BenchmarkRunner (orchestrator/runner.py)
├─ Loads: dataset.jsonl, scenario_db_dir/, agents_config.yaml, run_config.yaml
├─ Manages: PortPool, parallel execution with a semaphore
└─ Spawns: ConversationWorker per EvaluationRecord (max_concurrent)
   │
   ├─ AssistantServer (assistant/server.py)
   │  ├─ WebSocket server on allocated port
   │  ├─ Bot-to-bot protocol: start/audio/transcript/stop/response messages
   │  ├─ AgenticSystem (assistant/agentic/system.py)
   │  │  ├─ Single-agent mode: directly executes configured agent (no matching phase)
   │  │  └─ Agent execution: tool-calling loop with LLM
   │  │     └─ ToolExecutor.execute() → Python-based mocked responses
   │  │        ├─ Tool implementations from Python modules
   │  │        └─ Data from scenario database JSON (per-record)
   │  └─ Outputs: audit_log.json, transcript.jsonl, audio files
   │
   └─ UserSimulator (user_simulator/client.py)
      ├─ Connects to AssistantServer via WebSocket
      ├─ Sends user_goal and persona from EvaluationRecord
      └─ Generates responses (ElevenLabs conversational AI or mock)
```

**Post-execution** (independent):
```
MetricsRunner (metrics/runner.py)
├─ Loads: run_dir/records/* + dataset for ground_truth
├─ Per record: Computes metrics via MetricContext
└─ Saves: Individual metric scores per record
```

### Key Architectural Patterns

1. **Decoupled Components via WebSocket**: AssistantServer and UserSimulator are independent processes. This enables parallel execution, independent lifecycle management, and scalability.

2. **Stateful Single-Agent Orchestration**: `AgenticSystem` maintains conversation state across turns. Single-agent mode:
   - Directly executes the configured agent without matching
   - Tool-calling loop with agent (role + instructions + tools)
   - LLM decides which tools to call based on conversation context

3. **Python-Based Tool Mocking**: `ToolExecutor` provides reproducible tool responses via Python implementations:
   - **Tool Modules**: Python functions implementing tool logic (e.g., `airline_tools.py`)
   - **Scenario Database**: JSON files containing test data (reservations, flights, etc.)
   - **State Management**: Optional database mutations when enable_state=True
   - **Type Safety**: Full Python type hints and IDE support

   Tool behavior is defined in Python code with scenario data in JSON. This enables maintainable, testable tool implementations with full language features.

4. **Transfer to Live Agent**: Special handling for the `transfer_to_agent` tool:
   - When the agent calls `transfer_to_agent`, the system yields "Transferring you to a live agent. Please wait."
   - The conversation immediately ends (no mock response, no continued tool loop)
   - UserSimulator detects this exact message and sets end_reason="transfer"
   - Tool call is logged to audit log with status="transfer_initiated"

5. **Audit Trail for Metrics**: Every interaction is logged to `AuditLog` with sequence IDs:
   - `user_input`: User messages
   - `llm_call`: LLM calls for agent execution and tool selection
   - `tool_call`: Tool invocations with mocked responses
   - `assistant_output`: Assistant responses

   This complete lineage enables metric computation after execution.

6. **Decoupled Evaluation**: `MetricsRunner` is independent of benchmark execution. Same conversation can be re-evaluated with different metrics without re-running.

7. **Pluggable Metrics**: `BaseMetric` subclasses register via `MetricRegistry`. See [docs/metrics/README.md](docs/metrics/README.md) for the full metric list, categories, and scoring details.

8. **Resource Management**:
   - `PortPool`: Allocates/releases ports for AssistantServer instances
   - Asyncio semaphores: Controls max concurrent conversations
   - LiteLLM Router: Per-deployment `max_parallel_requests` and `rpm`/`tpm` limits via `EVA_MODEL_LIST`

### Critical Data Structures

**EvaluationRecord** (test case):
- `user_goal`: What the user wants to accomplish
- `user_persona`: ElevenLabs agent_id + goal_prompt + personality traits
- `ground_truth`: Expected tools, task completion, handoff behavior
- `category`: Grouping label (e.g., "hr_pto", "it_support")

**Scenario Database** (JSON):
- Per-record database containing tables (reservations, flights, etc.)
- `error_injections`: Configurable error scenarios
- Queryable and mutable (when stateful mode enabled)

**AuditLog** (conversation transcript):
- Ordered list of timestamped entries with sequence IDs
- Entry types: user_input, llm_call, tool_call, assistant_output
- Captures complete conversation lineage for metrics

**MetricContext** (evaluation data capsule):
- `ground_truth`: Expected behavior from EvaluationRecord
- `transcript`: Parsed transcript.jsonl
- `audit_log`: Parsed audit_log.json
- `tool_calls`: Extracted tool invocations
- `audio_paths`: Paths to audio files for audio-based metrics
- `turns_transcript`: Processed turn-by-turn transcript from postprocessor
- `turns_llm_response`: Processed LLM responses from postprocessor
- `turns_audio_timestamps_*`: Audio timing data for turn-taking analysis
- `tool_params`: Extracted tool parameters from audit log
- `conversation_finished`: Boolean flag from postprocessor
- Helper methods: `get_user_messages()`, `get_assistant_messages()`

### Metrics System

The metrics system uses a **three-stage pipeline**:

1. **Log Processing** → VoiceAgentMetricsPostprocessor
2. **Metric Computation** → Per-metric compute() methods
3. **Aggregation** → Dimension scoring and overall pass/fail

#### Stage 1: Log Processing (Postprocessor)

`VoiceAgentMetricsPostprocessor` transforms raw logs into metric-ready variables:

- **Input**: `audit_log.json`, `pipecat_events.jsonl`, `elevenlabs_events.jsonl`
- **Output**: Enriched `MetricContext` with processed fields
- **Processing**:
  - Parses audit log entries into turn-by-turn transcript
  - Extracts LLM responses (assistant and user simulator)
  - Extracts tool calls with parameters and responses
  - Computes audio timestamps for turn transitions
  - Detects conversation completion (goodbye messages)

The postprocessor is initialized with configurable goodbye messages and runs automatically during metrics computation.

#### Stage 2: Metric Computation

Each metric implements `BaseMetric.compute(context: MetricContext) -> MetricScore`:

**LLM-as-Judge Pattern:**
- Uses **integer/boolean ratings** (not decimals) to avoid precision issues
- Structured prompts with explicit rating scales in `configs/prompts.yaml`
- GPT-5.2 for text-only judges, Gemini 3 Pro for audio judges
- Batch processing with configurable batch size (default: 50)
- Normalization after response:
  - 3-point scale: 3→1.0, 2→0.5, 1→0.0
  - Turn-taking (-2 to +2): Normalized to 0.0-1.0 range
  - Boolean: true→1.0, false→0.0, null→excluded

**Audio Evaluation:**
- `turn_taking`: Per-turn segments (user_end - 0.8s to assistant_start + 0.8s)
- `agent_speech_fidelity` / `user_speech_fidelity`: Full conversation audio
- Audio encoded as base64 WAV, sent to Gemini via LiteLLM

**Code-Based Metrics:**
- Direct computation from transcript/audit_log (no LLM)
- F1 scores, WER calculations, latency measurements

#### Stage 3: Results Storage

**Individual Metrics:**
- Each metric score is stored separately in `metrics.json` per record
- Includes raw score, normalized score, and metric-specific details
- Errors are tracked per metric if computation fails

**Output Format:**
- `RecordMetrics` object containing dictionary of `MetricScore` objects
- Each `MetricScore` includes: name, score, normalized_score, details, error
- No aggregate scores computed - users analyze individual metrics as needed

**Metrics Summary:**
- Summary file lists all metrics computed across all records
- Enables custom aggregation and analysis by downstream tools

### Agent Configuration (YAML)

Agents are defined in `agents_config.yaml`:
- `id`, `name`, `description`: Agent identification
- `role`: System prompt role description
- `instructions`: Detailed agent instructions
- `tools`: List of tool definitions with parameters
  - `id`, `name`, `description`: Tool identification
  - `type`: script/subflow/action/fallback/live_agent_triage
  - `required_parameters`, `optional_parameters`: Tool schema

The AgenticSystem loads a single agent at runtime and uses it for:
1. System prompt construction (role + instructions + tools)
2. Tool schema validation (parameter types and constraints)
3. Tool execution via ToolExecutor

### Output Structure

```
output/<run_id>/
├── config.json              # Run configuration snapshot
├── summary.json             # Run-level statistics
├── results.csv              # Quick results table
├── metrics_summary.json     # Aggregate metrics (after metrics run)
├── metrics_summary.csv      # Per-category metrics table
└── records/<record_id>/
    ├── result.json          # ConversationResult (status, statistics, errors)
    ├── audio_assistant.wav  # Assistant audio channel
    ├── audio_user.wav       # User audio channel
    ├── audio_mixed.wav      # Mixed stereo audio
    ├── transcript.jsonl     # Turn-by-turn transcript
    ├── audit_log.json       # Complete interaction log
    ├── pipecat_events.jsonl # Pipecat framework events (if enabled)
    ├── elevenlabs_events.jsonl # ElevenLabs events (if enabled)
    └── metrics.json         # Per-record metric scores
```

## Development Guidelines

### Adding a New Metric

Metrics are organized by business domain. Choose the folder that best matches what the metric measures.


#### LLM-as-Judge Metric Example

```python
from eva.metrics.base import BaseMetric, MetricContext, MetricScore
from eva.metrics.registry import register_metric
from eva.utils.llm_client import LLMClient
import json

@register_metric
class MyJudgeMetric(BaseMetric):
    name = "my_judge_metric"
    description = "LLM-based evaluation"
    category = "accuracy"  #  or "experience"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        model = self.config.get("judge_model", "gpt-5.2")
        self.llm_client = LLMClient(model=model, temperature=0.0)
        self.prompts = self._load_prompts()  # From configs/prompts.yaml

    async def compute(self, context: MetricContext) -> MetricScore:
        # Format transcript
        transcript_text = self._format_transcript(context.turns_transcript)

        # Build prompt with integer rating scale
        user_prompt = self.prompts["user_prompt"].format(
            turns_transcript=transcript_text
        )

        # Call LLM with structured output
        messages = [{"role": "user", "content": user_prompt}]
        response_text = await self.llm_client.generate_text(
            messages, response_format={"type": "json_object"}
        )

        # Parse response
        response = json.loads(response_text)
        rating = response.get("rating")  # Should be 1, 2, or 3

        # Normalize: 3→1.0, 2→0.5, 1→0.0
        normalized_score = (rating - 1) / 2.0

        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized_score,
            details={"explanation": response.get("explanation", "")}
        )
```

#### Audio Metric Example

```python
from eva.metrics.base import BaseMetric, MetricContext, MetricScore
from eva.metrics.registry import register_metric
from eva.utils.llm_client import LLMClient
from pydub import AudioSegment
import base64

@register_metric
class MyAudioMetric(BaseMetric):
    name = "my_audio_metric"
    description = "Audio-based evaluation"
    category = "experience"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        # Use Gemini for audio analysis
        self.llm_client = LLMClient(model="gemini-3-pro", temperature=0.0)

    async def compute(self, context: MetricContext) -> MetricScore:
        # Load audio
        audio_path = context.audio_mixed_path
        audio = AudioSegment.from_wav(audio_path)

        # Encode as base64
        audio_b64 = self._audio_to_base64(audio)

        # Call Gemini with audio + text prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this conversation audio..."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:audio/wav;base64,{audio_b64}"
                        }
                    }
                ]
            }
        ]

        response_text = await self.llm_client.generate_text(
            messages, response_format={"type": "json_object"}
        )

        # Parse and normalize
        response = json.loads(response_text)
        rating = response.get("rating")
        normalized = (rating + 2) / 4.0  # -2 to +2 → 0.0 to 1.0

        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details=response
        )

    def _audio_to_base64(self, audio: AudioSegment) -> str:
        """Convert audio segment to base64 encoded WAV."""
        import io
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
```

#### Registering and Using Your Metric

1. Place your metric in the appropriate subdirectory:
   - `src/eva/metrics/accuracy/` — measures whether the agent is being accurate
   - `src/eva/metrics/experience/` — measure how good the experience is for the user
   - `src/eva/metrics/validation_metrics/` — quality control (unchanged)

2. Import in the corresponding `__init__.py`:
```python
from . import my_custom_metric  # noqa
```

3. Add prompt to `configs/prompts/judge.yaml` (for judge/audio metrics):
```yaml
judge:
  my_judge_metric:
    user_prompt: |
      You are an expert evaluator...
      Rating Scale:
      - 3: Excellent
      - 2: Acceptable
      - 1: Poor
```

4. Run your metric:
```bash
python main.py --metrics my_custom_metric
```

### Modifying Agent Behavior

**Agent Configuration (YAML)**: Edit `configs/example_agents.yaml` to modify agent metadata. No code changes needed for:
- Modifying agent instructions/roles/descriptions
- Adding/removing tools from the agent
- Changing tool parameter schemas

**Tool Implementations (Python)**: Edit the tool module (e.g., `airline_tools.py`) to modify tool behavior:
- Update tool logic and validations
- Change response formats
- Add new helper functions

### Tool Mocking Strategy

The system uses a **Python-based approach** where tool implementations are Python functions that query scenario databases.

#### Tool Implementation (Python)

Tools are implemented as Python functions in dedicated modules (e.g., `airline_tools.py`):

```python
from eva.assistant.tools.tool_executor import tool

@tool(
    name="get_reservation",
    description="Retrieve reservation details by confirmation number and last name"
)
def get_reservation(confirmation_number: str, last_name: str, db: dict) -> dict:
    """
    Get reservation details.

    Args:
        confirmation_number: Reservation confirmation code
        last_name: Passenger last name for verification
        db: Scenario database containing reservations

    Returns:
        Reservation details or error
    """
    reservations = db.get("reservations", {})
    reservation = reservations.get(confirmation_number.upper())

    if not reservation:
        return {
            "status": "error",
            "error_type": "not_found",
            "message": f"No reservation found with confirmation number {confirmation_number}"
        }

    # Verify last name matches
    if reservation.get("last_name", "").lower() != last_name.lower():
        return {
            "status": "error",
            "error_type": "verification_failed",
            "message": "Last name does not match reservation"
        }

    return {
        "status": "success",
        "reservation": reservation
    }
```

#### Scenario Database (JSON)

Per-record JSON files contain the test data that tools query:

```json
{
  "reservations": {
    "ABC123": {
      "confirmation_number": "ABC123",
      "last_name": "Smith",
      "status": "confirmed",
      "passengers": [...]
    }
  },
  "flights": {...}
}
```

#### Key Features

- **Type Safety**: Full Python type hints with IDE autocomplete
- **Testable**: Unit test individual tool functions
- **Maintainable**: Standard Python code with familiar patterns
- **Flexible**: Full language features (loops, conditionals, helpers)
- **Debuggable**: Use standard Python debugging tools
- **State Management**: Optional database mutations when enable_state=True

The `ToolExecutor` loads tool modules dynamically and provides tools with access to the scenario database and state management capabilities.

### Testing Bot-to-Bot Conversations

The UserSimulator can operate in two modes:
1. **ElevenLabs Conversational AI**: Real conversational agent (requires ELEVENLABS_API_KEY)
2. **Mock**: Simple rule-based responses (for unit testing)

For development, use mock mode to avoid API costs:
```python
# In test code
user_simulator = UserSimulator(mode="mock")
```

## Environment Variables

Required:
- `ELEVENLABS_API_KEY`: User simulator (ElevenLabs conversational AI)
- `OPENAI_API_KEY`: LLM for agent execution (if using OpenAI models)

Optional (provider-dependent):
- `GOOGLE_API_KEY`: For Gemini models
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`: For Azure OpenAI
- `DEEPGRAM_API_KEY`: For Deepgram STT
- `ASSEMBLYAI_API_KEY`: For AssemblyAI STT
- `CARTESIA_API_KEY`: For Cartesia TTS
- `EVA_LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

## Python Requirements

- Python >= 3.11 (required by type hints and asyncio features)
- Main dependencies: pipecat-ai, elevenlabs, pydantic, fastapi, websockets
- Dev dependencies: pytest, pytest-asyncio, pytest-cov, ruff, mypy

## Critical
Always check documentation for Pipecat and LiteLLM SDKs when working on these parts of the code, do not invent features that are already provided by these frameworks.
