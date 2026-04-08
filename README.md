<h1 align="center">A New End-to-end Framework for <br />Evaluating Voice Agents (EVA)</h1>

> *Most voice agent benchmarks evaluate either what the agent **does** or how it **sounds** — EVA evaluates both.*

[![Blog Post](https://img.shields.io/badge/Blog-Post-blue?style=flat-square&logo=huggingface)](https://huggingface.co/blog/ServiceNow-AI/eva)
[![Website](https://img.shields.io/badge/Website-EVA-green?style=flat-square&logo=googlechrome)](https://servicenow.github.io/eva/)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Rankings-orange?style=flat-square&logo=trophy)](https://servicenow.github.io/eva/#early-results)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/ServiceNow-AI/eva)
[![Demo](https://img.shields.io/badge/Demo-See%20It-purple?style=flat-square&logo=rocket)](https://servicenow.github.io/eva/#demo)

**EVA** is an open-source evaluation framework for conversational voice agents that scores complete, multi-turn spoken conversations across two fundamental dimensions:

- 🎯 **EVA-A (Accuracy)** — Did the agent complete the task correctly and faithfully?
- ✨ **EVA-X (Experience)** — Was the interaction natural, concise, and appropriate for spoken dialogue?

Using a realistic **bot-to-bot architecture**, EVA runs fully automated evaluations without human listeners — end to end, from speech in to judgment out.

### 📊 What's included
- **Metrics** for both EVA-A and EVA-X, fully documented and validated with judge prompts, code, etc.
- **50 airline scenarios** spanning flight rebooking, cancellations, vouchers, and more
- **Results** for 20 cascade and audio-native systems (speech-to-speech models, large audio language models) — see [Experiment Setup](docs/experiment_setup.md) for model configurations.

### 🔍 Key finding
Agents that score well on task completion tend to score worse on conversational experience — and vice versa. **The accuracy–experience tradeoff is real, consistent, and previously unmeasured.**

<details>
<summary><h2>Quick Start</h2></summary>

### Cloning the Repository

If you're only interested in running the latest stable version of EVA, you can clone with `--branch latest`, and optionally speed things up with `--depth 1 --no-tags --single-branch`.
```bash
git clone https://github.com/ServiceNow/eva.git --branch latest --depth 1 --no-tags --single-branch
```

Otherwise, for development, you can clone the default branch, `main`.
```bash
git clone https://github.com/ServiceNow/eva.git
```

### Installation

We recommend using [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management. If you don't have `uv` installed, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

This project requires **Python 3.11–3.13** (set via `requires-python` in `pyproject.toml`). `uv` will automatically select a compatible version. If you're using pip, make sure you're running a supported Python version.

```bash
cd eva

# Install all dependencies (uv automatically creates a virtual environment)
uv sync --all-extras

# Copy environment template
cp .env.example .env
# Edit .env with your API keys (ELEVENLABS_API_KEY, OPENAI_API_KEY required)
```

After installation, you can run EVA using either:
- `eva` — CLI entry point (e.g., `eva --help`)
- `python main.py` — script at the repo root (e.g., `python main.py --help`)

If using an IDE, point your Python interpreter to `.venv/bin/python` so commands run in the virtual environment automatically. Otherwise, prefix commands with `uv run` or activate the environment with `source .venv/bin/activate`.

<details>
<summary>Alternative: using pip</summary>

This project requires Python 3.11. If you need to manage multiple Python versions, consider using [pyenv](https://github.com/pyenv/pyenv).

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

</details>

### Environment Variables

**Required:**
- `OPENAI_API_KEY` (or another LLM provider): Powers the assistant LLM and text judge metrics
- `EVA_MODEL_LIST`: Model deployments that reference your API key (see `.env.example`). Also configurable via `--model-list` CLI flag. Only used for regular LLMs.
- `ELEVENLABS_API_KEY` + agent IDs: For user simulation
- STT/TTS API key and model: Passed via `EVA_MODEL__STT_PARAMS` / `EVA_MODEL__TTS_PARAMS` (default provider is Cartesia)

**For all metrics:**
- `OPENAI_API_KEY`: GPT-5.2 for text judge metrics (task completion, conciseness, turn taking, etc.)
- `GOOGLE_APPLICATION_CREDENTIALS`: Gemini via Vertex AI (audio judge metrics)
- `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`: Claude via Bedrock (faithfulness metric)

**Key Environment Variables:**
```bash
# Framework Configuration
EVA_DOMAIN=airline           # Domain-based path conventions
EVA_MAX_CONCURRENT_CONVERSATIONS=5   # Max parallel conversations
EVA_DEBUG=false                       # Run only 1 record for testing when enabled
EVA_RECORD_IDS=1.2.1,1.2.2            # Run specific records only (remove to run all records)

# Pipeline Model Configuration (nested under EVA_MODEL__)
EVA_MODEL__LLM=gpt-5-mini                # LLM model name (must match EVA_MODEL_LIST)
EVA_MODEL__STT=deepgram               # deepgram | openai_whisper
EVA_MODEL__TTS=cartesia               # cartesia | elevenlabs

EVA_MODEL__STT_PARAMS={"api_key":"", "alias": "deepgram-nova-3", "model": "nova-3"}
EVA_MODEL__TTS_PARAMS={"api_key":"", "alias": "cartesia-sonic-3", "model": "sonic-3"}

# Or speech-to-speech model (mutually exclusive with LLM)
# EVA_MODEL__S2S=gpt-realtime-mini    # Audio-native model name (S2S, S2T+TTS)

# Logging
EVA_LOG_LEVEL=INFO                    # DEBUG | INFO | WARNING | ERROR
```

See `.env.example` for the complete list of configuration options.

### Running the framework

```bash
# Run with domain-based conventions (easiest):
EVA_DOMAIN=airline python main.py
# Automatically uses:
#   data/airline_dataset.jsonl
#   configs/agents/airline_agent.yaml
#   data/airline_scenarios/

# Run with CLI overrides
python main.py --llm-model gpt-5-mini --max-concurrent 10
```

### Running Metrics

```bash
# Re-run specific metrics on an existing run
python main.py \
    --run-id <existing_run_id> \
    --metrics task_completion,faithfulness,conciseness
```

### Using Docker

```bash
# Build the image
docker compose build

# Run a benchmark
docker compose run --rm benchmark
```

### Development Setup

Install pre-commit hooks to lint and format code:

```bash
pre-commit install
```

### Running Tests

Install the `[dev]` extra dependencies as shown in the [Installation](#installation) section.

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_postprocessor_transcript.py -v

# Run with coverage
pytest tests/ --cov=eva

# Run metrics tests
pytest tests/integration/test_metrics.py -v
```


</details>

## Evaluation Gap

Existing benchmarks evaluate voice agent components in isolation — speech understanding, TTS quality, or conversational dynamics — but none assess the full pipeline end to end. In real deployed systems, errors compound across modules and failure modes interact in ways that component-level evaluation cannot capture. EVA addresses this by treating voice agent quality as an integrated whole, evaluating accuracy and experience jointly across complete multi-turn spoken conversations.

| **Framework** | **Interaction Mode** | **Multi-turn** | **Tool Calling** | **Goal Completion** | **Experience Metrics** | **Pass@k<br>Pass^k** | **Supported Systems** |
|---|---|---|---|---|---|--------------------|---|
| **EVA** | Live bot-to-bot | ✅ | ✅ | ✅ <br>Task Completion, Speech Fidelity, Faithfulness | ✅ <br>Conciseness, Turn-taking, Latency, Progression | ✅                  | Audio-native, Cascade |
| **VoiceAgent&shy;Bench** | Static, TTS-synthesized | ✅ | ✅ | ⚠️ | ❌ | ❌                  | Audio-native, Cascade |
| **CAVA** | Partial simulation | ✅ | ✅ | ⚠️ | ⚠️ <br>Latency, Tone-awareness | ❌                  | Audio-native, Cascade |
| **FDB-v2** | Live, automated examiner | ✅ | ❌ | ❌ | ✅ <br>Turn-taking fluency, Correction handling, Safety | ❌                  | Audio-native |
| **FDB-v1** | Static, pre-recorded | ❌ | ❌ | ❌ | ✅ <br>Turn-taking, Backchanneling, Interruption | ❌                  | Audio-native |
| **FD-Bench** | Live, simulated | ❌ | ❌ | ❌ | ✅ <br>Interruption, Delay, Robustness | ❌                  | Audio-native |
| **Talking Turns** | Static, curated | ❌ | ❌ | ❌ | ✅ <br>Turn change, Backchannel, Interruption | ❌                  | Audio-native, Cascade |

## 🏗️ Architecture

EVA evaluates agents using a **bot-to-bot audio architecture** — no human listeners, no text replays. Two conversational AIs speak to each other over a live WebSocket connection, producing realistic speech-to-speech interactions that capture real STT behavior and turn-taking dynamics.

| Component                           | Role                                                                                                                                                                                                                                                                                         |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🎭 **User Simulator** (ElevenAgent) | Plays the role of a caller with a defined goal and persona                                                                                                                                                                                                                                   |
| 🤖 **Voice Agent** (Pipecat)        | The system under evaluation — supports cascade (STT→LLM→TTS) and speech-to-speech models                                                                                                                                                                                                     |
| 🔧 **Tool Executor**                | The engine that provides deterministic, reproducible tool responses via custom Python functions. It dynamically queries and modifies a predefined per-scenario database.                                                                                                                     |
| ✅ **Validators**                    | Automated checks that verify conversations are complete and that the user simulator faithfully reproduced its intended goal — no human annotation required. Conversations that fail validation are automatically regenerated, ensuring only clean, correctly executed runs enter evaluation. |
| 📊 **Metrics Engine**               | Scores each conversation using the audio recording, transcripts, and tool call logs.                                                                                                                                                                                                         |


## Output Structure

```
output/<run_id>/
├── config.json              # Run configuration snapshot
├── results.csv              # Quick results table
├── metrics_summary.json     # Aggregate metrics (after metrics run)
├── metrics_summary.csv      # Per-category metrics breakdown
└── records/<record_id>/
    ├── result.json          # Conversation result
    ├── audio_assistant.wav  # Assistant audio channel
    ├── audio_user.wav       # User audio channel
    ├── audio_mixed.wav      # Mixed stereo audio
    ├── transcript.jsonl     # Turn-by-turn transcript
    ├── audit_log.json       # Complete interaction log
    ├── pipecat_logs.jsonl   # Pipecat framework events
    ├── elevenlabs_events.jsonl # ElevenLabs events
    └── metrics.json         # Per-record metric scores and details
```

## Metrics

| **🎯 EVA-A · Accuracy** | **✨ EVA-X · Experience** |
|---|---|
| *Did the agent complete the task correctly?* | *Was the conversational experience high quality?* |
|  **Task Completion** · Deterministic |  **Turn Taking** · LLM Judge `BETA` |
|  **Agent Speech Fidelity** · Audio LLM Judge `BETA` |  **Conciseness** · LLM Judge |
|  **Faithfulness** · LLM Judge |  **Conversation Progression** · LLM Judge |

See the [Metrics documentation](docs/metrics/README.md) for detailed scoring rubrics and judge prompts. For the data structures that metrics operate on, see [MetricContext documentation](docs/metric_context.md).

## 🗂️ Dataset

EVA includes **50 airline scenarios**, each specifying a user goal, persona, scenario database, and ground truth end state — making evaluations fully reproducible and directly comparable across agents and model versions. See the [Data documentation](docs/data.md) for a detailed breakdown of the data structure and scenario design, and the [Database & Tool Schema](docs/airline_database_tool_schema.md) for the airline scenario database format.

Flight rebooking is a strong initial domain: it is high-stakes, time-pressured, and demands temporal reasoning, policy following, constraint satisfaction, and accurate transcription of named entities (confirmation codes, flight numbers, passenger names, dates).

| Category | Description |
|---|---|
| ✈️ **IRROPS Rebooking** | Airline-initiated disruptions — user is entitled to rebooking at no cost |
| 🔄 **Voluntary Changes** | User-initiated changes subject to fare differences and change fees |
| 🔗 **Missed Connections** | Cascading disruptions across multiple legs |
| ⏱️ **Same-Day Changes** | Time-sensitive standby and same-day change requests |
| ⚠️ **Adversarial Scenarios** | Users seeking compensation they are not entitled to under policy |

## Project Structure

```
eva/
├── main.py                    # Main entry point
├── pyproject.toml             # Python project configuration
├── Dockerfile                 # Docker configuration
├── compose.yaml               # Docker Compose configuration
├── src/eva/
│   ├── cli.py                 # CLI interface
│   ├── run_benchmark.py       # Benchmark runner
│   ├── models/                # Pydantic data models
│   ├── orchestrator/          # Framework execution
│   │   ├── runner.py          # Main orchestrator
│   │   ├── worker.py          # Per-conversation worker
│   │   ├── validation_runner.py # Validation runner
│   │   └── port_pool.py       # Port management
│   ├── assistant/             # Pipecat-based assistant
│   │   ├── agentic/           # Agent orchestration
│   │   ├── tools/             # Python-based tool implementations
│   │   ├── pipeline/          # Audio/LLM processing pipeline
│   │   └── services/          # STT/TTS/LLM factories
│   ├── user_simulator/        # ElevenLabs user simulator
│   ├── metrics/               # Evaluation metrics
│   │   ├── base.py            # Base metric classes
│   │   ├── processor.py       # Metrics context processor
│   │   ├── runner.py          # Metrics execution
│   │   ├── registry.py        # Metric registry
│   │   ├── aggregation.py     # Metric aggregation
│   │   ├── accuracy/          # Task completion metrics
│   │   ├── experience/        # Responsiveness, progression, turn-taking
│   │   ├── diagnostic/        # Diagnostic metrics (not in final scores)
│   │   └── validation/        # Quality control metrics
│   └── utils/                 # Utilities (LLM client, log processing)
├── scripts/                   # Utility scripts
│   ├── run_text_only.py       # Text-only evaluation runner
│   ├── docker_entrypoint.py   # Docker entry point
│   ├── check_version_bump.py  # Version checking
│   └── push_to_hf.py         # Hugging Face push script
├── configs/                   # Configuration files
│   ├── prompts/               # Judge and simulation prompts
│   │   ├── judge.yaml         # Judge metric prompts
│   │   └── simulation.yaml    # User simulator prompts
│   └── agents/                # Agent configurations
│       └── airline_agent.yaml
├── docs/                      # Documentation
│   ├── metrics/               # Per-metric documentation
│   ├── data.md                # Data documentation
│   ├── experiment_setup.md    # Experiment setup guide
│   ├── llm_configuration.md   # LLM provider setup guide
│   ├── metric_context.md      # Metric context documentation
│   ├── limitations.md         # Known limitations
│   └── demo/                  # Demo audio files
├── data/                      # Data files
│   ├── airline_dataset.jsonl  # Evaluation dataset
│   └── airline_scenarios/     # Per-record scenario databases
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   ├── artifacts/             # Test artifacts and fixtures
│   └── fixtures/              # Shared test fixtures
└── website/                   # Project website (React/TypeScript)
```

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request. For larger features, we recommend reaching out first to ensure alignment with our roadmap.

## Limitations

See [Limitations](docs/limitations.md) for known limitations of the framework and metrics.
