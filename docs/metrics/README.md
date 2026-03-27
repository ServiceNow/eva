# Metrics System Documentation

## Overview

The EVA metrics system provides comprehensive evaluation of voice assistant conversations. The system includes 15 metrics organized into four categories, each answering a different question about the conversation.

## Why These Categories?

The categories form a layered evaluation, meant to be read in order:

1. **Validation** — *Is this simulation trustworthy?* Data quality gates that must pass before other metrics can be interpreted. If the user simulator behaved incorrectly or the conversation didn't finish, the agent is being evaluated against a corrupted scenario.

2. **Accuracy** — *Did the agent do the right thing?* These are the core pass/fail metrics. A conversation that fails accuracy is not a valid outcome, regardless of how smooth the experience was.

3. **Experience** — *Was it a good experience?* Given the agent did the right thing, these metrics capture whether a real user would find the conversation natural and efficient. They don't affect correctness but matter for production quality.

4. **Diagnostic** — *Why did something go wrong?* Breakdowns that help isolate root causes (STT errors, malformed tool calls, slow responses). These are not scored directly since they overlap with the categories above — they exist to make failures actionable.

## Metrics by Category

### Capabilities Legend

Each metric measures one or more voice agent capabilities:

- **VAD** — Voice activity detection: accurately detecting when the user has started and finished speaking.
- **Speech Recognition** — The ability to correctly understand spoken input: speech-to-text transcription in cascade, or audio understanding in audio-native (S2S) models.
- **Language Model** — The model's ability to understand context, follow instructions, reason, and generate correct responses and tool calls.
- **Speech Synthesis** — The ability to produce accurate spoken output: text-to-speech in cascade, or direct audio generation in audio-native (S2S) models.
- **Pipeline** — End-to-end system performance (latency) across all components, not attributable to a single model.

### Accuracy (3 metrics)

Measures whether the agent accomplished the user's goal correctly:

| Metric | Type | Capabilities | Description |
|--------|------|-------------|-------------|
| [`task_completion`](task_completion.md) | Deterministic | Speech Recognition, Language Model | Binary pass/fail via scenario DB state hash comparison (0-1) |
| [`faithfulness`](faithfulness.md) | Judge (Claude Opus) | Speech Recognition (audio-native only), Language Model | Faithfulness to information, policies, and instructions (1-3) |
| [`agent_speech_fidelity`](agent_speech_fidelity.md) | Audio Judge (Gemini) | Speech Synthesis | Whether assistant speech audio matches intended text (0-1) |

### Experience (3 metrics)

Measures the quality of the user's conversational experience:

| Metric | Type | Capabilities | Description |
|--------|------|-------------|-------------|
| [`turn_taking`](turn_taking.md) | Judge (text + timestamps) | VAD, Pipeline | Timing accuracy of turn transitions (-1 to +1) |
| [`conciseness`](conciseness.md) | Judge | Language Model | Whether responses are appropriately concise for voice (1-3) |
| [`conversation_progression`](conversation_progression.md) | Judge | Language Model | Whether assistant moves conversation forward without repetition (1-3) |

### Diagnostic (6 metrics)

Metrics that help isolate root causes of failures. These provide signals for understanding what went wrong, but are not directly used in final evaluation scores.

| Metric | Type | Capabilities | Description |
|--------|------|-------------|-------------|
| [`authentication_success`](authentication_success.md) | Deterministic | Speech Recognition, Language Model | Whether get_reservation was called successfully (0-1) |
| [`response_speed`](response_speed.md) | Deterministic | VAD, Pipeline | Latency between user utterance end and assistant response start (seconds) |
| [`speakability`](speakability.md) | Judge | Language Model | Whether text is voice-friendly and appropriate for TTS (0-1) |
| [`stt_wer`](stt_wer.md) | Deterministic | Speech Recognition | Speech-to-Text Word Error Rate using jiwer (0.0+) |
| [`tool_call_validity`](tool_call_validity.md) | Deterministic | Language Model | Fraction of tool calls with correctly formatted parameters (0.0-1.0) |
| [`transcription_accuracy_key_entities`](transcription_accuracy_key_entities.md) | Judge | Speech Recognition | STT accuracy for key entities (names, dates, numbers) (0.0-1.0) |

### Validation Metrics (3 metrics)

Quality control metrics that identify problematic simulations. These evaluate the simulation infrastructure, not agent capabilities.

| Metric | Type | Description |
|--------|------|-------------|
| [`user_behavioral_fidelity`](user_behavioral_fidelity.md) | Judge | Whether simulated user corrupted agent evaluation (0-1) |
| [`conversation_finished`](conversation_finished.md) | Deterministic | Whether conversation ended with proper end_call tool (0-1) |
| [`user_speech_fidelity`](user_speech_fidelity.md) | Audio (Gemini) | Whether user simulator speech audio matches intended text (1-3) |

## Metrics Pipeline

Each metric implements `BaseMetric.compute(context: MetricContext) -> MetricScore`. The `MetricContext` is constructed by the `MetricsContextProcessor`, which joins raw logs from three sources (audit log, Pipecat events, ElevenLabs events) into structured variables. See [metric_context.md](../metric_context.md) for full details on what data metrics receive and how it's produced.

### Metric Types

**LLM-as-Judge:**
- Integer/boolean ratings (not decimals) to avoid precision issues
- Structured prompts in `configs/prompts/judge.yaml`
- GPT-5.2 for text judges, Gemini 3.1 Pro for audio judges, Claude Opus for faithfulness

**Audio Evaluation:**
- Audio encoded as base64 WAV, sent to Gemini via LiteLLM
- Full conversation audio or per-turn segments depending on metric

**Deterministic:**
- Direct computation from MetricContext (no LLM)
- Hash comparisons, WER calculations, latency measurements

### Output

Each metric produces a `MetricScore` stored in `metrics.json` per record:
- `name`: Metric identifier
- `score`: Raw score in metric's native scale
- `normalized_score`: 0.0-1.0 scale (higher is better), or `null` if not normalizable
- `details`: Metric-specific details (explanations, per-turn data, etc.)
- `error`: Error message if computation failed

## Running Metrics

Metrics are run as part of the benchmark via `eva` (or `python main.py`). Use `--metrics` to select which metrics to compute:

```bash
# Run all metrics on benchmark output
python main.py \
    --run-id <existing_run_id>
```

### Running Specific Metrics

```bash
# Run accuracy metrics
python main.py \
    --run-id <existing_run_id> \
    --metrics task_completion,agent_speech_fidelity,faithfulness

# Run experience metrics (requires audio files for turn_taking)
python main.py \
    --run-id <existing_run_id> \
    --metrics turn_taking,conciseness,conversation_progression

# Run diagnostic metrics
python main.py \
    --run-id <existing_run_id> \
    --metrics authentication_success,response_speed,speakability,stt_wer,tool_call_validity,transcription_accuracy_key_entities

# Run validation metrics
python main.py \
    --run-id <existing_run_id> \
    --metrics user_behavioral_fidelity,conversation_finished,user_speech_fidelity
```

## Prompts and Customization

Judge metric prompts are defined in `configs/prompts/judge.yaml` under the `judge:` key. Each metric has a `user_prompt` with evaluation criteria, rating scale, and placeholders filled from MetricContext. To customize, edit the corresponding section in the YAML file.

## Validating the Judges 

LLM-as-judge evaluations are only as useful as the judges themselves. For each judge metric, we constructed a human-annotated validation dataset and measured judge accuracy against human labels. We use these datasets to improve our judge prompts as well as select the optimal LLM judge model for each metric. See [judge_validation_datasets/](judge_validation_datasets/) for the datasets and detailed judge accuracy results.

## Related Documentation

- [../llm_configuration.md](../llm_configuration.md) - LLM configuration
- [../metric_context.md](../metric_context.md) - MetricContext data structure
- [../../configs/prompts/judge.yaml](../../configs/prompts/judge.yaml) - Judge prompt templates
- [../../CLAUDE.md](../../CLAUDE.md) - Complete architecture documentation
