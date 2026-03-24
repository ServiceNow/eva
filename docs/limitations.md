# Limitations

EVA is designed to provide rigorous, end-to-end evaluation of conversational voice agents. The following limitations are important to understand when interpreting results or extending the framework.

---

## Metrics

### LLM Judge Reliability

Even with human-validated alignment scores, LLM judges carry their own biases and may systematically favor certain response styles independent of actual quality. LALM judges in particular are relatively new and are not yet as reliable as text-input-only judge models. There is also potential for conflict when the model being evaluated and the model serving as judge are from the same provider family. The turn-taking metric is additionally flagged as beta due to noisy timing data from real-time audio APIs, which limits the reliability of fine-grained latency-based judgments.

### Binary Task Completion

Task completion is currently scored as binary — a conversation either achieves the goal or it does not. This means no partial credit is awarded for conversations where the agent completed most of the task correctly but failed on a single sub-goal (e.g., rebooked the correct flight but failed to issue an entitled voucher). As a result, the metric may understate the relative quality of systems that fail gracefully versus those that fail catastrophically, and may obscure fine-grained differences between systems on task completion.

---

## Simulation

### Domain and Dataset Scope

The current dataset covers 50 scenarios in a single domain (airline flight rebooking). Results may not generalize to other voice agent use cases with different policy structures, entity types, or conversational dynamics. Flight rebooking has a relatively well-structured policy space; more open-ended or ambiguous domains may expose failure modes that EVA does not currently surface. All scenarios are in English with standard American accents, with no coverage of multilingual users, non-native speakers, or accented speech.

### User Simulator Fidelity

EVA uses ElevenLabs as a commercial user simulator. Because its behavior — including TTS quality, turn-taking, and latency — may change across versions, results may be difficult to reproduce exactly over time. More fundamentally, the simulator does not replicate the natural disfluencies, hesitations, self-corrections, or emotional variation exhibited by real callers, and does not model challenging speech conditions such as background noise, non-native accents, or intentional interruptions. The simulator's voice characteristics (prosody, pacing, accent) may also systematically favor ASR systems whose training distributions happen to align with it, introducing a bias that is difficult to quantify without a broader set of simulators. The simulator may also occasionally go off-policy; while EVA employs validators to detect such cases, perfect adherence cannot be guaranteed, particularly on subjective validator metrics.

---

## Framework

### Pipeline Assumptions

The PCM↔μ-law audio conversion used in EVA's bot-to-bot audio interface introduces quality degradation that may disproportionately affect some STT systems. More broadly, aspects of the audio interface — including how audio is buffered, bridged, and timed across the WebSocket connection — may not be fully representative of production voice agent deployments. Inaccurate pipeline event timing (e.g., VAD events) from differing sources may also lead to imprecise response speed values and timestamps. Log reconciliation between systems can have additional inaccuracies due to imprecise or misaligned timestamps across components.

### Voice Agent Framework Dependency

EVA uses Pipecat as the agent framework, which makes specific choices around VAD settings, turn-taking strategy, audio buffering, and pipeline orchestration. Other voice agent deployments — including custom frameworks and first-party APIs offered by model providers — may handle these concerns differently, and the same underlying model may exhibit meaningfully different behavior depending on the framework it is deployed in. As a result, EVA scores reflect a specific Pipecat-based deployment configuration and should not be interpreted as a measure of a model's inherent voice agent capability independent of its serving infrastructure.

### LLM Agent Integration

EVA invokes LLM agents via the chat completions API through LiteLLM. Reasoning state is not currently threaded across tool calls — extended thinking or chain-of-thought outputs from one call are discarded rather than passed forward as context for subsequent calls. Additionally, any text generated alongside a tool call is currently discarded before being surfaced to the user. This is currently an intentional design choice to prevent the user simulator from speaking in response to mid-task agent output and inadvertently triggering an interruption that disrupts the ongoing tool execution flow. Both behaviors may affect the measured performance of models that rely on persistent reasoning state or that use tool-call-adjacent text as a meaningful part of their response strategy.

### Reproducibility and Cost

Full reproduction of EVA results requires access to ElevenLabs and the evaluated commercial model APIs; the pipeline is not fully reproducible with open-source tooling alone. Running three trials per sample across multiple system configurations also carries non-trivial cost, which may limit community adoption for exhaustive comparisons. Additionally, latency measurements — which manifest in turn-taking and response speed metrics — will vary depending on API providers, deployment configurations, and hardware, potentially leading to variation in EVA-X results even within the same system across runs.
