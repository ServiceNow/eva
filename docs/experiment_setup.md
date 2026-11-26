# Experimental Setup

Below are the turn detection and model configurations for all evaluated and judge models, and details on the user simulator.

## Leaderboard Models Configurations

### Self-Hosted Models

All self-hosted models were served on NVIDIA H100 GPUs. Models served via vLLM used `vllm-openai v0.19.0`. The table below lists the hardware and serving configurations.

**Gemma-4-26B** and **Gemma-4-31B** were called with `temperature=1.0`, `top_p=0.95`, `top_k=64`, and `max_tokens=12000`. Thinking mode was disabled via `enable_thinking=false` and special tokens were preserved (`skip_special_tokens=false`).

**Qwen-3.5-27B** was called with `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, and `repetition_penalty=1.0`. Thinking mode was likewise disabled via `enable_thinking=false`.

| Model ID                             | Type | GPU     | CPU     | Precision | Deployment    |
|--------------------------------------|------|---------|---------|-----------|---------------|
| google/gemma-4-26B-A4B-it            | LLM  | 2× H100 | --      | BF16      | vLLM          |
| google/gemma-4-31B-it                | LLM  | 2× H100 | --      | BF16      | vLLM          |
| Qwen/Qwen3.5-27B                     | LLM  | 4× H100 | 8× 8GB  | BF16      | vLLM          |
| nvidia/parakeet-ctc-1.1b             | STT  | 1× H100 | --      | BF16      | Nvidia NIM    |
| openai/whisper-large-v3              | STT  | 1× H100 | 4× 64GB | FP16      | vLLM          |
| CohereLabs/cohere-transcribe-03-2026 | STT  | 1× H100 | 8× 64GB | BF16      | vLLM          |
| hexgrad/Kokoro-82M                   | TTS  | 1× H100 | 8× 32GB | FP32      | Remsky Kokoro |
| mistralai/Voxtral-4B-TTS-2603        | TTS  | 1× H100 | --      | BF16      | vLLM          |

### API-Hosted Models

For ElevenLabs, we used ElevenAgents with the following models: _Scribe-v2.2-Realtime_, _Claude Haiku 4.5_, and _Eleven Flash v2_. We used the default agent parameters, listed in the table below.

| Component | Parameter                 | Value                                              |
|-----------|---------------------------|----------------------------------------------------|
| STT       | Filter background speech  | disabled                                           |
| TTS       | Expressive mode           | disabled                                           |
|           | Voice                     | Lauren B - Friendly & Engaging Customer Care Agent |
| LLM       | Temperature               | 0                                                  |
|           | Reasoning effort          | minimal                                            |
|           | Limit token usage         | -1                                                 |
|           | Parallel tool calling     | disabled                                           |
|           | Cascade timeout           | 8 s                                                |
| Tools     | Wait for response         | enabled                                            |
|           | Pre-tool speech           | force                                              |
|           | Execution mode            | immediate                                          |
|           | Tool call sound           | none                                               |
|           | Response timeout          | 20 s                                               |
| Agent     | Eagerness                 | eager                                              |
|           | Spelling patience         | auto                                               |
|           | Speculative turn          | enabled                                            |
|           | Re-transcribe on timeout  | disabled                                           |
|           | Take turn after silence   | 15 s                                               |
|           | End call after silence    | disabled                                           |
|           | Max conversation duration | 600 s                                              |

The table below lists all the other API-hosted models.

| Model ID                                    | Provider    | Type | Parameters                  |
|---------------------------------------------|-------------|------|-----------------------------|
| gpt-5.4                                     | OpenAI      | LLM  | reasoning: default          |
| gpt-5.4-mini                                | OpenAI      | LLM  | reasoning: default          |
| gpt-realtime-2.0                            | OpenAI      | S2S  | reasoning: default; voice: Marin          |
| gpt-realtime-1.5                            | OpenAI      | S2S  | voice: Marin                         |
| gpt-realtime-mini                           | OpenAI      | S2S  | voice: Marin                         |
| gemini-3.1-flash-live-preview               | Google      | LALM | voice: Leda     |
| gemini-3.1-flash-tts-preview                | Google      | TTS  | voice: provider default     |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 | AWS Bedrock | LLM  | --                          |
| Ultravox-realtime                           | Ultravox    | LALM | --                          |
| ink-whisper                                 | Cartesia    | STT  | --                          |
| sonic-3                                     | Cartesia    | TTS  | voice:  Katie - Friendly Fixer  |
| nova-3                                      | Deepgram    | STT  | --                          |
| aura-2-helena-en                            | Deepgram    | TTS  | voice: helena; language: en |

### Cascade latency optimizations

The CASCADE pipeline exposes three optional latency levers. They are off or unset by default and affect only STT -> LLM -> TTS runs:

| Flag | Values | Effect |
|------|--------|--------|
| `EVA_MODEL__PRE_TOOL_SPEECH` | `off` / `auto` | Ask the model to speak a brief lead-in before running a tool. The lead-in is model-generated, not templated. |
| `EVA_MODEL__LLM_STREAMING` | `true` / `false` | Stream LLM output to TTS sentence-by-sentence. |
| `EVA_MODEL__PARALLEL_TOOL_CALLS` | unset / `true` / `false` | Forward the provider's `parallel_tool_calls` setting when tools are present. Leave unset for provider defaults; set `false` to match the ElevenAgents assistant config. |

Streaming still records one assistant audit entry per completed turn. If streaming is interrupted after speech starts, the spoken prefix is recorded so scored transcripts match emitted audio.

### Turn Detection Configurations

We use the default turn detection configurations for most framework in our experiments. Each framework offers varying levels of configurability, making it difficult to standardize exact parameters and turn strategies across evaluations.

- **Pipecat.** The default start strategy uses VAD (voice activity detection) or transcription receipt to determine when the user begins speaking, and the stop strategy uses AI-powered turn detection via `LocalSmartTurnAnalyzerV3` to determine when the user finishes speaking.
- **Cartesia STT.** `EVA_MODEL__STT=cartesia` uses ink-2 self-endpointing and auto-selects `TURN_START_STRATEGY=external`, `TURN_STOP_STRATEGY=external`, and `VAD=none`; `cartesia-multilingual` keeps the ink-whisper VAD/smart-turn path.
- **OpenAI Realtime.** We use the default server VAD, which uses periods of silence to detect turn boundaries. Default values are used for `threshold`, `prefix_padding_ms`, and `silence_duration`.
- **ElevenAgents.** The turn "eagerness" parameter was set to `eager`.
- **Gemini Live.** We use the default automatic VAD provided.

EVA-Bench makes turn detection parameters and options configurable via the CLI, so practitioners can run experiments using the turn detection settings available to their chosen framework. The only exception is ElevenAgents, where users must register and configure their agents separately prior to evaluation.

### Cartesia Managed Agents

`framework: cartesia` runs a Cartesia managed agent over a WebSocket connection. Managed Agents runs the whole agent loop: audio orchestration (listening, turn-taking, interruptions and speech) along with tool calling, reasoning orchestration and knowledge retrieval, around the reasoning model you choose. EVA's tools run as client tools, and EVA's system prompt and greeting are sent with each call as dynamic variables. One agent per domain is registered before evaluation, with its config in `scripts/provision_cartesia_agents.py`. Model fallback is disabled on these agents, so only the chosen model answers.

Some agent settings can't be changed per call, so they are set when the agent is provisioned from EVA's options. The assistant server checks them against the run config and refuses to start if they don't match:

| EVA option | Agent setting |
|---|---|
| `EVA_LANGUAGE` | `language.primary` (regional variants such as `fr-CA` use the base code) |
| `EVA_MODEL__PRE_TOOL_SPEECH` (`auto` or `off`) | each tool's `pre_tool_speech` |
| `EVA_TURN_END_FALLBACK_TIME` (5 to 60 seconds, or unset) | `turn.inactivity_check_in_secs` (unset turns check-ins off) |

```bash
# Create or update the agents; --model picks the LLM (default gpt-5.5). Prints each domain's s2s_params.
python scripts/provision_cartesia_agents.py --pre-tool-speech auto

EVA_FRAMEWORK=cartesia \
EVA_MODEL__S2S=cartesia \
EVA_MODEL__S2S_PARAMS='{"model": "gpt-5.5", "api_key": "<cartesia key>", "assistant_agent_id": "<airline agent id>"}' \
python main.py --domain airline
```

`--voice` sets the agent's voice; it isn't checked against the run config.

`s2s_params`: `api_key`, `model` and `assistant_agent_id` (the agent for the run's domain) are required, as for ElevenLabs. `model` must match the model configured on the agent; `alias` optionally labels the run. `base_url` (default `wss://api.cartesia.ai`) and `cartesia_version` are optional. `EVA_MODEL__LLM` must be unset, as for any S2S run.

### Krisp Managed Voice Agent

`framework: krisp` runs a published Krisp managed voice agent as a phone call over its Twilio-style media stream. The platform runs the whole agent loop: Krisp speech-to-text and text-to-speech, turn-taking, interruptions and tool calling, around the reasoning model below. Krisp provisions one agent per domain from EVA's realtime system prompt, greeting and tools, and hands over everything a run needs: the API URL, each domain's agent ID and team token, the gateway secret, and the tool address setup described below.

Known gaps against Cartesia's setup:

- **Scenario date.** By default (`"call_date": "session"`) the prompt carries EVA's own line, `Today is {{vars.datetime}}.`, and the server sends the scenario's date when each call starts; if the platform does not accept it, every call fails as unscoreable, so try one record first. With `"call_date": "clock"`, the line tells the agent to call `get_current_datetime`, a tool the assistant server answers with the scenario's date. Every other line of the prompt is EVA's.
- **Tools.** EVA's tools are HTTP tools that call a gateway hosted by the assistant server on `gateway_port` (default 8290). Krisp points each agent's tools at one fixed HTTPS address that forwards to that port: either a tunnel Krisp sets up, started next to EVA with `cloudflared tunnel run --token <tunnel token>` (outbound HTTPS only), or an address of your own that reaches the port. Each call carries the session id and the shared gateway secret. The domains share the address, so run them one after another.
- **Tool names.** Names are unique per platform team, and the three domains define shared names such as `transfer_to_agent` differently, so each domain's agent lives in its own team with its own token.
- **Transcript.** The media stream carries audio only. The server times each agent turn from the audio and reads the text from the platform's session history after the call; text the caller never heard because of an interruption is cut and marked `[interrupted]`.

The assistant server checks the published agent against the run config and refuses to start if they don't match:

| EVA option | Agent setting |
|---|---|
| `EVA_LANGUAGE` | the agent's supported languages and its greeting for that language |
| `EVA_MODEL__PRE_TOOL_SPEECH` (`auto` or `off`) | the system prompt (the lead-in instruction is part of it) |
| `s2s_params.model` | the agent's model |
| `s2s_params.preamble` (`on` or `off`) | each tool's lead-ins, except `get_current_datetime` |
| the domain's tools | the agent's tools (EVA's, plus `get_current_datetime` unless `call_date` is `session`) |

```bash
EVA_FRAMEWORK=krisp \
EVA_MODEL__S2S=krisp \
EVA_MODEL__PRE_TOOL_SPEECH=off \
KRISP_GATEWAY_SECRET='<gateway secret>' \
EVA_MODEL__S2S_PARAMS='{"model": "gpt-5.5", "api_url": "<API URL>", "api_key": "<airline team token>", "assistant_agent_id": "<airline agent id>"}' \
python main.py --domain airline
```

Krisp evaluates this configuration:

| Component | Parameter | Value |
|-----------|-----------|-------|
| LLM | Model | `gpt-5.5` |
| | Reasoning effort | `none` |
| | Service tier | `priority` |
| STT | Engine | Krisp STT |
| | Voice isolation | on |
| TTS | Engine and voice | Krisp TTS, voice "Eva" |
| Tools | Pre-tool speech | templated platform lead-ins ("One moment.", "Let me check.", ...) on every tool |
| | Execution | HTTP call to the assistant server's gateway; writes are never replayed |
| Prompt | Scenario date | sent per call (`call_date: session`) |
| Agent | Turn taking | the platform's default turn hold |
| Audio | Media stream | 8 kHz mu-law, both directions |

The platform speaks the lead-ins, so the run uses `EVA_MODEL__PRE_TOOL_SPEECH=off` and the prompt carries no lead-in instruction; the run declares the setting as `s2s_params.preamble`, which the server checks against the tools.

`s2s_params`: `api_url`, `api_key`, `model` and `assistant_agent_id` are required; `preamble` defaults to `on` and `call_date` to `session`. `gateway_secret` (or `KRISP_GATEWAY_SECRET`) must match the secret the agent's tools send; `gateway_port` defaults to 8290. `media_stream_url` overrides the stream endpoint derived from the session exchange. `EVA_MODEL__LLM` must be unset, as for any S2S run.

## Judge Models

The table below lists the API-hosted models used as a judge.

| Model ID                                    | Provider    | Type | Parameters                  |
|---------------------------------------------|-------------|------|-----------------------------|
| gpt-5.2                                     | OpenAI      | LLM  | reasoning: default          |
| gemini-3-flash-preview                      | Google      | LLM  | reasoning: default          |
| us.anthropic.claude-opus-4-6-v1             | AWS Bedrock | LLM  | reasoning: default          |



## ElevenLabs User Simulator

We use ElevenLabs ElevenAgents as the user simulator with the following cascade system: _Scribe v2.2 Realtime + GPT-5.1 + Eleven V3 Conversational_. We select these models for their high transcription accuracy, User Behavioral Fidelity, user realism for GPT-5.1, and for their naturalness and realism for Eleven v3 Conversational. ElevenLabs also provides a large voice library, enabling testing of a wide variety of user accents, languages, speaking styles, etc.

We created four ElevenLabs agents for the user simulator, covering two accents (English and French) and two genders each. When creating a new agent, select _Blank Agent_ as the starting template, then apply the configuration as described in the tables below. All parameters not listed are set to their default values provided by ElevenLabs at agent creation.


| Parameter                            | Value                                                                             |
|--------------------------------------|-----------------------------------------------------------------------------------|
| TTS model family                     | V3 Conversational                                                                 |
| Expressive mode                      | Enabled (no tags selected)                                                        |
| Language                             | English                                                                           |
| LLM                                  | GPT-5.1                                                                           |
| System prompt                        | {{prompt}}                                                                        |
| Default personality                  | Disabled                                                                          |
| First message                        | None (remove the default first message, as the agent speaks first)                |
| Interruptible                        | Disabled                                                                          |
| Advanced > Input audio               | μ-law telephony, 8000 Hz                                                          |
| Advanced > Eagerness                 | Eager                                                                             |
| Advanced > Take turn after silence   | 15s                                                                               |
| Advanced > Max conversation duration | 600s                                                                              |
| Tools > System tools                 | Enable "End conversation" (Name is `end_call`, and Description is provided below) |

Below are the voice names used for the user simulator with ElevenAgents, for English language:

| Accent  | Gender | Voice Name                            | Voice ID |
|---------|--------|---------------------------------------|---|
| English | Female | Natalee Champlin                      | KpTQ5yzwazQWLkvnK59A |
| English | Male   | Eric - Smooth, Trustworthy            | cjVigY5qzO86Huf0OWal |
| French  | Female | Mariva Viva Muse - Warm and Energetic | 1hIScOW98xkqE5ttC10C |
| French  | Male   | Jamie - French Accent \& Charismatic  | K8nDX2f6wjv6bCh5UeZi |

The simulator is prompted with a specific user goal and is instructed to stay on task, communicate all required named entities clearly, and terminate when the goal is accomplished or the task is clearly unlikely to succeed.

When enabling the "End Conversation" system tool, the name must be `end_call`, and the description to provide is shown below. This allows the simulator to hang up programmatically.

```
Use this to end the phone call and hang up.

Call this function when any ONE of the following is true:
1. The agent has confirmed your request is resolved and you have said goodbye
2. The agent says they are transferring you to a live agent
3. The agent has been unable to make progress for at least 5 consecutive turns
4. The agent says goodbye or indicates the conversation is over
5. The agent indicates that the remainder of your request cannot be fulfilled.
6. If the assistant says something along the lines of "I'm sorry I encountered an error processing your request."

IMPORTANT: never call this tool in the same turn that you provide the agent with data, an identifier, a request to transfer to a live agent, an approval to proceed, or any kind of additional information. 


Before calling this tool, always say a brief goodbye first.
```

Once the agent is configured, click "Publish" in the top-right corner. The `agent-id` can be retrieved from the "Widget" tab of the agent dashboard, under "Embed code".

The simulator is prompted in EVA-Bench with a specific user goal and is instructed to stay on task, communicate all required named entities clearly, and terminate the conversation when the goal is accomplished, or the task is clearly unlikely to succeed. The system prompts are defined in `configs/prompts/simulation.yaml`.
