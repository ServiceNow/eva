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

### Turn Detection Configurations

We use the default turn detection configurations for most framework in our experiments. Each framework offers varying levels of configurability, making it difficult to standardize exact parameters and turn strategies across evaluations.

- **Pipecat.** The default start strategy uses VAD (voice activity detection) or transcription receipt to determine when the user begins speaking, and the stop strategy uses AI-powered turn detection via `LocalSmartTurnAnalyzerV3` to determine when the user finishes speaking.
- **OpenAI Realtime.** We use the default server VAD, which uses periods of silence to detect turn boundaries. Default values are used for `threshold`, `prefix_padding_ms`, and `silence_duration`.
- **ElevenAgents.** The turn "eagerness" parameter was set to `eager`.
- **Gemini Live.** We use the default automatic VAD provided.

EVA-Bench makes turn detection parameters and options configurable via the CLI, so practitioners can run experiments using the turn detection settings available to their chosen framework. The only exception is ElevenAgents, where users must register and configure their agents separately prior to evaluation.

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
