# Cascade user simulator

The cascade caller combines streaming speech recognition, a caller LLM, and speech
synthesis. It waits for the assistant to greet, then takes ordinary conversational
turns using the shared persona, goal, and `end_call` rules.

## Setup

Set `ELEVENLABS_API_KEY` for the default streaming recognizer and `CARTESIA_API_KEY`
for caller speech. Configure a `user-llm` deployment in `EVA_MODEL_LIST`; the
deployment controls the caller model and its provider credentials independently
of the assistant being evaluated. See [LLM configuration](llm_configuration.md).

```dotenv
EVA_USER_SIMULATOR__PROVIDER=cascade
EVA_USER_SIMULATOR__STT=elevenlabs
EVA_USER_SIMULATOR__STT_PARAMS={"model":"scribe_v2_realtime"}
EVA_USER_SIMULATOR__LLM=user-llm
EVA_USER_SIMULATOR__TTS=cartesia
EVA_USER_SIMULATOR__TTS_PARAMS={"model":"sonic-3.5"}
```

`stt_params` supplies keyword arguments to the selected LiveKit STT plugin; the
ElevenLabs plugin is included. Other STT providers require their corresponding
LiveKit plugin. Caller TTS currently uses Cartesia and accepts `model`, `api_key`,
`female_voice`, and `male_voice` in `tts_params`.

Run a single debug record with your existing assistant configuration:

```bash
uv run eva --debug --user-simulator.provider cascade
```

The caller supports the existing behavior-prompt, background-noise, and
connection-degradation perturbations. Accent perturbations require the ElevenLabs
Agents caller.

## Transport and turn-taking

Each tick exchanges 200 ms of audio. The caller waits for assistant silence and
for a reply to its previous turn before generating another utterance. It waits
briefly for finalized transcripts, then falls back to available partial text.

OpenAI Realtime uses a tick-driven adapter: the server sends unpaced audio and the
caller releases one tick at a time. Quiet ticks still wait briefly for incoming
audio and send silence so provider turn detection can finish the caller's turn.
Other frameworks use the real-time adapter and retain output pacing.

## Artifacts and endings

- `user_simulator_events.jsonl` records speech, audio boundaries, and the terminal
  connection state.
- `audio_user_clean.wav` stores synthesized caller audio before perturbation.
- `user_simulator_decisions.jsonl` records each tick's speech state and audio level.

The caller ends with `goodbye` when its LLM calls `end_call`, or `timeout` when the
conversation budget expires. Sustained assistant silence ends with
`inactivity_timeout`. A stalled tick-driven provider ends with the distinct
`provider_stalled` reason and preserves partial artifacts for diagnosis.

## Experimental out-of-turn behaviors

These behaviors are opt-in and are still being tested. Enable them independently
in the cascade configuration:

```dotenv
EVA_USER_SIMULATOR__DECISION_LLM=user-llm
EVA_USER_SIMULATOR__ENABLE_BACKCHANNEL=true
EVA_USER_SIMULATOR__ENABLE_INTERRUPTIONS=true
EVA_USER_SIMULATOR__SPECULATIVE_GENERATION=true
```

All three behavior flags default to `false`. `decision_llm` selects the deployment
for listener and relevance checks; its default is `user-llm`.

Backchannels play cached continuers while the assistant speaks without taking an
ordinary caller turn. Interruptions generate goal-aware barge-ins, limited to one
per detected assistant turn. Speculative generation pre-renders candidate
interruptions and checks their relevance before use; it is useful when
interruptions are also enabled.

The tick-driven adapter reports the played audio position when a barge-in reaches
the wire, discards unheard buffered audio, and requests OpenAI Realtime item
truncation. The real-time adapter retains its normal streaming behavior.

`configs/caller_phrases.yaml` supplies localized continuers and interruption
openers. `scripts/add_culture_data.py` generates these phrase sets alongside other
language data. Phrase audio is cached across conversations.

The decision trace adds listener verdicts, skipped-check context, interruption
outcomes, and speculative-candidate decisions. Use it alongside the event log and
audio to evaluate actual timing; passing unit tests alone does not establish
behavioral quality.
