# Turn-Taking Development

## Overview

Turn-Taking Metric aims to measure whether a voice agent takes the conversational floor at the correct time — specifically, whether it begins responding after the user has genuinely completed their utterance, neither interrupting prematurely nor waiting so long that the silence becomes conversationally unnatural. Typically, turn-taking is quantified through timing-based metrics that can derive from either raw audio streams, speaker-aligned transcripts via forced-alignment and speaker diarization, or simulation logs. Therefore, we explore 3 complimentary approaches representing information extraction from 3 different sources.

## Raw Audio Stream

We explore direct audio-based evaluation by presenting the recorded audio stream to a capable LALM — specifically Gemini 2.5 — as the judge, without any intermediate preprocessing. While this approach eliminates the need for audio processing steps, it places significant demand on the judge model's ability to accurately perceive and reason over fine-grained acoustic timing signals.

## Speaker-Aligned Transcripts

We explore leveraging Whisper-X, a forced-alignment ASR system with speaker diarization support via Silero-VAD integration, to extract time-aligned speaker-attributed transcripts directly from the recorded audio. The resulting segment boundaries and transition timestamps are then provided to an LLM-as-Judge for turn-taking assessment.

## Simulation Logs

Our EVA framework natively captures fine-grained speaker-level timestamp information for both user's and agent's turns through the bot-to-bot simulation. These logs can be leveraged directly for turn-taking evaluation, eliminating the reliance on additional audio processing steps that would otherwise introduce errors in alignment and timing artifacts.

## Turn Labels

Each evaluated agent's turn (i.e. agent -> user) is assigned one of four labels based on the agent's response timing relative to the user's floor yield:

- **Early/Interruption (-1):** The agent begins speaking before the user has completed the utterance or cuts off or overlaps user speech.
- **On-Time (0):** The agent initiates response within the natural floor-transfer window of 200ms-4000ms.
- **Late (+1):** The agent's response is over 4000ms.
- **Other/Indeterminate (null):** Cases that cannot be reliably assessed, including tangled multi-party overlapping speech, ambiguous floor yields without clear utterance completion signals.

## Aggregation

The turn-taking metric is computed by mean-aggregating scores across all agent's turns in the simulation, followed by a normalization step to map the aggregated score to a standardized range of [0,1].

## Findings

Through our initial experimentation, we observe that the first approach is unreliable in practice as the LALM judge consistently assigns `On-Time` labels for agent's responses which are not on-time, and only produces reliable ratings when ground-truth timestamps are provided explicitly. This finding underscores the inability of current LALMs to accurately extract and reason over fine-grained acoustic timing information directly from audio streams, and is consistent with observations reported in existing literature [1][2][3].

The second approach introduces two distinct failure modes: first, speaker diarization and forced alignment are inherently imperfect, leading to word-level misattribution between speakers of different roles — a particularly acute problem when user utterances are prosodically minimal (e.g., short acknowledgments such as "Ok" or "Sure"); second, both diarization and forced alignment degrade significantly in the presence of overlapping speech or interruptions, precisely the conditions under which accurate turn boundary detection is most critical.

We therefore adopt the third approach as the primary evaluation pathway. Although pipeline-level latency offsets between components in Pipecat may introduce minor timing artifacts, the simulation logs provide the most faithful and directly accessible timestamp representation of the multi-turn dialogue structure, requiring minimal additional audio preprocessing and remaining robust to the acoustic conditions that undermine the reliability of the other two approaches.

## Interruption Detection

Since interruptions are an inevitable artifact of live bot-to-bot conversational simulation, we detect them programmatically by analyzing the recorded timestamp logs for overlapping speaker segments. The resulting interruption events are encoded as structured inline tags and injected into the input transcripts, providing the LLM judge with explicit, time-grounded signals about floor transfer dynamics that would otherwise be invisible from transcript text alone.

## References

- [1] Jiang et al., 2025. S2S-Arena, Evaluating Speech2Speech Protocols on Instruction Following with Paralinguistic Information. arXiv:2503.05085v1
- [2] Chandra et al., 2026. Hearing Between the Lines: Unlocking the Reasoning Power of LLMs for Speech Evaluation. EACL 2026 Findings
- [3] Shen et al., 2026. GSRM: Generative Speech Reward Model for Speech RLHF. arXiv:2602.13891
