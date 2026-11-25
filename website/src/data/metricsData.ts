import judgePrompts from 'virtual:judge-prompts';

export type MetricType = 'deterministic' | 'llm_judge' | 'lalm_judge';
export type MetricCategory = 'eva-a' | 'eva-x' | 'debug' | 'validation';

export interface MetricDefinition {
  id: string;
  displayName: string;
  category: MetricCategory;
  type: MetricType;
  badge?: string;
  judgeModel?: string;
  description: string;
  inputs: string;
  outputRange: string;
  passThreshold?: string;
  judgePrompt?: string;
  judgeAccuracy?: number;
  judgeScores?: { label: string; value: number; std?: number }[];
  judgeAlignment?: {
    measure: string;
    value: number;
    ci?: [number, number];
    notes?: string;
  };
  judgeDevelopmentNotes?: string;
  developmentDocUrl?: string;
}

export const metricTypeLabels: Record<MetricType, string> = {
  deterministic: 'Deterministic',
  llm_judge: 'LLM Judge',
  lalm_judge: 'Audio LLM Judge',
};

export const metricTypeColors: Record<MetricType, string> = {
  deterministic: '#06B6D4',
  llm_judge: '#8B5CF6',
  lalm_judge: '#F59E0B',
};

export const metrics: MetricDefinition[] = [
  // ─── EVA-A Core Metrics (3) ───
  {
    id: 'task_completion',
    displayName: 'Task Completion',
    category: 'eva-a',
    type: 'deterministic',
    description: 'Evaluates whether the agent correctly completed the task by comparing the expected end state of the scenario database against the actual end state after the conversation. This is a strict, deterministic comparison inspired by tau-bench-style evaluation.',
    inputs: 'Initial scenario database state, final scenario database state, expected end state database',
    outputRange: 'Binary: 0 (fail) or 1 (pass)',
    passThreshold: '1.0',
  },
  {
    id: 'agent_speech_fidelity',
    displayName: 'Speech Fidelity',
    badge: 'beta',
    category: 'eva-a',
    type: 'lalm_judge',
    judgeModel: 'Gemini 3.8 Flash',
    judgeAccuracy: 0.8957,
    judgeScores: [
      { label: 'accuracy', value: 0.8957, std: 0.0258 },
      { label: 'macro_f1', value: 0.856, std: 0.024 },
    ],
    judgeAlignment: {
      measure: "Unweighted Cohen's κ",
      value: 0.777,
      ci: [0.704, 0.835],
      notes: 'Binary metric',
    },
    description: 'Measures whether the agent\'s spoken audio accurately represents the key entities (dates, names, numbers, codes, addresses, etc.), using an audio LLM for multimodal analysis. If the agent garbles or misstates key information, the user receives incorrect information regardless of how good the text reasoning was. To keep the EVA score apples-to-apples across all pipeline setups, the same entity-focused metric runs for every pipeline type — cascade, S2S, and audio-LLM. It does not require any intended text.',
    inputs: 'Agent audio recording, conversation trace (user utterances and tool responses as the source of entities to listen for; assistant turns are redacted to a placeholder)',
    outputRange: 'Binary per turn (0=low fidelity, 1=high fidelity), aggregated as mean across scored turns (turns with no entities are skipped)',
    passThreshold: '≥ 0.95',
    judgePrompt: judgePrompts.agent_speech_fidelity.user_prompt,
    developmentDocUrl: 'https://github.com/ServiceNow/eva/blob/main/docs/metrics/metric_development/agent_speech_fidelity_development.md',
  },
  {
    id: 'faithfulness',
    displayName: 'Faithfulness',
    category: 'eva-a',
    type: 'llm_judge',
    judgeModel: 'GPT-5.2 (medium reasoning)',
    judgeAccuracy: 0.7956,
    judgeScores: [
      { label: 'accuracy', value: 0.7956, std: 0.0126 },
      { label: 'macro_f1', value: 0.7675, std: 0.0147 },
    ],
    judgeAlignment: {
      measure: "Quadratic-weighted Cohen's κ",
      value: 0.711,
      ci: [0.570, 0.820],
    },
    description: 'Measures whether the agent\'s responses were consistent with its instructions, provided policy, user inputs, and tool call results. Evaluates across 6 dimensions: fabricating tool parameters, misrepresenting tool results, violating policies, failing to disambiguate, hallucination, and wrong language (speaking a language other than the one in the agent instructions).',
    inputs: 'Agent role, agent instructions, available tools, current date/time, full conversation trace with tool calls',
    outputRange: '1-3 scale, normalized to 0-1 (1=unfaithful, 2=partially faithful, 3=fully faithful)',
    passThreshold: '≥ 0.5',
    judgePrompt: judgePrompts.faithfulness.user_prompt,
    developmentDocUrl: 'https://github.com/ServiceNow/eva/blob/main/docs/metrics/metric_development/faithfulness_development.md',
  },

  // ─── EVA-X Core Metrics (3) ───
  {
    id: 'turn_taking',
    displayName: 'Turn Taking',
    category: 'eva-x',
    type: 'deterministic',
    description: 'A timestamp-based metric measuring whether the agent took the floor at the right time \u2014 not interrupting the user, not letting silence linger.\n\nEach turn is classified by its interrupt condition and routed to one of three scoring functions:\n\nUninterrupted turns \u2014 scored against a piecewise-linear latency curve on the agent\'s response latency:\n- Hard zero below -500 ms (premature speech)\n- Ramp from 0 to 1 between -500 and 500 ms\n- Score = 1 in the 500\u20132000 ms sweet spot\n- Ramp back to 0 between 2000 and 3500 ms\n- A tool-call-aware variant stretches the upper breakpoints to 3000 ms and 5000 ms so infrastructure latency from tool execution isn\'t conflated with slow responsiveness.\n\nAgent-interrupted turns \u2014 take the minimum of three sub-scores, each capped at 0.5:\n- Total overlap duration (penalized up to a 2000 ms tolerance)\n- Count of distinct overlapping segments (penalized up to 3)\n- Post-interrupt recovery latency (scored on the same latency curve)\n\nUser-interrupted turns \u2014 scored on yield latency with a linear penalty up to a 2000 ms cap, rewarding agents that stop speaking immediately.\n\nThe per-record Turn Taking score is the mean of these per-turn scores in [0, 1].',
    inputs: 'Per-turn event timestamps and latencies from the simulation logs: audio start/end markers for both speakers, agent tool-call boundaries, and per-turn interrupt classification',
    outputRange: 'Continuous score in [0, 1], aggregated as the mean of per-turn scores',
    passThreshold: '≥ 0.80 (≈ 4 of 5 turns on-time)',
    developmentDocUrl: 'https://github.com/ServiceNow/eva/blob/main/docs/metrics/metric_development/turn_taking_development.md',
  },
  {
    id: 'conciseness',
    displayName: 'Conciseness',
    category: 'eva-x',
    type: 'llm_judge',
    judgeModel: 'GPT-5.2',
    judgeAccuracy: 0.9318,
    judgeScores: [
      { label: 'accuracy', value: 0.9318, std: 0.0012 },
      { label: 'macro_f1', value: 0.8462, std: 0.0032 },
    ],
    judgeAlignment: {
      measure: "Quadratic-weighted Cohen's κ",
      value: 0.751,
      ci: [0.685, 0.803],
    },
    description: 'Measures whether the agent\'s responses were appropriately brief and focused for spoken delivery. In voice, users cannot skim, re-read, or scroll back \u2014 presenting too many options, asking multiple questions per turn, or including unnecessary hedging degrades the interaction.',
    inputs: 'Full conversation trace with all turns',
    outputRange: '1-3 per turn (1=verbose, 2=adequate, 3=concise), normalized to 0-1',
    passThreshold: '≥ 0.5',
    judgePrompt: judgePrompts.conciseness.user_prompt,
    developmentDocUrl: 'https://github.com/ServiceNow/eva/blob/main/docs/metrics/metric_development/conciseness_development.md',
  },
  {
    id: 'conversation_progression',
    displayName: 'Conversation Progression',
    category: 'eva-x',
    type: 'llm_judge',
    judgeModel: 'GPT-5.2',
    judgeAccuracy: 0.7549,
    judgeScores: [
      { label: 'accuracy', value: 0.7549, std: 0.0236 },
      { label: 'macro_f1', value: 0.7339, std: 0.0209 },
    ],
    judgeAlignment: {
      measure: "Quadratic-weighted Cohen's κ",
      value: 0.665,
      ci: [0.527, 0.775],
    },
    description: 'Measures whether the agent moved the conversation forward effectively \u2014 avoiding unnecessary repetition, retaining context across turns, and driving toward task completion without stalling.',
    inputs: 'Full conversation trace',
    outputRange: '1-3 (1=clear progression issues, 2=minor issues, 3=smooth progression), normalized to 0-1',
    passThreshold: '≥ 0.5',
    judgePrompt: judgePrompts.conversation_progression.user_prompt,
    developmentDocUrl: 'https://github.com/ServiceNow/eva/blob/main/docs/metrics/metric_development/conversation_progression_development.md',
  },

  // ─── Debug Metrics (6) ───
  {
    id: 'transcription_accuracy_key_entities',
    displayName: 'Key Entity Transcription',
    category: 'debug',
    type: 'llm_judge',
    judgeModel: 'GPT-5.2',
    judgeAccuracy: 0.9453,
    judgeScores: [
      { label: 'entity_f1_lenient', value: 0.9453, std: 0.0031 },
      { label: 'correctness_with_penalty', value: 0.8623, std: 0.0081 },
    ],
    description: 'Evaluates whether the agent correctly transcribed key named entities from user speech \u2014 confirmation codes, names, flight numbers, dates, and similar entities where transcription errors are conversation-ending rather than merely cosmetic.',
    inputs: 'User intended speech text (what TTS was asked to say), agent\'s transcription of user speech',
    outputRange: 'Ratio 0-1 (proportion of correctly transcribed entities across all turns)',
    judgePrompt: judgePrompts.transcription_accuracy_key_entities.user_prompt,
    developmentDocUrl: 'https://github.com/ServiceNow/eva/blob/main/docs/metrics/metric_development/transcription_accuracy_key_entities.md',
  },
  {
    id: 'tts_fidelity',
    displayName: 'TTS Fidelity',
    category: 'debug',
    type: 'lalm_judge',
    judgeModel: 'Gemini 3.8 Flash',
    judgeAccuracy: 0.8957,
    judgeScores: [
      { label: 'accuracy', value: 0.8957, std: 0.0258 },
      { label: 'macro_f1', value: 0.856, std: 0.024 },
    ],
    judgeAlignment: {
      measure: "Unweighted Cohen's κ",
      value: 0.777,
      ci: [0.704, 0.835],
      notes: 'Binary metric',
    },
    description: 'Measures whether the agent correctly spoke the information it intended to communicate. TTS systems can mispronounce, skip, or distort words \u2014 in a voice context, if a confirmation code is not spoken correctly, the user cannot act on it regardless of whether the LLM produced the right answer.',
    inputs: 'Agent audio recording, intended assistant text (what LLM generated)',
    outputRange: 'Binary per turn (0=low fidelity, 1=high fidelity), aggregated as mean across turns',
    judgePrompt: judgePrompts.tts_fidelity.user_prompt,
    developmentDocUrl: 'https://github.com/ServiceNow/eva/blob/main/docs/metrics/tts_fidelity.md',
  },
  {
    id: 'authentication_success',
    displayName: 'Authentication Success',
    category: 'debug',
    type: 'deterministic',
    description: 'Checks whether the agent successfully authenticated the user by verifying identity through required credentials (e.g., confirmation number, last name).',
    inputs: 'Audit log tool calls, expected authentication parameters',
    outputRange: 'Binary: 0 (fail) or 1 (pass)',
  },
  {
    id: 'response_speed',
    displayName: 'Response Speed',
    category: 'debug',
    type: 'deterministic',
    description: 'Measures the elapsed time in seconds between the user\'s last audio and the agent\'s first audio response. A direct measurement of end-to-end latency.',
    inputs: 'Audio timestamp data from pipeline events',
    outputRange: 'Seconds (lower is better). Normalized: (5.0 - clamped_speed) / 3.0 for scores in 0-1 range',
  },
  {
    id: 'conversation_correctly_finished',
    displayName: 'Conversation Correctly Finished',
    category: 'debug',
    type: 'deterministic',
    description: 'Reports whether the conversation reached a clean close — both sides exchanged proper goodbyes and the user simulator invoked the end-call tool — rather than terminating because the agent went silent or otherwise failed to drive the call to completion. Useful for surfacing systems who systematically fail to respond to user turns leading the conversation to hang.',
    inputs: 'Transcript, audit log, end-call tool invocations',
    outputRange: 'Binary: 0 (agent failed to respond / conversation timed out) or 1 (user-driven close)',
  },
  {
    id: 'stt_wer',
    displayName: 'STT Accuracy (WER)',
    category: 'debug',
    type: 'deterministic',
    description: 'Speech-to-Text Word Error Rate computed using jiwer. Measures overall transcription quality by comparing what the user intended to say against what the agent\'s STT system actually transcribed. Score is reported as accuracy (1 - WER, clamped to 0-1).',
    inputs: 'Intended user turns (TTS text), transcribed user turns (STT output)',
    outputRange: 'Accuracy 0-1 (1.0 = perfect transcription, 0.0 = completely wrong)',
  },
  {
    id: 'tool_call_validity',
    displayName: 'Tool Call Validity',
    category: 'debug',
    type: 'deterministic',
    description: 'Checks whether all tool calls made by the agent used valid tool names and provided required parameters according to the tool schema.',
    inputs: 'Audit log tool calls, agent tool definitions',
    outputRange: 'Binary: 0 (invalid calls present) or 1 (all calls valid)',
  },

  // ─── Validation Metrics (3) ───
  {
    id: 'user_behavioral_fidelity',
    displayName: 'User Behavioral Fidelity',
    category: 'validation',
    type: 'llm_judge',
    judgeModel: 'GPT-5.2 (medium reasoning)',
    description: 'Determines whether the simulated user\'s behavior corrupted the voice agent evaluation \u2014 specifically, whether the user\'s actions led to the database being in a different state than it should be, or prevented the agent from completing actions it otherwise would have.',
    inputs: 'Agent-side transcript with tool calls, user-side text (ground truth), user goal, user persona, modification tools list',
    outputRange: 'Binary: 0 (corrupted) or 1 (clean)',
    judgePrompt: judgePrompts.user_behavioral_fidelity.user_prompt,
  },
  {
    id: 'conversation_finished',
    displayName: 'Conversation Valid End',
    category: 'validation',
    type: 'deterministic',
    description: 'Validation gate that decides whether a conversation enters scoring. Before any LLM judges are invoked, EVA-Bench runs a deterministic check on the conversation logs to verify that the simulation terminated correctly: a valid end state is one in which either the agent failed to respond to the user, or the user invoked the end-call tool. Conversations meeting this criterion proceed to the User Speech and Behavioral Fidelity judges; all others are rerun. This gate primarily catches infrastructure-level failures (WebSocket disconnects, simulator timeouts, conversations that failed to start) so judge calls are not spent on malformed simulations.',
    inputs: 'Transcript, audit log, end-call tool invocations',
    outputRange: 'Binary: 0 (invalid end / rerun) or 1 (valid end / proceed to scoring)',
  },
  {
    id: 'user_tts_fidelity',
    displayName: 'User Speech Fidelity',
    category: 'validation',
    type: 'lalm_judge',
    judgeModel: 'Gemini 3.8 Flash',
    description: 'Audio-based check that the user simulator\'s TTS output matches the intended text. Validates that the simulated user is actually saying what it\'s supposed to say.',
    inputs: 'User audio recording, intended user text',
    outputRange: '1-3 per turn (1=low fidelity, 2=medium, 3=high), aggregated as mean',
    judgePrompt: judgePrompts.user_speech_fidelity.user_prompt,
  },
];

export const evaAMetrics = metrics.filter(m => m.category === 'eva-a');
export const evaXMetrics = metrics.filter(m => m.category === 'eva-x');
export const debugMetrics = metrics.filter(m => m.category === 'debug');
export const validationMetrics = metrics.filter(m => m.category === 'validation');
export const judgeMetrics = metrics.filter(m => m.type === 'llm_judge' || m.type === 'lalm_judge');
