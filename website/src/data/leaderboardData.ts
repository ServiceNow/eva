export interface SystemScore {
  id: string;
  name: string;
  shortName: string;
  stt: string;
  llm: string;
  tts: string;
  type: 'cascade' | 's2s' | '2-part';
  evaA: number;
  evaX: number;
  accuracyMetrics: Record<string, number>;
  experienceMetrics: Record<string, number>;
  diagnosticMetrics: Record<string, number>;
  successRates: {
    accuracy: { pass_threshold: number; mean: number; pass_at_k: number; pass_k: number };
    experience: { pass_threshold: number; mean: number; pass_at_k: number; pass_k: number };
  };
}

export const accuracyMetricKeys = [
  'task_completion',
  'agent_tts_fidelity',
  'faithfulness',
] as const;

export const experienceMetricKeys = [
  'turn_taking',
  'conciseness',
  'conversation_progression',
] as const;

export const diagnosticMetricKeys = [
  'key_entity_transcription',
  'response_speed',
] as const;

export const accuracyMetricLabels: Record<string, string> = {
  task_completion: 'Task Completion',
  agent_tts_fidelity: 'Agent Speech Fidelity',
  faithfulness: 'Faithfulness',
};

export const experienceMetricLabels: Record<string, string> = {
  turn_taking: 'Turn Taking',
  conciseness: 'Conciseness',
  conversation_progression: 'Conversation Progression',
};

export const diagnosticMetricLabels: Record<string, string> = {
  key_entity_transcription: 'Key Entity Transcription',
  response_speed: 'Response Speed (s)',
};

export const invertedMetrics = new Set(['response_speed']);

export const ossSystems: SystemScore[] = [
  {
    id: 'ultravox-v0-7-kokoro',
    name: 'ultravox v0.7 + kokoro',
    shortName: 'ultravox v0.7 + kokoro',
    stt: '-', llm: 'ultravox v0.7', tts: 'kokoro',
    type: '2-part',
    evaA: 0.3933, evaX: 0.3067,
    accuracyMetrics: { task_completion: 0.5800, agent_tts_fidelity: 0.9662, faithfulness: 0.4500 },
    experienceMetrics: { turn_taking: 0.3782, conciseness: 0.7532, conversation_progression: 0.6400 },
    diagnosticMetrics: { key_entity_transcription: 0.7840, response_speed: 5.9224 },
    successRates: {
      accuracy: { pass_threshold: 0.3933, mean: 0.6654, pass_at_k: 0.5600, pass_k: 0.2793 },
      experience: { pass_threshold: 0.3067, mean: 0.5905, pass_at_k: 0.5400, pass_k: 0.1807 },
    },
  },
  {
    id: 'gpt-realtime-mini',
    name: 'gpt-realtime-mini',
    shortName: 'gpt-realtime-mini',
    stt: '-', llm: 'gpt-realtime-mini', tts: '-',
    type: 's2s',
    evaA: 0.1800, evaX: 0.4267,
    accuracyMetrics: { task_completion: 0.2667, agent_tts_fidelity: 0.9833, faithfulness: 0.1733 },
    experienceMetrics: { turn_taking: 0.7801, conciseness: 0.7477, conversation_progression: 0.3533 },
    diagnosticMetrics: { key_entity_transcription: 0.8925, response_speed: 3.6711 },
    successRates: {
      accuracy: { pass_threshold: 0.1800, mean: 0.4744, pass_at_k: 0.3200, pass_k: 0.1044 },
      experience: { pass_threshold: 0.4267, mean: 0.6270, pass_at_k: 0.7200, pass_k: 0.2163 },
    },
  },
  {
    id: 'ultravox-realtime',
    name: 'ultravox-realtime',
    shortName: 'ultravox-realtime',
    stt: '-', llm: 'ultravox-realtime', tts: '-',
    type: '2-part',
    evaA: 0.2800, evaX: 0.4400,
    accuracyMetrics: { task_completion: 0.4867, agent_tts_fidelity: 0.9426, faithfulness: 0.3167 },
    experienceMetrics: { turn_taking: 0.5190, conciseness: 0.6948, conversation_progression: 0.5933 },
    diagnosticMetrics: { key_entity_transcription: 0.8484, response_speed: 4.8470 },
    successRates: {
      accuracy: { pass_threshold: 0.2800, mean: 0.5820, pass_at_k: 0.4600, pass_k: 0.1511 },
      experience: { pass_threshold: 0.4400, mean: 0.6024, pass_at_k: 0.7600, pass_k: 0.2356 },
    },
  },
  {
    id: 'gpt-4o-mini-transcribe-gpt-5-mini-gpt-4o-mini-tts',
    name: 'gpt-4o-mini-transcribe + gpt-5-mini + gpt-4o-mini-tts',
    shortName: 'gpt-5-mini (gpt-4o-mini-transcribe)',
    stt: 'gpt-4o-mini-transcribe', llm: 'gpt-5-mini', tts: 'gpt-4o-mini-tts',
    type: 'cascade',
    evaA: 0.2095, evaX: 0.1267,
    accuracyMetrics: { task_completion: 0.3600, agent_tts_fidelity: 0.9707, faithfulness: 0.3000 },
    experienceMetrics: { turn_taking: 0.2703, conciseness: 0.7162, conversation_progression: 0.4533 },
    diagnosticMetrics: { key_entity_transcription: 0.6801, response_speed: 5.9619 },
    successRates: {
      accuracy: { pass_threshold: 0.2095, mean: 0.5398, pass_at_k: 0.5000, pass_k: 0.0694 },
      experience: { pass_threshold: 0.1267, mean: 0.4799, pass_at_k: 0.3200, pass_k: 0.0274 },
    },
  },
  {
    id: 'gpt-4o-mini-transcribe-sonnet-4-6-gpt-4o-mini-tts',
    name: 'gpt-4o-mini-transcribe + sonnet-4.6 + gpt-4o-mini-tts',
    shortName: 'sonnet-4.6 (gpt-4o-mini-transcribe)',
    stt: 'gpt-4o-mini-transcribe', llm: 'sonnet-4.6', tts: 'gpt-4o-mini-tts',
    type: 'cascade',
    evaA: 0.3600, evaX: 0.0200,
    accuracyMetrics: { task_completion: 0.5400, agent_tts_fidelity: 0.9605, faithfulness: 0.6433 },
    experienceMetrics: { turn_taking: 0.1043, conciseness: 0.8298, conversation_progression: 0.7767 },
    diagnosticMetrics: { key_entity_transcription: 0.6167, response_speed: 8.2609 },
    successRates: {
      accuracy: { pass_threshold: 0.3600, mean: 0.7146, pass_at_k: 0.6200, pass_k: 0.1867 },
      experience: { pass_threshold: 0.0200, mean: 0.5703, pass_at_k: 0.0600, pass_k: 0.0022 },
    },
  },
  {
    id: 'gpt-4o-mini-transcribe-gpt-oss-20b-gpt-4o-mini-tts',
    name: 'gpt-4o-mini-transcribe + gpt-oss-20b + gpt-4o-mini-tts',
    shortName: 'gpt-oss-20b (gpt-4o-mini-transcribe)',
    stt: 'gpt-4o-mini-transcribe', llm: 'gpt-oss-20b', tts: 'gpt-4o-mini-tts',
    type: 'cascade',
    evaA: 0.1267, evaX: 0.3000,
    accuracyMetrics: { task_completion: 0.3000, agent_tts_fidelity: 0.9516, faithfulness: 0.1767 },
    experienceMetrics: { turn_taking: 0.5225, conciseness: 0.6871, conversation_progression: 0.3567 },
    diagnosticMetrics: { key_entity_transcription: 0.6170, response_speed: 4.8793 },
    successRates: {
      accuracy: { pass_threshold: 0.1267, mean: 0.4761, pass_at_k: 0.2400, pass_k: 0.0541 },
      experience: { pass_threshold: 0.3000, mean: 0.5221, pass_at_k: 0.6000, pass_k: 0.1356 },
    },
  },
  {
    id: 'gpt-4o-mini-transcribe-gpt-oss-120b-gpt-4o-mini-tts',
    name: 'gpt-4o-mini-transcribe + gpt-oss-120b + gpt-4o-mini-tts',
    shortName: 'gpt-oss-120b (gpt-4o-mini-transcribe)',
    stt: 'gpt-4o-mini-transcribe', llm: 'gpt-oss-120b', tts: 'gpt-4o-mini-tts',
    type: 'cascade',
    evaA: 0.1733, evaX: 0.5400,
    accuracyMetrics: { task_completion: 0.2867, agent_tts_fidelity: 0.9668, faithfulness: 0.3433 },
    experienceMetrics: { turn_taking: 0.6251, conciseness: 0.7655, conversation_progression: 0.5367 },
    diagnosticMetrics: { key_entity_transcription: 0.5824, response_speed: 4.2443 },
    successRates: {
      accuracy: { pass_threshold: 0.1733, mean: 0.5323, pass_at_k: 0.3400, pass_k: 0.0770 },
      experience: { pass_threshold: 0.5400, mean: 0.6424, pass_at_k: 0.9400, pass_k: 0.2467 },
    },
  },
  {
    id: 'parakeet-ctc-1-1b-gpt-oss-20b-magpie',
    name: 'parakeet-ctc-1.1b + gpt-oss-20b + magpie',
    shortName: 'gpt-oss-20b (parakeet-ctc-1.1b)',
    stt: 'parakeet-ctc-1.1b', llm: 'gpt-oss-20b', tts: 'magpie',
    type: 'cascade',
    evaA: 0.1600, evaX: 0.3400,
    accuracyMetrics: { task_completion: 0.4000, agent_tts_fidelity: 0.9350, faithfulness: 0.1400 },
    experienceMetrics: { turn_taking: 0.4180, conciseness: 0.7205, conversation_progression: 0.4933 },
    diagnosticMetrics: { key_entity_transcription: 0.8148, response_speed: 5.9816 },
    successRates: {
      accuracy: { pass_threshold: 0.1600, mean: 0.4917, pass_at_k: 0.3200, pass_k: 0.0711 },
      experience: { pass_threshold: 0.3400, mean: 0.5439, pass_at_k: 0.7000, pass_k: 0.1444 },
    },
  },
  {
    id: 'parakeet-ctc-1-1b-gpt-oss-120b-magpie',
    name: 'parakeet-ctc-1.1b + gpt-oss-120b + magpie',
    shortName: 'gpt-oss-120b (parakeet-ctc-1.1b)',
    stt: 'parakeet-ctc-1.1b', llm: 'gpt-oss-120b', tts: 'magpie',
    type: 'cascade',
    evaA: 0.1678, evaX: 0.4200,
    accuracyMetrics: { task_completion: 0.3667, agent_tts_fidelity: 0.9065, faithfulness: 0.3600 },
    experienceMetrics: { turn_taking: 0.4663, conciseness: 0.7522, conversation_progression: 0.6300 },
    diagnosticMetrics: { key_entity_transcription: 0.8415, response_speed: 5.2856 },
    successRates: {
      accuracy: { pass_threshold: 0.1678, mean: 0.5440, pass_at_k: 0.3061, pass_k: 0.0718 },
      experience: { pass_threshold: 0.4200, mean: 0.6162, pass_at_k: 0.7200, pass_k: 0.2022 },
    },
  },
  {
    id: 'parakeet-ctc-1-1b-qwen3-5-27b-kokoro',
    name: 'parakeet-ctc-1.1b + qwen3.5-27b + kokoro',
    shortName: 'qwen3.5-27b (parakeet-ctc-1.1b)',
    stt: 'parakeet-ctc-1.1b', llm: 'qwen3.5-27b', tts: 'kokoro',
    type: 'cascade',
    evaA: 0.4133, evaX: 0.0600,
    accuracyMetrics: { task_completion: 0.5400, agent_tts_fidelity: 0.9896, faithfulness: 0.4700 },
    experienceMetrics: { turn_taking: 0.2249, conciseness: 0.6823, conversation_progression: 0.6167 },
    diagnosticMetrics: { key_entity_transcription: 0.8093, response_speed: 7.4968 },
    successRates: {
      accuracy: { pass_threshold: 0.4133, mean: 0.6665, pass_at_k: 0.7000, pass_k: 0.2104 },
      experience: { pass_threshold: 0.0600, mean: 0.5080, pass_at_k: 0.1400, pass_k: 0.0156 },
    },
  },
  {
    id: 'parakeet-ctc-1-1b-gpt-oss-120b-kokoro',
    name: 'parakeet-ctc-1.1b + gpt-oss-120b + kokoro',
    shortName: 'gpt-oss-120b (parakeet-ctc-1.1b)',
    stt: 'parakeet-ctc-1.1b', llm: 'gpt-oss-120b', tts: 'kokoro',
    type: 'cascade',
    evaA: 0.2333, evaX: 0.2400,
    accuracyMetrics: { task_completion: 0.3600, agent_tts_fidelity: 0.9601, faithfulness: 0.3267 },
    experienceMetrics: { turn_taking: 0.3569, conciseness: 0.7582, conversation_progression: 0.6167 },
    diagnosticMetrics: { key_entity_transcription: 0.7951, response_speed: 6.0521 },
    successRates: {
      accuracy: { pass_threshold: 0.2333, mean: 0.5489, pass_at_k: 0.4600, pass_k: 0.1059 },
      experience: { pass_threshold: 0.2400, mean: 0.5772, pass_at_k: 0.5000, pass_k: 0.0844 },
    },
  },
  {
    id: 'parakeet-ctc-1-1b-gpt-oss-120b-chatterbox-turbo',
    name: 'parakeet-ctc-1.1b + gpt-oss-120b + chatterbox-turbo',
    shortName: 'gpt-oss-120b (parakeet-ctc-1.1b)',
    stt: 'parakeet-ctc-1.1b', llm: 'gpt-oss-120b', tts: 'chatterbox-turbo',
    type: 'cascade',
    evaA: 0.1533, evaX: 0.0267,
    accuracyMetrics: { task_completion: 0.3600, agent_tts_fidelity: 0.8883, faithfulness: 0.3200 },
    experienceMetrics: { turn_taking: 0.0645, conciseness: 0.7841, conversation_progression: 0.4900 },
    diagnosticMetrics: { key_entity_transcription: 0.8053, response_speed: 9.8321 },
    successRates: {
      accuracy: { pass_threshold: 0.1533, mean: 0.5228, pass_at_k: 0.3200, pass_k: 0.0570 },
      experience: { pass_threshold: 0.0267, mean: 0.4462, pass_at_k: 0.0600, pass_k: 0.0074 },
    },
  },
  {
    id: 'parakeet-ctc-1-1b-qwen3-5-27b-chatterbox-turbo',
    name: 'parakeet-ctc-1.1b + qwen3.5-27b + chatterbox-turbo',
    shortName: 'qwen3.5-27b (parakeet-ctc-1.1b)',
    stt: 'parakeet-ctc-1.1b', llm: 'qwen3.5-27b', tts: 'chatterbox-turbo',
    type: 'cascade',
    evaA: 0.2533, evaX: 0.0000,
    accuracyMetrics: { task_completion: 0.5333, agent_tts_fidelity: 0.8513, faithfulness: 0.4200 },
    experienceMetrics: { turn_taking: 0.0225, conciseness: 0.6914, conversation_progression: 0.5600 },
    diagnosticMetrics: { key_entity_transcription: 0.8268, response_speed: 12.0952 },
    successRates: {
      accuracy: { pass_threshold: 0.2533, mean: 0.6015, pass_at_k: 0.5600, pass_k: 0.0815 },
      experience: { pass_threshold: 0.0000, mean: 0.4246, pass_at_k: 0.0000, pass_k: 0.0000 },
    },
  },
  {
    id: 'voxtral-mini-3b-gpt-oss-20b-magpie',
    name: 'voxtral-mini-3b + gpt-oss-20b + magpie',
    shortName: 'gpt-oss-20b (voxtral-mini-3b)',
    stt: 'voxtral-mini-3b', llm: 'gpt-oss-20b', tts: 'magpie',
    type: 'cascade',
    evaA: 0.1133, evaX: 0.3867,
    accuracyMetrics: { task_completion: 0.3733, agent_tts_fidelity: 0.9349, faithfulness: 0.1367 },
    experienceMetrics: { turn_taking: 0.5951, conciseness: 0.6917, conversation_progression: 0.3667 },
    diagnosticMetrics: { key_entity_transcription: 0.6618, response_speed: 4.4834 },
    successRates: {
      accuracy: { pass_threshold: 0.1133, mean: 0.4816, pass_at_k: 0.2400, pass_k: 0.0526 },
      experience: { pass_threshold: 0.3867, mean: 0.5512, pass_at_k: 0.7800, pass_k: 0.1541 },
    },
  },
  {
    id: 'voxtral-mini-3b-gpt-oss-120b-magpie',
    name: 'voxtral-mini-3b + gpt-oss-120b + magpie',
    shortName: 'gpt-oss-120b (voxtral-mini-3b)',
    stt: 'voxtral-mini-3b', llm: 'gpt-oss-120b', tts: 'magpie',
    type: 'cascade',
    evaA: 0.1477, evaX: 0.5667,
    accuracyMetrics: { task_completion: 0.3467, agent_tts_fidelity: 0.9467, faithfulness: 0.2967 },
    experienceMetrics: { turn_taking: 0.6659, conciseness: 0.7494, conversation_progression: 0.4767 },
    diagnosticMetrics: { key_entity_transcription: 0.6906, response_speed: 3.3998 },
    successRates: {
      accuracy: { pass_threshold: 0.1477, mean: 0.5279, pass_at_k: 0.3265, pass_k: 0.0620 },
      experience: { pass_threshold: 0.5667, mean: 0.6307, pass_at_k: 0.9200, pass_k: 0.3341 },
    },
  },
  {
    id: 'voxtral-mini-3b-gpt-oss-120b-chatterbox-turbo',
    name: 'voxtral-mini-3b + gpt-oss-120b + chatterbox-turbo',
    shortName: 'gpt-oss-120b (voxtral-mini-3b)',
    stt: 'voxtral-mini-3b', llm: 'gpt-oss-120b', tts: 'chatterbox-turbo',
    type: 'cascade',
    evaA: 0.1600, evaX: 0.0933,
    accuracyMetrics: { task_completion: 0.3600, agent_tts_fidelity: 0.9049, faithfulness: 0.3467 },
    experienceMetrics: { turn_taking: 0.2040, conciseness: 0.7701, conversation_progression: 0.5233 },
    diagnosticMetrics: { key_entity_transcription: 0.6376, response_speed: 7.2744 },
    successRates: {
      accuracy: { pass_threshold: 0.1600, mean: 0.5372, pass_at_k: 0.2800, pass_k: 0.0800 },
      experience: { pass_threshold: 0.0933, mean: 0.4991, pass_at_k: 0.2800, pass_k: 0.0104 },
    },
  },
  {
    id: 'voxtral-mini-3b-qwen3-5-27b-chatterbox-turbo',
    name: 'voxtral-mini-3b + qwen3.5-27b + chatterbox-turbo',
    shortName: 'qwen3.5-27b (voxtral-mini-3b)',
    stt: 'voxtral-mini-3b', llm: 'qwen3.5-27b', tts: 'chatterbox-turbo',
    type: 'cascade',
    evaA: 0.2067, evaX: 0.0000,
    accuracyMetrics: { task_completion: 0.5400, agent_tts_fidelity: 0.7960, faithfulness: 0.3967 },
    experienceMetrics: { turn_taking: 0.0296, conciseness: 0.7165, conversation_progression: 0.5167 },
    diagnosticMetrics: { key_entity_transcription: 0.7408, response_speed: 14.4124 },
    successRates: {
      accuracy: { pass_threshold: 0.2067, mean: 0.5775, pass_at_k: 0.5200, pass_k: 0.0452 },
      experience: { pass_threshold: 0.0000, mean: 0.4209, pass_at_k: 0.0000, pass_k: 0.0000 },
    },
  },
  {
    id: 'voxtral-mini-3b-qwen3-5-27b-kokoro',
    name: 'voxtral-mini-3b + qwen3.5-27b + kokoro',
    shortName: 'qwen3.5-27b (voxtral-mini-3b)',
    stt: 'voxtral-mini-3b', llm: 'qwen3.5-27b', tts: 'kokoro',
    type: 'cascade',
    evaA: 0.4933, evaX: 0.2467,
    accuracyMetrics: { task_completion: 0.5933, agent_tts_fidelity: 0.9949, faithfulness: 0.5067 },
    experienceMetrics: { turn_taking: 0.3740, conciseness: 0.6838, conversation_progression: 0.5433 },
    diagnosticMetrics: { key_entity_transcription: 0.7518, response_speed: 5.8276 },
    successRates: {
      accuracy: { pass_threshold: 0.4933, mean: 0.6983, pass_at_k: 0.7400, pass_k: 0.3348 },
      experience: { pass_threshold: 0.2467, mean: 0.5337, pass_at_k: 0.5000, pass_k: 0.0985 },
    },
  },
  {
    id: 'whisper-large-v3-gpt-oss-20b-chatterbox-turbo',
    name: 'whisper-large-v3 + gpt-oss-20b + chatterbox-turbo',
    shortName: 'gpt-oss-20b (whisper-large-v3)',
    stt: 'whisper-large-v3', llm: 'gpt-oss-20b', tts: 'chatterbox-turbo',
    type: 'cascade',
    evaA: 0.0733, evaX: 0.0400,
    accuracyMetrics: { task_completion: 0.3800, agent_tts_fidelity: 0.8849, faithfulness: 0.1533 },
    experienceMetrics: { turn_taking: 0.1816, conciseness: 0.7343, conversation_progression: 0.4400 },
    diagnosticMetrics: { key_entity_transcription: 0.6696, response_speed: 7.5545 },
    successRates: {
      accuracy: { pass_threshold: 0.0733, mean: 0.4728, pass_at_k: 0.1800, pass_k: 0.0170 },
      experience: { pass_threshold: 0.0400, mean: 0.4519, pass_at_k: 0.1200, pass_k: 0.0044 },
    },
  },
  {
    id: 'whisper-large-v3-gpt-oss-120b-kokoro',
    name: 'whisper-large-v3 + gpt-oss-120b + kokoro',
    shortName: 'gpt-oss-120b (whisper-large-v3)',
    stt: 'whisper-large-v3', llm: 'gpt-oss-120b', tts: 'kokoro',
    type: 'cascade',
    evaA: 0.1667, evaX: 0.5200,
    accuracyMetrics: { task_completion: 0.2800, agent_tts_fidelity: 0.9645, faithfulness: 0.2967 },
    experienceMetrics: { turn_taking: 0.6148, conciseness: 0.7536, conversation_progression: 0.5433 },
    diagnosticMetrics: { key_entity_transcription: 0.6573, response_speed: 4.1026 },
    successRates: {
      accuracy: { pass_threshold: 0.1667, mean: 0.5137, pass_at_k: 0.3200, pass_k: 0.0763 },
      experience: { pass_threshold: 0.5200, mean: 0.6372, pass_at_k: 0.8400, pass_k: 0.3244 },
    },
  }
];
