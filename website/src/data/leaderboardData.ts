import statsRaw from './leaderboardStats.json';

export type Domain = 'airline' | 'itsm' | 'medical_hr';
export type DomainOrPooled = Domain | 'pooled';
export const DOMAINS: Domain[] = ['airline', 'itsm', 'medical_hr'];

export const domainLabels: Record<DomainOrPooled, string> = {
  pooled: 'Pooled',
  airline: 'CSM',
  itsm: 'ITSM',
  medical_hr: 'HR',
};

export interface CIPoint {
  point: number;
  ci_lower: number;
  ci_upper: number;
  n?: number;
}

export interface CIPointWithSig extends CIPoint {
  corrected_p?: number;
  raw_p?: number;
  reject?: boolean;
}

export interface MetricBlock {
  pooled: CIPoint | null;
  per_domain: Partial<Record<Domain, CIPoint>>;
}

export interface PerturbationBlock {
  pooled: CIPointWithSig | null;
  per_domain: Partial<Record<Domain, CIPointWithSig>>;
}

export interface SystemStats {
  id: string;
  name: string;
  type: 'cascade' | 's2s' | '2-part';
  stt: string;
  llm: string;
  tts: string;
  clean: Record<string, MetricBlock>;
  perturbation_delta: Record<string, Record<string, PerturbationBlock>>;
}

const stats = statsRaw as { systems: SystemStats[] };
export const systems: SystemStats[] = stats.systems;

/** Return the CIPoint for `metric` on `system` at `domain` (or pooled). */
export function getValue(system: SystemStats, metric: string, domain: DomainOrPooled): CIPoint | null {
  const block = system.clean[metric];
  if (!block) return null;
  if (domain === 'pooled') return block.pooled;
  return block.per_domain[domain] ?? null;
}

export function getPertValue(
  system: SystemStats,
  metric: string,
  perturbation: string,
  domain: DomainOrPooled,
): CIPointWithSig | null {
  const block = system.perturbation_delta[metric]?.[perturbation];
  if (!block) return null;
  if (domain === 'pooled') return block.pooled;
  return block.per_domain[domain] ?? null;
}

// Metric keys (clean) used by the heatmap UI
export const accuracyMetricKeys = ['task_completion', 'agent_speech_fidelity', 'faithfulness'] as const;
export const experienceMetricKeys = ['turn_taking', 'conciseness', 'conversation_progression'] as const;
export const diagnosticMetricKeys = ['response_speed'] as const;

export const accuracyMetricLabels: Record<string, string> = {
  task_completion: 'Task Completion',
  agent_speech_fidelity: 'Speech Fidelity',
  faithfulness: 'Faithfulness',
};
export const experienceMetricLabels: Record<string, string> = {
  turn_taking: 'Turn Taking',
  conciseness: 'Conciseness',
  conversation_progression: 'Conversation Progression',
};
export const diagnosticMetricLabels: Record<string, string> = {
  response_speed: 'Response Speed (s)',
};

export const invertedMetrics = new Set(['response_speed']);

// List of perturbations available in the data (read once from the first system that has any).
export const perturbations: string[] = (() => {
  const seen = new Set<string>();
  for (const s of systems) {
    for (const m of Object.keys(s.perturbation_delta)) {
      for (const p of Object.keys(s.perturbation_delta[m])) {
        seen.add(p);
      }
    }
  }
  return Array.from(seen).sort();
})();

export const perturbationLabels: Record<string, string> = {
  accent: 'Accent',
  background_noise: 'Background Noise',
  both: 'Accent + Noise',
};
