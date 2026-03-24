import demoData from './demo_data.json';

export type DemoMetricCategory = 'eva-a' | 'eva-x' | 'diagnostic';
export type DemoMetricType = 'deterministic' | 'llm_judge' | 'lalm_judge';

export interface DemoMetricScore {
  name: string;
  displayName: string;
  category: DemoMetricCategory;
  type: DemoMetricType;
  score: number;
  normalizedScore: number;
  details: Record<string, unknown>;
}

export const demoMetrics: DemoMetricScore[] = demoData.metrics.map((m) => ({
  name: m.name,
  displayName: m.displayName,
  category: m.category as DemoMetricCategory,
  type: m.type as DemoMetricType,
  score: m.score,
  normalizedScore: m.normalizedScore,
  details: m.details as Record<string, unknown>,
}));

export const evaAMetrics = demoMetrics.filter((m) => m.category === 'eva-a');
export const evaXMetrics = demoMetrics.filter((m) => m.category === 'eva-x');
export const diagnosticMetrics = demoMetrics.filter((m) => m.category === 'diagnostic');
