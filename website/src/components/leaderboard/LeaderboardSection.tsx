import { Lightbulb } from 'lucide-react';
import { Section } from '../layout/Section';
import { ScatterPlot } from './ScatterPlot';
import { MetricHeatmap } from './MetricHeatmap';
import type { AggregateColumn } from './MetricHeatmap';
import { TurnTakingAnalysis } from './TurnTakingAnalysis';
import {
  ossSystems,
  accuracyMetricKeys, experienceMetricKeys,
  accuracyMetricLabels, experienceMetricLabels,
} from '../../data/leaderboardData';
import { useThemeColors } from '../../styles/theme';

const ossKeyInsights = [
  {
    title: 'Transcription failures cascade into low task completion',
    description: 'Transcription failures around last names and confirmation codes cascade into low task completion as the agent is unable to pull up the user\'s booking and proceed with the request.',
  },
  {
    title: 'Turn taking remains a key challenge',
    description: 'Effective turn taking remains a key challenge for cascade systems \u2014 most turns are late (>4 seconds). See the turn taking analysis below.',
  },
  {
    title: 'Speech synthesis struggles with alphanumeric codes',
    description: 'Speech synthesis systems generally produce the intended speech but struggle the most with alphanumeric codes, often dropping or switching characters and letters.',
  },
  {
    title: 'LLMs produce verbose, non-voice-appropriate content',
    description: 'LLMs struggle to produce concise, voice-appropriate content, particularly when trying to list flight options for the user.',
  },
  {
    title: 'Transcription failures reduce conversation efficiency',
    description: 'Transcription failures also lead to inefficient conversation progression, as the agent cannot move the conversation forward when it\'s stuck trying to retrieve the user\'s reservation.',
  },
{
  title: 'Audio-native systems show promise',
  description: 'Both audio-native systems sit on the Pareto frontier, while the single speech-to-speech system does not — we aim to benchmark more audio-native and s2s systems to see if this holds across the architectural classes.',
},
];

const ossInsights = [
  {
    title: 'Accuracy–experience trade-off',
    description: 'The Pareto frontier reveals a clear accuracy-experience tradeoff across systems, systems that push harder on accuracy are doing so at the cost of conversational experience, and vice versa.',
  },
  {
    title: 'Low Pass Rates',
    description: 'Performance remains far from saturated — no system clears 0.5 pass@1 on accuracy, and only a few systems exceed 0.50 EVA-X pass@1, suggesting ample opportunities for improvement.',
  },
  {
    title: 'Sparse Frontier',
    description: 'Only a few systems sit on the Pareto frontier, meaning most systems are strictly dominated. This concentrates the real decision space: only a small subset of system choices actually matter for navigating the accuracy–experience tradeoff.',
  },
];

const accuracyAggregates: AggregateColumn[] = [
  { key: 'eva_a_pass', label: 'EVA-A pass@1', getValue: (s) => s.successRates.accuracy.pass_threshold },
  { key: 'eva_a_mean', label: 'EVA-A Mean', getValue: (s) => s.successRates.accuracy.mean },
];

const experienceAggregates: AggregateColumn[] = [
  { key: 'eva_x_pass', label: 'EVA-X pass@1', getValue: (s) => s.successRates.experience.pass_threshold },
  { key: 'eva_x_mean', label: 'EVA-X Mean', getValue: (s) => s.successRates.experience.mean },
];

export function LeaderboardSection() {
  const colors = useThemeColors();
  const systems = ossSystems;

  return (
    <Section
      id="leaderboard"
      title="Early Results"
      subtitle="Early results on the airline domain (50 scenarios, 3 trials each)."
    >
      <div className="space-y-8">
        <ScatterPlot systems={systems} />

        <div className="rounded-xl border border-purple/20 bg-purple/5 p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-9 h-9 rounded-lg bg-purple/10 flex items-center justify-center">
              <Lightbulb className="w-5 h-5 text-purple-light" />
            </div>
            <h3 className="text-lg font-bold text-text-primary">Pareto Analysis</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {ossInsights.map((insight, i) => (
              <div key={i} className="rounded-lg bg-bg-secondary border border-border-default p-4">
                <div className="text-sm font-semibold text-text-primary mb-2">{insight.title}</div>
                <p className="text-sm text-text-secondary leading-relaxed">{insight.description}</p>
              </div>
            ))}
          </div>
        </div>

        <MetricHeatmap
          title="Accuracy Metrics (EVA-A)"
          description="Per-metric scores for accuracy. All values normalized to 0-1 (higher is better)."
          metricKeys={accuracyMetricKeys}
          metricLabels={accuracyMetricLabels}
          dataKey="accuracyMetrics"
          baseColor={colors.accent.purple}
          aggregateColumns={accuracyAggregates}
          aggregateColor="#F59E0B"
          systems={systems}
        />

        <MetricHeatmap
          title="Experience Metrics (EVA-X)"
          description="Per-metric scores for conversational experience. All values normalized to 0-1 (higher is better)."
          metricKeys={experienceMetricKeys}
          metricLabels={experienceMetricLabels}
          dataKey="experienceMetrics"
          baseColor={colors.accent.blue}
          aggregateColumns={experienceAggregates}
          aggregateColor="#F59E0B"
          systems={systems}
        />

        <div className="rounded-xl border border-purple/20 bg-purple/5 p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-9 h-9 rounded-lg bg-purple/10 flex items-center justify-center">
              <Lightbulb className="w-5 h-5 text-purple-light" />
            </div>
            <h3 className="text-lg font-bold text-text-primary">Key Insights</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {ossKeyInsights.map((insight, i) => (
              <div key={i} className="rounded-lg bg-bg-secondary border border-border-default p-4">
                <div className="text-sm font-semibold text-text-primary mb-2">{insight.title}</div>
                <p className="text-sm text-text-secondary leading-relaxed">{insight.description}</p>
              </div>
            ))}
          </div>
        </div>

        <TurnTakingAnalysis systems={systems} />

      </div>
    </Section>
  );
}
