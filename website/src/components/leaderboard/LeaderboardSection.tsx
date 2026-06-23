import { useState } from 'react';
import { Lightbulb } from 'lucide-react';
import { Section } from '../layout/Section';
import { ScatterPlot } from './ScatterPlot';
import { MetricHeatmap } from './MetricHeatmap';
import { Perturbations } from './Perturbations';
import { SttTranscription } from './SttTranscription';
import type { AggregateColumn } from './MetricHeatmap';
import {
  systems,
  accuracyMetricKeys, experienceMetricKeys,
  accuracyMetricLabels, experienceMetricLabels,
  domainLabels,
  type DomainOrPooled,
} from '../../data/leaderboardData';
import { useThemeColors } from '../../styles/theme';

// Three central findings from the EVA-Bench paper abstract / §5 Conclusion.
const paretoInsights = [
  {
    title: 'No system clears 0.6 on both axes pass@1',
    description:
      'Across 17 systems spanning all three architectures, no system simultaneously exceeds 0.6 on both EVA-A pass@1 and EVA-X pass@1 — joint accuracy–experience quality remains far from saturated.',
  },
  {
    title: 'Peak and reliable performance diverge',
    description:
      'Peak (pass@k) and reliable (pass^k) performance diverge substantially: the median pass@k–pass^k gap is 0.44 on EVA-A and 0.24 on EVA-X, indicating single-trial scores systematically overstate deployment-grade reliability.',
  },
  {
    title: 'Architecture and SDK implementation both shape results',
    description:
      'The Pareto frontier spans both S2S and cascade architectures. Cascade results vary significantly depending on the SDK implementation used, with some cascade configurations achieving turn-taking scores competitive with S2S models. This suggests that integration choices can matter as much as the underlying models.',
  },
];

// Supporting bullets drawn from §4.3 Robustness and §4.4 Failure Mode Analysis.
const keyInsights = [
  {
    title: 'Cascade accuracy–experience trade-off',
    description:
      'Among cascade systems we observe a consistent accuracy–experience trade-off: higher-accuracy cascades tend to have higher tool-call latencies, while faster cascades trade accuracy for lower latency.',
  },
  {
    title: 'Asymmetric degradation under perturbation',
    description:
      'Cascade systems are most vulnerable on accuracy under accented speech (task completion drops 10 points on average, up to 17), while S2S systems suffer most on experience under background noise (EVA-X mean ∆ = −0.16). Turn-taking is the most perturbation-sensitive metric overall (81% of pairs significant).',
  },
  {
    title: 'Named-entity transcription bottlenecks cascades',
    description:
      'Across nine cascade systems, mean key-entity transcription accuracy is strongly correlated with mean task completion. Cascades below 70% key-entity transcription accuracy show substantially lower task completion than those above it.',
  },
  {
    title: 'Faithfulness is decoupled from task completion',
    description:
      '72.2% of conversations with task completion = 1 still exhibit at least one faithfulness deviation, and 50.5% of faithfulness deviations co-occur with task completion = 0. Faithfulness must therefore be measured as an independent dimension.',
  },
  {
    title: 'Speech fidelity fails on alphanumeric content',
    description:
      'Entity errors — letter substitutions, digit omissions, spurious insertions, and phonetic confusions — are the dominant speech-fidelity failure mode. Even 1% per-turn fail rates compound over multi-turn interactions when the caller cannot detect the error from context.',
  },
  {
    title: 'Low-latency cascades close the experience gap',
    description:
      'Cascade systems built with low-latency models can outperform S2S models on experience. The fastest cascade system achieves the highest EVA-X pass@1 (0.82) of any system, with turn-taking (0.88) surpassing all S2S models — suggesting that latency, not architecture, is the primary driver of experience quality.',
  },
];

const DOMAIN_TABS: DomainOrPooled[] = ['pooled', 'airline', 'itsm', 'medical_hr'];

const accuracyAggregates: AggregateColumn[] = [
  { key: 'eva_a_pass', label: 'EVA-A pass@1', metric: 'EVA-A_pass' },
  { key: 'eva_a_mean', label: 'EVA-A Mean',  metric: 'EVA-A_mean' },
];
const experienceAggregates: AggregateColumn[] = [
  { key: 'eva_x_pass', label: 'EVA-X pass@1', metric: 'EVA-X_pass' },
  { key: 'eva_x_mean', label: 'EVA-X Mean',  metric: 'EVA-X_mean' },
];

export function LeaderboardSection() {
  const colors = useThemeColors();
  const [domain, setDomain] = useState<DomainOrPooled>('pooled');

  return (
    <Section
      id="leaderboard"
      title="Results"
      subtitle="Results across three domains (CSM, ITSM, HR). Pooled by default; toggle to inspect a single domain."
    >
      <div className="space-y-8">
        {/* Domain Toggle */}
        <div className="inline-flex rounded-lg border border-border-default bg-bg-secondary p-1">
          {DOMAIN_TABS.map(d => (
            <button
              key={d}
              onClick={() => setDomain(d)}
              className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                domain === d ? 'bg-bg-primary text-text-primary' : 'text-text-muted hover:text-text-secondary'
              }`}
            >
              {domainLabels[d]}
            </button>
          ))}
        </div>

        <ScatterPlot systems={systems} domain={domain} />

        <div className="rounded-xl border border-purple/20 bg-purple/5 p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-9 h-9 rounded-lg bg-purple/10 flex items-center justify-center">
              <Lightbulb className="w-5 h-5 text-purple-light" />
            </div>
            <h3 className="text-lg font-bold text-text-primary">Pareto Analysis</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {paretoInsights.map((insight, i) => (
              <div key={i} className="rounded-lg bg-bg-secondary border border-border-default p-4">
                <div className="text-sm font-semibold text-text-primary mb-2">{insight.title}</div>
                <p className="text-sm text-text-secondary leading-relaxed">{insight.description}</p>
              </div>
            ))}
          </div>
        </div>

        <MetricHeatmap
          title="Accuracy Metrics (EVA-A)"
          description="Per-metric scores for accuracy. All values normalized to 0-1 (higher is better). 95% bootstrap confidence intervals shown for each value."
          metricKeys={accuracyMetricKeys}
          metricLabels={accuracyMetricLabels}
          baseColor={colors.accent.purple}
          aggregateColumns={accuracyAggregates}
          aggregateColor="#F59E0B"
          systems={systems}
        />

        <MetricHeatmap
          title="Experience Metrics (EVA-X)"
          description="Per-metric scores for conversational experience. All values normalized to 0-1 (higher is better). 95% bootstrap confidence intervals shown for each value."
          metricKeys={experienceMetricKeys}
          metricLabels={experienceMetricLabels}
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
            {keyInsights.map((insight, i) => (
              <div key={i} className="rounded-lg bg-bg-secondary border border-border-default p-4">
                <div className="text-sm font-semibold text-text-primary mb-2">{insight.title}</div>
                <p className="text-sm text-text-secondary leading-relaxed">{insight.description}</p>
              </div>
            ))}
          </div>
          <p className="text-xs text-text-muted mt-4">
            *see <a href="https://arxiv.org/pdf/2605.13841" target="_blank" rel="noopener noreferrer" className="underline hover:text-text-secondary">paper</a> for full details
          </p>
        </div>

        <Perturbations systems={systems} />
        <SttTranscription systems={systems} />
      </div>
    </Section>
  );
}
