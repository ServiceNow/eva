import { useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import { Section } from '../layout/Section';
import { MetricNode } from './MetricNode';
import { evaAMetrics, evaXMetrics, debugMetrics, validationMetrics } from '../../data/metricsData';

export function MetricsExplorer() {
  const [showDebug, setShowDebug] = useState(false);

  return (
    <Section
      id="metrics"
      title="Evaluation Methodology"
      subtitle="EVA produces two fundamental scores composed of multiple sub-metrics. Click any metric to explore what it measures, its inputs, and the judge prompt."
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* EVA-A Column */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <div className="rounded-xl border-2 border-purple/30 bg-purple/5 p-5 mb-5 text-center">
            <div className="text-sm font-bold text-purple-light tracking-widest uppercase mb-1">EVA-A</div>
            <div className="text-2xl font-bold text-text-primary">Accuracy</div>
            <p className="text-sm text-text-secondary mt-1.5">Did the agent complete the task correctly?</p>
          </div>

          <div className="relative">
            <div className="absolute left-1/2 top-0 w-px h-4 bg-purple/30" />
          </div>

          <div className="space-y-3">
            {evaAMetrics.map(metric => (
              <MetricNode key={metric.id} metric={metric} />
            ))}
          </div>
        </motion.div>

        {/* EVA-X Column */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <div className="rounded-xl border-2 border-blue/30 bg-blue/5 p-5 mb-5 text-center">
            <div className="text-sm font-bold text-blue-light tracking-widest uppercase mb-1">EVA-X</div>
            <div className="text-2xl font-bold text-text-primary">Experience</div>
            <p className="text-sm text-text-secondary mt-1.5">Was the conversational experience high quality?</p>
          </div>

          <div className="relative">
            <div className="absolute left-1/2 top-0 w-px h-4 bg-blue/30" />
          </div>

          <div className="space-y-3">
            {evaXMetrics.map(metric => (
              <MetricNode key={metric.id} metric={metric} />
            ))}
          </div>
        </motion.div>
      </div>

      {/* Pass Metrics & Thresholds */}
      <div className="mt-10">
        <div className="mb-6">
          <h3 className="text-xl font-bold text-text-primary mb-2">Aggregate Metrics</h3>
          <p className="text-sm text-text-secondary leading-relaxed">
            EVA aggregates per-trial metric scores into four aggregate metrics, each capturing a different aspect of success and reliability.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="rounded-xl border border-border-default bg-bg-secondary p-5">
            <div className="text-base font-semibold text-text-primary mb-2">pass@1</div>
            <p className="text-sm text-text-secondary leading-relaxed">
              For each scenario, the proportion of trials where <em>all</em> metric thresholds are met (<em>c</em>/<em>n</em>), where <em>c</em> is the number of passing trials and <em>n</em> is the total number of trials (n=3), then averaged across all scenarios.
            </p>
          </div>
          <div className="rounded-xl border border-border-default bg-bg-secondary p-5">
            <div className="text-base font-semibold text-text-primary mb-2">pass@k (k=3)</div>
            <p className="text-sm text-text-secondary leading-relaxed">
              For each scenario, 1 if at least one of the k (3) trials meets pass criteria for all metrics, otherwise 0, then averaged across all scenarios. Measures whether the system <em>can</em> succeed.
            </p>
          </div>
          <div className="rounded-xl border border-border-default bg-bg-secondary p-5">
            <div className="text-base font-semibold text-text-primary mb-2">pass^k (k=3)</div>
            <p className="text-sm text-text-secondary leading-relaxed">
             For each scenario, we estimate the theoretical probability of passing k = 3 consecutive independent trials as  (<em>c</em>/<em>n</em>)<sup>k</sup> where c is the number of passing trials out of n = 3 total. We then average this value across all scenarios to measure consistency and reliability.
            </p>
          </div>
          <div className="rounded-xl border border-border-default bg-bg-secondary p-5">
            <div className="text-base font-semibold text-text-primary mb-2">Mean</div>
            <p className="text-sm text-text-secondary leading-relaxed">
              For each sample, we average sub-metric scores per dimension, then average across all 150 samples. Raw scores avoid binarizing near-boundary differences into a full pass/fail gap, capturing more nuanced system comparisons.
            </p>
          </div>
        </div>

      </div>

      {/* Debug & Validation Metrics */}
      <div className="mt-10">
        <button
          onClick={() => setShowDebug(!showDebug)}
          className="w-full flex items-center justify-between rounded-xl border border-border-default bg-bg-secondary px-6 py-5 hover:bg-bg-hover/30 transition-colors"
        >
          <div>
            <div className="text-base font-semibold text-text-secondary text-left">Diagnostic & Validation Metrics</div>
            <div className="text-sm text-text-muted mt-1 text-left">
              {debugMetrics.length + validationMetrics.length} additional metrics for diagnostics and quality control
            </div>
          </div>
          <ChevronDown
            className={`w-5 h-5 text-text-muted transition-transform ${showDebug ? 'rotate-180' : ''}`}
          />
        </button>

        {showDebug && (
          <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-8 opacity-80">
            {/* Debug Metrics */}
            <div>
              <div className="px-1 mb-4">
                <div className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-1">Diagnostic Metrics</div>
                <p className="text-sm text-text-muted leading-relaxed">
                  Diagnostic metrics for understanding <em>why</em> the core scores look the way they do. These help identify which pipeline component (STT, LLM, TTS) is contributing to failures but are not part of the EVA-A or EVA-X scores.
                </p>
              </div>
              <div className="space-y-3">
                {debugMetrics.map(metric => (
                  <MetricNode key={metric.id} metric={metric} />
                ))}
              </div>
            </div>

            {/* Validation Metrics */}
            <div>
              <div className="px-1 mb-4">
                <div className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-1">Validation Metrics</div>
                <p className="text-sm text-text-muted leading-relaxed">
                  Validators run before evaluation. Any conversation that fails validation is regenerated so that core metrics are only computed on conversations with a well-behaved user simulator and properly completed interactions.
                </p>
              </div>
              <div className="space-y-3">
                {validationMetrics.map(metric => (
                  <MetricNode key={metric.id} metric={metric} />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-6 mt-8">
        <div className="flex items-center gap-2 text-sm text-text-secondary">
          <div className="w-3.5 h-3.5 rounded bg-cyan/20 border border-cyan/40" />
          Deterministic (Code)
        </div>
        <div className="flex items-center gap-2 text-sm text-text-secondary">
          <div className="w-3.5 h-3.5 rounded bg-purple/20 border border-purple/40" />
          LLM Judge (Text)
        </div>
        <div className="flex items-center gap-2 text-sm text-text-secondary">
          <div className="w-3.5 h-3.5 rounded bg-amber/20 border border-amber/40" />
          Audio LLM Judge (LALM)
        </div>
      </div>
    </Section>
  );
}
