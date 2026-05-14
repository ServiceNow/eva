import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { SystemStats } from '../../data/leaderboardData';
import { perturbations, perturbationLabels } from '../../data/leaderboardData';
import { PerturbationBarChart } from './PerturbationBarChart';
import { useThemeColors } from '../../styles/theme';

const PERT_COLOR_KEYS: Record<string, 'amber' | 'cyan' | 'purple'> = {
  accent: 'amber',
  background_noise: 'cyan',
  both: 'purple',
};

interface PerturbationsProps {
  systems: SystemStats[];
}

interface MetricSpec {
  key: string;
  label: string;
}

const METRICS: MetricSpec[] = [
  { key: 'task_completion', label: 'Task Completion' },
  { key: 'agent_speech_fidelity', label: 'Speech Fidelity' },
  { key: 'faithfulness', label: 'Faithfulness' },
  { key: 'turn_taking', label: 'Turn Taking' },
  { key: 'conciseness', label: 'Conciseness' },
  { key: 'conversation_progression', label: 'Conversation Progression' },
  { key: 'EVA-A_pass', label: 'EVA-A pass@1' },
  { key: 'EVA-X_pass', label: 'EVA-X pass@1' },
  { key: 'conversation_correctly_finished', label: 'Conversation Correctly Finished' },
];

export function Perturbations({ systems }: PerturbationsProps) {
  const colors = useThemeColors();
  const [sectionOpen, setSectionOpen] = useState(true);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggleMetric = (key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <div className="rounded-xl border border-border-default bg-bg-secondary overflow-hidden">
      <button
        onClick={() => setSectionOpen((o) => !o)}
        className="w-full flex items-center gap-3 p-5 hover:bg-bg-hover transition-colors text-left"
      >
        {sectionOpen ? (
          <ChevronDown className="w-5 h-5 text-text-muted flex-shrink-0" />
        ) : (
          <ChevronRight className="w-5 h-5 text-text-muted flex-shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-bold text-text-primary">Perturbations</h3>
          <p className="text-sm text-text-muted mt-0.5">
            For each domain we select <span className="font-semibold text-text-secondary">30 scenarios</span> and run
            each system with <span className="font-semibold text-text-secondary">k = 3 trials per scenario</span> under
            accent, background-noise, and combined perturbations. Each bar shows the mean Δ vs. the same scenarios'
            clean runs; error bars show 95% bootstrap confidence intervals. Asterisks (<span className="text-amber-400">*</span>)
            indicate that the perturbation effect is statistically significant after{' '}
            <span className="font-semibold text-text-secondary">Holm–Bonferroni</span> correction across the family of
            metric × perturbation × system tests.
          </p>
        </div>
      </button>

      {sectionOpen && (
        <div className="border-t border-border-default p-4 space-y-3">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-2 py-3 rounded-lg bg-bg-primary border border-border-default">
            {perturbations.map((p) => {
              const k = PERT_COLOR_KEYS[p];
              const fill = k ? colors.accent[k] : colors.accent.blue;
              return (
                <div key={p} className="flex items-center gap-2 text-xs">
                  <span
                    className="w-3 h-3 rounded-sm flex-shrink-0"
                    style={{ backgroundColor: fill }}
                  />
                  <span className="text-text-secondary">{perturbationLabels[p] ?? p}</span>
                </div>
              );
            })}
            <div className="text-xs text-text-muted ml-auto">
              <span className="text-amber-400 font-bold">*</span> significant after correction (reject = true)
            </div>
          </div>
          {METRICS.map((m) => {
            const open = expanded.has(m.key);
            return (
              <div key={m.key} className="rounded-lg border border-border-default bg-bg-primary overflow-hidden">
                <button
                  onClick={() => toggleMetric(m.key)}
                  className="w-full flex items-center gap-2 px-4 py-3 hover:bg-bg-hover transition-colors text-left"
                >
                  {open ? (
                    <ChevronDown className="w-4 h-4 text-text-muted flex-shrink-0" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-text-muted flex-shrink-0" />
                  )}
                  <span className="text-sm font-semibold text-text-primary">{m.label}</span>
                </button>
                {open && (
                  <div className="border-t border-border-default p-4">
                    <PerturbationBarChart
                      metric={m.key}
                      metricLabel={m.label}
                      systems={systems}
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
