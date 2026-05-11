import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { SystemStats, DomainOrPooled } from '../../data/leaderboardData';
import { PerturbationBarChart } from './PerturbationBarChart';

interface PerturbationsProps {
  systems: SystemStats[];
  domain: DomainOrPooled;
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

export function Perturbations({ systems, domain }: PerturbationsProps) {
  const [sectionOpen, setSectionOpen] = useState(false);
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
            Per-metric deltas (Δ = perturbed − clean) under accent, background noise, and combined perturbations.
            Error bars show 95% CIs; <span className="text-amber-400">*</span> marks statistically significant pairs.
          </p>
        </div>
      </button>

      {sectionOpen && (
        <div className="border-t border-border-default p-4 space-y-3">
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
                      domain={domain}
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
