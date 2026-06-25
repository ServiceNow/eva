import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { SystemStats } from '../../data/leaderboardData';
import { perturbationLabels } from '../../data/leaderboardData';
import { PerturbationBarChart } from './PerturbationBarChart';
import { PerturbationMetricValueBarChart } from './PerturbationMetricValueBarChart';
import { useThemeColors } from '../../styles/theme';

const PERT_COLOR_KEYS: Record<string, 'amber' | 'cyan' | 'purple'> = {
  accent: 'amber',
  background_noise: 'cyan',
  both: 'purple',
};

const PERTURBATIONS = ['accent', 'background_noise', 'both'] as const;

export function SttTranscription({ systems }: { systems: SystemStats[] }) {
  const colors = useThemeColors();
  const cascadeSystems = systems.filter((s) => s.type === 'cascade');
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
          <h3 className="text-lg font-bold text-text-primary">
            STT Robustness - Transcription Accuracy (Key Entities)
          </h3>
          <p className="text-sm text-text-muted mt-0.5">
            Cascade systems first transcribe the caller's audio into text using a STT (speech-to-text) model, so we can
            directly measure their accuracy on key entities (names, IDs, numbers, dates). In contrast, S2S and hybrid
            systems process audio end-to-end and never produce an intermediate transcript.
            <br /> <br /> For each model we report on clean audio and under the three perturbations presented above:
            accent, background noise, and the two combined. The first plot shows accuracy (higher is better), and the
            second plot shows the mean delta vs. the same scenario's clean runs (greater negative is more impacted).
            Error bars show 95% bootstrap confidence intervals. Asterisks (<span className="text-amber-400">*</span>)
            indicate that the delta between the perturbation effect and clean baseline is statistically significant after
            Holm-Bonferroni correction across the family of perturbation{' '}
            <span style={{ fontFamily: 'inherit' }}>×</span> system tests.
          </p>
        </div>
      </button>

      {sectionOpen && (
        <div className="border-t border-border-default p-4 space-y-3">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-2 py-3 rounded-lg bg-bg-primary border border-border-default">
            {/* Clean baseline */}
            <div className="flex items-center gap-2 text-xs">
              <span
                className="w-3 h-3 rounded-sm flex-shrink-0"
                style={{ backgroundColor: colors.text.muted }}
              />
              <span className="text-text-secondary">Clean</span>
            </div>
            {/* Three perturbations */}
            {PERTURBATIONS.map((p) => {
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
              <span className="text-amber-400 font-bold">*</span> significant perturbation effect:  <span className="text-amber-400">*</span> p &lt; 0.05, <span className="text-amber-400">**</span> p &lt; 0.01, <span className="text-amber-400">***</span> p &lt; 0.001
            </div>
          </div>

          <div className="rounded-lg border border-border-default bg-bg-primary overflow-hidden">
            <button
              onClick={() => toggleMetric('accuracy')}
              className="w-full flex items-center gap-2 px-4 py-3 hover:bg-bg-hover transition-colors text-left"
            >
              {expanded.has('accuracy') ? (
                <ChevronDown className="w-4 h-4 text-text-muted flex-shrink-0" />
              ) : (
                <ChevronRight className="w-4 h-4 text-text-muted flex-shrink-0" />
              )}
              <span className="text-sm font-semibold text-text-primary">Accuracy</span>
            </button>
            {expanded.has('accuracy') && (
              <div className="border-t border-border-default p-4">
                <PerturbationMetricValueBarChart
                  metric="transcription_accuracy_key_entities"
                  metricLabel="Accuracy on key entities (higher is better)"
                  systems={cascadeSystems}
                />
              </div>
            )}
          </div>

          <div className="rounded-lg border border-border-default bg-bg-primary overflow-hidden">
            <button
              onClick={() => toggleMetric('delta')}
              className="w-full flex items-center gap-2 px-4 py-3 hover:bg-bg-hover transition-colors text-left"
            >
              {expanded.has('delta') ? (
                <ChevronDown className="w-4 h-4 text-text-muted flex-shrink-0" />
              ) : (
                <ChevronRight className="w-4 h-4 text-text-muted flex-shrink-0" />
              )}
              <span className="text-sm font-semibold text-text-primary">Delta vs. clean baseline</span>
            </button>
            {expanded.has('delta') && (
              <div className="border-t border-border-default p-4">
                <PerturbationBarChart
                  metric="transcription_accuracy_key_entities"
                  metricLabel="Mean Δ vs. clean (more negative = greater impact)"
                  systems={cascadeSystems}
                  amberFirst
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
