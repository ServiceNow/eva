import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Code, ExternalLink, MessageSquare, Volume2 } from 'lucide-react';
import { metricTypeLabels, metricTypeColors } from '../../data/metricsData';
import type { MetricDefinition } from '../../data/metricsData';
import { JudgePromptViewer } from './JudgePromptViewer';

interface MetricNodeProps {
  metric: MetricDefinition;
}

const typeIcons = {
  deterministic: Code,
  llm_judge: MessageSquare,
  lalm_judge: Volume2,
};

export function MetricNode({ metric }: MetricNodeProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showDevAccuracy, setShowDevAccuracy] = useState(false);
  const Icon = typeIcons[metric.type];
  const badgeColor = metricTypeColors[metric.type];

  return (
    <motion.div
      layout
      className="rounded-xl border border-border-default bg-bg-secondary overflow-hidden"
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-5 text-left hover:bg-bg-hover/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ backgroundColor: badgeColor + '20' }}
          >
            <Icon className="w-5 h-5" style={{ color: badgeColor }} />
          </div>
          <div>
            <div className="text-base font-semibold text-text-primary">
              {metric.displayName}
              {metric.badge && (
                <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded-full bg-amber/10 text-amber border border-amber/20 font-medium uppercase align-middle">
                  {metric.badge}
                </span>
              )}
            </div>
            <div className="text-xs font-medium mt-0.5" style={{ color: badgeColor }}>
              {metricTypeLabels[metric.type]}
              {metric.judgeModel && <span className="text-text-muted"> &middot; {metric.judgeModel}</span>}
            </div>
          </div>
        </div>
        <ChevronDown
          className={`w-5 h-5 text-text-muted transition-transform ${isExpanded ? 'rotate-180' : ''}`}
        />
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 space-y-4">
              <div className="border-t border-border-default pt-4">
                <p className="text-sm text-text-secondary leading-relaxed">{metric.description}</p>
              </div>

              <div className="space-y-3">
                <div className="rounded-lg bg-bg-primary p-4">
                  <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Inputs</div>
                  <div className="text-sm text-text-secondary leading-relaxed">{metric.inputs}</div>
                </div>
                <div className="rounded-lg bg-bg-primary p-4">
                  <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Output</div>
                  <div className="text-sm text-text-secondary leading-relaxed">{metric.outputRange}</div>
                </div>
                {metric.passThreshold && (
                  <div className="rounded-lg bg-bg-primary p-4">
                    <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Pass Threshold</div>
                    <div className="text-sm text-text-secondary leading-relaxed">{metric.passThreshold}</div>
                  </div>
                )}
              </div>

              {metric.judgePrompt && (
                <JudgePromptViewer prompt={metric.judgePrompt} model={metric.judgeModel} />
              )}

              {/* Development and Accuracy nested expander */}
              {metric.judgeScores && metric.judgeScores.length > 0 && (
                <div className="rounded-lg border border-border-default overflow-hidden">
                  <button
                    onClick={() => setShowDevAccuracy(!showDevAccuracy)}
                    className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-bg-hover/30 transition-colors"
                  >
                    <div className="text-sm font-semibold text-text-secondary">Judge Accuracy</div>
                    <ChevronDown
                      className={`w-4 h-4 text-text-muted transition-transform ${showDevAccuracy ? 'rotate-180' : ''}`}
                    />
                  </button>
                  <AnimatePresence>
                    {showDevAccuracy && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        className="overflow-hidden"
                      >
                        <div className="px-4 pb-4 space-y-3">
                          <div className="border-t border-border-default pt-3">
                            <div className="flex flex-wrap gap-3">
                              {metric.judgeScores.map(({ label, value, std }) => (
                                <div key={label} className="flex items-center gap-2 rounded-lg bg-bg-primary px-3 py-2">
                                  <span className="text-xs text-text-muted font-mono">{label}</span>
                                  <span className="text-sm font-semibold text-text-primary">
                                    {(value * 100).toFixed(1)}%
                                    {std != null && (
                                      <span className="text-text-muted font-normal text-xs ml-1">
                                        (&plusmn;{(std * 100).toFixed(1)}%)
                                      </span>
                                    )}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                          {metric.developmentDocUrl && (
                            <a
                              href={metric.developmentDocUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-1.5 text-sm text-accent-primary hover:text-accent-hover transition-colors"
                            >
                              View judge development details
                              <ExternalLink className="w-3.5 h-3.5" />
                            </a>
                          )}
                          {metric.judgeDevelopmentNotes && (
                            <p className="text-sm text-text-secondary leading-relaxed">{metric.judgeDevelopmentNotes}</p>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
