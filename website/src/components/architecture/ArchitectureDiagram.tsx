import { motion } from 'framer-motion';
import { Section } from '../layout/Section';

function Node({ label, sublabel, color, delay = 0 }: { label: string; sublabel?: string; color: string; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay }}
      className="relative rounded-xl border bg-bg-secondary px-6 py-5 text-center"
      style={{ borderColor: color + '40' }}
    >
      <div className="text-base font-semibold text-text-primary">{label}</div>
      {sublabel && <div className="text-sm text-text-muted mt-1">{sublabel}</div>}
      <div className="absolute inset-0 rounded-xl opacity-10" style={{ background: `radial-gradient(ellipse at center, ${color}, transparent 70%)` }} />
    </motion.div>
  );
}

function Connector({ color, className = '' }: { color: string; className?: string }) {
  return (
    <div
      className={`mx-auto ${className}`}
      style={{
        width: '2px',
        background: `repeating-linear-gradient(to bottom, ${color}80 0px, ${color}80 6px, transparent 6px, transparent 10px)`,
      }}
    />
  );
}

function HLine({ color, className = '' }: { color: string; className?: string }) {
  return (
    <div
      className={className}
      style={{
        height: '2px',
        background: `repeating-linear-gradient(to right, ${color}80 0px, ${color}80 6px, transparent 6px, transparent 10px)`,
      }}
    />
  );
}

export function ArchitectureDiagram() {
  return (
    <Section
      id="architecture"
      title="Bot-to-Bot Architecture"
      subtitle="EVA evaluates voice agents using realistic bot-to-bot audio conversations over WebSocket, then computes metrics independently on the validated conversations."
    >
      <div className="max-w-5xl mx-auto relative">
        {/* Row 1: BenchmarkRunner */}
        <div className="flex justify-center">
          <div className="w-72">
            <Node label="Evaluation Runner" sublabel="Orchestrates parallel evaluation" color="#8B5CF6" delay={0} />
          </div>
        </div>

        <Connector color="#8B5CF6" className="h-8" />

        {/* Row 2: ConversationWorker */}
        <div className="flex justify-center">
          <div className="w-64">
            <Node label="ConversationWorker" sublabel="Per-scenario execution" color="#8B5CF6" delay={0.1} />
          </div>
        </div>

        {/* Branching connector: Worker -> Assistant + User */}
        <div className="flex justify-center">
          <div className="relative w-[60%]">
            <Connector color="#8B5CF6" className="h-5" />
            <HLine color="#8B5CF6" className="w-full" />
            <div className="flex justify-between">
              <Connector color="#8B5CF6" className="h-5" />
              <Connector color="#38BDF8" className="h-5" />
            </div>
          </div>
        </div>

        {/* Row 3: Assistant + WebSocket + User */}
        <div className="grid grid-cols-[1fr_auto_1fr] gap-4 items-start">
          {/* AssistantServer */}
          <div>
            <Node label="Voice Agent" sublabel="Pipecat Server" color="#8B5CF6" delay={0.2} />
            <div className="mt-4 space-y-2.5 pl-4">
              <div className="text-sm text-text-muted border-l-2 border-purple/30 pl-3 py-1">
                <span className="text-purple-light font-medium">Cascade Pipeline</span> — STT + LLM + TTS
              </div>
              <div className="text-sm text-text-muted border-l-2 border-purple/30 pl-3 py-1">
                <span className="text-purple-light font-medium">Speech-to-Speech</span> — Realtime models
              </div>
              <div className="text-sm text-text-muted border-l-2 border-purple/30 pl-3 py-1">
                <span className="text-purple-light font-medium">Turn Detection</span> — Built-in Pipecat Silero VAD + Smart Turn Analyzer (unless overridden by external VAD)
              </div>
              <div className="text-sm text-text-muted border-l-2 border-purple/30 pl-3 py-1">
                <span className="text-purple-light font-medium">Tool Executor</span> — Dynamic python tools
              </div>
            </div>
          </div>

          {/* WebSocket bidirectional arrow */}
          <div className="flex flex-col items-center justify-center pt-6 px-2">
            <div className="flex items-center gap-1">
              <div className="text-cyan font-bold">&larr;</div>
              <div
                style={{
                  width: '60px',
                  height: '2px',
                  background: `repeating-linear-gradient(to right, #06B6D480 0px, #06B6D480 8px, transparent 8px, transparent 12px)`,
                }}
              />
              <div className="text-cyan font-bold">&rarr;</div>
            </div>
            <div className="mt-2 px-3 py-1.5 rounded-full bg-cyan/10 border border-cyan/30 text-cyan text-xs font-medium whitespace-nowrap">
              WebSocket Audio
            </div>
          </div>

          {/* UserSimulator */}
          <div>
            <Node label="User Simulator" color="#38BDF8" delay={0.3} />
            <div className="mt-4 space-y-2.5 pl-4">
              <div className="text-sm text-text-muted border-l-2 border-blue/30 pl-3 py-1">
                <span className="text-blue-light font-medium">Scenario-specific</span> — Unique goal, specific decision logic, persona &amp; constraints per conversation
              </div>
              <div className="text-sm text-text-muted border-l-2 border-blue/30 pl-3 py-1">
                <span className="text-blue-light font-medium">Human-like voice</span> — Conversational TTS
              </div>
            </div>
          </div>
        </div>

        {/* Merging connector: Assistant + User -> single line */}
        <div className="flex justify-center">
          <div className="relative w-[60%]">
            <div className="flex justify-between">
              <Connector color="#F59E0B" className="h-5" />
              <Connector color="#F59E0B" className="h-5" />
            </div>
            <HLine color="#F59E0B" className="w-full" />
            <Connector color="#F59E0B" className="h-5" />
          </div>
        </div>

        {/* Row 4: Outputs */}
        <div className="flex justify-center mb-2">
          <div className="flex gap-5 flex-wrap justify-center">
            <div className="px-5 py-3 rounded-lg bg-bg-tertiary border border-border-default text-center">
              <div className="text-sm font-medium text-text-primary">Audio Files</div>
              <div className="text-xs text-text-muted mt-1">WAV recordings (assistant, user, mixed)</div>
            </div>
            <div className="px-5 py-3 rounded-lg bg-bg-tertiary border border-border-default text-center">
              <div className="text-sm font-medium text-text-primary">Logs &amp; Transcripts</div>
              <div className="text-xs text-text-muted mt-1">audit_log.json, transcript.jsonl, events</div>
            </div>
          </div>
        </div>

        <Connector color="#F59E0B" className="h-8" />

        {/* Row 5: ValidationRunner — positioned with the rerun track on the right */}
        <div className="flex justify-center">
          <div className="w-80">
            <Node label="Validators" sublabel="Reruns invalid conversations" color="#F59E0B" delay={0.4} />
          </div>
        </div>

        <Connector color="#F59E0B" className="h-8" />

        {/* Row 6: MetricsRunner */}
        <div className="flex justify-center">
          <div className="w-[28rem]">
            <Node label="Metrics Suite" sublabel="Independent post-execution evaluation" color="#F59E0B" delay={0.5} />
            <div className="grid grid-cols-3 gap-4 mt-5">
              <div className="rounded-xl border border-purple/25 bg-purple/5 px-4 py-5 text-center">
                <div className="text-base font-bold text-purple-light">EVA-A</div>
                <div className="text-sm text-text-muted mt-1.5">3 accuracy metrics</div>
              </div>
              <div className="rounded-xl border border-blue/25 bg-blue/5 px-4 py-5 text-center">
                <div className="text-base font-bold text-blue-light">EVA-X</div>
                <div className="text-sm text-text-muted mt-1.5">3 experience metrics</div>
              </div>
              <div className="rounded-xl border border-amber/25 bg-amber/5 px-4 py-5 text-center">
                <div className="text-base font-bold text-amber">Diagnostic</div>
                <div className="text-sm text-text-muted mt-1.5">6 diagnostic metrics</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}
