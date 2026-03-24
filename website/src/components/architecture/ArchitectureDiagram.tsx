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
        <div className="flex justify-center px-4">
          <div className="w-full max-w-72">
            <Node label="Evaluation Runner" sublabel="Orchestrates parallel evaluation" color="#8B5CF6" delay={0} />
          </div>
        </div>

        <Connector color="#8B5CF6" className="h-8" />

        {/* Row 2: ConversationWorker */}
        <div className="flex justify-center px-4">
          <div className="w-full max-w-64">
            <Node label="Conversation Worker" sublabel="Per-scenario execution" color="#8B5CF6" delay={0.1} />
          </div>
        </div>

        {/* Branching connector: Worker -> User + Assistant */}
        {/* Desktop: branching pattern */}
        <div className="hidden md:flex justify-center">
          <div className="relative w-[60%]">
            <Connector color="#8B5CF6" className="h-5" />
            <HLine color="#8B5CF6" className="w-full" />
            <div className="flex justify-between">
              <Connector color="#38BDF8" className="h-5" />
              <Connector color="#8B5CF6" className="h-5" />
            </div>
          </div>
        </div>
        {/* Mobile: simple vertical connector */}
        <div className="md:hidden">
          <Connector color="#38BDF8" className="h-8" />
        </div>

        {/* Row 3: User + WebSocket + Assistant */}
        {/* Desktop: horizontal layout */}
        <div className="hidden md:grid grid-cols-[1fr_auto_1fr] gap-4 items-start">
          {/* UserSimulator */}
          <div>
            <Node label="User Simulator" color="#38BDF8" delay={0.2} />
            <div className="mt-4 space-y-2.5 pl-4">
              <div className="text-sm text-text-muted border-l-2 border-blue/30 pl-3 py-1">
                <span className="text-blue-light font-medium">Scenario-specific</span> — Unique goal, specific decision logic, persona &amp; constraints per conversation
              </div>
              <div className="text-sm text-text-muted border-l-2 border-blue/30 pl-3 py-1">
                <span className="text-blue-light font-medium">Human-like voice</span> — Conversational TTS
              </div>
            </div>
          </div>

          {/* WebSocket bidirectional arrow - horizontal */}
          <div className="flex flex-col items-center justify-center pt-6 px-2">
            <div className="flex items-center justify-center">
              <span className="text-cyan font-bold">&larr;</span>
              <div
                style={{
                  width: '56px',
                  height: '2px',
                  background: `repeating-linear-gradient(to right, #06B6D480 0px, #06B6D480 6px, transparent 6px, transparent 10px)`,
                }}
              />
              <span className="text-cyan font-bold">&rarr;</span>
            </div>
            <div className="mt-2 px-3 py-1.5 rounded-full bg-cyan/10 border border-cyan/30 text-cyan text-xs font-medium whitespace-nowrap">
              WebSocket Audio
            </div>
          </div>

          {/* AssistantServer */}
          <div>
            <Node label="Voice Agent" sublabel="Pipecat Server" color="#8B5CF6" delay={0.3} />
            <div className="mt-4 space-y-2.5 pl-4">
              <div className="text-sm text-text-muted border-l-2 border-purple/30 pl-3 py-1">
                <span className="text-purple-light font-medium">Cascade Pipeline</span> — STT + LLM + TTS
              </div>
              <div className="text-sm text-text-muted border-l-2 border-purple/30 pl-3 py-1">
                <span className="text-purple-light font-medium">Audio-native models</span> — S2S, LALM + TTS
              </div>
              <div className="text-sm text-text-muted border-l-2 border-purple/30 pl-3 py-1">
                <span className="text-purple-light font-medium">Turn Detection</span> — Built-in Pipecat Silero VAD + Smart Turn Analyzer (unless overridden by external VAD)
              </div>
              <div className="text-sm text-text-muted border-l-2 border-purple/30 pl-3 py-1">
                <span className="text-purple-light font-medium">Tool Executor</span> — Dynamic python tools
              </div>
            </div>
          </div>
        </div>

        {/* Mobile: vertical stacked layout */}
        <div className="md:hidden flex flex-col items-center gap-4">
          {/* UserSimulator */}
          <div className="w-full max-w-sm">
            <Node label="User Simulator" color="#38BDF8" delay={0.2} />
            <div className="mt-4 space-y-2.5 pl-4">
              <div className="text-sm text-text-muted border-l-2 border-blue/30 pl-3 py-1">
                <span className="text-blue-light font-medium">Scenario-specific</span> — Unique goal, decision logic, persona &amp; constraints
              </div>
              <div className="text-sm text-text-muted border-l-2 border-blue/30 pl-3 py-1">
                <span className="text-blue-light font-medium">Human-like voice</span> — Conversational TTS
              </div>
            </div>
          </div>

          {/* WebSocket bidirectional arrow - vertical */}
          <div className="flex flex-col items-center py-2">
            <div className="flex flex-col items-center">
              <span className="text-cyan font-bold">&uarr;</span>
              <div
                style={{
                  width: '2px',
                  height: '36px',
                  background: `repeating-linear-gradient(to bottom, #06B6D480 0px, #06B6D480 6px, transparent 6px, transparent 10px)`,
                }}
              />
              <span className="text-cyan font-bold">&darr;</span>
            </div>
            <div className="mt-2 px-3 py-1.5 rounded-full bg-cyan/10 border border-cyan/30 text-cyan text-xs font-medium whitespace-nowrap">
              WebSocket Audio
            </div>
          </div>

          {/* AssistantServer */}
          <div className="w-full max-w-sm">
            <Node label="Voice Agent" sublabel="Pipecat Server" color="#8B5CF6" delay={0.3} />
            <div className="mt-4 space-y-2.5 pl-4">
              <div className="text-sm text-text-muted border-l-2 border-purple/30 pl-3 py-1">
                <span className="text-purple-light font-medium">Cascade Pipeline</span> — STT + LLM + TTS
              </div>
              <div className="text-sm text-text-muted border-l-2 border-purple/30 pl-3 py-1">
                <span className="text-purple-light font-medium">Audio-native models</span> — S2S, LALM + TTS
              </div>
              <div className="text-sm text-text-muted border-l-2 border-purple/30 pl-3 py-1">
                <span className="text-purple-light font-medium">Turn Detection</span> — VAD + Smart Turn Analyzer
              </div>
              <div className="text-sm text-text-muted border-l-2 border-purple/30 pl-3 py-1">
                <span className="text-purple-light font-medium">Tool Executor</span> — Dynamic python tools
              </div>
            </div>
          </div>
        </div>

        {/* Branching connector: User/Assistant -> Outputs */}
        {/* Desktop: branching pattern aligned with purple connector above */}
        <div className="hidden md:flex justify-center mt-3">
          <div className="relative w-[60%]">
            <div className="flex justify-between">
              <Connector color="#F59E0B" className="h-5" />
              <Connector color="#F59E0B" className="h-5" />
            </div>
            <HLine color="#F59E0B" className="w-full" />
          </div>
        </div>
        {/* Mobile: simple vertical connector */}
        <div className="md:hidden">
          <Connector color="#F59E0B" className="h-8" />
        </div>

        {/* Row 4: Outputs */}
        <div className="flex justify-center px-4 my-4 w-full">
          <div className="flex flex-col md:flex-row items-stretch justify-center gap-4 md:gap-6 w-full max-w-md md:max-w-none md:w-auto">
            <div className="md:w-[250px] py-3 rounded-lg bg-bg-tertiary border border-border-default text-center">
              <div className="text-sm font-medium text-text-primary">Audio Files</div>
              <div className="text-xs text-text-muted mt-1">WAV recordings (assistant, user, mixed)</div>
            </div>
            <div className="md:w-[250px] py-3 rounded-lg bg-bg-tertiary border border-border-default text-center">
              <div className="text-sm font-medium text-text-primary">Logs &amp; Transcripts</div>
              <div className="text-xs text-text-muted mt-1">audit_log.json, transcript.jsonl, events</div>
            </div>
          </div>
        </div>

        {/* Merging connector: Outputs -> Validators */}
        <div className="hidden md:flex justify-center">
          <div className="relative" style={{ width: '524px' }}>
            <div className="flex justify-between" style={{ padding: '0 80px' }}>
              <Connector color="#F59E0B" className="h-5" />
              <Connector color="#F59E0B" className="h-5" />
            </div>
            <HLine color="#F59E0B" className="w-full" />
            <Connector color="#F59E0B" className="h-8" />
          </div>
        </div>
        {/* Mobile: simple vertical connector */}
        <div className="md:hidden">
          <Connector color="#F59E0B" className="h-8" />
        </div>

        {/* Row 5: ValidationRunner — positioned with the rerun track on the right */}
        <div className="flex justify-center px-4 -mt-2">
          <div className="w-full max-w-80">
            <Node label="Validators" sublabel="Reruns invalid conversations" color="#F59E0B" delay={0.4} />
          </div>
        </div>

        <Connector color="#F59E0B" className="h-8" />

        {/* Row 6: MetricsRunner */}
        <div className="flex justify-center px-4">
          <div className="w-full max-w-[28rem]">
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
                <div className="text-sm text-text-muted mt-1.5">5 diagnostic metrics</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}
