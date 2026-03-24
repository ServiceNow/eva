import { motion } from 'framer-motion';
import { AlertTriangle, Rocket } from 'lucide-react';
import { Section } from '../layout/Section';

const limitations = [
  {
    category: 'Metrics',
    items: [
      {
        title: 'LLM Judge Reliability',
        description: 'Judge models carry their own biases and may favor certain response styles. Potential conflicts when evaluated model and judge are from the same family. LALM judges in particular are relatively new and are not as reliable as text-input-only judge models.',
      },
      {
        title: 'Binary Task Completion',
        description: 'No partial credit for conversations where the agent completed most of the task but failed on a single sub-goal. This may obscure fine grained differences between systems on task completion.',
      },
    ],
  },
  {
    category: 'Simulation',
    items: [
      {
        title: 'Domain & Simulation Scope',
        description: '50 scenarios in a single domain (airline). All scenarios are in English with standard American accents — no multilingual or accented speech coverage. Our user simulator does not include challenging speech conditions such as background noise, non-native accents, or intended interruptions.',
      },
      {
        title: 'User Simulator Fidelity',
        description: 'As our user simulator relies on a commercial system, its behavior may change across versions. The simulator may not fully replicate the natural disfluencies, hesitations, or emotional variations exhibited by real callers. The simulator may also occasionally go off policy; while we employ validators to detect such cases, perfect adherence cannot be guaranteed, particularly on subjective validator metrics.',
      },
    ],
  },
  {
    category: 'Framework',
    items: [
      {
        title: 'Pipeline Assumptions',
        description: 'PCM-to-mulaw audio conversion introduces quality degradation. Bot-to-bot audio interface timing may not fully represent production deployments. Inaccurate pipeline event timing (VAD events, etc) from differing sources may also lead to imperfect response speed values and timestamps. Log reconciliation between various systems can also have inaccuracies due to imprecise timestamps. ',
      },
      {
        title: 'Reproducibility & Reliability',
        description: 'Full reproduction requires access to commercial model APIs. There is a non-trivial cost for generating three validated trials across multiple configurations. Latency measurements (which manifest in turn taking and response speed metrics) will vary depending on APIs, deployments, and hardware, potentially leading to variation in EVA-X results within the same system.',
      },
    ],
  },
];

const futurePlans = [
  {
    category: 'Metrics',
    items: [
      {
        title: 'Naturalness & Prosody Evaluation',
        description: 'Evaluating pronunciation, rhythm, and expressiveness of agent speech. Current audio judges cannot reliably assess prosodic quality.',
      },
      {
        title: 'Emotion & Affect',
        description: 'Evaluating whether agents respond appropriately to user distress — both in content and vocal tone.',
      },
    ],
  },
{
    category: 'Framework',
    items: [
      {
        title: 'User Simulator Options',
        description: 'Introduce additional options for the user simulator, including open-source model support for greater flexibility and ease of adoption.',
      },
    ],
  },
  {
    category: 'Simulation',
    items: [
      {
        title: 'Expand Data and Domain Scope',
        description: 'Compound requests, multi-step follow-ups, and scenarios requiring longer conversational memory. New voice agent domains with distinct policy structures, entity types, and conversational dynamics.',
      },
      {
        title: 'Robustness Testing',
        description: 'Evaluate voice agent performance in noisy environments, with diverse user accents, multilingual users, and varied speaker behaviors (slow, interruptive, etc.).',
      },
    ],
  },
  {
    category: 'Extended Leaderboard',
    items: [
      {
        title: 'Extended Leaderboard',
        description: 'More cascade and speech-to-speech systems evaluated and added to the findings including native voice agent APIs beyond Pipecat (ElevenAgents, Gemini Live, OpenAI Realtime, Deepgram Voice Agent, etc.)',
      },
    ],
  },
];

const categoryColors: Record<string, { border: string; bg: string; badge: string; badgeBorder: string; text: string }> = {
  Metrics:              { border: 'border-amber/20',        bg: 'bg-amber/5',        badge: 'bg-amber/10',        badgeBorder: 'border-amber/20',        text: 'text-amber' },
  Data:                 { border: 'border-cyan-500/20',     bg: 'bg-cyan-500/5',     badge: 'bg-cyan-500/10',     badgeBorder: 'border-cyan-500/20',     text: 'text-cyan-400' },
  Framework:            { border: 'border-purple/20',       bg: 'bg-purple/5',       badge: 'bg-purple/10',       badgeBorder: 'border-purple/20',       text: 'text-purple-light' },
  Evaluation:           { border: 'border-amber/20',        bg: 'bg-amber/5',        badge: 'bg-amber/10',        badgeBorder: 'border-amber/20',        text: 'text-amber' },
  Simulation:             { border: 'border-cyan-500/20',     bg: 'bg-cyan-500/5',     badge: 'bg-cyan-500/10',     badgeBorder: 'border-cyan-500/20',     text: 'text-cyan-400' },
  'Extended Leaderboard': { border: 'border-emerald-500/20', bg: 'bg-emerald-500/5', badge: 'bg-emerald-500/10', badgeBorder: 'border-emerald-500/20', text: 'text-emerald-400' },
};

const defaultColor = { border: 'border-border-default', bg: 'bg-bg-secondary', badge: 'bg-bg-tertiary', badgeBorder: 'border-border-default', text: 'text-text-muted' };

export function LimitationsSection() {
  return (
    <Section
      id="limitations"
      title="Limitations & Future"
      subtitle="Known limitations of the current release and our near-term roadmap."
    >
      <div className="space-y-12">
        {/* Limitations */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-amber/10 flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-amber" />
            </div>
            <h3 className="text-xl font-bold text-text-primary">Current Limitations</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {limitations.flatMap((group) => {
              const c = categoryColors[group.category] ?? defaultColor;
              return group.items.map((item, i) => (
                <div key={`${group.category}-${i}`} className={`rounded-xl border ${c.border} ${c.bg} p-5 flex flex-col`}>
                  <div className="flex items-center gap-2 mb-2">
                    <span className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full ${c.badge} ${c.text} border ${c.badgeBorder}`}>{group.category}</span>
                  </div>
                  <div className="text-sm font-semibold text-text-primary mb-1.5">{item.title}</div>
                  <p className="text-sm text-text-secondary leading-relaxed flex-1">{item.description}</p>
                </div>
              ));
            })}
          </div>
        </motion.div>

        {/* Future */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-purple/10 flex items-center justify-center">
              <Rocket className="w-5 h-5 text-purple" />
            </div>
            <h3 className="text-xl font-bold text-text-primary">Roadmap</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {futurePlans.flatMap((group) => {
              const c = categoryColors[group.category] ?? defaultColor;
              return group.items.map((item, i) => (
                <div key={`${group.category}-${i}`} className={`rounded-xl border ${c.border} ${c.bg} p-5 flex flex-col`}>
                  <div className="flex items-center gap-2 mb-2">
                    <span className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full ${c.badge} ${c.text} border ${c.badgeBorder}`}>{group.category}</span>
                  </div>
                  <div className="text-sm font-semibold text-text-primary mb-1.5">{item.title}</div>
                  <p className="text-sm text-text-secondary leading-relaxed flex-1">{item.description}</p>
                </div>
              ));
            })}
          </div>
        </motion.div>
      </div>
    </Section>
  );
}
