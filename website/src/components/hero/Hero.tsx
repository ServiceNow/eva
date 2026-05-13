import { motion } from 'framer-motion';
import { Github, ExternalLink, Database, Plane, Wrench, Stethoscope } from 'lucide-react';

const DOMAINS = [
  {
    id: 'airline',
    label: 'CSM',
    icon: Plane,
    blurb: 'Customers calling a customer-service line to rebook disrupted flights — IRROPS rebooking, voluntary changes, cancellations, and vouchers.',
    tools: 15,
    scenarios: 50,
  },
  {
    id: 'itsm',
    label: 'ITSM',
    icon: Wrench,
    blurb: 'Employees calling IT support to resolve enterprise IT and service-management issues.',
    tools: 59,
    scenarios: 80,
  },
  {
    id: 'medical-hr',
    label: 'HR',
    icon: Stethoscope,
    blurb: 'Healthcare workers calling HR for benefits, scheduling, leave, and policy questions.',
    tools: 47,
    scenarios: 83,
  },
];

export function Hero() {
  return (
    <section id="hero" className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto text-center">
        <div>
          <h1
            className="text-4xl sm:text-5xl lg:text-6xl font-extrabold mb-3 leading-tight bg-clip-text text-transparent"
            style={{ backgroundImage: 'linear-gradient(to right, #7C3AED, #818CF8, #60A5FA)' }}
          >
            EVA-Bench
          </h1>
          <p
            className="text-xl sm:text-2xl lg:text-[1.75rem] font-semibold max-w-3xl mx-auto mb-4 leading-tight bg-clip-text text-transparent"
            style={{ backgroundImage: 'linear-gradient(to right, #7C3AED, #818CF8, #60A5FA)' }}
          >
            A New End-to-end Framework for Evaluating Voice Agents
          </p>
          <p className="text-sm sm:text-base font-bold text-[#A78BFA] max-w-3xl mx-auto mb-2.5">
            Tara Bogavelli, Gabrielle Gauthier Melançon, Katrina Stankiewicz, Oluwanifemi Bamgbose, Fanny Riols, Hoang Nguyen, Raghav Mehndiratta, Lindsay Brin, Hari Subramani, Joseph Marinier*
          </p>
          <p className="text-base sm:text-lg font-semibold text-text-secondary max-w-3xl mx-auto mb-4">
            ServiceNow AI Research
          </p>
          <p className="text-base sm:text-lg text-text-muted max-w-3xl mx-auto mb-14 leading-relaxed">
            An open-source evaluation framework that measures voice agents over complete, multi-turn
            spoken conversations using a realistic bot-to-bot architecture. EVA captures the
            compounding failure modes that component-level benchmarks miss.
          </p>
        </div>

        {/* Data & Evaluation Dimensions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="flex flex-col gap-10 max-w-5xl mx-auto mb-14"
        >
          {/* Data Section */}
          <div className="flex flex-col">
            <h3 className="text-xl font-bold text-text-primary text-center mb-5">Data</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {DOMAINS.map(d => (
                <div key={d.id} className="rounded-xl border border-border-default bg-bg-secondary p-5 flex flex-col">
                  <div className="flex items-center justify-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-lg bg-amber/10 flex items-center justify-center flex-shrink-0">
                      <d.icon className="w-5 h-5 text-amber" />
                    </div>
                    <div className="text-base font-semibold text-text-primary">{d.label}</div>
                  </div>
                  <p className="text-xs text-text-secondary leading-relaxed mb-3 text-center">{d.blurb}</p>
                  <div className="grid grid-cols-2 gap-2 mt-auto">
                    <div className="rounded-lg bg-bg-primary px-2 py-2 text-center">
                      <div className="text-xl font-bold text-text-primary">{d.tools}</div>
                      <div className="text-[10px] text-text-muted">Tools</div>
                    </div>
                    <div className="rounded-lg bg-bg-primary px-2 py-2 text-center">
                      <div className="text-xl font-bold text-text-primary">{d.scenarios}</div>
                      <div className="text-[10px] text-text-muted">Scenarios</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Evaluation Dimensions Section */}
          <div className="flex flex-col">
            <h3 className="text-xl font-bold text-text-primary text-center mb-5">Evaluation Dimensions</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="rounded-xl border border-purple/30 bg-purple/5 p-7 flex flex-col items-center justify-center text-center">
                <div className="text-sm font-semibold text-purple-light tracking-wide uppercase mb-1">EVA-A</div>
                <div className="text-xl font-bold text-text-primary">Accuracy</div>
                <p className="text-sm text-text-secondary mt-2">Did the agent complete the task correctly?</p>
              </div>
              <div className="rounded-xl border border-blue/30 bg-blue/5 p-7 flex flex-col items-center justify-center text-center">
                <div className="text-sm font-semibold text-blue-light tracking-wide uppercase mb-1">EVA-X</div>
                <div className="text-xl font-bold text-text-primary">Experience</div>
                <p className="text-sm text-text-secondary mt-2">Was the conversational experience high quality?</p>
              </div>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="flex flex-wrap justify-center gap-3"
        >
          <a
            href="https://github.com/ServiceNow/eva"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-purple text-white font-medium text-sm hover:bg-purple-dim transition-colors"
          >
            <Github className="w-4 h-4" /> View on GitHub
          </a>
          <a
            href="https://huggingface.co/datasets/ServiceNow-AI/eva"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-bg-tertiary text-text-primary font-medium text-sm hover:bg-bg-hover border border-border-default transition-colors"
          >
            <Database className="w-4 h-4" /> Dataset
          </a>
          <a
            href="https://huggingface.co/blog/ServiceNow-AI/eva"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-bg-tertiary text-text-primary font-medium text-sm hover:bg-bg-hover border border-border-default transition-colors"
          >
            <ExternalLink className="w-4 h-4" /> Blog Post
          </a>
        </motion.div>

        <p className="text-xs text-text-muted mt-6">
          *Full list of contributors found in the Contributors tab
        </p>
      </div>
    </section>
  );
}
