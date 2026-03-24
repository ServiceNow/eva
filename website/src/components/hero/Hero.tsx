import { motion } from 'framer-motion';
import { Github, ExternalLink, Plane } from 'lucide-react';

export function Hero() {
  return (
    <section id="hero" className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto text-center">
        <div>
          <h1
            className="text-3xl sm:text-4xl lg:text-[2.75rem] font-extrabold mb-2 leading-tight bg-clip-text text-transparent"
            style={{ backgroundImage: 'linear-gradient(to right, #7C3AED, #818CF8, #60A5FA)' }}
          >
            A New End-to-end Framework for<br />Evaluating Voice Agents (EVA)
          </h1>
          <p className="text-sm sm:text-base font-bold text-[#A78BFA] max-w-3xl mx-auto mb-2.5">
            Tara Bogavelli, Gabrielle Gauthier Melançon, Katrina Stankiewicz, Oluwanifemi Bamgbose, Hoang Nguyen, Raghav Mehndiratta, Hari Subramani*
          </p>
          <p className="text-base sm:text-lg font-semibold text-text-secondary max-w-3xl mx-auto mb-4">
            ServiceNow Research
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
          className="grid grid-cols-1 sm:grid-cols-2 gap-10 max-w-5xl mx-auto mb-14"
        >
          {/* Data Section */}
          <div className="flex flex-col">
            <h3 className="text-xl font-bold text-text-primary text-center mb-5">Data</h3>
            <div className="rounded-xl border border-border-default bg-bg-secondary p-7 flex-1 flex flex-col">
              <div className="flex items-center justify-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-amber/10 flex items-center justify-center flex-shrink-0">
                  <Plane className="w-5 h-5 text-amber" />
                </div>
                <div className="text-base font-semibold text-text-primary">Airline</div>
              </div>
              <p className="text-sm text-text-secondary leading-relaxed mb-4 text-center">
                Passengers calling to rebook disrupted flights — IRROPS rebooking, voluntary changes, cancellations, and vouchers.
              </p>
              <div className="flex flex-wrap justify-center gap-1.5 mb-6">
                {['IRROPS Rebooking', 'Voluntary Changes', 'Cancellations', 'Vouchers', 'Standby'].map(cat => (
                  <span key={cat} className="px-2 py-0.5 rounded-full bg-amber/10 text-amber text-xs font-medium">{cat}</span>
                ))}
              </div>
              <div className="grid grid-cols-4 gap-3 mt-auto">
                <div className="rounded-lg bg-bg-primary px-3 py-3 text-center">
                  <div className="text-2xl font-bold text-text-primary">15</div>
                  <div className="text-xs text-text-muted">Tools</div>
                </div>
                <div className="rounded-lg bg-bg-primary px-3 py-3 text-center">
                  <div className="text-2xl font-bold text-text-primary">50</div>
                  <div className="text-xs text-text-muted">Scenarios</div>
                </div>
                <div className="rounded-lg bg-bg-primary px-3 py-3 text-center">
                  <div className="text-2xl font-bold text-text-primary">3</div>
                  <div className="text-xs text-text-muted">Trials each</div>
                </div>
                <div className="rounded-lg bg-bg-primary px-3 py-3 text-center">
                  <div className="text-2xl font-bold text-text-primary">150</div>
                  <div className="text-xs text-text-muted leading-tight">Simulated<br />Conversations</div>
                </div>
              </div>
              <p className="text-sm font-bold text-white text-center mt-4">More domains coming soon!</p>
            </div>
          </div>

          {/* Evaluation Dimensions Section */}
          <div className="flex flex-col">
            <h3 className="text-xl font-bold text-text-primary text-center mb-5">Evaluation Dimensions</h3>
            <div className="space-y-4 flex-1 flex flex-col">
              <div className="rounded-xl border border-purple/30 bg-purple/5 p-7 flex-1 flex flex-col items-center justify-center text-center">
                <div className="text-sm font-semibold text-purple-light tracking-wide uppercase mb-1">EVA-A</div>
                <div className="text-xl font-bold text-text-primary">Accuracy</div>
                <p className="text-sm text-text-secondary mt-2">Did the agent complete the task correctly?</p>
              </div>
              <div className="rounded-xl border border-blue/30 bg-blue/5 p-7 flex-1 flex flex-col items-center justify-center text-center">
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
