import { motion } from 'framer-motion';
import { Section } from '../layout/Section';

export function AcknowledgementsSection() {
  return (
    <Section
      id="acknowledgements"
      title="Contributions & Acknowledgements"
      subtitle=""
    >
      <div className="max-w-3xl mx-auto space-y-8">
        {/* Core Contributors */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <div className="rounded-xl border border-purple/30 bg-purple/5 p-6">
            <h3 className="text-base font-semibold text-purple-light mb-3">Core Contributors</h3>
            <p className="text-sm font-semibold text-text-primary">Tara Bogavelli, Gabrielle Gauthier Melançon, Katrina Stankiewicz, Oluwanifemi Bamgbose, Hoang Nguyen, Raghav Mehndiratta, Hari Subramani</p>
          </div>
        </motion.div>

        {/* Secondary Contributors */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <div className="rounded-xl border border-blue/30 bg-blue/5 p-6">
            <h3 className="text-base font-semibold text-blue-light mb-2">Secondary Contributors</h3>
            <p className="text-sm text-text-secondary mb-3">We thank the following individuals for their careful data review and thoughtful contributions to the framework.</p>
            <p className="text-sm font-semibold text-text-primary">Lindsay Brin, Akshay Kalkunte, Joseph Marinier, Jishnu Nair, Aman Tiwari</p>
          </div>
        </motion.div>

        {/* Management and Leadership */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="rounded-xl border border-amber/30 bg-amber/5 p-6">
            <h3 className="text-base font-semibold text-amber mb-2">Management and Leadership</h3>
            <p className="text-sm text-text-secondary mb-4">We are grateful to the following individuals for their management, leadership, and support.</p>
            <div className="space-y-2">
              <div className="flex justify-between items-baseline gap-4">
                <span className="text-sm font-semibold text-text-primary">Fanny Riols</span>
                <span className="text-xs text-text-muted">Research Scientist Manager</span>
              </div>
              <div className="flex justify-between items-baseline gap-4">
                <span className="text-sm font-semibold text-text-primary">Anil Madamala</span>
                <span className="text-xs text-text-muted">Director, Machine Learning Engineering Management</span>
              </div>
              <div className="flex justify-between items-baseline gap-4">
                <span className="text-sm font-semibold text-text-primary">Sridhar Nemala</span>
                <span className="text-xs text-text-muted">Senior Director, Machine Learning Engineering</span>
              </div>
              <div className="flex justify-between items-baseline gap-4">
                <span className="text-sm font-semibold text-text-primary">Srinivas Sunkara</span>
                <span className="text-xs text-text-muted">VP, Research Engineering Management</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Upstream Contributors */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <div className="rounded-xl border border-cyan/30 bg-cyan/5 p-6">
            <h3 className="text-base font-semibold text-cyan mb-2">Upstream Contributors</h3>
            <p className="text-sm text-text-secondary">We extend our thanks to the <span className="font-bold text-text-primary">PAVA</span> and <span className="font-bold text-text-primary">CLAE</span> teams whose prior work on evaluations and voice agents provided valuable inspiration for this project.</p>
          </div>
        </motion.div>

        {/* Citation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="rounded-xl border border-border-default bg-bg-secondary p-6"
        >
          <h3 className="text-base font-semibold text-text-primary mb-3">Citation</h3>
          <pre className="text-xs text-text-muted bg-bg-primary rounded-lg p-4 overflow-x-auto font-mono">
{`@misc{eva-2026,
  title={EVA: A New End-to-end Framework for Evaluating Voice Agents},
  author={Bogavelli, Tara and Gauthier Melançon, Gabrielle
          and Stankiewicz, Katrina and Bamgbose, Oluwanifemi
          and Nguyen, Hoang and Mehndiratta, Raghav
          and Subramani, Hari},
  year={2026},
  url={https://github.com/ServiceNow/eva}
}`}
          </pre>
        </motion.div>
      </div>
    </Section>
  );
}
