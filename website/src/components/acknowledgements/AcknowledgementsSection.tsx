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
            <p className="text-sm font-semibold text-text-primary">Tara Bogavelli, Gabrielle Gauthier Melançon, Katrina Stankiewicz, Oluwanifemi Bamgbose, Fanny Riols, Hoang Nguyen, Raghav Mehndiratta, Lindsay Brin, Hari Subramani, Joseph Marinier</p>
          </div>
        </motion.div>

        {/* Linguists */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-6">
            <h3 className="text-base font-semibold text-emerald-400 mb-2">Machine Learning Data Linguists</h3>
            <p className="text-sm text-text-secondary mb-3">We thank our linguist collaborators for their work on carefully reviewing the HR and ITSM data scenarios, providing feedback on domain design, and annotating conversation samples with ratings for us to measure human-judge alignment.</p>
            <p className="text-sm font-semibold text-text-primary">Tiffany Do, Ryan Dux, Maria Kossenko, Keerthana Gopinathan, Anne Heaton-Dunlap, Nidhi Kumari, Ranjani Iyer</p>
          </div>
        </motion.div>

        {/* Secondary Contributors */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.15 }}
        >
          <div className="rounded-xl border border-blue/30 bg-blue/5 p-6">
            <h3 className="text-base font-semibold text-blue-light mb-2">Secondary Contributors</h3>
            <p className="text-sm text-text-secondary mb-3">We thank the following individuals for their careful data review of the CSM domain and thoughtful contributions to the framework.</p>
            <p className="text-sm font-semibold text-text-primary">Akshay Kalkunte, Jishnu Nair, Aman Tiwari</p>
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
            <div className="space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-1">
                <span className="text-sm font-semibold text-text-primary">Anil Madamala</span>
                <span className="text-xs text-text-muted">Director, Machine Learning Engineering Management</span>
              </div>
              <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-1">
                <span className="text-sm font-semibold text-text-primary">Sridhar Nemala</span>
                <span className="text-xs text-text-muted">Senior Director, Machine Learning Engineering</span>
              </div>
              <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-1">
                <span className="text-sm font-semibold text-text-primary">Srinivas Sunkara</span>
                <span className="text-xs text-text-muted">VP, Research Engineering Management</span>
              </div>
              <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-1">
                <span className="text-sm font-semibold text-text-primary">Joyce Li</span>
                <span className="text-xs text-text-muted">Principal Product Manager</span>
              </div>
              <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-1">
                <span className="text-sm font-semibold text-text-primary">Nitin Aggarwal</span>
                <span className="text-xs text-text-muted">Senior Director, Product Management</span>
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
{`@misc{bogavelli2026evabenchnewendtoendframework,
      title={EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents},
      author={Tara Bogavelli and Gabrielle Gauthier Melançon and Katrina Stankiewicz and Oluwanifemi Bamgbose and Fanny Riols and Hoang H. Nguyen and Raghav Mehndiratta and Lindsay Devon Brin and Joseph Marinier and Hari Subramani and Anil Madamala and Sridhar Krishna Nemala and Srinivas Sunkara},
      year={2026},
      eprint={2605.13841},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2605.13841},
}`}
          </pre>
        </motion.div>
      </div>
    </Section>
  );
}
