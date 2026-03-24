import { Github, ExternalLink } from 'lucide-react';

export function Footer() {
  return (
    <footer className="border-t border-border-default bg-bg-secondary py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Acknowledgements */}
        <div className="mb-10 pb-10 border-b border-border-default">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Acknowledgements</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <p className="text-sm text-text-muted mb-2">Data review and framework code improvements</p>
              <p className="text-sm text-text-secondary">Aman Tiwari, Jishnu S Nair, Lindsay Brin, Joseph Marinier</p>
            </div>
            <div>
              <p className="text-sm text-text-muted mb-2">Organization leadership</p>
              <p className="text-sm text-text-secondary">Fanny Riols, Anil Madamala, Sridhar Nemala</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-lg font-semibold text-text-primary mb-3">EVA</h3>
            <p className="text-sm text-text-secondary">
              End-to-end evaluation framework for conversational voice agents.
              Built with Pipecat and ElevenLabs.
            </p>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-text-primary mb-3">Links</h3>
            <div className="flex flex-col gap-2">
              <a href="https://github.com/ServiceNow/EVA-Bench" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-sm text-text-secondary hover:text-purple-light transition-colors">
                <Github className="w-4 h-4" /> GitHub Repository
              </a>
              <a href="https://huggingface.co/datasets/ServiceNow/EVA-Bench" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-sm text-text-secondary hover:text-purple-light transition-colors">
                <ExternalLink className="w-4 h-4" /> HuggingFace Dataset
              </a>
              <a href="https://huggingface.co/blog/ServiceNow/eva-bench" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-sm text-text-secondary hover:text-purple-light transition-colors">
                <ExternalLink className="w-4 h-4" /> Blog Post
              </a>
            </div>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-text-primary mb-3">Citation</h3>
            <pre className="text-xs text-text-muted bg-bg-primary rounded-lg p-3 overflow-x-auto font-mono">
{`@misc{eva-2026,
  title={EVA: End-to-End Evaluation
         of Conversational Voice Agents},
  author={ServiceNow Research},
  year={2026},
  url={https://github.com/ServiceNow/EVA-Bench}
}`}
            </pre>
          </div>
        </div>
        <div className="mt-8 pt-8 border-t border-border-default text-center text-sm text-text-muted">
          ServiceNow Research
        </div>
      </div>
    </footer>
  );
}
