import { useState, useEffect, useCallback } from 'react';
import { X } from 'lucide-react';

interface JudgePromptViewerProps {
  prompt: string;
  model?: string;
}

export function JudgePromptViewer({ prompt, model }: JudgePromptViewerProps) {
  const [isOpen, setIsOpen] = useState(false);

  const close = useCallback(() => setIsOpen(false), []);

  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    document.addEventListener('keydown', handleKey);
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', handleKey);
      document.body.style.overflow = '';
    };
  }, [isOpen, close]);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="mt-3 flex items-center gap-1.5 text-xs font-medium text-purple-light hover:text-purple transition-colors"
      >
        View Judge Prompt
        {model && <span className="text-text-muted">({model})</span>}
      </button>

      {isOpen && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center p-4"
          onClick={(e) => { if (e.target === e.currentTarget) close(); }}
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />

          {/* Modal */}
          <div className="relative w-full max-w-4xl max-h-[85vh] flex flex-col rounded-xl border border-border-default bg-bg-primary shadow-2xl">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-border-default flex-shrink-0">
              <div>
                <div className="text-sm font-semibold text-text-primary">Judge Prompt</div>
                {model && <div className="text-xs text-text-muted mt-0.5">Model: {model}</div>}
              </div>
              <button
                onClick={close}
                className="p-1.5 rounded-lg hover:bg-bg-hover transition-colors text-text-muted hover:text-text-primary"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Content */}
            <div className="overflow-y-auto flex-1">
              <pre className="px-6 py-5 text-[13px] leading-relaxed text-text-primary font-mono whitespace-pre-wrap break-words">
                {prompt}
              </pre>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
