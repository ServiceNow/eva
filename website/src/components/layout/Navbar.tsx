import { Sun, Moon } from 'lucide-react';
import type { TabId } from '../../App';
import type { ThemeMode } from '../../styles/theme';

const tabs: { id: TabId; label: string }[] = [
  { id: 'intro', label: 'Intro' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'metrics', label: 'Methodology' },
  { id: 'leaderboard-oss', label: 'Early Results' },
  { id: 'demo', label: 'Demo' },
  { id: 'limitations', label: 'Limitations & Future' },
  { id: 'acknowledgements', label: 'Contributors' },
];

interface NavbarProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
}

export function Navbar({ activeTab, onTabChange, theme, onToggleTheme }: NavbarProps) {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-bg-primary/80 backdrop-blur-xl border-b border-border-default">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="w-0" />
          <div className="hidden sm:flex items-center gap-4">
            {tabs.map(t => (
              <a
                key={t.id}
                href={`#${t.id}`}
                onClick={(e) => { e.preventDefault(); onTabChange(t.id); }}
                className={`px-4 py-2 rounded-lg text-base font-semibold transition-colors no-underline ${
                  activeTab === t.id
                    ? 'text-purple-light bg-purple/15 border border-purple/30'
                    : 'text-text-primary/80 hover:text-text-primary hover:bg-bg-hover border border-transparent'
                }`}
              >
                {t.label}
              </a>
            ))}
          </div>
          <button
            onClick={onToggleTheme}
            className="w-9 h-9 rounded-lg flex items-center justify-center text-text-muted hover:text-text-primary hover:bg-bg-hover transition-colors"
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? <Sun className="w-4.5 h-4.5" /> : <Moon className="w-4.5 h-4.5" />}
          </button>
        </div>
      </div>
    </nav>
  );
}
