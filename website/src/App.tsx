import { useState, useEffect, useCallback, useMemo } from 'react';
import { Navbar } from './components/layout/Navbar';
import { AcknowledgementsSection } from './components/acknowledgements/AcknowledgementsSection';

import { Hero } from './components/hero/Hero';
import { ArchitectureDiagram } from './components/architecture/ArchitectureDiagram';
import { MetricsExplorer } from './components/metrics/MetricsExplorer';
import { LeaderboardSection } from './components/leaderboard/LeaderboardSection';
import { ConversationDemo } from './components/conversation/ConversationDemo';
import { LimitationsSection } from './components/limitations/LimitationsSection';
import { ThemeContext, themeColors, type ThemeMode } from './styles/theme';

export type TabId = 'intro' | 'architecture' | 'metrics' | 'leaderboard-oss' | 'demo' | 'limitations' | 'acknowledgements';

const validTabs: Set<string> = new Set<TabId>([
  'intro', 'architecture', 'metrics', 'leaderboard-oss', 'demo', 'limitations', 'acknowledgements',
]);

function getTabFromHash(): TabId {
  const hash = window.location.hash.slice(1); // remove '#'
  if (hash && validTabs.has(hash)) return hash as TabId;
  return 'intro';
}

function getInitialTheme(): ThemeMode {
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem('eva-theme');
    if (stored === 'light' || stored === 'dark') return stored;
  }
  return 'dark';
}

function App() {
  const [activeTab, setActiveTab] = useState<TabId>(getTabFromHash);
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);

  // Sync hash -> tab on back/forward navigation
  useEffect(() => {
    const onHashChange = () => setActiveTab(getTabFromHash());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  // Update hash when tab changes
  const changeTab = useCallback((tab: TabId) => {
    setActiveTab(tab);
    window.history.pushState(null, '', `#${tab}`);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('eva-theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  }, []);

  const themeCtx = useMemo(() => ({
    mode: theme,
    colors: themeColors[theme],
  }), [theme]);

  return (
    <ThemeContext.Provider value={themeCtx}>
      <div className="min-h-screen bg-bg-primary">
        <Navbar activeTab={activeTab} onTabChange={changeTab} theme={theme} onToggleTheme={toggleTheme} />
        <main>
          {activeTab === 'intro' && <Hero />}
          {activeTab === 'architecture' && <ArchitectureDiagram />}
          {activeTab === 'metrics' && <MetricsExplorer />}
          {activeTab === 'leaderboard-oss' && <LeaderboardSection />}
          {activeTab === 'demo' && <ConversationDemo />}
          {activeTab === 'limitations' && <LimitationsSection />}
          {activeTab === 'acknowledgements' && <AcknowledgementsSection />}
        </main>
      </div>
    </ThemeContext.Provider>
  );
}

export default App;
