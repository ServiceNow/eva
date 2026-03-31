import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, ArrowUp, ArrowDown } from 'lucide-react';
import { invertedMetrics } from '../../data/leaderboardData';
import type { SystemScore } from '../../data/leaderboardData';
import { getScaledHeatmapColor, useThemeColors, useThemeMode } from '../../styles/theme';

export interface AggregateColumn {
  key: string;
  label: string;
  getValue: (s: SystemScore) => number;
}

// Per-category color palettes — each category uses maximally distinct colors so
// components you actually compare (e.g. two STT models) never look alike.
const categoryPalettesDark: Record<string, string[]> = {
  stt: [
    '#F59E0B',  // amber
    '#38BDF8',  // sky blue
    '#34D399',  // emerald
    '#F87171',  // red
    '#A78BFA',  // purple
    '#FACC15',  // yellow
  ],
  llm: [
    '#22D3EE',  // cyan
    '#FB923C',  // orange
    '#818CF8',  // indigo
    '#4ADE80',  // green
    '#F59E0B',  // amber
    '#F472B6',  // pink
    '#94A3B8',  // slate
    '#A3E635',  // lime
    '#E879F9',  // fuchsia
    '#F87171',  // red
  ],
  tts: [
    '#A3E635',  // lime
    '#FB7185',  // rose
    '#67E8F9',  // light cyan
    '#C084FC',  // violet
    '#FDBA74',  // peach
    '#2DD4BF',  // teal
  ],
};

const categoryPalettesLight: Record<string, string[]> = {
  stt: [
    '#B45309',  // amber
    '#0369A1',  // sky blue
    '#047857',  // emerald
    '#B91C1C',  // red
    '#6D28D9',  // purple
    '#A16207',  // yellow
  ],
  llm: [
    '#0E7490',  // cyan
    '#C2410C',  // orange
    '#4338CA',  // indigo
    '#15803D',  // green
    '#B45309',  // amber
    '#BE185D',  // pink
    '#475569',  // slate
    '#65A30D',  // lime
    '#A21CAF',  // fuchsia
    '#B91C1C',  // red
  ],
  tts: [
    '#65A30D',  // lime
    '#E11D48',  // rose
    '#0891B2',  // light cyan
    '#7C3AED',  // violet
    '#EA580C',  // peach
    '#0D9488',  // teal
  ],
};

function getComponentColorMap(systems: SystemScore[], isDark: boolean): Map<string, string> {
  const palettes = isDark ? categoryPalettesDark : categoryPalettesLight;

  // Collect unique names per category
  const sttNames: string[] = [];
  const llmNames: string[] = [];
  const ttsNames: string[] = [];
  const seen = new Set<string>();
  for (const s of systems) {
    if (s.stt !== '-' && !seen.has('stt:' + s.stt)) { sttNames.push(s.stt); seen.add('stt:' + s.stt); }
    if (!seen.has('llm:' + s.llm)) { llmNames.push(s.llm); seen.add('llm:' + s.llm); }
    if (s.tts !== '-' && !seen.has('tts:' + s.tts)) { ttsNames.push(s.tts); seen.add('tts:' + s.tts); }
  }

  const map = new Map<string, string>();
  const assign = (names: string[], pal: string[]) => {
    names.forEach((name, i) => map.set(name, pal[i % pal.length]));
  };
  assign(sttNames, palettes.stt);
  assign(llmNames, palettes.llm);
  assign(ttsNames, palettes.tts);
  return map;
}

function SystemName({ system, componentColors }: { system: SystemScore; componentColors: Map<string, string> }) {
  if (system.type === 's2s' || system.type === '2-part') {
    if (system.tts !== '-') {
      return (
        <span className="text-sm leading-relaxed inline-flex flex-wrap items-baseline">
          <span className="whitespace-nowrap" style={{ color: componentColors.get(system.llm) }}>{system.llm}</span>
          <span className="text-text-muted whitespace-nowrap">&nbsp;+&nbsp;</span>
          <span className="whitespace-nowrap" style={{ color: componentColors.get(system.tts) }}>{system.tts}</span>
        </span>
      );
    }
    const color = componentColors.get(system.llm) || '#F1F5F9';
    return <span style={{ color }}>{system.llm}</span>;
  }
  return (
    <span className="text-sm leading-relaxed inline-flex flex-wrap items-baseline">
      <span className="whitespace-nowrap" style={{ color: componentColors.get(system.stt) }}>{system.stt}</span>
      <span className="text-text-muted whitespace-nowrap">&nbsp;+&nbsp;</span>
      <span className="whitespace-nowrap" style={{ color: componentColors.get(system.llm) }}>{system.llm}</span>
      <span className="text-text-muted whitespace-nowrap">&nbsp;+&nbsp;</span>
      <span className="whitespace-nowrap" style={{ color: componentColors.get(system.tts) }}>{system.tts}</span>
    </span>
  );
}

type SortDir = 'asc' | 'desc';

const systemSortOptions = [
  { key: null, label: 'Default' },
  { key: 'system_stt', label: 'STT' },
  { key: 'system_llm', label: 'LLM' },
  { key: 'system_tts', label: 'TTS' },
] as const;

function SortIndicator({ active, dir }: { active: boolean; dir: SortDir }) {
  if (!active) return null;
  return dir === 'desc'
    ? <ArrowDown className="w-3 h-3 inline ml-0.5" />
    : <ArrowUp className="w-3 h-3 inline ml-0.5" />;
}

interface MetricHeatmapProps {
  title: string;
  description: string;
  metricKeys: readonly string[];
  metricLabels: Record<string, string>;
  dataKey: 'accuracyMetrics' | 'experienceMetrics' | 'diagnosticMetrics';
  baseColor: string;
  aggregateColumns?: AggregateColumn[];
  aggregateColor?: string;
  systems: SystemScore[];
}

export function MetricHeatmap({ title, description, metricKeys, metricLabels, dataKey, baseColor, aggregateColumns, aggregateColor = '#F59E0B', systems }: MetricHeatmapProps) {
  const themeColors = useThemeColors();
  const themeMode = useThemeMode();
  const aggCols = aggregateColumns ?? [];
  const isDark = themeMode !== 'light';
  const componentColors = useMemo(() => getComponentColorMap(systems, isDark), [systems, isDark]);

  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [systemMenuOpen, setSystemMenuOpen] = useState(false);
  const [menuPos, setMenuPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  // Mobile view: tabs to switch between aggregate scores and individual metrics
  const [mobileTab, setMobileTab] = useState<'scores' | 'metrics'>('scores');

  const openMenu = useCallback(() => {
    if (buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      setMenuPos({ top: rect.bottom + 4, left: rect.left });
    }
    setSystemMenuOpen(o => !o);
  }, []);

  // Close menu on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (
        menuRef.current && !menuRef.current.contains(e.target as Node) &&
        buttonRef.current && !buttonRef.current.contains(e.target as Node)
      ) {
        setSystemMenuOpen(false);
      }
    }
    if (systemMenuOpen) {
      document.addEventListener('mousedown', handleClick);
      return () => document.removeEventListener('mousedown', handleClick);
    }
  }, [systemMenuOpen]);

  function handleHeaderClick(key: string) {
    if (sortKey === key) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  }

  function handleSystemSort(key: string | null) {
    if (key === null) {
      setSortKey(null);
    } else {
      if (sortKey === key) {
        setSortDir(d => d === 'desc' ? 'asc' : 'desc');
      } else {
        setSortKey(key);
        setSortDir('asc'); // alphabetical default
      }
    }
    setSystemMenuOpen(false);
  }

  const sorted = useMemo(() => {
    if (!sortKey) {
      return [...systems].sort((a, b) => {
        const aS2S = a.type === 's2s' || a.type === '2-part';
        const bS2S = b.type === 's2s' || b.type === '2-part';
        if (aS2S && !bS2S) return -1;
        if (!aS2S && bS2S) return 1;
        return a.stt.localeCompare(b.stt);
      });
    }

    const getValue = (s: SystemScore): number | string => {
      // System component sorts (string)
      if (sortKey === 'system_stt') return s.stt;
      if (sortKey === 'system_llm') return s.llm;
      if (sortKey === 'system_tts') return s.tts;
      // Aggregate column
      const aggCol = aggCols.find(c => c.key === sortKey);
      if (aggCol) return aggCol.getValue(s);
      // Metric column
      return s[dataKey][sortKey] ?? 0;
    };

    const compare = (a: SystemScore, b: SystemScore): number => {
      const va = getValue(a);
      const vb = getValue(b);
      if (typeof va === 'string' && typeof vb === 'string') {
        return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      const na = va as number;
      const nb = vb as number;
      return sortDir === 'desc' ? nb - na : na - nb;
    };

    return [...systems].sort(compare);
  }, [sortKey, sortDir, aggCols, dataKey]);

  // Compute min/max per metric for scaled coloring
  const metricRanges: Record<string, { min: number; max: number }> = {};
  for (const k of metricKeys) {
    const values = systems.map(s => s[dataKey][k] ?? 0);
    metricRanges[k] = { min: Math.min(...values), max: Math.max(...values) };
  }

  // Compute min/max for aggregate columns
  const aggRanges: Record<string, { min: number; max: number }> = {};
  for (const col of aggCols) {
    const values = systems.map(s => col.getValue(s));
    aggRanges[col.key] = { min: Math.min(...values), max: Math.max(...values) };
  }

  const totalDataCols = aggCols.length + metricKeys.length;
  const systemPct = 35;
  const dataColWidth = `${(100 - systemPct) / totalDataCols}%`;
  const systemColWidth = `${systemPct}%`;

  const headerClass = "text-center py-3 px-1 font-bold text-xs leading-snug cursor-pointer select-none hover:bg-bg-hover/50 transition-colors";

  // Determine which columns to show based on mobile tab
  const showAggCols = mobileTab === 'scores' ? aggCols : [];
  const showMetricKeys = mobileTab === 'metrics' ? metricKeys : [];

  // Calculate column widths based on what's shown
  const mobileTotalDataCols = mobileTab === 'scores' ? aggCols.length : metricKeys.length;
  const mobileDataColWidth = `${(100 - systemPct) / mobileTotalDataCols}%`;

  return (
    <div className="bg-bg-secondary rounded-xl border border-border-default p-4 sm:p-6">
      <h3 className="text-lg font-semibold text-text-primary mb-1">{title}</h3>
      <p className="text-sm text-text-secondary mb-4">{description}</p>

      {/* Mobile tabs - only show if we have both aggregate columns and metrics */}
      {aggCols.length > 0 && metricKeys.length > 0 && (
        <div className="flex gap-2 mb-4 md:hidden">
          <button
            onClick={() => setMobileTab('scores')}
            className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
              mobileTab === 'scores'
                ? 'bg-purple/20 text-purple-light'
                : 'bg-bg-hover text-text-muted hover:text-text-secondary'
            }`}
          >
            Aggregate Scores
          </button>
          <button
            onClick={() => setMobileTab('metrics')}
            className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
              mobileTab === 'metrics'
                ? 'bg-purple/20 text-purple-light'
                : 'bg-bg-hover text-text-muted hover:text-text-secondary'
            }`}
          >
            Individual Metrics
          </button>
        </div>
      )}

      {/* Desktop table - shows all columns */}
      <div className="hidden md:block overflow-x-auto">
        <table className="w-full text-sm" style={{ tableLayout: 'fixed' }}>
          <thead>
            <tr className="border-b border-border-default">
              <th className="text-left py-3 px-3 text-text-muted font-medium text-sm sticky left-0 bg-bg-secondary z-10" style={{ width: systemColWidth }}>
                <button
                  ref={buttonRef}
                  onClick={openMenu}
                  className="flex items-center gap-1 hover:text-text-primary transition-colors"
                >
                  System
                  <ChevronDown className="w-3.5 h-3.5" />
                  {sortKey?.startsWith('system_') && <SortIndicator active dir={sortDir} />}
                </button>
                {systemMenuOpen && createPortal(
                  <div
                    ref={menuRef}
                    className="bg-bg-tertiary border border-border-default rounded-lg shadow-xl py-1 min-w-[100px]"
                    style={{ position: 'fixed', top: menuPos.top, left: menuPos.left, zIndex: 9999 }}
                  >
                    {systemSortOptions.map(opt => (
                      <button
                        key={opt.key ?? 'default'}
                        onClick={() => handleSystemSort(opt.key)}
                        className={`block w-full text-left px-3 py-1.5 text-xs hover:bg-bg-hover transition-colors ${sortKey === opt.key || (opt.key === null && sortKey === null) ? 'text-purple-light font-medium' : 'text-text-secondary'}`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>,
                  document.body
                )}
              </th>
              {aggCols.map((col, i) => (
                <th
                  key={col.key}
                  className={`${headerClass} ${i === aggCols.length - 1 ? 'border-r-2 border-border-default' : ''}`}
                  style={{ color: aggregateColor, width: dataColWidth }}
                  onClick={() => handleHeaderClick(col.key)}
                >
                  {col.label}
                  <SortIndicator active={sortKey === col.key} dir={sortDir} />
                </th>
              ))}
              {metricKeys.map(k => (
                <th
                  key={k}
                  className={`${headerClass} text-text-primary`}
                  style={{ width: dataColWidth }}
                  onClick={() => handleHeaderClick(k)}
                >
                  {metricLabels[k] || k}
                  <SortIndicator active={sortKey === k} dir={sortDir} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((s) => (
              <tr key={s.id} className="border-b border-border-default/30">
                <td className="py-2.5 px-3 sticky left-0 bg-bg-secondary z-10 whitespace-nowrap">
                  <SystemName system={s} componentColors={componentColors} />
                </td>
                {aggCols.map((col, i) => {
                  const val = col.getValue(s);
                  const { min, max } = aggRanges[col.key];
                  const { bg, text } = getScaledHeatmapColor(val, min, max, aggregateColor, false, themeColors);
                  return (
                    <td key={col.key} className={`py-1.5 px-1 text-center ${i === aggCols.length - 1 ? 'border-r-2 border-border-default' : ''}`}>
                      <div
                        className="rounded-md px-0.5 py-1.5 font-mono text-xs font-medium"
                        style={{ backgroundColor: bg, color: text }}
                      >
                        {val.toFixed(2)}
                      </div>
                    </td>
                  );
                })}
                {metricKeys.map(k => {
                  const val = s[dataKey][k] ?? 0;
                  const { min, max } = metricRanges[k];
                  const invert = invertedMetrics.has(k);
                  const { bg, text } = getScaledHeatmapColor(val, min, max, baseColor, invert, themeColors);
                  return (
                    <td key={k} className="py-1.5 px-1 text-center">
                      <div
                        className="rounded-md px-0.5 py-1.5 font-mono text-xs font-medium"
                        style={{ backgroundColor: bg, color: text }}
                      >
                        {val.toFixed(2)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mobile table - shows only selected tab columns */}
      <div className="md:hidden overflow-x-auto">
        <table className="w-full text-sm" style={{ tableLayout: 'fixed' }}>
          <thead>
            <tr className="border-b border-border-default">
              <th className="text-left py-3 px-2 text-text-muted font-medium text-xs sticky left-0 bg-bg-secondary z-10" style={{ width: systemColWidth }}>
                System
              </th>
              {showAggCols.map((col) => (
                <th
                  key={col.key}
                  className={`${headerClass} text-[10px] sm:text-xs`}
                  style={{ color: aggregateColor, width: mobileDataColWidth }}
                  onClick={() => handleHeaderClick(col.key)}
                >
                  {col.label.replace('EVA-A ', '')}
                  <SortIndicator active={sortKey === col.key} dir={sortDir} />
                </th>
              ))}
              {showMetricKeys.map(k => (
                <th
                  key={k}
                  className={`${headerClass} text-text-primary text-[10px] sm:text-xs`}
                  style={{ width: mobileDataColWidth }}
                  onClick={() => handleHeaderClick(k)}
                >
                  {metricLabels[k] || k}
                  <SortIndicator active={sortKey === k} dir={sortDir} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((s) => (
              <tr key={s.id} className="border-b border-border-default/30">
                <td className="py-2 px-2 sticky left-0 bg-bg-secondary z-10 text-xs">
                  <SystemName system={s} componentColors={componentColors} />
                </td>
                {showAggCols.map((col) => {
                  const val = col.getValue(s);
                  const { min, max } = aggRanges[col.key];
                  const { bg, text } = getScaledHeatmapColor(val, min, max, aggregateColor, false, themeColors);
                  return (
                    <td key={col.key} className="py-1 px-0.5 text-center">
                      <div
                        className="rounded-md px-0.5 py-1 font-mono text-[10px] sm:text-xs font-medium"
                        style={{ backgroundColor: bg, color: text }}
                      >
                        {val.toFixed(2)}
                      </div>
                    </td>
                  );
                })}
                {showMetricKeys.map(k => {
                  const val = s[dataKey][k] ?? 0;
                  const { min, max } = metricRanges[k];
                  const invert = invertedMetrics.has(k);
                  const { bg, text } = getScaledHeatmapColor(val, min, max, baseColor, invert, themeColors);
                  return (
                    <td key={k} className="py-1 px-0.5 text-center">
                      <div
                        className="rounded-md px-0.5 py-1 font-mono text-[10px] sm:text-xs font-medium"
                        style={{ backgroundColor: bg, color: text }}
                      >
                        {val.toFixed(2)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
