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

// Palette of distinct colors — ordered so adjacent colors always contrast (warm/cool alternating)
const componentPaletteDark = [
  '#F59E0B', '#38BDF8', '#34D399', '#A78BFA',  // amber, sky, emerald, purple
  '#F87171', '#22D3EE', '#FB923C', '#818CF8',  // red, cyan, orange, indigo
  '#F472B6', '#4ADE80', '#FACC15', '#2DD4BF',  // pink, green, yellow, teal
  '#C084FC', '#FB7185', '#67E8F9', '#A3E635',  // violet, rose, light-cyan, lime
];

const componentPaletteLight = [
  '#B45309', '#0369A1', '#047857', '#6D28D9',  // amber, sky, emerald, purple
  '#B91C1C', '#0E7490', '#C2410C', '#4338CA',  // red, cyan, orange, indigo
  '#BE185D', '#15803D', '#A16207', '#0D9488',  // pink, green, yellow, teal
  '#7C3AED', '#E11D48', '#0891B2', '#65A30D',  // violet, rose, light-cyan, lime
];

function getComponentColorMap(systems: SystemScore[], palette: string[]): Map<string, string> {
  const allComponents = new Set<string>();
  for (const s of systems) {
    if (s.stt !== '-') allComponents.add(s.stt);
    allComponents.add(s.llm);
    if (s.tts !== '-') allComponents.add(s.tts);
  }
  const map = new Map<string, string>();
  let i = 0;
  for (const name of allComponents) {
    map.set(name, palette[i % palette.length]);
    i++;
  }
  return map;
}

function SystemName({ system, componentColors }: { system: SystemScore; componentColors: Map<string, string> }) {
  if (system.type === 's2s' || system.type === '2-part') {
    if (system.tts !== '-') {
      return (
        <span className="text-sm leading-relaxed">
          <span style={{ color: componentColors.get(system.llm) }}>{system.llm}</span>
          <span className="text-text-muted"> + </span>
          <span style={{ color: componentColors.get(system.tts) }}>{system.tts}</span>
        </span>
      );
    }
    const color = componentColors.get(system.llm) || '#F1F5F9';
    return <span style={{ color }}>{system.llm}</span>;
  }
  return (
    <span className="text-sm leading-relaxed">
      <span style={{ color: componentColors.get(system.stt) }}>{system.stt}</span>
      <span className="text-text-muted"> + </span>
      <span style={{ color: componentColors.get(system.llm) }}>{system.llm}</span>
      <span className="text-text-muted"> + </span>
      <span style={{ color: componentColors.get(system.tts) }}>{system.tts}</span>
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
  const palette = themeMode === 'light' ? componentPaletteLight : componentPaletteDark;
  const componentColors = useMemo(() => getComponentColorMap(systems, palette), [systems, palette]);

  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [systemMenuOpen, setSystemMenuOpen] = useState(false);
  const [menuPos, setMenuPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

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

  return (
    <div className="bg-bg-secondary rounded-xl border border-border-default p-6">
      <h3 className="text-lg font-semibold text-text-primary mb-1">{title}</h3>
      <p className="text-sm text-text-secondary mb-4">{description}</p>
      <div className="overflow-x-auto">
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
    </div>
  );
}
