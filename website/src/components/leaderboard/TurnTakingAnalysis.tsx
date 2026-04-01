import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, ArrowUp, ArrowDown } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { SystemScore } from '../../data/leaderboardData';
import { turnTakingData } from '../../data/turnTakingData';
import type { TurnTakingEntry, TurnLabelPcts, LateTurnBreakdown } from '../../data/turnTakingData';
import { useThemeColors, useThemeMode } from '../../styles/theme';

// ─── Shared constants & utilities ────────────────────────────────────────────

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

const distributionColors = {
  dark:  { onTime: '#34D399', late: '#F68EC4', early: '#F59E0B', indeterminate: '#64748B' },
  light: { onTime: '#059669', late: '#DB2777', early: '#D97706', indeterminate: '#94A3B8' },
};

const breakdownColors = {
  dark:  { withToolCalls: '#37C4DC', withoutToolCalls: '#8D76D4' },
  light: { withToolCalls: '#0891B2', withoutToolCalls: '#7C3AED' },
};

function getComponentColorMap(systems: SystemScore[], isDark: boolean): Map<string, string> {
  const palettes = isDark ? categoryPalettesDark : categoryPalettesLight;

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

type SortDir = 'asc' | 'desc';
type SectionKey = 'allTurns' | 'toolCallTurns' | 'noToolCallTurns';
type ViewKey = SectionKey | 'lateTurnBreakdown';

const sectionLabels: Record<SectionKey, string> = {
  allTurns: 'Overall',
  toolCallTurns: 'Tool Calls',
  noToolCallTurns: 'No Tool Calls',
};

const viewLabels: Record<ViewKey, string> = {
  ...sectionLabels,
  lateTurnBreakdown: 'Late Breakdown',
};

// Shorter labels for mobile view selector
const viewLabelsShort: Record<ViewKey, string> = {
  allTurns: 'All',
  toolCallTurns: 'Tools',
  noToolCallTurns: 'No Tools',
  lateTurnBreakdown: 'Late',
};

const sections: SectionKey[] = ['allTurns', 'toolCallTurns', 'noToolCallTurns'];
const allViews: ViewKey[] = ['allTurns', 'toolCallTurns', 'noToolCallTurns', 'lateTurnBreakdown'];

const systemSortOptions = [
  { key: null, label: 'Default' },
  { key: 'system_stt', label: 'STT' },
  { key: 'system_llm', label: 'LLM' },
  { key: 'system_tts', label: 'TTS' },
] as const;

const metricSortOptions = [
  { key: 'onTime', label: 'On Time %' },
  { key: 'late', label: 'Late %' },
  { key: 'early', label: 'Early / Interruption %' },
  { key: 'indeterminate', label: 'Indeterminate %' },
  { key: 'withToolCalls', label: 'Late w/ Tool Calls %' },
  { key: 'withoutToolCalls', label: 'Late w/o Tool Calls %' },
];

function SortIndicator({ active, dir }: { active: boolean; dir: SortDir }) {
  if (!active) return null;
  return dir === 'desc'
    ? <ArrowDown className="w-3 h-3 inline ml-0.5" />
    : <ArrowUp className="w-3 h-3 inline ml-0.5" />;
}

const ttDataMap = new Map<string, TurnTakingEntry>();
for (const entry of turnTakingData) {
  ttDataMap.set(entry.systemId, entry);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function SystemTickLabel({ x, y, payload, systemLookup, componentColors, textColor }: any) {
  const system: SystemScore | undefined = systemLookup.get(payload.value);
  if (!system) return null;

  if (system.type === 's2s' || system.type === '2-part') {
    if (system.tts !== '-') {
      return (
        <text x={x} y={y} textAnchor="end" dominantBaseline="central" fontSize={14}>
          <tspan fill={componentColors.get(system.llm) || textColor}>{system.llm}</tspan>
          <tspan fill={textColor}>{' + '}</tspan>
          <tspan fill={componentColors.get(system.tts) || textColor}>{system.tts}</tspan>
        </text>
      );
    }
    return (
      <text x={x} y={y} textAnchor="end" dominantBaseline="central" fontSize={14}>
        <tspan fill={componentColors.get(system.llm) || textColor}>{system.llm}</tspan>
      </text>
    );
  }

  return (
    <text x={x} y={y} textAnchor="end" dominantBaseline="central" fontSize={14}>
      <tspan fill={componentColors.get(system.stt) || textColor}>{system.stt}</tspan>
      <tspan fill={textColor}>{' + '}</tspan>
      <tspan fill={componentColors.get(system.llm) || textColor}>{system.llm}</tspan>
      <tspan fill={textColor}>{' + '}</tspan>
      <tspan fill={componentColors.get(system.tts) || textColor}>{system.tts}</tspan>
    </text>
  );
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

// Shared sort hook
function useSortState(systemsWithData: SystemScore[], section: SectionKey) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const handleSort = useCallback((key: string | null) => {
    if (key === null) {
      setSortKey(null);
    } else {
      setSortKey(prev => {
        if (prev === key) {
          setSortDir(d => d === 'desc' ? 'asc' : 'desc');
          return prev;
        }
        setSortDir(key.startsWith('system_') ? 'asc' : 'desc');
        return key;
      });
    }
  }, []);

  const sorted = useMemo(() => {
    if (!sortKey) {
      return [...systemsWithData].sort((a, b) => {
        const aS2S = a.type === 's2s' || a.type === '2-part';
        const bS2S = b.type === 's2s' || b.type === '2-part';
        if (aS2S && !bS2S) return -1;
        if (!aS2S && bS2S) return 1;
        return a.stt.localeCompare(b.stt);
      });
    }

    const getValue = (s: SystemScore): number | string => {
      if (sortKey === 'system_stt') return s.stt;
      if (sortKey === 'system_llm') return s.llm;
      if (sortKey === 'system_tts') return s.tts;
      const entry = ttDataMap.get(s.id);
      if (!entry) return 0;
      if (sortKey === 'withToolCalls') return entry.lateTurnBreakdown.withToolCalls;
      if (sortKey === 'withoutToolCalls') return entry.lateTurnBreakdown.withoutToolCalls;
      const pcts: TurnLabelPcts = entry[section];
      if (sortKey === 'onTime') return pcts.onTime;
      if (sortKey === 'late') return pcts.late;
      if (sortKey === 'early') return pcts.early;
      if (sortKey === 'indeterminate') return pcts.indeterminate;
      return 0;
    };

    return [...systemsWithData].sort((a, b) => {
      const va = getValue(a);
      const vb = getValue(b);
      if (typeof va === 'string' && typeof vb === 'string') {
        return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      return sortDir === 'desc' ? (vb as number) - (va as number) : (va as number) - (vb as number);
    });
  }, [sortKey, sortDir, systemsWithData, section]);

  return { sortKey, sortDir, handleSort, sorted };
}

// Shared sort menu component
function SortMenu({ sortKey, sortDir, onSort }: { sortKey: string | null; sortDir: SortDir; onSort: (key: string | null) => void }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [menuPos, setMenuPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const openMenu = useCallback(() => {
    if (buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      setMenuPos({ top: rect.bottom + 4, left: rect.left });
    }
    setMenuOpen(o => !o);
  }, []);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (
        menuRef.current && !menuRef.current.contains(e.target as Node) &&
        buttonRef.current && !buttonRef.current.contains(e.target as Node)
      ) {
        setMenuOpen(false);
      }
    }
    if (menuOpen) {
      document.addEventListener('mousedown', handleClick);
      return () => document.removeEventListener('mousedown', handleClick);
    }
  }, [menuOpen]);

  function handleClick(key: string | null) {
    onSort(key);
    setMenuOpen(false);
  }

  return (
    <div className="relative">
      <button
        ref={buttonRef}
        onClick={openMenu}
        className="flex items-center gap-1 text-sm text-text-muted hover:text-text-primary transition-colors"
      >
        Sort
        <ChevronDown className="w-3.5 h-3.5" />
        {sortKey && <SortIndicator active dir={sortDir} />}
      </button>
      {menuOpen && createPortal(
        <div
          ref={menuRef}
          className="bg-bg-tertiary border border-border-default rounded-lg shadow-xl py-1 min-w-[160px]"
          style={{ position: 'fixed', top: menuPos.top, left: menuPos.left, zIndex: 9999 }}
        >
          <div className="px-3 py-1 text-xs text-text-muted font-medium">System</div>
          {systemSortOptions.map(opt => (
            <button
              key={opt.key ?? 'default'}
              onClick={() => handleClick(opt.key)}
              className={`block w-full text-left px-3 py-1.5 text-xs hover:bg-bg-hover transition-colors ${sortKey === opt.key || (opt.key === null && sortKey === null) ? 'text-purple-light font-medium' : 'text-text-secondary'}`}
            >
              {opt.label}
            </button>
          ))}
          <div className="border-t border-border-default my-1" />
          <div className="px-3 py-1 text-xs text-text-muted font-medium">Metric</div>
          {metricSortOptions.map(opt => (
            <button
              key={opt.key}
              onClick={() => handleClick(opt.key)}
              className={`block w-full text-left px-3 py-1.5 text-xs hover:bg-bg-hover transition-colors ${sortKey === opt.key ? 'text-purple-light font-medium' : 'text-text-secondary'}`}
            >
              {opt.label}
            </button>
          ))}
        </div>,
        document.body
      )}
    </div>
  );
}

// Shared legend
function DistributionLegend({ colors }: { colors: typeof distributionColors.dark }) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2 sm:gap-x-6 mt-4 text-xs sm:text-sm text-text-primary">
      <span className="flex items-center gap-1.5 sm:gap-2">
        <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-sm" style={{ backgroundColor: colors.onTime }} />
        On Time
      </span>
      <span className="flex items-center gap-1.5 sm:gap-2">
        <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-sm" style={{ backgroundColor: colors.late }} />
        Late
      </span>
      <span className="flex items-center gap-1.5 sm:gap-2">
        <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-sm" style={{ backgroundColor: colors.early }} />
        Early
      </span>
      <span className="flex items-center gap-1.5 sm:gap-2">
        <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-sm" style={{ backgroundColor: colors.indeterminate }} />
        Indeterminate
      </span>
    </div>
  );
}

interface SharedChartProps {
  systems: SystemScore[];
  componentColors: Map<string, string>;
}

// ─── Option B: Toggle/Dropdown ──────────────────────────────────────────────

function BreakdownLegend({ colors }: { colors: typeof breakdownColors.dark }) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2 sm:gap-x-6 mt-4 text-xs sm:text-sm text-text-primary">
      <span className="flex items-center gap-1.5 sm:gap-2">
        <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-sm" style={{ backgroundColor: colors.withToolCalls }} />
        With Tool Calls
      </span>
      <span className="flex items-center gap-1.5 sm:gap-2">
        <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-sm" style={{ backgroundColor: colors.withoutToolCalls }} />
        Without Tool Calls
      </span>
    </div>
  );
}

function ToggleChart({ systems, componentColors }: SharedChartProps) {
  const themeColors = useThemeColors();
  const themeMode = useThemeMode();
  const colors = themeMode === 'light' ? distributionColors.light : distributionColors.dark;
  const bdColors = themeMode === 'light' ? breakdownColors.light : breakdownColors.dark;
  const [activeView, setActiveView] = useState<ViewKey>('allTurns');
  const isBreakdown = activeView === 'lateTurnBreakdown';

  const systemsWithData = useMemo(() => systems.filter(s => ttDataMap.has(s.id)), [systems]);
  // For sorting, use 'allTurns' when in breakdown mode since breakdown doesn't have the 4-label fields
  const sortSection: SectionKey = isBreakdown ? 'allTurns' : activeView as SectionKey;
  const { sortKey, sortDir, handleSort, sorted } = useSortState(systemsWithData, sortSection);

  const systemLookup = useMemo(() => {
    const map = new Map<string, SystemScore>();
    for (const s of systemsWithData) map.set(s.id, s);
    return map;
  }, [systemsWithData]);

  const data = useMemo(() =>
    [...sorted].reverse().map(s => {
      const entry = ttDataMap.get(s.id)!;
      if (isBreakdown) {
        return {
          systemId: s.id,
          withToolCalls: entry.lateTurnBreakdown.withToolCalls,
          withoutToolCalls: entry.lateTurnBreakdown.withoutToolCalls,
        };
      }
      const pcts: TurnLabelPcts = entry[activeView as SectionKey];
      return {
        systemId: s.id,
        onTime: pcts.onTime,
        late: pcts.late,
        early: pcts.early,
        indeterminate: pcts.indeterminate,
      };
    }),
    [sorted, activeView, isBreakdown]
  );

  const chartHeight = data.length * 48 + 100;

  return (
    <div>
      <div className="flex justify-center sm:justify-end mb-4">
        <SortMenu sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
      </div>
      <div className="flex justify-center sm:justify-start mb-4">
        <div className="flex gap-1 bg-bg-tertiary rounded-lg p-1 overflow-x-auto">
          {allViews.map(view => (
            <button
              key={view}
              onClick={() => setActiveView(view)}
              className={`px-2 sm:px-3 py-1.5 text-xs sm:text-sm rounded-md transition-colors whitespace-nowrap ${
                activeView === view
                  ? 'bg-purple-light/20 text-purple-light font-medium'
                  : 'text-text-muted hover:text-text-primary hover:bg-bg-hover'
              }`}
            >
              <span className="sm:hidden">{viewLabelsShort[view]}</span>
              <span className="hidden sm:inline">{viewLabels[view]}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Mobile: Legend at top */}
      <div className="md:hidden mb-3">
        {isBreakdown ? <BreakdownLegend colors={bdColors} /> : <DistributionLegend colors={colors} />}
      </div>

      {/* Desktop: Bar chart */}
      <div className="hidden md:block">
        <ResponsiveContainer width="100%" height={chartHeight}>
          <BarChart data={data} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 25 }}>
            <XAxis
              type="number"
              domain={[0, 100]}
              ticks={[0, 20, 40, 60, 80, 100]}
              allowDataOverflow={true}
              tick={{ fill: themeColors.text.primary, fontSize: 11 }}
              label={{ value: isBreakdown ? '% of late turns' : '% of turns', position: 'insideBottom', offset: -5, fill: themeColors.text.muted, fontSize: 11 }}
            />
            <YAxis
              type="category"
              dataKey="systemId"
              width={280}
              tick={<SystemTickLabel systemLookup={systemLookup} componentColors={componentColors} textColor={themeColors.text.muted} />}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: themeColors.bg.tertiary,
                border: `1px solid ${themeColors.text.muted}`,
                borderRadius: 8,
                color: themeColors.text.primary,
                fontSize: 12,
              }}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={((value: any, name: any) => [`${value}%`, name]) as any}
              labelStyle={{ color: themeColors.text.primary, fontWeight: 600 }}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              labelFormatter={(label: any) => {
                const system = systemLookup.get(label);
                if (!system) return label;
                if (system.type === 's2s' || system.type === '2-part') return system.tts !== '-' ? `${system.llm} + ${system.tts}` : system.llm;
                return `${system.stt} + ${system.llm} + ${system.tts}`;
              }}
              itemSorter={() => 0}
              cursor={false}
            />
            {isBreakdown ? (
              <>
                <Bar dataKey="withToolCalls" name="With Tool Calls" stackId="a" fill={bdColors.withToolCalls} barSize={20} />
                <Bar dataKey="withoutToolCalls" name="Without Tool Calls" stackId="a" fill={bdColors.withoutToolCalls} barSize={20} radius={[0, 4, 4, 0]} />
              </>
            ) : (
              <>
                <Bar dataKey="onTime" name="On Time" stackId="a" fill={colors.onTime} barSize={20} />
                <Bar dataKey="late" name="Late" stackId="a" fill={colors.late} barSize={20} />
                <Bar dataKey="early" name="Early / Interruption" stackId="a" fill={colors.early} barSize={20} />
                <Bar dataKey="indeterminate" name="Indeterminate" stackId="a" fill={colors.indeterminate} barSize={20} radius={[0, 4, 4, 0]} />
              </>
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Mobile: Table with mini bars */}
      <div className="md:hidden overflow-x-auto">
        <table className="w-full text-sm" style={{ tableLayout: 'fixed' }}>
          <thead>
            <tr className="border-b border-border-default">
              <th className="text-left py-2 px-2 text-text-muted font-medium text-xs" style={{ width: '45%' }}>System</th>
              <th className="text-center py-2 px-1 text-text-muted font-medium text-xs" style={{ width: '55%' }}>
                {viewLabels[activeView]}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(s => {
              const entry = ttDataMap.get(s.id);
              if (!entry) return null;
              return (
                <tr key={s.id} className="border-b border-border-default/30">
                  <td className="py-2 px-2 text-xs">
                    <SystemName system={s} componentColors={componentColors} />
                  </td>
                  <td className="py-2 px-1">
                    {isBreakdown ? (
                      <MiniBreakdownBar breakdown={entry.lateTurnBreakdown} colors={bdColors} />
                    ) : (
                      <MiniStackedBar pcts={entry[activeView as SectionKey]} colors={colors} section={activeView as SectionKey} />
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Desktop: Legend at bottom */}
      <div className="hidden md:block">
        {isBreakdown ? <BreakdownLegend colors={bdColors} /> : <DistributionLegend colors={colors} />}
      </div>
    </div>
  );
}

// ─── Option C: Side-by-Side ─────────────────────────────────────────────────

function MiniStackedBar({ pcts, colors, section }: { pcts: TurnLabelPcts; colors: typeof distributionColors.dark; section: SectionKey }) {
  const [hovered, setHovered] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);
  const [tooltipPos, setTooltipPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const themeColors = useThemeColors();

  const handleMouseEnter = useCallback(() => {
    if (barRef.current) {
      const rect = barRef.current.getBoundingClientRect();
      setTooltipPos({ top: rect.top - 8, left: rect.left + rect.width / 2 });
    }
    setHovered(true);
  }, []);

  const segments = [
    { key: 'onTime', value: pcts.onTime, color: colors.onTime, label: 'On Time' },
    { key: 'late', value: pcts.late, color: colors.late, label: 'Late' },
    { key: 'early', value: pcts.early, color: colors.early, label: 'Early / Interruption' },
    { key: 'indeterminate', value: pcts.indeterminate, color: colors.indeterminate, label: 'Indeterminate' },
  ];

  return (
    <>
      <div
        ref={barRef}
        className="flex h-5 rounded-md overflow-hidden cursor-default"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={() => setHovered(false)}
      >
        {segments.map(seg => (
          seg.value > 0 ? (
            <div
              key={seg.key}
              style={{ width: `${seg.value}%`, backgroundColor: seg.color }}
              className="min-w-0"
            />
          ) : null
        ))}
      </div>
      {hovered && createPortal(
        <div
          className="rounded-lg shadow-xl py-2 px-3 text-xs pointer-events-none"
          style={{
            position: 'fixed',
            top: tooltipPos.top,
            left: tooltipPos.left,
            transform: 'translate(-50%, -100%)',
            zIndex: 9999,
            backgroundColor: themeColors.bg.tertiary,
            border: `1px solid ${themeColors.text.muted}`,
            color: themeColors.text.primary,
          }}
        >
          <div className="font-semibold mb-1">{sectionLabels[section]}</div>
          {segments.map(seg => (
            <div key={seg.key} className="flex items-center gap-2 py-0.5">
              <span className="inline-block w-2 h-2 rounded-sm" style={{ backgroundColor: seg.color }} />
              <span>{seg.label}: {seg.value}%</span>
            </div>
          ))}
        </div>,
        document.body
      )}
    </>
  );
}

function MiniBreakdownBar({ breakdown, colors }: { breakdown: LateTurnBreakdown; colors: typeof breakdownColors.dark }) {
  const [hovered, setHovered] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);
  const [tooltipPos, setTooltipPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const themeColors = useThemeColors();

  const handleMouseEnter = useCallback(() => {
    if (barRef.current) {
      const rect = barRef.current.getBoundingClientRect();
      setTooltipPos({ top: rect.top - 8, left: rect.left + rect.width / 2 });
    }
    setHovered(true);
  }, []);

  const segments = [
    { key: 'withToolCalls', value: breakdown.withToolCalls, color: colors.withToolCalls, label: 'With Tool Calls' },
    { key: 'withoutToolCalls', value: breakdown.withoutToolCalls, color: colors.withoutToolCalls, label: 'Without Tool Calls' },
  ];

  return (
    <>
      <div
        ref={barRef}
        className="flex h-5 rounded-md overflow-hidden cursor-default"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={() => setHovered(false)}
      >
        {segments.map(seg => (
          seg.value > 0 ? (
            <div
              key={seg.key}
              style={{ width: `${seg.value}%`, backgroundColor: seg.color }}
              className="min-w-0"
            />
          ) : null
        ))}
      </div>
      {hovered && createPortal(
        <div
          className="rounded-lg shadow-xl py-2 px-3 text-xs pointer-events-none"
          style={{
            position: 'fixed',
            top: tooltipPos.top,
            left: tooltipPos.left,
            transform: 'translate(-50%, -100%)',
            zIndex: 9999,
            backgroundColor: themeColors.bg.tertiary,
            border: `1px solid ${themeColors.text.muted}`,
            color: themeColors.text.primary,
          }}
        >
          <div className="font-semibold mb-1">Late Turn Breakdown</div>
          {segments.map(seg => (
            <div key={seg.key} className="flex items-center gap-2 py-0.5">
              <span className="inline-block w-2 h-2 rounded-sm" style={{ backgroundColor: seg.color }} />
              <span>{seg.label}: {seg.value}%</span>
            </div>
          ))}
        </div>,
        document.body
      )}
    </>
  );
}

function SideBySideChart({ systems, componentColors }: SharedChartProps) {
  const themeMode = useThemeMode();
  const colors = themeMode === 'light' ? distributionColors.light : distributionColors.dark;
  const bdColors = themeMode === 'light' ? breakdownColors.light : breakdownColors.dark;

  const systemsWithData = useMemo(() => systems.filter(s => ttDataMap.has(s.id)), [systems]);
  const { sortKey, sortDir, handleSort, sorted } = useSortState(systemsWithData, 'allTurns');

  // Combined legend for SideBySideChart
  const SideBySideLegend = () => (
    <div className="flex flex-wrap items-center justify-center gap-x-3 sm:gap-x-6 gap-y-2 text-xs sm:text-sm text-text-primary">
      <span className="flex items-center gap-1.5 sm:gap-2">
        <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-sm" style={{ backgroundColor: colors.onTime }} />
        On Time
      </span>
      <span className="flex items-center gap-1.5 sm:gap-2">
        <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-sm" style={{ backgroundColor: colors.late }} />
        Late
      </span>
      <span className="flex items-center gap-1.5 sm:gap-2">
        <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-sm" style={{ backgroundColor: colors.early }} />
        Early
      </span>
      <span className="flex items-center gap-1.5 sm:gap-2">
        <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-sm" style={{ backgroundColor: colors.indeterminate }} />
        Indeterminate
      </span>
      <span className="hidden sm:inline text-text-muted">|</span>
      <span className="flex items-center gap-1.5 sm:gap-2">
        <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-sm" style={{ backgroundColor: bdColors.withToolCalls }} />
        <span className="hidden sm:inline">Late w/</span><span className="sm:hidden">w/</span> Tool Calls
      </span>
      <span className="flex items-center gap-1.5 sm:gap-2">
        <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-sm" style={{ backgroundColor: bdColors.withoutToolCalls }} />
        <span className="hidden sm:inline">Late w/o</span><span className="sm:hidden">w/o</span> Tool Calls
      </span>
    </div>
  );

  return (
    <div>
      <div className="flex justify-center sm:justify-end mb-4">
        <SortMenu sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
      </div>
      {/* Mobile: Legend at top */}
      <div className="md:hidden mb-3">
        <SideBySideLegend />
      </div>
      {/* Desktop table - all columns */}
      <div className="hidden md:block overflow-x-auto">
        <table className="w-full text-sm" style={{ tableLayout: 'fixed' }}>
          <thead>
            <tr className="border-b border-border-default">
              <th className="text-left py-2 px-3 text-text-muted font-medium text-xs" style={{ width: '36%' }}>System</th>
              {sections.map(sec => (
                <th key={sec} className="text-center py-2 px-2 text-text-muted font-medium text-xs" style={{ width: '16%' }}>
                  {sectionLabels[sec]}
                </th>
              ))}
              <th className="text-center py-2 px-2 text-text-muted font-medium text-xs" style={{ width: '16%' }}>
                Late Breakdown
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(s => {
              const entry = ttDataMap.get(s.id);
              if (!entry) return null;
              return (
                <tr key={s.id} className="border-b border-border-default/30">
                  <td className="py-3 px-3 whitespace-nowrap">
                    <SystemName system={s} componentColors={componentColors} />
                  </td>
                  {sections.map(sec => (
                    <td key={sec} className="py-3 px-2">
                      <MiniStackedBar pcts={entry[sec]} colors={colors} section={sec} />
                    </td>
                  ))}
                  <td className="py-3 px-2">
                    <MiniBreakdownBar breakdown={entry.lateTurnBreakdown} colors={bdColors} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {/* Mobile table - simplified with just Overall and Late Breakdown */}
      <div className="md:hidden overflow-x-auto">
        <table className="w-full text-sm" style={{ tableLayout: 'fixed' }}>
          <thead>
            <tr className="border-b border-border-default">
              <th className="text-left py-2 px-2 text-text-muted font-medium text-xs" style={{ width: '45%' }}>System</th>
              <th className="text-center py-2 px-1 text-text-muted font-medium text-xs" style={{ width: '27.5%' }}>
                Overall
              </th>
              <th className="text-center py-2 px-1 text-text-muted font-medium text-xs" style={{ width: '27.5%' }}>
                Late Breakdown
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(s => {
              const entry = ttDataMap.get(s.id);
              if (!entry) return null;
              return (
                <tr key={s.id} className="border-b border-border-default/30">
                  <td className="py-2 px-2 text-xs">
                    <SystemName system={s} componentColors={componentColors} />
                  </td>
                  <td className="py-2 px-1">
                    <MiniStackedBar pcts={entry.allTurns} colors={colors} section="allTurns" />
                  </td>
                  <td className="py-2 px-1">
                    <MiniBreakdownBar breakdown={entry.lateTurnBreakdown} colors={bdColors} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {/* Desktop: Legend at bottom */}
      <div className="hidden md:block mt-4">
        <SideBySideLegend />
      </div>
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────────────────

type LayoutMode = 'sideBySide' | 'toggle';

interface TurnTakingAnalysisProps {
  systems: SystemScore[];
}

export function TurnTakingAnalysis({ systems }: TurnTakingAnalysisProps) {
  const themeMode = useThemeMode();
  const isDark = themeMode !== 'light';
  const componentColors = useMemo(() => getComponentColorMap(systems, isDark), [systems, isDark]);
  const [isOpen, setIsOpen] = useState(false);
  const [layout, setLayout] = useState<LayoutMode>('sideBySide');

  return (
    <div className="bg-bg-secondary rounded-xl border border-border-default">
      <button
        onClick={() => setIsOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 sm:px-6 py-4 text-left hover:bg-bg-hover/50 transition-colors rounded-xl"
      >
        <h3 className="text-base sm:text-lg font-semibold text-text-primary">Turn-Taking Analysis</h3>
        <ChevronDown className={`w-5 h-5 text-text-muted transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      {isOpen && (
        <div className="px-4 sm:px-6 pb-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
            <p className="text-xs sm:text-sm text-text-secondary text-center sm:text-left">Distribution of turn-taking labels across systems. Hover/tap for details.</p>
            <div className="flex justify-center sm:justify-end">
              <div className="flex gap-1 bg-bg-tertiary rounded-lg p-1">
                <button
                  onClick={() => setLayout('sideBySide')}
                  className={`px-2 sm:px-3 py-1 text-xs rounded-md transition-colors whitespace-nowrap ${
                    layout === 'sideBySide'
                      ? 'bg-purple-light/20 text-purple-light font-medium'
                      : 'text-text-muted hover:text-text-primary hover:bg-bg-hover'
                  }`}
                >
                  Side-by-Side
                </button>
                <button
                  onClick={() => setLayout('toggle')}
                  className={`px-2 sm:px-3 py-1 text-xs rounded-md transition-colors whitespace-nowrap ${
                    layout === 'toggle'
                      ? 'bg-purple-light/20 text-purple-light font-medium'
                      : 'text-text-muted hover:text-text-primary hover:bg-bg-hover'
                  }`}
                >
                  Toggle View
                </button>
              </div>
            </div>
          </div>
          {layout === 'sideBySide' && <SideBySideChart systems={systems} componentColors={componentColors} />}
          {layout === 'toggle' && <ToggleChart systems={systems} componentColors={componentColors} />}
        </div>
      )}
    </div>
  );
}
