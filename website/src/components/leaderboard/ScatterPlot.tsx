import { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, useXAxisScale, useYAxisScale } from 'recharts';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { SystemScore } from '../../data/leaderboardData';
import { useThemeColors } from '../../styles/theme';

/** Renders "EVA-A" + subscript in HTML */
function Sub({ base, sub }: { base: string; sub: string }) {
  return <>{base}<sub className="text-[0.7em]">{sub}</sub></>;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: SystemScore & { plotX: number; plotY: number } }>;
  xSub: string;
  ySub: string;
}

function CustomTooltip({ active, payload, xSub, ySub }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  const s = payload[0].payload;
  const typeLabel = s.type === 'cascade' ? 'Cascade' : s.type === '2-part' ? '2-Part' : 'Speech-to-Speech';
  const typeColor = s.type === 'cascade' ? 'bg-purple/20 text-purple-light' : s.type === '2-part' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-blue/20 text-blue-light';
  return (
    <div className="bg-bg-tertiary border border-border-default rounded-lg p-3 shadow-xl max-w-xs">
      <div className="text-sm font-semibold text-text-primary mb-1">{s.name}</div>
      <div className="flex gap-4 text-xs">
        <div><span className="text-text-muted"><Sub base="EVA-A" sub={xSub} />:</span> <span className="text-purple-light font-mono">{s.plotX.toFixed(2)}</span></div>
        <div><span className="text-text-muted"><Sub base="EVA-X" sub={ySub} />:</span> <span className="text-blue-light font-mono">{s.plotY.toFixed(2)}</span></div>
      </div>
      {s.type === 'cascade' && (
        <div className="text-[10px] text-text-muted mt-1.5 space-y-0.5">
          <div>STT: {s.stt}</div>
          <div>LLM: {s.llm}</div>
          <div>TTS: {s.tts}</div>
        </div>
      )}
      <div className="mt-1.5">
        <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${typeColor}`}>
          {typeLabel}
        </span>
      </div>
    </div>
  );
}

/** SVG axis label with subscript via <tspan> */
function AxisLabel({ x, y, base, sub, suffix, fill, angle }: { x: number; y: number; base: string; sub: string; suffix?: string; fill: string; angle?: number }) {
  return (
    <text
      x={x}
      y={y}
      fill={fill}
      fontSize={20}
      fontWeight={600}
      textAnchor="middle"
      transform={angle ? `rotate(${angle}, ${x}, ${y})` : undefined}
    >
      {base}
      <tspan fontSize={14} dy={5}>{sub}</tspan>
      {suffix && <tspan fontSize={20} dy={-5}>{suffix}</tspan>}
    </text>
  );
}

const axisTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];

const cascadeColor = '#A78BFA';
const cascadeStroke = '#C4B5FD';
const s2sColor = '#F59E0B';
const s2sStroke = '#FBBF24';
const twoPartColor = '#34D399';
const twoPartStroke = '#6EE7B7';
const frontierColor = '#06B6D4';

interface ScatterPoint extends SystemScore {
  plotX: number;
  plotY: number;
}

function computeParetoFrontier(points: ScatterPoint[]): { plotX: number; plotY: number }[] {
  const frontier: ScatterPoint[] = [];
  for (const p of points) {
    const dominated = points.some(
      q => q.plotY >= p.plotY && q.plotX >= p.plotX && (q.plotY > p.plotY || q.plotX > p.plotX)
    );
    if (!dominated) frontier.push(p);
  }
  return frontier
    .sort((a, b) => a.plotX - b.plotX)
    .map(p => ({ plotX: p.plotX, plotY: p.plotY }));
}

/** Renders Pareto frontier as a raw SVG polyline using Recharts 3 scale hooks. */
function ParetoLine({ frontier }: { frontier: { plotX: number; plotY: number }[] }) {
  const xScale = useXAxisScale();
  const yScale = useYAxisScale();

  if (!xScale || !yScale || frontier.length < 2) return null;

  const points = frontier
    .map(p => `${xScale(p.plotX)},${yScale(p.plotY)}`)
    .join(' ');

  return (
    <polyline
      points={points}
      fill="none"
      stroke={frontierColor}
      strokeWidth={2}
      strokeDasharray="6 3"
      pointerEvents="none"
    />
  );
}

interface PlotConfig {
  title: string;
  description: React.ReactNode;
  subscript: string;
  getX: (s: SystemScore) => number;
  getY: (s: SystemScore) => number;
  domain: [number, number];
}

const plots: PlotConfig[] = [
  {
    title: 'pass@1',
    description: (<>
      Average of per-sample scores, where each sample scores 1 if all metrics in category surpass metric-specific threshold, else 0.
    </>),
    subscript: 'pass@1',
    getX: (s) => s.successRates.accuracy.pass_threshold,
    getY: (s) => s.successRates.experience.pass_threshold,
    domain: [0, 1],
  },
  {
    title: 'pass@k (k=3)',
    description: (<>
      Percent of scenarios where at least 1 of k=3 trials surpasses metric-specific thresholds in all metrics in the category. .
    </>),
    subscript: 'pass@k',
    getX: (s) => s.successRates.accuracy.pass_at_k,
    getY: (s) => s.successRates.experience.pass_at_k,
    domain: [0, 1],
  },
  {
    title: 'pass^k (k=3)',
    description: (<>
      Per-scenario probability of all k=3 trials succeeding (scenario pass rate raised to the k-th power) for that category, averaged across scenarios.
    </>),
    subscript: 'pass^k',
    getX: (s) => s.successRates.accuracy.pass_k,
    getY: (s) => s.successRates.experience.pass_k,
    domain: [0, 1],
  },
  {
    title: 'Mean',
    description: (<>
      Average of per-sample scores, where each sample's score is the mean of the submetrics in that category.
    </>),
    subscript: 'mean',
    getX: (s) => s.successRates.accuracy.mean,
    getY: (s) => s.successRates.experience.mean,
    domain: [0, 1],
  },
];

function getPointColor(type: string): { fill: string; stroke: string } {
  if (type === '2-part') return { fill: twoPartColor, stroke: twoPartStroke };
  if (type === 's2s') return { fill: s2sColor, stroke: s2sStroke };
  return { fill: cascadeColor, stroke: cascadeStroke };
}

interface ScatterPlotProps {
  systems: SystemScore[];
}

export function ScatterPlot({ systems }: ScatterPlotProps) {
  const colors = useThemeColors();
  const [index, setIndex] = useState(0);
  const plot = plots[index];
  const data: ScatterPoint[] = systems.map(s => ({ ...s, plotX: plot.getX(s), plotY: plot.getY(s) }));
  const frontierLine = computeParetoFrontier(data);

  const prev = () => setIndex((i) => (i - 1 + plots.length) % plots.length);
  const next = () => setIndex((i) => (i + 1) % plots.length);

  return (
    <div className="bg-bg-secondary rounded-xl border border-border-default p-6">
      {/* Dots indicator at top */}
      <div className="flex justify-center gap-2 mb-6">
        {plots.map((p, i) => (
          <button
            key={i}
            onClick={() => setIndex(i)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${i === index ? 'bg-purple/20 text-purple-light' : 'bg-bg-hover text-text-muted hover:text-text-secondary'}`}
          >
            {p.title}
          </button>
        ))}
      </div>

      {/* Header with nav */}
      <div className="flex items-center justify-between mb-6">
        <button
          onClick={prev}
          className="p-2 rounded-lg hover:bg-bg-hover transition-colors text-text-muted hover:text-text-primary"
        >
          <ChevronLeft className="w-6 h-6" />
        </button>
        <div className="text-center flex-1 px-4">
          <h3 className="text-xl font-semibold text-text-primary mb-2">{plot.title}</h3>
          <p className="text-sm text-text-muted leading-loose max-w-xl mx-auto">{plot.description}</p>
        </div>
        <button
          onClick={next}
          className="p-2 rounded-lg hover:bg-bg-hover transition-colors text-text-muted hover:text-text-primary"
        >
          <ChevronRight className="w-6 h-6" />
        </button>
      </div>

      {/* Chart + legend side by side */}
      <div className="flex items-center gap-6 max-w-4xl mx-auto">
        {/* Chart */}
        <div className="flex-1">
          <div style={{ width: '100%', aspectRatio: '1' }} className="[&_.recharts-surface]:overflow-visible">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 15, right: 20, bottom: 60, left: 45 }} style={{ overflow: 'visible' }}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.bg.tertiary} />
                <XAxis
                  type="number"
                  dataKey="plotX"
                  domain={plot.domain}
                  ticks={axisTicks}
                  allowDataOverflow={true}
                  tickFormatter={(v: number) => v.toFixed(1)}
                  stroke={colors.text.muted}
                  tick={{ fill: colors.text.secondary, fontSize: 12 }}
                  label={({ viewBox }) => {
                    const { x, y, width } = viewBox as { x: number; y: number; width: number };
                    return <AxisLabel x={x + width / 2} y={y + 50} base="Accuracy (EVA-A" sub={plot.subscript} suffix=")" fill={colors.accent.purpleLight} />;
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="plotY"
                  domain={plot.domain}
                  ticks={axisTicks}
                  allowDataOverflow={true}
                  tickFormatter={(v: number) => v.toFixed(1)}
                  stroke={colors.text.muted}
                  tick={{ fill: colors.text.secondary, fontSize: 12 }}
                  label={({ viewBox }) => {
                    const { x, y, height } = viewBox as { x: number; y: number; height: number };
                    return <AxisLabel x={x - 8} y={y + height / 2} base="Experience (EVA-X" sub={plot.subscript} suffix=")" fill={colors.accent.blueLight} angle={-90} />;
                  }}
                />
                <Tooltip content={<CustomTooltip xSub={plot.subscript} ySub={plot.subscript} />} cursor={false} />
                <ParetoLine frontier={frontierLine} />
                <Scatter data={data} fill={cascadeColor}>
                  {data.map((s) => {
                    const { fill, stroke } = getPointColor(s.type);
                    return (
                      <Cell
                        key={s.id}
                        fill={fill}
                        stroke={stroke}
                        strokeWidth={1.5}
                        r={8}
                      />
                    );
                  })}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Legend on the side */}
        <div className="flex flex-col gap-3 flex-shrink-0 pr-2">
          <div className="flex items-center gap-2 text-sm text-text-secondary">
            <div className="w-3.5 h-3.5 rounded-full flex-shrink-0" style={{ backgroundColor: cascadeColor }} />
            <span className="whitespace-nowrap">Cascade (STT + LLM + TTS)</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-text-secondary">
            <div className="w-3.5 h-3.5 rounded-full flex-shrink-0" style={{ backgroundColor: twoPartColor }} />
            <span className="whitespace-nowrap">Audio Native (AudioLLM + TTS)</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-text-secondary">
            <div className="w-3.5 h-3.5 rounded-full flex-shrink-0" style={{ backgroundColor: s2sColor }} />
            <span className="whitespace-nowrap">Speech-to-Speech</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-text-secondary">
            <div className="w-6 h-0 border-t-2 border-dashed flex-shrink-0" style={{ borderColor: frontierColor }} />
            <span className="whitespace-nowrap">Pareto Frontier</span>
          </div>
        </div>
      </div>
    </div>
  );
}
