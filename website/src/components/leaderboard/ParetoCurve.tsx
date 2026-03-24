import {
  Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, Line, ComposedChart,
} from 'recharts';
import { ossSystems as systems } from '../../data/leaderboardData';
import type { SystemScore } from '../../data/leaderboardData';
import { useThemeColors } from '../../styles/theme';

const cascadeColor = '#A78BFA';
const cascadeStroke = '#C4B5FD';
const s2sColor = '#F59E0B';
const s2sStroke = '#FBBF24';
const frontierColor = '#06B6D4';

interface ScatterPoint extends SystemScore {
  plotX: number;
  plotY: number;
}

function computeParetoFrontier(points: ScatterPoint[]): ScatterPoint[] {
  // Find non-dominated points (maximizing both EVA-A and EVA-X)
  const frontier: ScatterPoint[] = [];
  for (const p of points) {
    const dominated = points.some(
      q => q.plotY >= p.plotY && q.plotX >= p.plotX && (q.plotY > p.plotY || q.plotX > p.plotX)
    );
    if (!dominated) frontier.push(p);
  }
  // Sort by EVA-A descending for line drawing
  return frontier.sort((a, b) => b.plotY - a.plotY);
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: ScatterPoint }> }) {
  if (!active || !payload?.length) return null;
  const s = payload[0].payload;
  return (
    <div className="bg-bg-tertiary border border-border-default rounded-lg p-3 shadow-xl max-w-xs">
      <div className="text-sm font-semibold text-text-primary mb-1">{s.name}</div>
      <div className="flex gap-4 text-xs">
        <div>
          <span className="text-text-muted">EVA-A<sub className="text-[0.7em]">pass</sub>:</span>{' '}
          <span className="text-purple-light font-mono">{s.plotY.toFixed(4)}</span>
        </div>
        <div>
          <span className="text-text-muted">EVA-X<sub className="text-[0.7em]">pass</sub>:</span>{' '}
          <span className="text-blue-light font-mono">{s.plotX.toFixed(4)}</span>
        </div>
      </div>
      {s.type === 'cascade' && (
        <div className="text-[10px] text-text-muted mt-1.5 space-y-0.5">
          <div>STT: {s.stt}</div>
          <div>LLM: {s.llm}</div>
          <div>TTS: {s.tts}</div>
        </div>
      )}
      <div className="mt-1.5">
        <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${s.type === 'cascade' ? 'bg-purple/20 text-purple-light' : 'bg-blue/20 text-blue-light'}`}>
          {s.type === 'cascade' ? 'Cascade' : 'Speech-to-Speech'}
        </span>
      </div>
    </div>
  );
}

function AxisLabel({ x, y, base, sub, fill, angle }: { x: number; y: number; base: string; sub: string; fill: string; angle?: number }) {
  return (
    <text x={x} y={y} fill={fill} fontSize={20} fontWeight={600} textAnchor="middle"
      transform={angle ? `rotate(${angle}, ${x}, ${y})` : undefined}>
      {base}<tspan fontSize={14} dy={5}>{sub}</tspan>
    </text>
  );
}

export function ParetoCurve() {
  const colors = useThemeColors();
  const data: ScatterPoint[] = systems.map(s => ({
    ...s,
    plotX: s.evaX,
    plotY: s.evaA,
  }));

  const frontier = computeParetoFrontier(data);
  const frontierLine = frontier.map(p => ({ plotX: p.plotX, plotY: p.plotY }));

  return (
    <div className="bg-bg-secondary rounded-xl border border-border-default p-6">
      <div className="text-center mb-6">
        <h3 className="text-xl font-semibold text-text-primary mb-2">
          Accuracy vs Experience Pareto Frontier
        </h3>
        <p className="text-sm text-text-muted leading-loose max-w-2xl mx-auto">
          The Pareto frontier connects models where no other model scores higher on <span className="text-purple-light">both</span> accuracy and experience.
          Models on the frontier represent the best accuracy–experience tradeoffs.
        </p>
      </div>

      <div className="flex items-center gap-6 max-w-4xl mx-auto">
        <div className="flex-1">
          <div style={{ width: '100%', aspectRatio: '1' }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart margin={{ top: 15, right: 20, bottom: 60, left: 45 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.bg.tertiary} />
                <XAxis
                  type="number"
                  dataKey="plotX"
                  domain={[0, 0.55]}
                  tickFormatter={(v: number) => v.toFixed(1)}
                  stroke={colors.text.muted}
                  tick={{ fill: colors.text.secondary, fontSize: 12 }}
                  label={({ viewBox }) => {
                    const { x, y, width } = viewBox as { x: number; y: number; width: number };
                    return <AxisLabel x={x + width / 2} y={y + 50} base="EVA-X" sub="pass" fill={colors.accent.blueLight} />;
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="plotY"
                  domain={[0, 0.7]}
                  tickFormatter={(v: number) => v.toFixed(1)}
                  stroke={colors.text.muted}
                  tick={{ fill: colors.text.secondary, fontSize: 12 }}
                  label={({ viewBox }) => {
                    const { x, y, height } = viewBox as { x: number; y: number; height: number };
                    return <AxisLabel x={x - 8} y={y + height / 2} base="EVA-A" sub="pass" fill={colors.accent.purpleLight} angle={-90} />;
                  }}
                />
                <Tooltip content={<CustomTooltip />} />

                {/* Pareto frontier line */}
                <Line
                  data={frontierLine}
                  type="linear"
                  dataKey="plotY"
                  stroke={frontierColor}
                  strokeWidth={2}
                  strokeDasharray="6 3"
                  dot={false}
                  isAnimationActive={false}
                />

                {/* Scatter points */}
                <Scatter data={data} fill={cascadeColor}>
                  {data.map((s) => (
                    <Cell
                      key={s.id}
                      fill={s.type === 'cascade' ? cascadeColor : s2sColor}
                      stroke={s.type === 'cascade' ? cascadeStroke : s2sStroke}
                      strokeWidth={1.5}
                      r={8}
                    />
                  ))}
                </Scatter>
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-col gap-3 flex-shrink-0 pr-2">
          <div className="flex items-center gap-2 text-sm text-text-secondary">
            <div className="w-3.5 h-3.5 rounded-full flex-shrink-0" style={{ backgroundColor: cascadeColor }} />
            <span className="whitespace-nowrap">Cascade</span>
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
