import type React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ErrorBar, ReferenceLine, Customized, useXAxisScale, useYAxisScale } from 'recharts';
import type { SystemStats, DomainOrPooled } from '../../data/leaderboardData';
import { getPertValue, perturbations, perturbationLabels, groupedSystems } from '../../data/leaderboardData';
import { useThemeColors } from '../../styles/theme';

interface PerturbationBarChartProps {
  metric: string;
  metricLabel: string;
  systems: SystemStats[];
  domain: DomainOrPooled;
}

interface ChartRow {
  name: string;
  type: SystemStats['type'];
  [key: string]: string | number | [number, number] | boolean | null | undefined;
}

interface TooltipPayloadItem {
  dataKey: string;
  value: number;
  color: string;
  payload: ChartRow;
}

interface TooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: string;
}

function CustomTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-tertiary border border-border-default rounded-lg p-3 shadow-xl max-w-xs">
      <div className="text-sm font-semibold text-text-primary mb-2">{label}</div>
      <div className="flex flex-col gap-1 text-xs">
        {payload.map((item) => {
          // dataKey of the form `<pert>_point`
          const pertKey = item.dataKey.replace(/_point$/, '');
          const sig = item.payload[`${pertKey}_sig`] as boolean | undefined;
          const err = item.payload[`${pertKey}_err`] as [number, number] | undefined;
          if (item.value === null || item.value === undefined || Number.isNaN(item.value)) return null;
          const lower = err ? item.value - err[0] : item.value;
          const upper = err ? item.value + err[1] : item.value;
          return (
            <div key={item.dataKey} className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ backgroundColor: item.color }} />
              <span className="text-text-muted">{perturbationLabels[pertKey] ?? pertKey}:</span>
              <span className="font-mono text-text-primary">
                {item.value >= 0 ? '+' : ''}{item.value.toFixed(3)}
                {sig ? <span className="text-amber-400 ml-0.5">*</span> : null}
              </span>
              <span className="font-mono text-text-muted">
                [{lower.toFixed(2)}, {upper.toFixed(2)}]
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const PERT_COLORS: Record<string, keyof ReturnType<typeof useThemeColors>['accent']> = {
  accent: 'amber',
  background_noise: 'cyan',
  both: 'purple',
};

function colorFor(pert: string, colors: ReturnType<typeof useThemeColors>): string {
  const key = PERT_COLORS[pert];
  if (key) return colors.accent[key];
  // Fallback rotation
  return colors.accent.blue;
}

function tierLabel(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return '';
  if (p < 0.001) return '***';
  if (p < 0.01) return '**';
  if (p < 0.05) return '*';
  return '';
}

interface BandScale {
  (v: string): number | undefined;
  bandwidth?: () => number;
}

function StarsLayer({ data, amberColor }: { data: ChartRow[]; amberColor: string }) {
  const xScale = useXAxisScale() as BandScale | undefined;
  const yScale = useYAxisScale() as ((v: number) => number | undefined) | undefined;
  if (!xScale || !yScale || typeof xScale.bandwidth !== 'function') return null;
  const bandwidth = xScale.bandwidth();
  const n = perturbations.length;
  const elements: React.ReactElement[] = [];
  data.forEach((row) => {
    const bandStart = xScale(row.name as string);
    if (bandStart == null) return;
    perturbations.forEach((pert, i) => {
      const label = row[`${pert}_sig_label`] as string | undefined;
      if (!label) return;
      const point = row[`${pert}_point`] as number | null | undefined;
      const err = row[`${pert}_err`] as [number, number] | undefined;
      if (point == null || !err) return;
      const cx = bandStart + (bandwidth * (i + 0.5)) / n;
      const isPos = point >= 0;
      const yTip = isPos
        ? (yScale(point + err[1]) ?? 0) - 6
        : (yScale(point - err[0]) ?? 0) + 14;
      elements.push(
        <text
          key={`sig-${row.name}-${pert}`}
          x={cx}
          y={yTip}
          fill={amberColor}
          fontSize={14}
          fontWeight={700}
          textAnchor="middle"
        >
          {label}
        </text>,
      );
    });
  });
  return <g>{elements}</g>;
}

export function PerturbationBarChart({ metric, metricLabel, systems, domain }: PerturbationBarChartProps) {
  const colors = useThemeColors();

  // Order systems by architecture group: S2S → Hybrid (2-part) → Cascade.
  const ordered = groupedSystems(systems);

  // Build data rows: one per system that has any perturbation data for this metric.
  const data: ChartRow[] = ordered.flatMap((s) => {
    const row: ChartRow = { name: s.name, type: s.type };
    let any = false;
    for (const p of perturbations) {
      const v = getPertValue(s, metric, p, domain);
      if (v) {
        const label = tierLabel(v.corrected_p);
        row[`${p}_point`] = v.point;
        row[`${p}_err`] = [v.point - v.ci_lower, v.ci_upper - v.point];
        row[`${p}_sig`] = label !== '';
        row[`${p}_sig_label`] = label;
        any = true;
      } else {
        row[`${p}_point`] = null;
        row[`${p}_err`] = undefined;
        row[`${p}_sig`] = false;
        row[`${p}_sig_label`] = '';
      }
    }
    return any ? [row] : [];
  });

  if (data.length === 0) {
    return (
      <div className="text-sm text-text-muted italic px-4 py-6">
        No perturbation data available for {metricLabel} at this domain.
      </div>
    );
  }

  // Compute group boundary indices: positions where the type changes from the previous row.
  // The ReferenceLine x value is the `name` of the first row in the new group; recharts will
  // draw the line at that category's tick.
  const separators: string[] = [];
  for (let i = 1; i < data.length; i++) {
    if (data[i].type !== data[i - 1].type) separators.push(data[i].name);
  }

  const minWidth = Math.max(720, data.length * 80);

  return (
    <div>
      <div className="overflow-x-auto">
        <div className="h-[440px]" style={{ minWidth: `${minWidth}px` }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 24, right: 16, bottom: 70, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={colors.bg.tertiary} />
              <XAxis
                dataKey="name"
                stroke={colors.text.muted}
                tick={{ fill: colors.text.secondary, fontSize: 10 }}
                tickFormatter={(v: string) =>
                  v.startsWith('Scribe v2.2 Realtime')
                    ? 'Scribe + Gemini 3 Flash + Conversational v3'
                    : v
                }
                interval={0}
                angle={-30}
                textAnchor="end"
                height={80}
              />
              <YAxis
                stroke={colors.text.muted}
                tick={{ fill: colors.text.secondary, fontSize: 11 }}
                domain={[-0.5, 0.5]}
                ticks={[-0.5, -0.25, 0, 0.25, 0.5]}
                tickFormatter={(v: number) => v.toFixed(2)}
                allowDataOverflow
                width={56}
                label={{ value: 'Δ vs clean', angle: -90, position: 'insideLeft', offset: 0, fill: colors.text.secondary, style: { fontSize: 12 } }}
              />
              <ReferenceLine y={0} stroke={colors.text.muted} />
              <Customized
                component={(props: unknown) => {
                  const p = props as {
                    xAxisMap?: Record<string, { scale?: { (v: string): number | undefined; bandwidth?: () => number; step?: () => number } }>;
                    offset?: { top?: number; height?: number };
                  };
                  const xMap = p.xAxisMap;
                  if (!xMap) return null;
                  const xAxis = Object.values(xMap)[0];
                  const scale = xAxis?.scale;
                  if (!scale || typeof scale.bandwidth !== 'function') return null;
                  const top = p.offset?.top ?? 0;
                  const height = p.offset?.height ?? 0;
                  const bandwidth = scale.bandwidth();
                  const step = typeof scale.step === 'function' ? scale.step() : bandwidth;
                  const gapHalf = (step - bandwidth) / 2;
                  return (
                    <g>
                      {separators.map((name) => {
                        const start = scale(name);
                        if (start == null) return null;
                        const x = start - gapHalf;
                        return (
                          <line
                            key={`sep-${name}`}
                            x1={x}
                            x2={x}
                            y1={top}
                            y2={top + height}
                            stroke={colors.text.muted}
                            strokeDasharray="4 4"
                            strokeOpacity={0.6}
                          />
                        );
                      })}
                    </g>
                  );
                }}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: colors.bg.hover, opacity: 0.3 }} />
              {perturbations.map((p) => (
                <Bar key={p} dataKey={`${p}_point`} fill={colorFor(p, colors)} radius={[2, 2, 0, 0]}>
                  <ErrorBar dataKey={`${p}_err`} direction="y" width={4} strokeWidth={1} stroke={colors.text.muted} />
                </Bar>
              ))}
              <StarsLayer data={data} amberColor={colors.accent.amber} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="mt-2 text-xs text-text-muted px-2">
        <span className="font-medium text-text-secondary">{metricLabel}</span>
        {' '}— Δ = perturbed − clean
      </div>
    </div>
  );
}
