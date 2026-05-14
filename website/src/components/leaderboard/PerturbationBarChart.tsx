import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ErrorBar, ReferenceLine, Customized, LabelList, useYAxisScale } from 'recharts';
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
          const sigLabel = item.payload[`${pertKey}_sig_label`] as string | undefined;
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
                {sigLabel ? <span className="text-amber-400 ml-0.5">{sigLabel}</span> : null}
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

/** Renders a single significance marker above a bar+CI structure. Uses the
 *  YAxis scale hook so the y position is exact regardless of chart layout. */
function StarMark({
  vb,
  label,
  upperValue,
  amberColor,
}: {
  vb: { x: number; width: number };
  label: string;
  upperValue: number;
  amberColor: string;
}) {
  const yScale = useYAxisScale() as ((v: number) => number | undefined) | undefined;
  if (!yScale) return null;
  // Position the star above the higher of (upper CI cap, zero line).
  // For deeply-negative bars whose entire CI sits below zero, this clamps
  // the star to the zero line so it stays well above the x-axis instead of
  // drifting to the bottom of the plot.
  const target = Math.max(0, upperValue);
  const ySc = yScale(target);
  if (ySc == null) return null;
  return (
    <text
      x={vb.x + vb.width / 2}
      y={ySc - 14}
      fill={amberColor}
      fontSize={14}
      fontWeight={700}
      textAnchor="middle"
    >
      {label}
    </text>
  );
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
                  <LabelList
                    dataKey={`${p}_sig_label`}
                    content={(props: unknown) => {
                      const cp = props as {
                        viewBox?: { x?: number; width?: number };
                        value?: string;
                        index?: number;
                      };
                      const label = cp.value;
                      const vb = cp.viewBox;
                      if (!label || !vb || vb.x == null || vb.width == null || cp.index == null) {
                        return null;
                      }
                      const row = data[cp.index];
                      const point = row?.[`${p}_point`] as number | null | undefined;
                      const err = row?.[`${p}_err`] as [number, number] | undefined;
                      if (point == null || !err) return null;
                      return (
                        <StarMark
                          vb={{ x: vb.x, width: vb.width }}
                          label={label}
                          upperValue={point + err[1]}
                          amberColor={colors.accent.amber}
                        />
                      );
                    }}
                  />
                </Bar>
              ))}
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
