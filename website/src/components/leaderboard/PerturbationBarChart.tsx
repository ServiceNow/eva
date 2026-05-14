import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ErrorBar, ReferenceLine, Customized, LabelList, useXAxisScale, useYAxisScale } from 'recharts';
import type { SystemStats } from '../../data/leaderboardData';
import { getPertValue, perturbations, perturbationLabels, groupedSystems } from '../../data/leaderboardData';
import { useThemeColors } from '../../styles/theme';

interface PerturbationBarChartProps {
  metric: string;
  metricLabel: string;
  systems: SystemStats[];
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

/** Renders dashed vertical separators between architecture groups. Uses the
 *  XAxis scale hook to look up category positions; falls back to interpolating
 *  the gap between adjacent left-edge positions when bandwidth/step methods
 *  aren't exposed by recharts. */
function SeparatorsLayer({
  separators,
  strokeColor,
}: {
  separators: { name: string; prevName: string }[];
  strokeColor: string;
}) {
  const xScale = useXAxisScale() as
    | (((v: string) => number | undefined) & {
        bandwidth?: () => number;
        step?: () => number;
      })
    | undefined;
  const yScale = useYAxisScale() as ((v: number) => number | undefined) | undefined;
  if (!xScale || !yScale) return null;
  const top = yScale(0.5);
  const bottom = yScale(-0.5);
  if (top == null || bottom == null) return null;
  const bandwidth = typeof xScale.bandwidth === 'function' ? xScale.bandwidth() : undefined;
  return (
    <g>
      {separators.map(({ name, prevName }) => {
        const curr = xScale(name);
        const prev = xScale(prevName);
        if (curr == null || prev == null) return null;
        // Place line at the center of the gap between the previous band's
        // right edge and the current band's left edge.
        const step = curr - prev;
        const bw = bandwidth ?? step * 0.9;
        const x = curr - (step - bw) / 2;
        return (
          <line
            key={`sep-${name}`}
            x1={x}
            x2={x}
            y1={top}
            y2={bottom}
            stroke={strokeColor}
            strokeDasharray="4 4"
            strokeOpacity={0.7}
          />
        );
      })}
    </g>
  );
}

/** Renders a single significance marker just outside a bar+CI structure in
 *  the bar's direction (above for positive deltas, below for negative). Font
 *  size scales with bar width so "***" always fits. */
function StarMark({
  vb,
  label,
  point,
  ciLower,
  ciUpper,
  amberColor,
}: {
  vb: { x: number; width: number };
  label: string;
  point: number;
  ciLower: number;
  ciUpper: number;
  amberColor: string;
}) {
  const yScale = useYAxisScale() as ((v: number) => number | undefined) | undefined;
  if (!yScale) return null;
  // "***" width ≈ 3 chars × 0.6 × fontSize. Solve for fontSize that fits vb.width.
  const fontSize = Math.max(7, Math.min(13, Math.floor(vb.width / (3 * 0.6))));
  const clearance = 5;
  const above = point >= 0;
  const capPx = yScale(above ? ciUpper : ciLower);
  if (capPx == null) return null;
  // SVG text y is the baseline. For above-bar placement, baseline sits just
  // above the cap so the glyphs hover over the cap; for below-bar placement,
  // baseline sits one fontSize below the cap so the glyphs hover under it.
  let y = above ? capPx - clearance : capPx + clearance + fontSize;
  // Clamp inside the plot area in case the cap is outside the visible domain.
  const topPx = yScale(0.5);
  const bottomPx = yScale(-0.5);
  if (topPx != null) y = Math.max(y, topPx + fontSize);
  if (bottomPx != null) y = Math.min(y, bottomPx - 2);
  return (
    <text
      x={vb.x + vb.width / 2}
      y={y}
      fill={amberColor}
      fontSize={fontSize}
      fontWeight={700}
      textAnchor="middle"
    >
      {label}
    </text>
  );
}

export function PerturbationBarChart({ metric, metricLabel, systems }: PerturbationBarChartProps) {
  const colors = useThemeColors();

  // Order systems by architecture group: S2S → Hybrid (2-part) → Cascade.
  const ordered = groupedSystems(systems);

  // Perturbation results are always shown pooled across domains; the domain
  // pills at the top of the leaderboard scope only the scatter plot.
  const data: ChartRow[] = ordered.flatMap((s) => {
    const row: ChartRow = { name: s.name, type: s.type };
    let any = false;
    for (const p of perturbations) {
      const v = getPertValue(s, metric, p, 'pooled');
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
        No perturbation data available for {metricLabel}.
      </div>
    );
  }

  // Compute group boundary indices: positions where the type changes from the previous row.
  // The ReferenceLine x value is the `name` of the first row in the new group; recharts will
  // draw the line at that category's tick.
  const separators: { name: string; prevName: string }[] = [];
  for (let i = 1; i < data.length; i++) {
    if (data[i].type !== data[i - 1].type) {
      separators.push({ name: data[i].name, prevName: data[i - 1].name });
    }
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
                component={() => (
                  <SeparatorsLayer separators={separators} strokeColor={colors.text.secondary} />
                )}
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
                          point={point}
                          ciLower={point - err[0]}
                          ciUpper={point + err[1]}
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
