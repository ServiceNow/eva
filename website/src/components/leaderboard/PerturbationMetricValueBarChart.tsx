import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ErrorBar, Customized, LabelList, useXAxisScale, useYAxisScale } from 'recharts';
import type { SystemStats } from '../../data/leaderboardData';
import { getPertValue, getPertMetricValue, perturbations, perturbationLabels, groupedSystems } from '../../data/leaderboardData';
import { useThemeColors } from '../../styles/theme';

interface PerturbationMetricValueBarChartProps {
  metric: string;
  metricLabel: string;
  systems: SystemStats[];
}

interface ChartRow {
  name: string;
  type: SystemStats['type'];
  [key: string]: string | number | [number, number] | boolean | null | undefined;
}

// Conditions shown as bars: clean baseline first, then each perturbation.
const CONDITIONS: string[] = ['clean', ...perturbations];
const conditionLabels: Record<string, string> = { clean: 'Clean', ...perturbationLabels };

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
          // dataKey of the form `<condition>_point`
          const condKey = item.dataKey.replace(/_point$/, '');
          const sigLabel = item.payload[`${condKey}_sig_label`] as string | undefined;
          const err = item.payload[`${condKey}_err`] as [number, number] | undefined;
          if (item.value === null || item.value === undefined || Number.isNaN(item.value)) return null;
          const lower = err ? item.value - err[0] : item.value;
          const upper = err ? item.value + err[1] : item.value;
          return (
            <div key={item.dataKey} className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ backgroundColor: item.color }} />
              <span className="text-text-muted">{conditionLabels[condKey] ?? condKey}:</span>
              <span className="font-mono text-text-primary">
                {item.value.toFixed(3)}
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

function colorFor(condition: string, colors: ReturnType<typeof useThemeColors>): string {
  if (condition === 'clean') return colors.text.muted; // neutral baseline bar
  const key = PERT_COLORS[condition];
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
  // Metric values live in [0, 1] (unlike the delta chart's [-0.5, 0.5]).
  const top = yScale(1);
  const bottom = yScale(0);
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

/** Renders a single significance marker just above a bar+CI structure. Metric
 *  values are non-negative, so bars always grow upward and the marker sits
 *  above the upper CI cap. Font size scales with bar width so "***" always fits. */
function StarMark({
  vb,
  label,
  ciUpper,
  amberColor,
}: {
  vb: { x: number; width: number };
  label: string;
  ciUpper: number;
  amberColor: string;
}) {
  const yScale = useYAxisScale() as ((v: number) => number | undefined) | undefined;
  if (!yScale) return null;
  // "***" width ≈ 3 chars × 0.6 × fontSize. Solve for fontSize that fits vb.width.
  const fontSize = Math.max(7, Math.min(13, Math.floor(vb.width / (3 * 0.6))));
  const clearance = 5;
  const capPx = yScale(ciUpper);
  if (capPx == null) return null;
  // Baseline sits just above the upper cap so the glyphs hover over it.
  let y = capPx - clearance;
  // Clamp inside the plot area ([0, 1] domain) in case the cap is off-domain.
  const topPx = yScale(1);
  const bottomPx = yScale(0);
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

export function PerturbationMetricValueBarChart({ metric, metricLabel, systems }: PerturbationMetricValueBarChartProps) {
  const colors = useThemeColors();

  // Order systems by architecture group: S2S → Hybrid (2-part) → Cascade.
  const ordered = groupedSystems(systems);

  // Always shown pooled across domains; the domain pills scope only the scatter plot.
  const data: ChartRow[] = ordered.flatMap((s) => {
    const row: ChartRow = { name: s.name, type: s.type };
    let any = false;
    for (const c of CONDITIONS) {
      const v = getPertMetricValue(s, metric, c, 'pooled'); // bar height + CI
      if (v) {
        row[`${c}_point`] = v.point;
        row[`${c}_err`] = [v.point - v.ci_lower, v.ci_upper - v.point];
        // asterisks: clean has none; perturbations reuse the delta significance
        const sig = c === 'clean' ? null : getPertValue(s, metric, c, 'pooled');
        row[`${c}_sig_label`] = sig ? tierLabel(sig.corrected_p) : '';
        any = true;
      } else {
        row[`${c}_point`] = null;
        row[`${c}_err`] = undefined;
        row[`${c}_sig_label`] = '';
      }
    }
    return any ? [row] : [];
  });

  if (data.length === 0) {
    return (
      <div className="text-sm text-text-muted italic px-4 py-6">
        No metric-value data available for {metricLabel}.
      </div>
    );
  }

  // Group boundaries: each entry pairs the new-group's first row with the previous row,
  // so SeparatorsLayer can place a dashed line at the midpoint of the gap between them.
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
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tickFormatter={(v: number) => v.toFixed(2)}
                allowDataOverflow
                width={56}
                label={{ value: 'metric value', angle: -90, position: 'insideLeft', offset: 0, fill: colors.text.secondary, style: { fontSize: 12 } }}
              />
              <Customized
                component={() => (
                  <SeparatorsLayer separators={separators} strokeColor={colors.text.secondary} />
                )}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: colors.bg.hover, opacity: 0.3 }} />
              {CONDITIONS.map((c) => (
                <Bar key={c} dataKey={`${c}_point`} fill={colorFor(c, colors)} radius={[2, 2, 0, 0]}>
                  <ErrorBar dataKey={`${c}_err`} direction="y" width={4} strokeWidth={1} stroke={colors.text.muted} />
                  <LabelList
                    // Encode the row's significance + CI into cp.value via valueAccessor
                    // rather than reading `data[cp.index]` in content: Bar drops zero-dimension
                    // rectangles, so cp.index is into a filtered array and would misalign rows.
                    valueAccessor={(entry: { payload?: ChartRow }) => {
                      const r = entry?.payload;
                      const label = r?.[`${c}_sig_label`] as string | undefined;
                      const point = r?.[`${c}_point`] as number | null | undefined;
                      const err = r?.[`${c}_err`] as [number, number] | undefined;
                      if (!label || point == null || !err) return '';
                      return `${label}|${point}|${err[0]}|${err[1]}`;
                    }}
                    content={(props: unknown) => {
                      const cp = props as { viewBox?: { x?: number; width?: number }; value?: string };
                      const vb = cp.viewBox;
                      if (!cp.value || !vb || vb.x == null || vb.width == null) return null;
                      const [label, pointStr, , errHiStr] = cp.value.split('|');
                      const point = parseFloat(pointStr);
                      const errHi = parseFloat(errHiStr);
                      if (!Number.isFinite(point) || !Number.isFinite(errHi)) {
                        return null;
                      }
                      return (
                        <StarMark
                          vb={{ x: vb.x, width: vb.width }}
                          label={label}
                          ciUpper={point + errHi}
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
        {' '}— metric value, pooled across domains; asterisks mark significant change vs. clean
      </div>
    </div>
  );
}
