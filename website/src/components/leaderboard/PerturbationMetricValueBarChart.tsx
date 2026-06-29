import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ErrorBar, LabelList } from 'recharts';
import type { SystemStats } from '../../data/leaderboardData';
import { getPertValue, getPertMetricValue, perturbations, perturbationLabels, groupedSystems } from '../../data/leaderboardData';
import { useThemeColors } from '../../styles/theme';
import { tierLabel, colorFor, CustomTick, type CustomTickProps, StarMark } from './perturbationChartUtils';

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



export function PerturbationMetricValueBarChart({ metric, metricLabel, systems }: PerturbationMetricValueBarChartProps) {
  const colors = useThemeColors();

  // Always shown pooled across domains; the domain pills scope only the scatter plot.
  const data: ChartRow[] = groupedSystems(systems).flatMap((s) => {
    const row: ChartRow = { name: s.stt, type: s.type };
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

  data.sort((a, b) => {
    const va = (a.clean_point as number | null) ?? -Infinity;
    const vb = (b.clean_point as number | null) ?? -Infinity;
    return vb - va;
  });

  // When multiple systems share the same STT name, keep only the one with the
  // highest clean score (already first after the sort above).
  const seen = new Set<string>();
  const deduped_data = data.filter((row) => {
    const name = row.name as string;
    if (seen.has(name)) return false;
    seen.add(name);
    return true;
  });

  if (deduped_data.length === 0) {
    return (
      <div className="text-sm text-text-muted italic px-4 py-6">
        No metric-value data available for {metricLabel}.
      </div>
    );
  }

  const minWidth = Math.max(720, deduped_data.length * 80);

  return (
    <div>
      <div className="overflow-x-auto">
        <div className="h-[440px]" style={{ minWidth: `${minWidth}px` }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={deduped_data} margin={{ top: 24, right: 16, bottom: 70, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={colors.bg.tertiary} />
              <XAxis
                dataKey="name"
                stroke={colors.text.muted}
                tick={(props: unknown) => (
                  <CustomTick
                    {...(props as CustomTickProps)}
                    fill={colors.text.secondary}
                    fontSize={10}
                    angle={-30}
                    textAnchor="end"

                  />
                )}
                interval={0}
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
              <Tooltip content={<CustomTooltip />} cursor={{ fill: colors.bg.hover, opacity: 0.3 }} />
              {CONDITIONS.map((c) => (
                <Bar key={c} dataKey={`${c}_point`} fill={colorFor(c, colors, true)} radius={[2, 2, 0, 0]}>
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
                      const [label, pointStr, errLoStr, errHiStr] = cp.value.split('|');
                      const point = parseFloat(pointStr);
                      const errLo = parseFloat(errLoStr);
                      const errHi = parseFloat(errHiStr);
                      if (!Number.isFinite(point) || !Number.isFinite(errLo) || !Number.isFinite(errHi)) {
                        return null;
                      }
                      return (
                        <StarMark
                          vb={{ x: vb.x, width: vb.width }}
                          label={label}
                          point={point}
                          ciLower={point - errLo}
                          ciUpper={point + errHi}
                          amberColor={colors.accent.amber}
                          yTop={1}
                          yBottom={0}
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
