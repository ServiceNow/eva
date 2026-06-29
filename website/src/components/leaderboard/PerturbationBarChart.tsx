import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ErrorBar, ReferenceLine, Customized, LabelList } from 'recharts';
import type { SystemStats } from '../../data/leaderboardData';
import { getPertValue, perturbations, perturbationLabels, groupedSystems } from '../../data/leaderboardData';
import { useThemeColors } from '../../styles/theme';
import { tierLabel, colorFor, CustomTick, type CustomTickProps, SeparatorsLayer, StarMark } from './perturbationChartUtils';

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
        row[`${p}_point`] = v.point;
        row[`${p}_err`] = [v.point - v.ci_lower, v.ci_upper - v.point];
        row[`${p}_sig_label`] = tierLabel(v.corrected_p);
        any = true;
      } else {
        row[`${p}_point`] = null;
        row[`${p}_err`] = undefined;
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
                  <SeparatorsLayer separators={separators} strokeColor={colors.text.secondary} yTop={0.5} yBottom={-0.5} />
                )}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: colors.bg.hover, opacity: 0.3 }} />
              {perturbations.map((p) => (
                <Bar key={p} dataKey={`${p}_point`} fill={colorFor(p, colors)} radius={[2, 2, 0, 0]}>
                  <ErrorBar dataKey={`${p}_err`} direction="y" width={4} strokeWidth={1} stroke={colors.text.muted} />
                  <LabelList
                    // Encode the row's significance + CI into cp.value via valueAccessor
                    // rather than reading `data[cp.index]` in content: Bar drops zero-dimension
                    // rectangles, so cp.index is into a filtered array and would misalign rows
                    // after any all-zero row.
                    valueAccessor={(entry: { payload?: ChartRow }) => {
                      const r = entry?.payload;
                      const label = r?.[`${p}_sig_label`] as string | undefined;
                      const point = r?.[`${p}_point`] as number | null | undefined;
                      const err = r?.[`${p}_err`] as [number, number] | undefined;
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
                          yTop={0.5}
                          yBottom={-0.5}
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
