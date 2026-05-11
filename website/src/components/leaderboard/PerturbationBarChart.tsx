import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ErrorBar, ReferenceLine } from 'recharts';
import type { SystemStats, DomainOrPooled } from '../../data/leaderboardData';
import { getPertValue, perturbations, perturbationLabels } from '../../data/leaderboardData';
import { useThemeColors } from '../../styles/theme';

interface PerturbationBarChartProps {
  metric: string;
  metricLabel: string;
  systems: SystemStats[];
  domain: DomainOrPooled;
}

interface ChartRow {
  name: string;
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

export function PerturbationBarChart({ metric, metricLabel, systems, domain }: PerturbationBarChartProps) {
  const colors = useThemeColors();

  // Build data rows: one per system that has any perturbation data for this metric.
  const data: ChartRow[] = systems.flatMap((s) => {
    const row: ChartRow = { name: s.name };
    let any = false;
    for (const p of perturbations) {
      const v = getPertValue(s, metric, p, domain);
      if (v) {
        row[`${p}_point`] = v.point;
        row[`${p}_err`] = [v.point - v.ci_lower, v.ci_upper - v.point];
        row[`${p}_sig`] = !!v.reject;
        any = true;
      } else {
        row[`${p}_point`] = null;
        row[`${p}_err`] = undefined;
        row[`${p}_sig`] = false;
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

  return (
    <div>
      <div style={{ width: '100%', height: 360 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 16, right: 16, bottom: 60, left: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={colors.bg.tertiary} />
            <XAxis
              dataKey="name"
              stroke={colors.text.muted}
              tick={{ fill: colors.text.secondary, fontSize: 10 }}
              interval={0}
              angle={-30}
              textAnchor="end"
              height={70}
            />
            <YAxis
              stroke={colors.text.muted}
              tick={{ fill: colors.text.secondary, fontSize: 11 }}
              label={{ value: 'Δ vs clean', angle: -90, position: 'insideLeft', fill: colors.text.secondary, style: { fontSize: 12 } }}
            />
            <ReferenceLine y={0} stroke={colors.text.muted} />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: colors.bg.hover, opacity: 0.3 }} />
            <Legend
              formatter={(value: string) => {
                const k = value.replace(/_point$/, '');
                return <span style={{ color: colors.text.secondary }}>{perturbationLabels[k] ?? k}</span>;
              }}
            />
            {perturbations.map((p) => (
              <Bar key={p} dataKey={`${p}_point`} fill={colorFor(p, colors)} radius={[2, 2, 0, 0]}>
                <ErrorBar dataKey={`${p}_err`} direction="y" width={4} strokeWidth={1} stroke={colors.text.muted} />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-text-muted px-2">
        <div>
          <span className="font-medium text-text-secondary">{metricLabel}</span>
          {' '}— Δ = perturbed − clean
        </div>
        <div>
          <span className="text-amber-400">*</span> significant after correction (reject = true)
        </div>
      </div>
    </div>
  );
}
