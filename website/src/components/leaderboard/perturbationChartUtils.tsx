import { useXAxisScale, useYAxisScale } from 'recharts';
import { useThemeColors } from '../../styles/theme';

export const PERT_COLOR_KEYS: Record<string, 'amber' | 'cyan' | 'purple'> = {
  accent: 'amber',
  background_noise: 'cyan',
  both: 'purple',
};

export function tierLabel(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return '';
  if (p < 0.001) return '***';
  if (p < 0.01) return '**';
  if (p < 0.05) return '*';
  return '';
}

export function colorFor(condition: string, colors: ReturnType<typeof useThemeColors>, includeClean = false): string {
  if (includeClean && condition === 'clean') return colors.text.muted;
  const key = PERT_COLOR_KEYS[condition];
  if (key) return colors.accent[key];
  return colors.accent.blue;
}

export interface CustomTickProps {
  x?: number;
  y?: number;
  payload?: { value: string };
  fill?: string;
  fontSize?: number;
  angle?: number;
  textAnchor?: 'start' | 'middle' | 'end' | 'inherit';
  amberFirst?: boolean;
  dy?: number;
}

export function CustomTick({ x, y, payload, fill, fontSize = 10, angle = -30, textAnchor = 'end', amberFirst = false, dy = 8 }: CustomTickProps) {
  const colors = useThemeColors();
  if (!payload?.value || x == null || y == null) return null;
  const parts = payload.value.split(' + ');
  const sttModel = parts[0];
  const rest = parts.slice(1).join(' + ');
  return (
    <g transform={`translate(${x},${y})`}>
      <text
        transform={`rotate(${angle})`}
        textAnchor={textAnchor}
        fill={fill}
        fontSize={fontSize}
        dy={dy}
      >
        <tspan fill={amberFirst ? colors.accent.amber : fill}>{sttModel}</tspan>
        {rest && <tspan>{` + ${rest}`}</tspan>}
      </text>
    </g>
  );
}

/** Dashed vertical separators between architecture groups. `yTop`/`yBottom` are
 *  the chart's y-domain bounds (e.g. 0.5/-0.5 for deltas, 1/0 for metric values). */
export function SeparatorsLayer({
  separators,
  strokeColor,
  yTop,
  yBottom,
}: {
  separators: { name: string; prevName: string }[];
  strokeColor: string;
  yTop: number;
  yBottom: number;
}) {
  const xScale = useXAxisScale() as
    | (((v: string) => number | undefined) & { bandwidth?: () => number; step?: () => number })
    | undefined;
  const yScale = useYAxisScale() as ((v: number) => number | undefined) | undefined;
  if (!xScale || !yScale) return null;
  const top = yScale(yTop);
  const bottom = yScale(yBottom);
  if (top == null || bottom == null) return null;
  const bandwidth = typeof xScale.bandwidth === 'function' ? xScale.bandwidth() : undefined;
  return (
    <g>
      {separators.map(({ name, prevName }) => {
        const curr = xScale(name);
        const prev = xScale(prevName);
        if (curr == null || prev == null) return null;
        // Center the line in the gap between the previous band's right edge and the current's left.
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

/** Significance marker just outside a bar+CI in the bar's direction (above for
 *  point ≥ 0, below otherwise). `yTop`/`yBottom` are the y-domain bounds used to
 *  clamp inside the plot. Font size scales with bar width so "***" always fits. */
export function StarMark({
  vb,
  label,
  point,
  ciLower,
  ciUpper,
  amberColor,
  yTop,
  yBottom,
}: {
  vb: { x: number; width: number };
  label: string;
  point: number;
  ciLower: number;
  ciUpper: number;
  amberColor: string;
  yTop: number;
  yBottom: number;
}) {
  const yScale = useYAxisScale() as ((v: number) => number | undefined) | undefined;
  if (!yScale) return null;
  // "***" width ≈ 3 chars × 0.6 × fontSize. Solve for fontSize that fits vb.width.
  const fontSize = Math.max(7, Math.min(13, Math.floor(vb.width / (3 * 0.6))));
  const clearance = 5;
  const above = point >= 0;
  const capPx = yScale(above ? ciUpper : ciLower);
  if (capPx == null) return null;
  // Baseline hovers just past the relevant CI cap.
  let y = above ? capPx - clearance : capPx + clearance + fontSize;
  const topPx = yScale(yTop);
  const bottomPx = yScale(yBottom);
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
