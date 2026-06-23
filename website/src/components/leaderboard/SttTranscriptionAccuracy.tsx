import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ErrorBar,
  LabelList,
  useYAxisScale,
} from 'recharts';
import { useThemeColors } from '../../styles/theme';

// The four audio conditions shown side-by-side for each STT model. `clean` is the
// baseline; the other three are perturbations tested for significance vs. clean.
const CONDITIONS = [
  { key: 'clean', label: 'Clean' },
  { key: 'accent', label: 'Accent' },
  { key: 'background_noise', label: 'Background Noise' },
  { key: 'both', label: 'Accent + Noise' },
] as const;

type ConditionKey = (typeof CONDITIONS)[number]['key'];

function colorFor(key: ConditionKey, colors: ReturnType<typeof useThemeColors>): string {
  switch (key) {
    case 'clean':
      return colors.text.secondary;
    case 'accent':
      return colors.accent.amber;
    case 'background_noise':
      return colors.accent.cyan;
    case 'both':
      return colors.accent.purple;
  }
}

/** Significance tier from a Holm–Bonferroni-corrected p-value. */
function tierLabel(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return '';
  if (p < 0.001) return '***';
  if (p < 0.01) return '**';
  if (p < 0.05) return '*';
  return '';
}

interface SttRaw {
  name: string;
  clean: number;
  accent: number;
  background_noise: number;
  both: number;
  // Holm–Bonferroni-corrected p-values for each perturbation effect vs. clean.
  p: Record<'accent' | 'background_noise' | 'both', number>;
}

// Key-entity transcription accuracy is a property of the STT (speech-to-text)
// module, so only cascade systems expose it. The values below are the six STT
// models listed on this page, each measured under four audio conditions.
// NOTE: placeholder values — replace with measured key-entity transcription accuracy
// and corrected p-values.
const STT_ACCURACY: SttRaw[] = [
  { name: 'Scribe v2.2 Realtime', clean: 0.97, accent: 0.96, background_noise: 0.95, both: 0.94, p: { accent: 0., background_noise: 0., both: 0. } },
  { name: 'Universal 3.5 Pro', clean: 0.80, accent: 0., background_noise: 0., both: 0., p: { accent: 0., background_noise: 0., both: 0. } },
  { name: 'Parakeet 1.1', clean: 0.78, accent: 0.75, background_noise: 0.74, both: 0.72, p: { accent: 0., background_noise: 0., both: 0. } },
  { name: 'Nova 3', clean: 0.75, accent: 0.62, background_noise: 0.57, both: 0.47, p: { accent: 0., background_noise: 0., both: 0. } },
  { name: 'Whisper Large v3', clean: 0.66, accent: 0.52, background_noise: 0.47, both: 0.44, p: { accent: 0., background_noise: 0., both: 0. } },
  { name: 'Cohere Transcribe', clean: 0.62, accent: 0.48, background_noise: 0.46, both: 0.38, p: { accent: 0., background_noise: 0., both: 0. } },
  { name: 'Ink Whisper', clean: 0.60, accent: 0.48, background_noise: 0.49, both: 0.34, p: { accent: 0., background_noise: 0., both: 0. } },
  // { name: 'Scribe v2.2 Realtime', clean: 0.91, accent: 0.86, background_noise: 0.83, both: 0.78, p: { accent: 0.04, background_noise: 0.012, both: 0.002 } },
  // { name: 'Nova 3', clean: 0.84, accent: 0.78, background_noise: 0.74, both: 0.68, p: { accent: 0.02, background_noise: 0.004, both: 0.0003 } },
  // { name: 'Cohere Transcribe', clean: 0.76, accent: 0.69, background_noise: 0.65, both: 0.58, p: { accent: 0.008, background_noise: 0.001, both: 0.00005 } },
  // { name: 'Ink Whisper', clean: 0.72, accent: 0.64, background_noise: 0.6, both: 0.53, p: { accent: 0.003, background_noise: 0.0008, both: 0.00002 } },
  // { name: 'Parakeet 1.1', clean: 0.69, accent: 0.61, background_noise: 0.57, both: 0.49, p: { accent: 0.002, background_noise: 0.0005, both: 0.00001 } },
  // { name: 'Whisper Large v3', clean: 0.65, accent: 0.57, background_noise: 0.52, both: 0.44, p: { accent: 0.001, background_noise: 0.0003, both: 0.000005 } },
];

// Placeholder 95% confidence-interval half-width applied symmetrically to every bar.
const CI_HALF_WIDTH = 0.03;

type ChartRow = { name: string } & Record<string, number | [number, number] | string>;

const chartData: ChartRow[] = STT_ACCURACY.map((row) => {
  const out: ChartRow = { name: row.name };
  for (const { key } of CONDITIONS) {
    out[key] = row[key];
    out[`${key}_err`] = [CI_HALF_WIDTH, CI_HALF_WIDTH];
    out[`${key}_sig`] = key === 'clean' ? '' : tierLabel(row.p[key as 'accent' | 'background_noise' | 'both']);
  }
  return out;
});

/** Renders a significance marker just above a bar's upper CI cap. Font size scales
 *  with bar width so "***" always fits. */
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
  const fontSize = Math.max(7, Math.min(13, Math.floor(vb.width / (3 * 0.6))));
  const clearance = 5;
  const capPx = yScale(ciUpper);
  if (capPx == null) return null;
  let y = capPx - clearance;
  const topPx = yScale(1);
  if (topPx != null) y = Math.max(y, topPx + fontSize);
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

function CustomTooltip({
  active,
  payload,
  colors,
}: {
  active?: boolean;
  payload?: Array<{ payload: ChartRow }>;
  colors: ReturnType<typeof useThemeColors>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0].payload;
  return (
    <div className="bg-bg-tertiary border border-border-default rounded-lg p-3 shadow-xl max-w-xs">
      <div className="text-sm font-semibold text-text-primary mb-2">{row.name}</div>
      <div className="flex flex-col gap-1 text-xs">
        {CONDITIONS.map(({ key, label }) => {
          const value = row[key] as number;
          const sig = row[`${key}_sig`] as string;
          const lower = value - CI_HALF_WIDTH;
          const upper = value + CI_HALF_WIDTH;
          return (
            <div key={key} className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                style={{ backgroundColor: colorFor(key, colors) }}
              />
              <span className="text-text-muted">{label}:</span>
              <span className="font-mono text-text-primary">
                {(value * 100).toFixed(0)}%
                {sig ? <span className="text-amber-400 ml-0.5">{sig}</span> : null}
              </span>
              <span className="font-mono text-text-muted">
                [{(lower * 100).toFixed(0)}, {(upper * 100).toFixed(0)}]
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function SttTranscriptionAccuracy() {
  const colors = useThemeColors();
  const [sectionOpen, setSectionOpen] = useState(false);

  const minWidth = Math.max(720, STT_ACCURACY.length * 150);

  return (
    <div className="rounded-xl border border-border-default bg-bg-secondary overflow-hidden">
      <button
        onClick={() => setSectionOpen((o) => !o)}
        className="w-full flex items-center gap-3 p-5 hover:bg-bg-hover transition-colors text-left"
      >
        {sectionOpen ? (
          <ChevronDown className="w-5 h-5 text-text-muted flex-shrink-0" />
        ) : (
          <ChevronRight className="w-5 h-5 text-text-muted flex-shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-bold text-text-primary">
            STT Performance: Transcription Accuracy (Key Entities)
          </h3>
          <p className="text-sm text-text-muted mt-0.5">
            This section covers <span className="font-semibold text-text-secondary">STT (speech-to-text) models only</span>.
            Cascade systems first transcribe the caller's audio into text, so their accuracy on
            key entities (names, IDs, numbers, dates) can be measured directly. S2S and hybrid
            systems are excluded because they process audio end-to-end and never produce an
            intermediate transcript. For each model we report accuracy on clean audio and under
            three perturbations — accent, background noise, and the two combined — with 95%
            confidence intervals. Asterisks (<span className="text-amber-400">*</span>) indicate that
            the perturbation effect vs. clean baseline is statistically significant after{' '}
            <span className="font-semibold text-text-secondary">Holm–Bonferroni</span> correction
            across the perturbation × model tests. Higher is better.
          </p>
        </div>
      </button>

      {sectionOpen && (
        <div className="border-t border-border-default p-4 space-y-3">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-2 py-3 rounded-lg bg-bg-primary border border-border-default">
            {CONDITIONS.map(({ key, label }) => (
              <div key={key} className="flex items-center gap-2 text-xs">
                <span
                  className="w-3 h-3 rounded-sm flex-shrink-0"
                  style={{ backgroundColor: colorFor(key, colors) }}
                />
                <span className="text-text-secondary">{label}</span>
              </div>
            ))}
            <div className="text-xs text-text-muted ml-auto">
              <span className="text-amber-400 font-bold">*</span> significant perturbation effect
            </div>
          </div>
          <div className="overflow-x-auto">
            <div className="h-[440px]" style={{ minWidth: `${minWidth}px` }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  margin={{ top: 24, right: 16, bottom: 70, left: 16 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={colors.bg.tertiary} />
                  <XAxis
                    dataKey="name"
                    stroke={colors.text.muted}
                    tick={{ fill: colors.text.secondary, fontSize: 10 }}
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
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                    width={56}
                    label={{
                      // value: 'Key-entity accuracy',
                      angle: -90,
                      position: 'insideLeft',
                      offset: 0,
                      fill: colors.text.secondary,
                      style: { fontSize: 12 },
                    }}
                  />
                  <Tooltip
                    content={<CustomTooltip colors={colors} />}
                    cursor={{ fill: colors.bg.hover, opacity: 0.3 }}
                  />
                  {CONDITIONS.map(({ key }) => (
                    <Bar key={key} dataKey={key} fill={colorFor(key, colors)} radius={[2, 2, 0, 0]}>
                      <ErrorBar
                        dataKey={`${key}_err`}
                        direction="y"
                        width={4}
                        strokeWidth={1}
                        stroke={colors.text.muted}
                      />
                      <LabelList
                        // Encode significance + point + CI into the label value so StarMark
                        // can place the marker above the upper CI cap.
                        valueAccessor={(entry: { payload?: ChartRow }) => {
                          const r = entry?.payload;
                          const sig = r?.[`${key}_sig`] as string | undefined;
                          const point = r?.[key] as number | undefined;
                          const err = r?.[`${key}_err`] as [number, number] | undefined;
                          if (!sig || point == null || !err) return '';
                          return `${sig}|${point}|${err[1]}`;
                        }}
                        content={(props: unknown) => {
                          const cp = props as { viewBox?: { x?: number; width?: number }; value?: string };
                          const vb = cp.viewBox;
                          if (!cp.value || !vb || vb.x == null || vb.width == null) return null;
                          const [label, pointStr, errHiStr] = cp.value.split('|');
                          const point = parseFloat(pointStr);
                          const errHi = parseFloat(errHiStr);
                          if (!Number.isFinite(point) || !Number.isFinite(errHi)) return null;
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
            <span className="font-medium text-text-secondary">Transcription Accuracy (Key Entities)</span>
            {' '}— share of key entities transcribed correctly
          </div>
        </div>
      )}
    </div>
  );
}
