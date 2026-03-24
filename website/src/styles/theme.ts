import { createContext, useContext } from 'react';

export type ThemeMode = 'dark' | 'light';

export interface ThemeColors {
  bg: { primary: string; secondary: string; tertiary: string; hover: string };
  accent: { purple: string; purpleLight: string; purpleDim: string; blue: string; blueLight: string; blueDim: string; cyan: string; amber: string };
  text: { primary: string; secondary: string; muted: string };
  heatmap: { bad: string; badBg: string; mid: string; midBg: string; good: string; goodBg: string };
}

const darkColors: ThemeColors = {
  bg: { primary: '#0B0D17', secondary: '#12152A', tertiary: '#1A1F3D', hover: '#222850' },
  accent: { purple: '#8B5CF6', purpleLight: '#A78BFA', purpleDim: '#6D28D9', blue: '#38BDF8', blueLight: '#7DD3FC', blueDim: '#0284C7', cyan: '#06B6D4', amber: '#F59E0B' },
  text: { primary: '#F1F5F9', secondary: '#94A3B8', muted: '#64748B' },
  heatmap: { bad: '#EF4444', badBg: '#7F1D1D', mid: '#EAB308', midBg: '#713F12', good: '#22C55E', goodBg: '#14532D' },
};

const lightColors: ThemeColors = {
  bg: { primary: '#FFFFFF', secondary: '#F8FAFC', tertiary: '#F1F5F9', hover: '#E2E8F0' },
  accent: { purple: '#7C3AED', purpleLight: '#6D28D9', purpleDim: '#5B21B6', blue: '#0284C7', blueLight: '#0369A1', blueDim: '#075985', cyan: '#0891B2', amber: '#D97706' },
  text: { primary: '#0F172A', secondary: '#334155', muted: '#64748B' },
  heatmap: { bad: '#DC2626', badBg: '#FEE2E2', mid: '#CA8A04', midBg: '#FEF9C3', good: '#16A34A', goodBg: '#DCFCE7' },
};

/** Default export for backwards compatibility — dark theme */
export const colors = darkColors;

export const themeColors: Record<ThemeMode, ThemeColors> = {
  dark: darkColors,
  light: lightColors,
};

/** React context for current theme colors (used by Recharts and other JS-driven color consumers) */
export const ThemeContext = createContext<{ mode: ThemeMode; colors: ThemeColors }>({
  mode: 'dark',
  colors: darkColors,
});

export function useThemeColors(): ThemeColors {
  return useContext(ThemeContext).colors;
}

export function useThemeMode(): ThemeMode {
  return useContext(ThemeContext).mode;
}

export function getHeatmapColor(value: number, themeOverride?: ThemeColors): { bg: string; text: string } {
  const c = themeOverride ?? darkColors;
  if (value < 0.4) return { bg: c.heatmap.badBg, text: c.heatmap.bad };
  if (value < 0.7) return { bg: c.heatmap.midBg, text: c.heatmap.mid };
  return { bg: c.heatmap.goodBg, text: c.heatmap.good };
}

/**
 * Returns a background color scaled within a metric's range.
 * Better scores = darker/more saturated, worse = lighter/more transparent.
 */
export function getScaledHeatmapColor(
  value: number,
  min: number,
  max: number,
  baseColor: string,
  invert = false,
  themeOverride?: ThemeColors,
): { bg: string; text: string } {
  const c = themeOverride ?? darkColors;
  const range = max - min;
  let t = range === 0 ? 0.5 : (value - min) / range;
  if (invert) t = 1 - t;
  // t: 0 = worst, 1 = best
  // Map to opacity: best = 0.65, worst = 0.10
  const opacity = 0.10 + t * 0.55;
  return {
    bg: hexToRgba(baseColor, opacity),
    text: c.text.primary,
  };
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(2)})`;
}
