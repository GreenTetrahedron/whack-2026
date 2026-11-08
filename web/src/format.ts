// Number formatting and colour scales shared by the map and the panel.

const MINUS = "−";

export function egp(v: number, signed = false): string {
  const a = Math.abs(v);
  const sign = v < 0 ? MINUS : signed && v > 0 ? "+" : "";
  if (a >= 1e9) return `${sign}EGP ${(a / 1e9).toFixed(2)}bn`;
  if (a >= 1e6) return `${sign}EGP ${(a / 1e6).toFixed(1)}m`;
  if (a >= 1e3) return `${sign}EGP ${(a / 1e3).toFixed(0)}k`;
  return `${sign}EGP ${a.toFixed(0)}`;
}

export function pct(v: number, digits = 1): string {
  const sign = v > 0 ? "+" : v < 0 ? MINUS : "";
  return `${sign}${Math.abs(v).toFixed(digits)}%`;
}

export function num(v: number): string {
  return Math.round(v).toLocaleString("en-GB");
}

export type RGBA = [number, number, number, number];

export const LOSS: RGBA = [214, 64, 69, 230];
export const GAIN: RGBA = [33, 113, 181, 230];
export const NEUTRAL: RGBA = [150, 150, 150, 150];

function mix(a: RGBA, b: RGBA, t: number): RGBA {
  return [0, 1, 2, 3].map((i) => Math.round(a[i] + (b[i] - a[i]) * t)) as RGBA;
}

/** Diverging colour for a % change: red loss, blue gain, grey near zero. Saturates at ±cap %. */
export function divergingColor(v: number, cap = 5): RGBA {
  if (!Number.isFinite(v) || Math.abs(v) < 0.1) return NEUTRAL;
  const t = Math.min(1, Math.abs(v) / cap);
  return mix([200, 200, 200, 170], v < 0 ? LOSS : GAIN, 0.25 + 0.75 * t);
}

/** Sequential colour (yellow → dark orange) for a non-negative value scaled to max. */
export function sequentialColor(v: number, max: number, alpha = 170): RGBA {
  const t = max > 0 ? Math.min(1, Math.max(0, v / max)) : 0;
  return mix([255, 237, 160, alpha], [189, 54, 0, alpha], Math.sqrt(t));
}

export function hexToRgb(hex: string, alpha = 255): RGBA {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255, alpha];
}

export const RISK_COLOR: Record<string, string> = { green: "#2e8b57", amber: "#d18b00", red: "#c62828" };
