/** Utilitaires purs de formatage d'affichage pour l'interface utilisateur. */

export function countKdaValue(value: unknown): number {
  return Array.isArray(value) ? value.length : Number(value || 0);
}

export interface KdaInput {
  kills?: unknown;
  deaths?: unknown;
  assists?: unknown;
}

/** Formate un triplet de KDA en chaîne `K/D/A · Ratio` (ou `Perfect` si 0 mort). */
export function formatKda(kda?: KdaInput | null): string | null {
  if (!kda) return null;
  const kills = countKdaValue(kda.kills);
  const deaths = countKdaValue(kda.deaths);
  const assists = countKdaValue(kda.assists);
  const ratio = deaths === 0 ? "Perfect" : ((kills + assists) / deaths).toFixed(2);
  return `${kills}/${deaths}/${assists} · ${ratio}`;
}

/** Formate une durée en minutes (ex: `28 min`), ou null si absente ou non valide. */
export function formatDuration(minutes?: unknown): string | null {
  const num = Number(minutes);
  return Number.isFinite(num) && num > 0 ? `${Math.round(num)} min` : null;
}

/** Formate un taux de confiance (ex: `80%` ou `Confidence 80%`). */
export function formatConfidence(confidence?: unknown, prefix = ""): string {
  if (confidence === null || confidence === undefined || confidence === "") return "—";
  const num = Number(confidence);
  if (!Number.isFinite(num)) return "—";
  const pct = `${Math.round(num * 100)}%`;
  return prefix ? `${prefix} ${pct}` : pct;
}

/** Formate un pourcentage (ex: `52 %`), ou `—` si absent. */
export function formatPercent(value: unknown): string {
  return value === null || value === undefined ? "—" : `${Math.round(Number(value) * 100)} %`;
}

/** Formate un nombre à N décimales, ou `—` si absent. */
export function formatNumber(value: unknown, digits = 1): string {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

/** Formate un écart signé (ex: `+150` ou `-50`). */
export function formatDiff(value: unknown, suffix = ""): string {
  if (value === null || value === undefined) return "—";
  const number = Number(value);
  return `${number > 0 ? "+" : ""}${value}${suffix}`;
}

/** Classe CSS pour coloration d'écart. */
export function diffClass(value: unknown): string {
  if (value === null || value === undefined || Number(value) === 0) return "diff-neutral";
  return Number(value) > 0 ? "diff-pos" : "diff-neg";
}

/** Parse de manière tolérante une valeur temporelle (timestamp numérique ou chaîne ISO). */
export function parseDateValue(value?: number | string | null): Date | null {
  if (value == null || value === "" || value === 0) return null;
  const num = typeof value === "number" ? value : Number(value);
  if (!Number.isNaN(num)) {
    if (num <= 0) return null;
    return new Date(num);
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

/** Formate une date courte en anglais (ex: `15 Oct 2026`). */
export function formatDate(value?: number | string | null): string {
  if (value == null || value === "") return "—";
  const date = parseDateValue(value);
  if (!date) return String(value).slice(0, 10);
  return new Intl.DateTimeFormat("en-US", {
    day: "numeric", month: "short", year: "numeric",
  }).format(date);
}

/** Formate une date de partie courte (ex: `15 Oct 2026`), ou chaîne vide si invalide. */
export function formatGameDate(value?: number | string | null): string {
  const date = parseDateValue(value);
  if (!date) return "";
  return new Intl.DateTimeFormat("en-US", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(date);
}

/** Formate une date détaillée avec heure (ex: `Thursday, 15 October 2026, 14:30`). */
export function formatFullDate(value?: number | string | null): string {
  const date = parseDateValue(value);
  if (!date) return "";
  return new Intl.DateTimeFormat("en-US", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

