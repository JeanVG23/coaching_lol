import {
  championIcon,
  fallbackChampionIcon,
} from "./champion-assets";
import {
  formatDuration as formatDurationValue,
  formatKda as formatKdaValue,
} from "./formatters";

export interface GameReview {
  ts: string;
  match_id?: string;
  meta?: Record<string, any>;
  payload?: { meta?: Record<string, any> };
  review?: {
    confidence?: number;
    next_focus?: string;
    strengths?: any[];
    mistakes?: any[];
  };
}

export function gameMeta(review?: GameReview | null): Record<string, any> {
  return review?.payload?.meta || review?.meta || {};
}

export const gameChampion = (review?: GameReview | null): string => gameMeta(review).champion || "Analyzed game";
export const gameOpponent = (review?: GameReview | null): string | null => gameMeta(review).opponent || null;
export const gamePatch = (review?: GameReview | null): string | null => gameMeta(review).patch || null;
export const gameMatchId = (review?: GameReview | null): string => review?.match_id || gameMeta(review).match_id || "—";

export function gameResult(review?: GameReview | null): string {
  const win = gameMeta(review).win;
  return win === true ? "Victory" : win === false ? "Defeat" : "Analysis";
}

export function gameDuration(review?: GameReview | null): string | null {
  return formatDurationValue(gameMeta(review).duration_min);
}

export function gameKda(review?: GameReview | null): string | null {
  return formatKdaValue(gameMeta(review).kda);
}

export function gameIcon(review?: GameReview | null): string {
  const meta = gameMeta(review);
  return championIcon(meta.champion, meta.patch);
}

export function iconFallback(event: Event, champion: string): void {
  fallbackChampionIcon(event, champion);
}
