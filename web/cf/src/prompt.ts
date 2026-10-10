/** Prompts de coaching. Le texte vient de shared/prompts/ via le module généré :
 *  src/04_coaching/prompt.py et ce fichier lisent la MÊME source. Seul le rendu du
 *  message utilisateur (render/renderGame) est du code, et il reste ici. */

import { SYSTEM, SYSTEM_GAME } from "./generated/shared";

export { SYSTEM, SYSTEM_GAME };

export function render(payload: Record<string, any>): [string, string] {
  const meta = payload.meta;
  let user = `Signals from your latest ${meta.n_games_me} games `
    + `(${meta.scope}, outcome=${meta.outcome_focus}, vs ${meta.target}):\n\n`
    + JSON.stringify(payload, undefined, 2)
    + "\n\nProduce the review.";
  if (meta.qualitative_mode === "unbalanced" || meta.unbalanced_causes) {
    const losses = meta.n_game_reviews_available_losses ?? meta.n_game_reviews_losses ?? 0;
    const available = meta.n_game_reviews_available ?? 1;
    user += "\n\nSAMPLING-BIAS NOTE: The player has only analyzed "
      + (losses > 0 ? "losses" : "wins")
      + ` individually (${available} available; only one included). `
      + "This isolated cause is NOT representative of the full profile. "
      + "Base your conclusions primarily on the real statistical signals.";
  }
  return [SYSTEM, user];
}

export function renderGame(payload: Record<string, any>): [string, string] {
  const meta = payload.meta;
  const issue = meta.win ? "win" : "loss";
  const user = `Journal for your game ${meta.match_id} — ${meta.champion} vs `
    + `${meta.opponent || "?"} (${meta.role}, ${issue}, ${meta.duration_min} min), `
    + `${meta.target} benchmarks:\n\n${JSON.stringify(payload, undefined, 2)}`
    + "\n\nProduce the review for this game.";
  return [SYSTEM_GAME, user];
}

const promptVersionCache = new Map<string, string>();

export async function versionOf(system: string): Promise<string> {
  const cached = promptVersionCache.get(system);
  if (cached !== undefined) return cached;
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(system));
  const version = Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 12);
  promptVersionCache.set(system, version);
  return version;
}
