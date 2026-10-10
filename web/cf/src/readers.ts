/** Lectures KV — portage de web/backend/readers.py (mêmes sémantiques). */
import { paginate, type Page } from "./http";

export interface KVLike {
  get(key: string): Promise<string | null>;
  put(key: string, value: string, options?: { expirationTtl?: number }): Promise<void>;
}

export const KEYS = {
  games: (slug: string) => `silver:${slug}:games`,
  rank: (slug: string) => `silver:${slug}:rank`,
  gold: (slug: string, scope: string) => `gold:${slug}:${scope}`,
  ref: (rank: string, scope: string) => `ref:${rank}:${scope}`,
  pred: (slug: string) => `pred:${slug}`,
  shap: (slug: string) => `shap:${slug}:drivers`,
  role_shap: (slug: string) => `shap:${slug}:role`,
  scan_index: (slug: string) => `riftsense:${slug}:scan-index`,
  reviews: (slug: string) => `riftsense:${slug}:reviews`,
  feedback: (slug: string) => `riftsense:${slug}:feedback`,
  chats: (slug: string) => `riftsense:${slug}:chats`,
  game_payloads: (slug: string) => `riftsense:${slug}:game-payloads`,
  account: (slug: string) => `account:${slug}`,
  accounts_index: () => `accounts:index`,
};

export interface GamePayloadBundle {
  schema_version: number;
  generated_at: string | null;
  target: string;
  max_games: number;
  items: Record<string, {
    payload_hash: string;
    benchmark_scope: string;
    payload: Record<string, unknown>;
  }>;
  unavailable: Array<{ match_id: string; reason: string }>;
}

export async function readGamePayloadBundle(
  kv: KVLike,
  slug: string,
): Promise<GamePayloadBundle> {
  const empty: GamePayloadBundle = {
    schema_version: 1, generated_at: null, target: "challenger", max_games: 0,
    items: {}, unavailable: [],
  };
  const value = await readJson<Partial<GamePayloadBundle>>(kv, KEYS.game_payloads(slug));
  if (!value || typeof value.items !== "object" || value.items === null
      || Array.isArray(value.items)) return empty;
  return {
    schema_version: typeof value.schema_version === "number" ? value.schema_version : 1,
    generated_at: typeof value.generated_at === "string" ? value.generated_at : null,
    target: typeof value.target === "string" ? value.target : "challenger",
    max_games: typeof value.max_games === "number" ? value.max_games : 0,
    items: value.items as GamePayloadBundle["items"],
    unavailable: Array.isArray(value.unavailable) ? value.unavailable : [],
  };
}

export function matchSeq(matchId: string): number {
  const tail = matchId.split("_").pop() ?? "";
  const n = Number(tail);
  return Number.isFinite(n) && tail !== "" ? n : 0;
}

export type GamesPage = Page<Record<string, unknown>>;

export async function readGames(
  kv: KVLike,
  slug: string,
  page = 1,
  size = 20,
): Promise<GamesPage> {
  const rows = await readJsonl<Record<string, unknown>>(kv, KEYS.games(slug));
  const items = [...rows].sort(
    (a, b) => matchSeq(String(b.match_id ?? "")) - matchSeq(String(a.match_id ?? "")),
  );
  return paginate(items, { page, size });
}

export async function readRank(
  kv: KVLike,
  slug: string,
): Promise<Record<string, unknown> | null> {
  return readJson<Record<string, unknown>>(kv, KEYS.rank(slug));
}

export async function readPred(
  kv: KVLike,
  slug: string,
): Promise<Record<string, unknown> | null> {
  return readJson<Record<string, unknown>>(kv, KEYS.pred(slug));
}

export async function readShap(
  kv: KVLike,
  slug: string,
): Promise<{ available: boolean; drivers: unknown[] }> {
  const value = await readJson(kv, KEYS.shap(slug));
  return Array.isArray(value)
    ? { available: true, drivers: value }
    : { available: false, drivers: [] };
}

export interface RoleAnalysis {
  schema_version: number;
  available: boolean;
  role: string | null;
  scope?: string | null;
  reason?: string;
  [key: string]: unknown;
}

/** Analyse ML du rôle du joueur.
 *
 * Une clé absente devient un motif et non un null : l'interface n'a alors qu'un seul
 * cas à traiter, « available est faux, voici pourquoi ». Le producteur réécrit cette
 * clé à chaque ingestion, donc une analyse qui reste disponible l'est vraiment.
 */
export async function readRoleShap(kv: KVLike, slug: string): Promise<RoleAnalysis> {
  const value = await readJson<Record<string, unknown>>(kv, KEYS.role_shap(slug));
  if (!value || Array.isArray(value) || typeof value.available !== "boolean") {
    return { schema_version: 1, available: false, reason: "not_ingested", role: null };
  }
  return value as unknown as RoleAnalysis;
}

export async function readJsonl<T = Record<string, unknown>>(
  kv: KVLike,
  key: string,
): Promise<T[]> {
  const raw = await kv.get(key);
  if (raw === null) return [];
  const items: T[] = [];
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      items.push(JSON.parse(trimmed) as T);
    } catch {
      // Ignorer une ligne corrompue pour éviter un crash 500 sur l'ensemble du dataset
    }
  }
  return items;
}

export async function countJsonl(
  kv: KVLike,
  key: string,
): Promise<number> {
  const raw = await kv.get(key);
  if (raw === null) return 0;
  let count = 0;
  for (const line of raw.split("\n")) {
    if (line.trim() !== "") count++;
  }
  return count;
}

export async function findLastJsonl<T = Record<string, unknown>>(
  kv: KVLike,
  key: string,
  predicate: (item: T) => boolean,
): Promise<T | null> {
  const raw = await kv.get(key);
  if (raw === null) return null;
  const lines = raw.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const trimmed = lines[i].trim();
    if (!trimmed) continue;
    try {
      const parsed = JSON.parse(trimmed) as T;
      if (predicate(parsed)) return parsed;
    } catch {
      // Ignorer une ligne corrompue
    }
  }
  return null;
}

export async function readJson<T = unknown>(
  kv: KVLike,
  key: string,
): Promise<T | null> {
  const raw = await kv.get(key);
  if (raw === null) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export async function writeJsonl(
  kv: KVLike,
  key: string,
  rows: Record<string, unknown>[],
): Promise<void> {
  await kv.put(key, rows.map((row) => JSON.stringify(row)).join("\n"));
}

export async function appendJsonl(
  kv: KVLike,
  key: string,
  record: Record<string, unknown>,
): Promise<void> {
  const lines = await readJsonl<Record<string, unknown>>(kv, key);
  lines.push(record);
  await writeJsonl(kv, key, lines);
}

/**
 * Remplace la ligne portant la même valeur de `idKey` (sinon ajoute).
 * `feedback.ts` refaisait ce read-modify-write à la main.
 */
export async function upsertJsonl(
  kv: KVLike,
  key: string,
  record: Record<string, unknown>,
  idKey = "ts",
): Promise<void> {
  const existing = await readJsonl<Record<string, unknown>>(kv, key);
  const kept = existing.filter((line) => line[idKey] !== record[idKey]);
  kept.push(record);
  await writeJsonl(kv, key, kept);
}
