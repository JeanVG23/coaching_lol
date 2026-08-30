/** Lectures KV — portage de web/backend/readers.py (mêmes sémantiques). */
export interface KVLike {
  get(key: string): Promise<string | null>;
  put(key: string, value: string): Promise<void>;
}

export const KEYS = {
  games: (slug: string) => `silver:${slug}:games`,
  rank: (slug: string) => `silver:${slug}:rank`,
  gold: (slug: string, scope: string) => `gold:${slug}:${scope}`,
  ref: (rank: string, scope: string) => `ref:${rank}:${scope}`,
  pred: (slug: string) => `pred:${slug}`,
  shap: (slug: string) => `shap:${slug}:drivers`,
  reviews: (slug: string) => `coaching:${slug}:reviews`,
  feedback: (slug: string) => `coaching:${slug}:feedback`,
};

export function matchSeq(matchId: string): number {
  const tail = matchId.split("_").pop() ?? "";
  const n = Number(tail);
  return Number.isFinite(n) && tail !== "" ? n : 0;
}

export interface GamesPage {
  items: Record<string, unknown>[];
  page: number;
  size: number;
  total: number;
}

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
  const start = (page - 1) * size;
  return { items: items.slice(start, start + size), page, size, total: items.length };
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

export async function readJsonl<T = Record<string, unknown>>(
  kv: KVLike,
  key: string,
): Promise<T[]> {
  const raw = await kv.get(key);
  if (raw === null) return [];
  return raw
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line) as T);
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

export async function appendJsonl(
  kv: KVLike,
  key: string,
  record: Record<string, unknown>,
): Promise<void> {
  const lines = await readJsonl<Record<string, unknown>>(kv, key);
  lines.push(record);
  await kv.put(key, lines.map((line) => JSON.stringify(line)).join("\n"));
}
