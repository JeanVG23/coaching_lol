import { describe, expect, it } from "vitest";
import { handle, type Env } from "../src/index";
import { KEYS, type KVLike } from "../src/readers";

class MemoryKV implements KVLike {
  store = new Map<string, string>();
  async get(key: string) { return this.store.get(key) ?? null; }
  async put(key: string, value: string) { this.store.set(key, value); }
}

const SPA_HTML = "<!doctype html><title>spa</title>";

function makeEnv(): { env: Env; kv: MemoryKV } {
  const kv = new MemoryKV();
  const env = {
    DATA: kv,
    ASSETS: { fetch: async () => new Response(SPA_HTML) },
  } as unknown as Env;
  return { env, kv };
}

async function seed(): Promise<{ env: Env; kv: MemoryKV }> {
  const { env, kv } = makeEnv();
  await kv.put(KEYS.games("spadzze"), [
    JSON.stringify({ match_id: "EUW1_10", champion: "Zeri", win: true }),
    JSON.stringify({ match_id: "EUW1_30", champion: "Jinx", win: false }),
    JSON.stringify({ match_id: "EUW1_20", champion: "Caitlyn", win: true }),
  ].join("\n"));
  await kv.put(KEYS.rank("spadzze"), JSON.stringify({
    tier: "MASTER", league_points: 300, fetched_at: "2026-08-30T10:00:00",
  }));
  await kv.put(KEYS.pred("spadzze"), JSON.stringify({
    predicted_rank: "master", proba: 0.61, n_games_used: 30, predicted_lp: 412,
  }));
  await kv.put(KEYS.shap("spadzze"), JSON.stringify([{ feature: "gd10", sv: 0.3 }]));
  await kv.put(KEYS.reviews("spadzze"), [
    JSON.stringify({ ts: "2026-08-30T11:00:00", model: "kimi-k2.6", review: {} }),
    JSON.stringify({
      ts: "2026-08-30T12:00:00", kind: "game", model: "kimi-k2.6", match_id: "EUW1_30",
      payload: { meta: { champion: "Jinx", win: false } },
      review: { strengths: [], mistakes: [{ point: "m", evidence: "12:30", cause: "c" }], next_focus: "focus", confidence: 0.7 },
    }),
  ].join("\n"));
  await kv.put(KEYS.feedback("spadzze"), JSON.stringify({
    ts: "2026-08-30T11:00:00", items: [],
  }));
  return { env, kv };
}

describe("GET /api/accounts", () => {
  it("résume chaque compte : games_count + last_review_ts", async () => {
    const { env } = await seed();
    const r = await handle(new Request("http://x/api/accounts"), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toEqual([{
      slug: "spadzze", riot_id: "Spadzze#euw", region: "euw1",
      games_count: 3, last_review_ts: "2026-08-30T11:00:00",
    }]);
  });
});

describe("GET /api/c/{slug}/games", () => {
  it("défauts page=1 size=20, tri séquence décroissante", async () => {
    const { env } = await seed();
    const r = await handle(new Request("http://x/api/c/spadzze/games"), env);
    expect(await r.json()).toEqual({
      items: [
        { match_id: "EUW1_30", champion: "Jinx", win: false },
        { match_id: "EUW1_20", champion: "Caitlyn", win: true },
        { match_id: "EUW1_10", champion: "Zeri", win: true },
      ], page: 1, size: 20, total: 3,
    });
  });

  it("422 si page<1 ou size hors [1,200]", async () => {
    const { env } = await seed();
    for (const query of ["page=0", "size=0", "size=201", "page=abc", "size=abc"]) {
      const r = await handle(new Request(`http://x/api/c/spadzze/games?${query}`), env);
      expect(r.status).toBe(422);
      expect((await r.json() as { detail: string }).detail).toBe("page>=1 et size in [1,200]");
    }
  });

  it("slug inconnu → liste vide (pas de 404, parité FastAPI)", async () => {
    const { env } = await seed();
    expect(await (await handle(new Request("http://x/api/c/inconnu/games"), env)).json())
      .toEqual({ items: [], page: 1, size: 20, total: 0 });
  });
});

describe("GET /api/c/{slug}/rank + predicted-rank", () => {
  it("rank présent / vide structuré si absent", async () => {
    const { env } = await seed();
    expect(await (await handle(new Request("http://x/api/c/spadzze/rank"), env)).json())
      .toEqual({ tier: "MASTER", league_points: 300, fetched_at: "2026-08-30T10:00:00" });
    expect(await (await handle(new Request("http://x/api/c/inconnu/rank"), env)).json())
      .toEqual({
        tier: null, division: null, league_points: null,
        wins: null, losses: null, fetched_at: null,
      });
  });

  it("predicted-rank lit pred:{slug} (précalculé)", async () => {
    const { env } = await seed();
    expect(await (await handle(new Request("http://x/api/c/spadzze/predicted-rank"), env)).json())
      .toEqual({ predicted_rank: "master", proba: 0.61, n_games_used: 30, predicted_lp: 412 });
    expect(await (await handle(new Request("http://x/api/c/inconnu/predicted-rank"), env)).json())
      .toEqual({ predicted_rank: null, proba: null, n_games_used: 0 });
  });
});

describe("GET /api/c/{slug}/reviews|feedback|shap", () => {
  it("listes jsonl + shap structuré", async () => {
    const { env } = await seed();
    expect(await (await handle(new Request("http://x/api/c/spadzze/reviews"), env)).json())
      .toHaveLength(2);
    expect(await (await handle(new Request("http://x/api/c/spadzze/feedback"), env)).json())
      .toHaveLength(1);
    expect(await (await handle(new Request("http://x/api/c/spadzze/shap"), env)).json())
      .toEqual({ available: true, drivers: [{ feature: "gd10", sv: 0.3 }] });
    expect(await (await handle(new Request("http://x/api/c/inconnu/shap"), env)).json())
      .toEqual({ available: false, drivers: [] });
  });

  it("sépare la liste légère des analyses de parties et leur détail", async () => {
    const { env } = await seed();
    const page = await handle(new Request("http://x/api/c/spadzze/reviews?kind=game&page=1&size=1"), env);
    expect(page.status).toBe(200);
    const data = await page.json() as { items: Array<Record<string, unknown>>; page: number; size: number; total: number };
    expect(data).toMatchObject({ page: 1, size: 1, total: 1 });
    expect(data.items[0]).toMatchObject({
      ts: "2026-08-30T12:00:00", kind: "game", match_id: "EUW1_30",
      summary: { strengths_count: 0, mistakes_count: 1, next_focus: "focus", confidence: 0.7 },
    });
    expect(data.items[0]).not.toHaveProperty("payload");

    const detail = await handle(new Request("http://x/api/c/spadzze/reviews/2026-08-30T12%3A00%3A00"), env);
    expect(detail.status).toBe(200);
    expect(await detail.json()).toMatchObject({ kind: "game", payload: { meta: { champion: "Jinx" } } });
  });

  it("valide les paramètres de pagination des reviews", async () => {
    const { env } = await seed();
    expect((await handle(new Request("http://x/api/c/spadzze/reviews?kind=game&page=0"), env)).status).toBe(422);
    expect((await handle(new Request("http://x/api/c/spadzze/reviews?kind=autre"), env)).status).toBe(422);
  });
});

describe("routes API inconnues", () => {
  it("POST /api/fetch et GET /api/jobs/{id} n'existent plus (V1)", async () => {
    const { env } = await seed();
    const r1 = await handle(new Request("http://x/api/fetch", { method: "POST", body: "{}" }), env);
    const r2 = await handle(new Request("http://x/api/jobs/abc"), env);
    expect(r1.status).toBe(404);
    expect(r2.status).toBe(404);
  });
});
