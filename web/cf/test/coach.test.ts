import { describe, expect, it } from "vitest";
import { apiCoach, coachFlow, type CoachParams } from "../src/coach";
import { KEYS, readJsonl, type KVLike } from "../src/readers";
import type { Env } from "../src/index";

class MemoryKV implements KVLike {
  store = new Map<string, string>();
  async get(key: string) { return this.store.get(key) ?? null; }
  async put(key: string, value: string) { this.store.set(key, value); }
}

const AGGREGATE = (games: number) => ({
  n_games: games,
  patch: "16.13",
  winrate: 0.5,
  overall: { deaths_per_game: 5 },
  win: { deaths_per_game: 4 },
  loss: { deaths_per_game: 6 },
});

const REVIEW = {
  strengths: [{ point: "s1", evidence: "e1" }],
  mistakes: [
    { point: "m1", evidence: "e1" },
    { point: "m2", evidence: "e2" },
    { point: "m3", evidence: "e3" },
  ],
  habits: ["h1", "h2"],
  next_focus: "focus",
  confidence: 0.6,
};

const PARAMS: CoachParams = {
  slug: "spadzze",
  scope: "adc",
  outcome: "loss",
  target: "challenger",
  model: "kimi-k2.6",
};

async function seed(): Promise<MemoryKV> {
  const kv = new MemoryKV();
  await kv.put(KEYS.gold("spadzze", "adc"), JSON.stringify(AGGREGATE(18)));
  await kv.put(KEYS.ref("challenger", "adc"), JSON.stringify(AGGREGATE(400)));
  return kv;
}

async function collect(generator: AsyncGenerator<{ event: string; data: any }>) {
  const events: { event: string; data: any }[] = [];
  for await (const event of generator) events.push(event);
  return events;
}

describe("coachFlow", () => {
  it("flux payload -> llm -> review, persiste l'enregistrement complet", async () => {
    const kv = await seed();
    const events = await collect(coachFlow({
      kv,
      now: () => "2026-08-30T12:00:00",
      generate: async () => REVIEW,
    }, PARAMS));
    expect(events.map((event) => event.event)).toEqual(["payload", "llm", "review"]);
    const record = events[2].data;
    expect(record).toMatchObject({
      ts: "2026-08-30T12:00:00",
      model: "kimi-k2.6",
      scope: "adc",
      target: "challenger",
      outcome_focus: "loss",
    });
    expect(record.review).toEqual(REVIEW);
    expect(record.payload.meta).toBeDefined();
    expect(await readJsonl(kv, KEYS.reviews("spadzze"))).toEqual([record]);
  });

  it("sortie non conforme -> 2 tentatives puis error, rien persisté", async () => {
    const kv = await seed();
    let calls = 0;
    const events = await collect(coachFlow({
      kv,
      now: () => "t",
      generate: async () => { calls += 1; return { strengths: [] }; },
    }, PARAMS));
    expect(calls).toBe(2);
    expect(events.map((event) => event.event)).toEqual(["payload", "llm", "error"]);
    expect(events[2].data.error).toContain("schéma");
    expect(await readJsonl(kv, KEYS.reviews("spadzze"))).toEqual([]);
  });

  it("agrégat absent -> error explicite, pas d'appel LLM", async () => {
    const kv = new MemoryKV();
    let calls = 0;
    const events = await collect(coachFlow({
      kv,
      now: () => "t",
      generate: async () => { calls += 1; return REVIEW; },
    }, PARAMS));
    expect(events.map((event) => event.event)).toEqual(["error"]);
    expect(events[0].data.error).toContain("sync");
    expect(calls).toBe(0);
  });

  it("première sortie invalide, deuxième valide -> review persistée", async () => {
    const kv = await seed();
    let attempt = 0;
    const events = await collect(coachFlow({
      kv,
      now: () => "t",
      generate: async () => (attempt++ === 0 ? { bad: 1 } : REVIEW),
    }, PARAMS));
    expect(events.map((event) => event.event)).toEqual(["payload", "llm", "review"]);
  });

  it("erreur LLM -> event error et aucune persistance", async () => {
    const kv = await seed();
    const events = await collect(coachFlow({
      kv,
      now: () => "t",
      generate: async () => { throw new Error("Ollama indisponible"); },
    }, PARAMS));
    expect(events.map((event) => event.event)).toEqual(["payload", "llm", "error"]);
    expect(events[2].data.error).toContain("Ollama indisponible");
    expect(await readJsonl(kv, KEYS.reviews("spadzze"))).toEqual([]);
  });
});

describe("apiCoach", () => {
  function env(apiKey?: string): Env {
    return {
      DATA: new MemoryKV(),
      ASSETS: { fetch: async () => new Response("spa") },
      OLLAMA_API_KEY: apiKey,
    } as unknown as Env;
  }

  const request = (slug: string) => new Request("http://x/api/coach", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ slug }),
  });

  it("404 pour un compte inconnu", async () => {
    const response = await apiCoach(request("inconnu"), env("secret"));
    expect(response.status).toBe(404);
    expect(await response.json()).toEqual({ detail: "compte inconnu" });
  });

  it("500 si OLLAMA_API_KEY est absent", async () => {
    const response = await apiCoach(request("spadzze"), env());
    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({ detail: "OLLAMA_API_KEY non configuré" });
  });
});
