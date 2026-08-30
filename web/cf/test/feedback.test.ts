import { describe, expect, it } from "vitest";
import { apiFeedback } from "../src/feedback";
import { KEYS, readJsonl, type KVLike } from "../src/readers";
import type { Env } from "../src/index";

class MemoryKV implements KVLike {
  store = new Map<string, string>();
  async get(key: string) { return this.store.get(key) ?? null; }
  async put(key: string, value: string) { this.store.set(key, value); }
}

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

async function makeEnv(): Promise<Env> {
  const kv = new MemoryKV();
  await kv.put(KEYS.reviews("spadzze"), JSON.stringify({
    ts: "T1",
    model: "kimi-k2.6",
    scope: "adc",
    target: "challenger",
    payload: {},
    review: REVIEW,
    outcome_focus: "loss",
  }));
  return {
    DATA: kv,
    ASSETS: { fetch: async () => new Response("spa") },
  } as unknown as Env;
}

const post = (env: Env, body: unknown) => apiFeedback(new Request("http://x/api/feedback", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
}), env);

describe("apiFeedback", () => {
  it("persiste un feedback ordonné et répond ok", async () => {
    const env = await makeEnv();
    const response = await post(env, {
      slug: "spadzze",
      ts: "T1",
      responses: {
        "mistake,1": { useful: false, tag: "trop-vague" },
        "strength,0": { useful: true },
        "focus,0": { useful: true, note: "clair" },
      },
    });
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ ok: true });
    const lines = await readJsonl(env.DATA, KEYS.feedback("spadzze"));
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({
      ts: "T1",
      player: "spadzze",
      model: "kimi-k2.6",
      overall_useful: null,
    });
    expect((lines[0] as any).items).toEqual([
      { kind: "strength", index: 0, useful: true, tag: null, note: null },
      { kind: "mistake", index: 1, useful: false, tag: "trop-vague", note: null },
      { kind: "focus", index: 0, useful: true, tag: null, note: "clair" },
    ]);
  });

  it("ré-annoter le même ts écrase", async () => {
    const env = await makeEnv();
    await post(env, {
      slug: "spadzze", ts: "T1", responses: { "strength,0": { useful: true } },
    });
    await post(env, {
      slug: "spadzze",
      ts: "T1",
      responses: { "strength,0": { useful: false, tag: "autre" } },
    });
    const lines = await readJsonl(env.DATA, KEYS.feedback("spadzze"));
    expect(lines).toHaveLength(1);
    expect((lines[0] as any).items[0]).toMatchObject({ useful: false, tag: "autre" });
  });

  it("404 compte inconnu / review introuvable", async () => {
    const env = await makeEnv();
    expect((await post(env, { slug: "inconnu", ts: "T1", responses: {} })).status).toBe(404);
    expect((await post(env, { slug: "spadzze", ts: "PAS_LA", responses: {} })).status).toBe(404);
  });

  it("422 pour clé, useful, tag ou tag requis invalides", async () => {
    const env = await makeEnv();
    expect((await post(env, {
      slug: "spadzze", ts: "T1", responses: { mistake: { useful: true } },
    })).status).toBe(422);
    expect((await post(env, {
      slug: "spadzze", ts: "T1", responses: { "mistake,0": { useful: "oui" } },
    })).status).toBe(422);
    expect((await post(env, {
      slug: "spadzze", ts: "T1", responses: {
        "mistake,0": { useful: false, tag: "pas-un-tag" },
      },
    })).status).toBe(422);
    expect((await post(env, {
      slug: "spadzze", ts: "T1", responses: { "mistake,0": { useful: false } },
    })).status).toBe(422);
  });

  it("422 si le body ne respecte pas la surface FastAPI", async () => {
    const env = await makeEnv();
    expect((await post(env, { slug: "spadzze", ts: "T1" })).status).toBe(422);
  });
});
