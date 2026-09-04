import { describe, expect, it } from "vitest";
import {
  asksForHiddenEnemyPosition,
  chatTurn,
  SYSTEM_CHAT,
  type ChatMessage,
} from "../src/chat";
import { KEYS, readJsonl, type KVLike } from "../src/readers";

class MemoryKV implements KVLike {
  store = new Map<string, string>();
  async get(key: string) { return this.store.get(key) ?? null; }
  async put(key: string, value: string) { this.store.set(key, value); }
}

async function seeded(): Promise<MemoryKV> {
  const kv = new MemoryKV();
  await kv.put(KEYS.reviews("spadzze"), JSON.stringify({
    ts: "review-1", kind: "game",
    payload: {
      meta: { match_id: "EUW1_42" },
      context: { comp: { enemy_adc: "Caitlyn", enemy_jungle: "Skarner" } },
      journal: { deaths: [{ clock: "11:06", unspent_gold: 1268 }] },
    },
    review: { mistakes: [{ point: "reset", evidence: "11:06" }] },
  }));
  return kv;
}

describe("garde-fou d'asymétrie du chat", () => {
  it("détecte la question explicite sur le jungler ennemi", () => {
    const messages: ChatMessage[] = [{
      role: "user", content: "Où était le jungler ennemi à ce moment-là ?",
    }];
    expect(asksForHiddenEnemyPosition(messages, {})).toBe(true);
    expect(SYSTEM_CHAT).toContain("refuse explicitement");
    expect(SYSTEM_CHAT).toContain("timeline complète");
  });

  it("refuse mécaniquement sans appeler le LLM et persiste l'échange", async () => {
    const kv = await seeded();
    let calls = 0;
    const result = await chatTurn({
      kv, now: () => "2026-09-04T20:00:00Z",
      generate: async () => { calls += 1; return {}; },
    }, {
      slug: "spadzze", reviewTs: "review-1", model: "m",
      messages: [{ role: "user", content: "Où était Caitlyn ?" }],
    });
    expect(calls).toBe(0);
    expect(result.response.refused_hidden_info).toBe(true);
    expect(await readJsonl(kv, KEYS.chats("spadzze"))).toHaveLength(1);
  });

  it("répond via le modèle à une justification du joueur", async () => {
    const kv = await seeded();
    const result = await chatTurn({
      kv, now: () => "t",
      generate: async (_model, system, user) => {
        expect(system).toBe(SYSTEM_CHAT);
        expect(user).toContain("1268");
        return { answer: "Oui, attendre la B.F. Sword peut être légitime.",
                 refused_hidden_info: false };
      },
    }, {
      slug: "spadzze", reviewTs: "review-1", model: "m",
      messages: [{ role: "user", content: "J'attendais ma B.F. Sword." }],
    });
    expect(result.response.refused_hidden_info).toBe(false);
  });
});
