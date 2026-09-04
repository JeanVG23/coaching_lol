import { describe, expect, it } from "vitest";
import { readEval, TARGET_N, TARGET_RATE } from "../src/evaluation";
import { KEYS, type KVLike } from "../src/readers";

class MemoryKV implements KVLike {
  store = new Map<string, string>();
  async get(key: string) { return this.store.get(key) ?? null; }
  async put(key: string, value: string) { this.store.set(key, value); }
}

const jsonl = (rows: unknown[]) => rows.map((r) => JSON.stringify(r)).join("\n") + "\n";

function kvWith(reviews: unknown[], feedback: unknown[]): MemoryKV {
  const kv = new MemoryKV();
  kv.store.set(KEYS.reviews("s"), jsonl(reviews));
  kv.store.set(KEYS.feedback("s"), jsonl(feedback));
  return kv;
}

const mistake = (useful: boolean, tag?: string) =>
  ({ kind: "mistake", index: 0, useful, tag: tag ?? null });

describe("readEval", () => {
  it("ne compte que les mistakes des analyses par-partie", async () => {
    // La review agrégée est annotée elle aussi : ses items ne doivent pas
    // gonfler la métrique par-partie (objectif défini sur les parties).
    const kv = kvWith(
      [{ ts: "g1", kind: "game" }, { ts: "a1" }],
      [
        { ts: "g1", items: [mistake(true), mistake(false, "trop-vague")] },
        { ts: "a1", items: [mistake(true), mistake(true)] },
      ],
    );
    const report = await readEval(kv, "s");
    expect(report.objective.n_game_reviews_annotated).toBe(1);
    expect(report.objective.mistake_useful_rate).toBe(0.5);
    expect(report.n_game_reviews).toBe(1);
    expect(report.n_reviews_annotated).toBe(2);          // toutes sections confondues
    expect(report.target_met).toBe(false);               // 1 review < TARGET_N
  });

  it("atteint la cible sur 10 analyses annotées à 70 %", async () => {
    const reviews = Array.from({ length: TARGET_N }, (_, i) => ({ ts: `g${i}`, kind: "game" }));
    const feedback = reviews.map((r, i) => ({ ts: r.ts, items: [mistake(i < 7)] }));
    const report = await readEval(kvWith(reviews, feedback), "s");
    expect(report.objective.mistake_useful_rate).toBeCloseTo(TARGET_RATE);
    expect(report.target_met).toBe(true);
  });

  it("rend un taux null plutôt que 0 quand rien n'est annoté", async () => {
    const report = await readEval(kvWith([{ ts: "g1", kind: "game" }], []), "s");
    expect(report.objective.mistake_useful_rate).toBeNull();
    expect(report.global_rate).toBeNull();
    expect(report.target_met).toBe(false);
  });

  it("classe les tags des insights rejetés", async () => {
    const kv = kvWith(
      [{ ts: "g1", kind: "game" }],
      [{ ts: "g1", items: [
        mistake(false, "non-actionnable"),
        mistake(false, "non-actionnable"),
        mistake(false, "trop-vague"),
        mistake(true),
      ] }],
    );
    const report = await readEval(kv, "s");
    expect(report.top_tags).toEqual([
      { tag: "non-actionnable", n: 2 },
      { tag: "trop-vague", n: 1 },
    ]);
    expect(report.by_kind.mistake).toEqual({ n: 4, useful: 1, rate: 0.25 });
  });
});
