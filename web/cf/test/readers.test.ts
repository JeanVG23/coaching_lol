import { describe, expect, it } from "vitest";
import { ACCOUNTS, accountFor } from "../src/accounts";
import {
  appendJsonl,
  KEYS,
  matchSeq,
  readGames,
  readJson,
  readJsonl,
  readPred,
  readRank,
  readShap,
  type KVLike,
} from "../src/readers";

class MemoryKV implements KVLike {
  store = new Map<string, string>();
  async get(key: string) { return this.store.get(key) ?? null; }
  async put(key: string, value: string) { this.store.set(key, value); }
}

const GAME = (id: string, champ: string) =>
  JSON.stringify({ match_id: id, champion: champ, win: true });

describe("KEYS", () => {
  it("fabrique les clés du layout spec §5", () => {
    expect(KEYS.games("spadzze")).toBe("silver:spadzze:games");
    expect(KEYS.rank("spadzze")).toBe("silver:spadzze:rank");
    expect(KEYS.gold("spadzze", "adc")).toBe("gold:spadzze:adc");
    expect(KEYS.ref("challenger", "adc")).toBe("ref:challenger:adc");
    expect(KEYS.pred("spadzze")).toBe("pred:spadzze");
    expect(KEYS.shap("spadzze")).toBe("shap:spadzze:drivers");
    expect(KEYS.reviews("spadzze")).toBe("coaching:spadzze:reviews");
    expect(KEYS.feedback("spadzze")).toBe("coaching:spadzze:feedback");
  });
});

describe("matchSeq", () => {
  it("extrait le suffixe numérique, fallback 0", () => {
    expect(matchSeq("EUW1_123456789")).toBe(123456789);
    expect(matchSeq("EUW1_1_42")).toBe(42);
    expect(matchSeq("pasunid")).toBe(0);
    expect(matchSeq("")).toBe(0);
  });
});

describe("readGames", () => {
  it("trie par séquence décroissante et pagine", async () => {
    const kv = new MemoryKV();
    await kv.put(
      KEYS.games("p"),
      ["EUW1_100", "EUW1_300", "EUW1_200"].map((id) => GAME(id, "Zeri")).join("\n"),
    );
    const p1 = await readGames(kv, "p", 1, 2);
    expect(p1.total).toBe(3);
    expect(p1.items.map((game) => game.match_id)).toEqual(["EUW1_300", "EUW1_200"]);
    const p2 = await readGames(kv, "p", 2, 2);
    expect(p2.items.map((game) => game.match_id)).toEqual(["EUW1_100"]);
    expect((await readGames(kv, "p", 3, 2)).items).toEqual([]);
  });

  it("clé absente → liste vide, pas d'erreur", async () => {
    expect(await readGames(new MemoryKV(), "inconnu", 1, 20)).toEqual({
      items: [], page: 1, size: 20, total: 0,
    });
  });
});

describe("readRank / readPred / readShap", () => {
  it("rank absent → null, présent → objet parsé", async () => {
    const kv = new MemoryKV();
    expect(await readRank(kv, "p")).toBeNull();
    await kv.put(KEYS.rank("p"), JSON.stringify({ tier: "MASTER", league_points: 300 }));
    expect(await readRank(kv, "p")).toEqual({ tier: "MASTER", league_points: 300 });
  });

  it("pred absent → null", async () => {
    expect(await readPred(new MemoryKV(), "p")).toBeNull();
  });

  it("shap : la valeur KV EST le tableau de drivers", async () => {
    const kv = new MemoryKV();
    expect(await readShap(kv, "p")).toEqual({ available: false, drivers: [] });
    await kv.put(KEYS.shap("p"), JSON.stringify([{ feature: "gd10", sv: 0.3 }]));
    expect(await readShap(kv, "p")).toEqual({
      available: true, drivers: [{ feature: "gd10", sv: 0.3 }],
    });
  });
});

describe("readJsonl / appendJsonl / readJson", () => {
  it("parse les lignes en ignorant les vides, append conserve l'existant", async () => {
    const kv = new MemoryKV();
    await kv.put("k", `${JSON.stringify({ a: 1 })}\n\n${JSON.stringify({ a: 2 })}\n`);
    expect(await readJsonl(kv, "k")).toEqual([{ a: 1 }, { a: 2 }]);
    await appendJsonl(kv, "k", { a: 3 });
    expect(await readJsonl(kv, "k")).toEqual([{ a: 1 }, { a: 2 }, { a: 3 }]);
    expect(await readJsonl(new MemoryKV(), "absent")).toEqual([]);
  });

  it("readJson : absent → null, JSON invalide → null", async () => {
    const kv = new MemoryKV();
    expect(await readJson(kv, "x")).toBeNull();
    await kv.put("x", "{pas du json");
    expect(await readJson(kv, "x")).toBeNull();
    await kv.put("x", "{\"ok\":true}");
    expect(await readJson(kv, "x")).toEqual({ ok: true });
  });
});

describe("accounts", () => {
  it("compte préconfiguré + lookup", () => {
    expect(ACCOUNTS).toContainEqual({ slug: "spadzze", riot_id: "Spadzze#euw", region: "euw1" });
    expect(accountFor("spadzze")?.riot_id).toBe("Spadzze#euw");
    expect(accountFor("inconnu")).toBeUndefined();
  });
});
