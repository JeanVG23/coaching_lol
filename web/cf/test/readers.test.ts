import { describe, expect, it } from "vitest";
import {
  appendJsonl,
  countJsonl,
  findLastJsonl,
  KEYS,
  matchSeq,
  readGames,
  readGamePayloadBundle,
  readJson,
  readJsonl,
  readPred,
  readRank,
  readRoleShap,
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
    expect(KEYS.role_shap("spadzze")).toBe("shap:spadzze:role");
    expect(KEYS.scan_index("spadzze")).toBe("riftsense:spadzze:scan-index");
    expect(KEYS.reviews("spadzze")).toBe("riftsense:spadzze:reviews");
    expect(KEYS.feedback("spadzze")).toBe("riftsense:spadzze:feedback");
    expect(KEYS.chats("spadzze")).toBe("riftsense:spadzze:chats");
    expect(KEYS.game_payloads("spadzze")).toBe("riftsense:spadzze:game-payloads");
  });
});

describe("readGamePayloadBundle", () => {
  it("dégrade vers un bundle vide et lit une valeur valide", async () => {
    const kv = new MemoryKV();
    expect((await readGamePayloadBundle(kv, "p")).items).toEqual({});
    await kv.put(KEYS.game_payloads("p"), JSON.stringify({
      generated_at: "2026-09-06T10:00:00Z", target: "challenger", max_games: 50,
      items: { m1: { payload_hash: "abc", benchmark_scope: "adc", payload: { meta: {} } } },
      unavailable: [],
    }));
    expect((await readGamePayloadBundle(kv, "p")).items.m1.payload_hash).toBe("abc");
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

describe("readRoleShap", () => {
  it("rend l'enveloppe telle qu'elle a été publiée", async () => {
    const kv = new MemoryKV();
    await kv.put("shap:jean:role", JSON.stringify({
      schema_version: 1, available: false, reason: "role_closed", role: "JUNGLE",
    }));
    expect(await readRoleShap(kv, "jean")).toEqual({
      schema_version: 1, available: false, reason: "role_closed", role: "JUNGLE",
    });
  });

  it("transforme une clé absente en motif, pour que l'interface n'ait qu'un cas", async () => {
    // Sans ce repli, le client devrait distinguer « pas de clé » de « clé qui dit
    // non », et finirait par afficher un vide là où il y a une raison.
    expect(await readRoleShap(new MemoryKV(), "jean")).toEqual({
      schema_version: 1, available: false, reason: "not_ingested", role: null,
    });
  });

  it("refuse un contenu qui n'est pas une enveloppe", async () => {
    const kv = new MemoryKV();
    await kv.put("shap:jean:role", JSON.stringify([1, 2, 3]));
    expect((await readRoleShap(kv, "jean")).reason).toBe("not_ingested");
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

  it("readJsonl : ignore les lignes malformées sans lever d'erreur", async () => {
    const kv = new MemoryKV();
    await kv.put("k", `{"valid":1}\n{ligne-cassee\n{"valid":2}\n`);
    expect(await readJsonl(kv, "k")).toEqual([{ valid: 1 }, { valid: 2 }]);
  });

  it("countJsonl : compte les lignes non vides sans parser le JSON", async () => {
    const kv = new MemoryKV();
    expect(await countJsonl(kv, "absent")).toBe(0);
    await kv.put("k", "\nline1\n\nline2\nline3\n  \n");
    expect(await countJsonl(kv, "k")).toBe(3);
  });

  it("findLastJsonl : scanne depuis la fin et ignore les lignes corrompues", async () => {
    const kv = new MemoryKV();
    expect(await findLastJsonl(kv, "absent", () => true)).toBeNull();
    await kv.put("reviews", [
      JSON.stringify({ id: 1, kind: "global" }),
      JSON.stringify({ id: 2, kind: "game" }),
      "{broken json line",
      JSON.stringify({ id: 3, kind: "game" }),
    ].join("\n"));
    const lastGlobal = await findLastJsonl<{ id: number; kind: string }>(
      kv, "reviews", (r) => r.kind === "global",
    );
    expect(lastGlobal).toEqual({ id: 1, kind: "global" });
    const lastGame = await findLastJsonl<{ id: number; kind: string }>(
      kv, "reviews", (r) => r.kind === "game",
    );
    expect(lastGame).toEqual({ id: 3, kind: "game" });
  });
});
