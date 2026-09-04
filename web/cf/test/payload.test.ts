import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { addGameReviewCauses, buildPayload } from "../src/payload";

const goldenDirectory = new URL("./golden/", import.meta.url);

describe("parité golden payload (Python == TS)", () => {
  const files = readdirSync(goldenDirectory)
    .filter((file) => file.startsWith("payload_") && file.endsWith(".json"));

  it("a des fixtures à rejouer", () => expect(files.length).toBeGreaterThanOrEqual(7));
  for (const file of files) {
    const golden = JSON.parse(
      readFileSync(new URL(`./golden/${file}`, import.meta.url), "utf8"),
    );
    it(`rejoue ${file}`, () => {
      expect(buildPayload(golden.me, golden.ref, golden.args)).toEqual(golden.expected);
    });
  }
});

describe("map des reviews par partie", () => {
  it("conserve point+cause, retire evidence et filtre le scope", () => {
    const payload = { meta: { scope: "adc" }, signals: [], context: {} };
    const result = addGameReviewCauses(payload, [
      {
        ts: "2026-09-02", kind: "game", scope: "adc", match_id: "EUW1_42",
        payload: { meta: { champion: "Jinx", win: false } },
        review: { strengths: [], mistakes: [
          { point: "greed reset", cause: "attente", evidence: "1 268 g à 11:06" },
        ] },
      },
      {
        ts: "2026-09-03", kind: "game", scope: "mid",
        payload: { meta: { champion: "Ahri", win: true } },
        review: { strengths: [], mistakes: [{ point: "mid", cause: "mid" }] },
      },
    ], "adc");
    expect(result.meta.n_game_reviews_used).toBe(1);
    expect(result.game_review_causes).toEqual([{
      champion: "Jinx", outcome: "loss", strengths: [],
      mistakes: [{ point: "greed reset", cause: "attente" }],
    }]);
    expect(JSON.stringify(result)).not.toContain("1 268");
    expect(JSON.stringify(result)).not.toContain("EUW1_42");
  });
});
