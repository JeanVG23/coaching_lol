import { describe, expect, it } from "vitest";
import { render, SYSTEM } from "../src/prompt";

describe("prompt", () => {
  it("SYSTEM porte les règles d'asymétrie et de benchmark", () => {
    expect(SYSTEM).toContain("ASYMÉTRIE");
    expect(SYSTEM).toContain("BENCHMARK-RELATIF");
    expect(SYSTEM.length).toBeGreaterThan(500);
  });

  it("render(payload) sérialise le contexte utilisateur", () => {
    const payload = {
      meta: {
        n_games_me: 18,
        scope: "adc",
        outcome_focus: "loss",
        target: "challenger",
      },
      signals: [],
    };
    const [system, user] = render(payload);
    expect(system).toBe(SYSTEM);
    expect(user).toContain("Signaux de tes 18 dernières games");
    expect(user).toContain("adc");
    expect(user).toContain("challenger");
    expect(user).toContain(JSON.stringify(payload, undefined, 2));
    expect(user.trimEnd().endsWith("Produis la review.")).toBe(true);
  });
});
