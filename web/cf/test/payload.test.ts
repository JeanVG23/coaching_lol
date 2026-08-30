import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { buildPayload } from "../src/payload";

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
