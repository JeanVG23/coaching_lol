import { describe, expect, it } from "vitest";
import {
  countKdaValue,
  diffClass,
  formatConfidence,
  formatDate,
  formatDiff,
  formatDuration,
  formatFullDate,
  formatGameDate,
  formatKda,
  formatNumber,
  formatPercent,
  parseDateValue,
} from "./formatters";

describe("formatters", () => {
  it("countKdaValue counts array length or parses number", () => {
    expect(countKdaValue([1, 2, 3])).toBe(3);
    expect(countKdaValue(5)).toBe(5);
    expect(countKdaValue(0)).toBe(0);
    expect(countKdaValue(null)).toBe(0);
    expect(countKdaValue(undefined)).toBe(0);
  });

  it("formatKda formats numbers or arrays correctly", () => {
    expect(formatKda({ kills: 5, deaths: 2, assists: 3 })).toBe("5/2/3 · 4.00");
    expect(formatKda({ kills: 10, deaths: 0, assists: 5 })).toBe("10/0/5 · Perfect");
    expect(formatKda({ kills: [1, 2], deaths: [1], assists: [1, 2, 3] })).toBe("2/1/3 · 5.00");
    expect(formatKda(null)).toBeNull();
    expect(formatKda(undefined)).toBeNull();
  });

  it("formatDuration formats minutes properly", () => {
    expect(formatDuration(28.4)).toBe("28 min");
    expect(formatDuration(30)).toBe("30 min");
    expect(formatDuration(0)).toBeNull();
    expect(formatDuration(-5)).toBeNull();
    expect(formatDuration(null)).toBeNull();
    expect(formatDuration("abc")).toBeNull();
  });

  it("formatConfidence formats confidence with optional prefix", () => {
    expect(formatConfidence(0.85)).toBe("85%");
    expect(formatConfidence(0.854)).toBe("85%");
    expect(formatConfidence(0.856)).toBe("86%");
    expect(formatConfidence(0.85, "Confidence")).toBe("Confidence 85%");
    expect(formatConfidence(null)).toBe("—");
    expect(formatConfidence(undefined)).toBe("—");
  });

  it("formatPercent formats percentage", () => {
    expect(formatPercent(0.52)).toBe("52 %");
    expect(formatPercent(null)).toBe("—");
    expect(formatPercent(undefined)).toBe("—");
  });

  it("formatNumber formats fixed digits", () => {
    expect(formatNumber(12.345, 1)).toBe("12.3");
    expect(formatNumber(12.345, 2)).toBe("12.35");
    expect(formatNumber(null)).toBe("—");
  });

  it("formatDiff and diffClass handle positive, negative, and neutral values", () => {
    expect(formatDiff(150, " g")).toBe("+150 g");
    expect(formatDiff(-50, " g")).toBe("-50 g");
    expect(formatDiff(0, " g")).toBe("0 g");
    expect(formatDiff(null)).toBe("—");

    expect(diffClass(10)).toBe("diff-pos");
    expect(diffClass(-5)).toBe("diff-neg");
    expect(diffClass(0)).toBe("diff-neutral");
    expect(diffClass(null)).toBe("diff-neutral");
  });

  it("parseDateValue parses timestamps, numeric strings, and ISO strings", () => {
    expect(parseDateValue(null)).toBeNull();
    expect(parseDateValue(undefined)).toBeNull();
    expect(parseDateValue(0)).toBeNull();
    expect(parseDateValue(-100)).toBeNull();
    expect(parseDateValue("invalid")).toBeNull();

    const timestamp = 1710000000000;
    const date = parseDateValue(timestamp);
    expect(date).toBeInstanceOf(Date);
    expect(date?.getTime()).toBe(timestamp);

    const strDate = parseDateValue("2026-03-09T16:00:00Z");
    expect(strDate).toBeInstanceOf(Date);
  });

  it("formatDate formats dates or falls back gracefully", () => {
    expect(formatDate(null)).toBe("—");
    expect(formatDate("")).toBe("—");
    expect(formatDate(1710000000000)).toMatch(/2024/);
  });

  it("formatGameDate and formatFullDate format correctly or return empty string", () => {
    expect(formatGameDate(null)).toBe("");
    expect(formatGameDate(1710000000000)).toMatch(/2024/);

    expect(formatFullDate(null)).toBe("");
    expect(formatFullDate(1710000000000)).toMatch(/2024/);
  });
});
