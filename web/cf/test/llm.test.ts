import { describe, expect, it, vi } from "vitest";
import { generateJson, LLMError } from "../src/llm_client";

const OK_BODY = JSON.stringify({
  message: { content: JSON.stringify({ strengths: [], ok: true }) },
});
const noSleep = async () => {};

function response(status: number, body: string = OK_BODY): Response {
  return new Response(body, { status });
}

describe("generateJson", () => {
  it("retente 429/5xx puis réussit avec backoff 2s, 4s", async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response(500))
      .mockResolvedValueOnce(response(429))
      .mockResolvedValueOnce(response(200));
    const sleepImpl = vi.fn(noSleep);
    const output = await generateJson("kimi-k2.6", "s", "u", {}, {
      apiKey: "k", fetchImpl, sleepImpl,
    });
    expect(output).toEqual({ strengths: [], ok: true });
    expect(fetchImpl).toHaveBeenCalledTimes(3);
    expect(sleepImpl.mock.calls).toEqual([[2_000], [4_000]]);
  });

  it("401 -> LLMError immédiate, sans retry", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(response(401));
    await expect(generateJson("m", "s", "u", {}, {
      apiKey: "k", fetchImpl, sleepImpl: noSleep,
    })).rejects.toThrow(LLMError);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("contenu non-JSON -> retry puis LLMError après 4 tentatives", async () => {
    const fetchImpl = vi.fn().mockImplementation(async () => response(
      200,
      JSON.stringify({ message: { content: "{pas du json" } }),
    ));
    await expect(generateJson("m", "s", "u", {}, {
      apiKey: "k", fetchImpl, sleepImpl: noSleep,
    })).rejects.toThrow("dernier motif : contenu LLM non JSON");
    expect(fetchImpl).toHaveBeenCalledTimes(4);
  });

  it("conserve le dernier statut retryable dans l'erreur finale", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(response(503));
    await expect(generateJson("m", "s", "u", {}, {
      apiKey: "k", fetchImpl, sleepImpl: noSleep,
    })).rejects.toThrow("dernier motif : HTTP 503");
  });

  it("erreur réseau -> retry, puis réussit", async () => {
    const fetchImpl = vi.fn()
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockResolvedValueOnce(response(200));
    const output = await generateJson("m", "s", "u", {}, {
      apiKey: "k", fetchImpl, sleepImpl: noSleep,
    });
    expect(output).toEqual({ strengths: [], ok: true });
  });

  it("body : format=schema, stream=false, température défaut 0.2", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(response(200));
    await generateJson("kimi-k2.6", "sys", "usr", { type: "object" }, {
      apiKey: "k", fetchImpl, sleepImpl: noSleep,
    });
    const init = fetchImpl.mock.calls[0][1] as RequestInit;
    const body = JSON.parse(init.body as string);
    expect(body.model).toBe("kimi-k2.6");
    expect(body.messages).toEqual([
      { role: "system", content: "sys" },
      { role: "user", content: "usr" },
    ]);
    expect(body.format).toEqual({ type: "object" });
    expect(body.stream).toBe(false);
    expect(body.options.temperature).toBe(0.2);
  });
});
