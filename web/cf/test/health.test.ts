import { describe, expect, it } from "vitest";
import { handle, type Env } from "../src/index";

const SPA_HTML = "<!doctype html><html><title>spa</title></html>";

function makeEnv(): Env {
  return {
    ASSETS: {
      fetch: async (req: Request) => {
        if (new URL(req.url).pathname === "/style.css") {
          return new Response("body{}", { headers: { "content-type": "text/css" } });
        }
        return new Response(SPA_HTML, { headers: { "content-type": "text/html" } });
      },
    },
  } as unknown as Env;
}

describe("handle", () => {
  it("GET /api/health répond ok", async () => {
    const r = await handle(new Request("http://x/api/health"), makeEnv());
    expect(r.status).toBe(200);
    const j = await r.json() as Record<string, unknown>;
    expect(j.status).toBe("ok");
    expect(j.service).toBe("coaching-lol");
    expect(typeof j.server_time).toBe("string");
  });

  it("GET /api/inconnu répond 404 JSON", async () => {
    const r = await handle(new Request("http://x/api/inconnu"), makeEnv());
    expect(r.status).toBe(404);
    expect(await r.json()).toEqual({ detail: "Not Found" });
  });

  it("les chemins non-API délèguent aux assets (SPA fallback /c/{slug})", async () => {
    for (const p of ["/", "/c/spadzze", "/readme"]) {
      const r = await handle(new Request(`http://x${p}`), makeEnv());
      expect(r.status).toBe(200);
      expect(await r.text()).toBe(SPA_HTML);
    }
    const css = await handle(new Request("http://x/style.css"), makeEnv());
    expect(css.headers.get("content-type")).toBe("text/css");
  });
});
