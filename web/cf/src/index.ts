export interface Env {
  ASSETS: Fetcher;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return handle(request, env);
  },
};

export async function handle(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  if (url.pathname === "/api/health") {
    return Response.json({
      status: "ok",
      service: "coaching-lol",
      server_time: new Date().toISOString(),
    });
  }
  if (url.pathname.startsWith("/api/")) {
    return Response.json({ detail: "Not Found" }, { status: 404 });
  }
  return env.ASSETS.fetch(request);
}
