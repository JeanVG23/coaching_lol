import { ACCOUNTS } from "./accounts";
import { apiCoach } from "./coach";
import {
  KEYS,
  readGames,
  readJsonl,
  readPred,
  readRank,
  readShap,
  type KVLike,
} from "./readers";

export interface Env {
  DATA: KVLike;
  ASSETS: Fetcher;
  OLLAMA_API_KEY?: string;
  OLLAMA_MODEL?: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return handle(request, env);
  },
};

const EMPTY_RANK = {
  tier: null,
  division: null,
  league_points: null,
  wins: null,
  losses: null,
  fetched_at: null,
};
const EMPTY_PRED = { predicted_rank: null, proba: null, n_games_used: 0 };

async function apiAccounts(env: Env): Promise<Response> {
  const out: unknown[] = [];
  for (const account of ACCOUNTS) {
    const games = await readGames(env.DATA, account.slug, 1, 1);
    const reviews = await readJsonl<{ ts?: string }>(env.DATA, KEYS.reviews(account.slug));
    out.push({
      slug: account.slug,
      riot_id: account.riot_id,
      region: account.region,
      games_count: games.total,
      last_review_ts: reviews.length ? reviews[reviews.length - 1].ts ?? null : null,
    });
  }
  return Response.json(out);
}

async function apiGames(env: Env, slug: string, params: URLSearchParams): Promise<Response> {
  const page = Number(params.get("page") ?? 1);
  const size = Number(params.get("size") ?? 20);
  const okPage = Number.isInteger(page) && page >= 1;
  const okSize = Number.isInteger(size) && size >= 1 && size <= 200;
  if (!okPage || !okSize) {
    return Response.json({ detail: "page>=1 et size in [1,200]" }, { status: 422 });
  }
  return Response.json(await readGames(env.DATA, slug, page, size));
}

export async function handle(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  if (url.pathname === "/api/health") {
    return Response.json({
      status: "ok",
      service: "coaching-lol",
      server_time: new Date().toISOString(),
    });
  }
  if (url.pathname === "/api/accounts" && request.method === "GET") {
    return apiAccounts(env);
  }
  const match = url.pathname.match(/^\/api\/c\/([^/]+)\/([a-z-]+)$/);
  if (match) {
    const [, slug, tail] = match;
    if (request.method !== "GET") {
      return Response.json({ detail: "Method Not Allowed" }, { status: 405 });
    }
    if (tail === "games") return apiGames(env, slug, url.searchParams);
    if (tail === "rank") return Response.json((await readRank(env.DATA, slug)) ?? EMPTY_RANK);
    if (tail === "predicted-rank") {
      return Response.json((await readPred(env.DATA, slug)) ?? EMPTY_PRED);
    }
    if (tail === "reviews") {
      return Response.json(await readJsonl(env.DATA, KEYS.reviews(slug)));
    }
    if (tail === "feedback") {
      return Response.json(await readJsonl(env.DATA, KEYS.feedback(slug)));
    }
    if (tail === "shap") return Response.json(await readShap(env.DATA, slug));
  }
  if (url.pathname === "/api/coach" && request.method === "POST") {
    return apiCoach(request, env);
  }
  if (url.pathname.startsWith("/api/")) {
    return Response.json({ detail: "Not Found" }, { status: 404 });
  }
  return env.ASSETS.fetch(request);
}
