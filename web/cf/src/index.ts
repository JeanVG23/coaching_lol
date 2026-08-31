import { ACCOUNTS } from "./accounts";
import { apiCoach } from "./coach";
import { apiFeedback } from "./feedback";
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

type StoredReview = Record<string, unknown> & {
  ts?: string;
  kind?: string;
  model?: string;
  match_id?: string;
  payload?: unknown;
  review?: unknown;
};

function recordOf(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function pageParams(params: URLSearchParams): { page: number; size: number } | null {
  const page = Number(params.get("page") ?? 1);
  const size = Number(params.get("size") ?? 20);
  return Number.isInteger(page) && page >= 1 && Number.isInteger(size) && size >= 1 && size <= 100
    ? { page, size }
    : null;
}

function gameReviewSummary(item: StoredReview): Record<string, unknown> {
  const payload = recordOf(item.payload);
  const meta = recordOf(payload?.meta) ?? {};
  const review = recordOf(item.review);
  return {
    ts: item.ts ?? null,
    model: item.model ?? null,
    kind: "game",
    match_id: item.match_id ?? meta.match_id ?? null,
    meta,
    summary: {
      strengths_count: Array.isArray(review?.strengths) ? review.strengths.length : 0,
      mistakes_count: Array.isArray(review?.mistakes) ? review.mistakes.length : 0,
      next_focus: typeof review?.next_focus === "string" ? review.next_focus : null,
      confidence: typeof review?.confidence === "number" ? review.confidence : null,
    },
  };
}

async function apiAccounts(env: Env): Promise<Response> {
  const out: unknown[] = [];
  for (const account of ACCOUNTS) {
    const games = await readGames(env.DATA, account.slug, 1, 1);
    const reviews = await readJsonl<{ ts?: string; kind?: string }>(
      env.DATA,
      KEYS.reviews(account.slug),
    );
    const latestGlobalReview = [...reviews].reverse().find((review) => review.kind !== "game");
    out.push({
      slug: account.slug,
      riot_id: account.riot_id,
      region: account.region,
      games_count: games.total,
      last_review_ts: latestGlobalReview?.ts ?? null,
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

async function apiReviews(env: Env, slug: string, params: URLSearchParams): Promise<Response> {
  const kind = params.get("kind");
  const reviews = await readJsonl<StoredReview>(env.DATA, KEYS.reviews(slug));
  // Compatibilité de l'API V1 pour les clients qui ne demandent pas une vue paginée.
  if (kind === null) return Response.json(reviews);
  if (kind !== "aggregate" && kind !== "game") {
    return Response.json({ detail: "kind doit être aggregate ou game" }, { status: 422 });
  }
  const paging = pageParams(params);
  if (!paging) {
    return Response.json({ detail: "page>=1 et size in [1,100]" }, { status: 422 });
  }
  const filtered = reviews
    .filter((review) => kind === "game" ? review.kind === "game" : review.kind !== "game")
    .reverse();
  const start = (paging.page - 1) * paging.size;
  const slice = filtered.slice(start, start + paging.size);
  return Response.json({
    items: kind === "game" ? slice.map(gameReviewSummary) : slice,
    page: paging.page,
    size: paging.size,
    total: filtered.length,
  });
}

async function apiReviewDetail(env: Env, slug: string, ts: string): Promise<Response> {
  const reviews = await readJsonl<StoredReview>(env.DATA, KEYS.reviews(slug));
  const review = reviews.find((item) => item.ts === ts && item.kind === "game");
  return review
    ? Response.json(review)
    : Response.json({ detail: "analyse de partie introuvable" }, { status: 404 });
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
  const reviewDetail = url.pathname.match(/^\/api\/c\/([^/]+)\/reviews\/([^/]+)$/);
  if (reviewDetail) {
    if (request.method !== "GET") return Response.json({ detail: "Method Not Allowed" }, { status: 405 });
    return apiReviewDetail(env, reviewDetail[1], decodeURIComponent(reviewDetail[2]));
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
    if (tail === "reviews") return apiReviews(env, slug, url.searchParams);
    if (tail === "feedback") {
      return Response.json(await readJsonl(env.DATA, KEYS.feedback(slug)));
    }
    if (tail === "shap") return Response.json(await readShap(env.DATA, slug));
  }
  if (url.pathname === "/api/coach" && request.method === "POST") {
    return apiCoach(request, env);
  }
  if (url.pathname === "/api/feedback" && request.method === "POST") {
    return apiFeedback(request, env);
  }
  if (url.pathname.startsWith("/api/")) {
    return Response.json({ detail: "Not Found" }, { status: 404 });
  }
  return env.ASSETS.fetch(request);
}
