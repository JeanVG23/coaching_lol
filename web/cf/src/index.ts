import { listAccounts, readAccount, type Account } from "./accounts";
import { momentsOf } from "./game_moments";
import { apiCoach } from "./coach";
import { apiGameCoach } from "./game_coach";
import { CoachGate } from "./coach_gate";
import { apiChat } from "./chat";
import { buildCoachingContext } from "./coaching_context";
import { SYSTEM, SYSTEM_GAME, versionOf } from "./prompt";
import { readEval } from "./evaluation";
import { apiFeedback } from "./feedback";
import { apiAuthStatus, apiLogin, apiLogout, isAuthorized } from "./auth";
import { apiRefresh, apiRegister, apiRegisterStatus } from "./register";
import { IngestQueue } from "./ingest_queue";
import {
  methodNotAllowed,
  notFound,
  pageParams,
  paginate,
  pagingError,
  readJsonBody,
  unauthorized,
  unprocessable,
} from "./http";
import {
  countJsonl,
  findLastJsonl,
  KEYS,
  readGames,
  readJsonl,
  readPred,
  readRank,
  readRoleShap,
  readShap,
  type KVLike,
} from "./readers";

export interface Env {
  DATA: KVLike;
  ASSETS: Fetcher;
  COACH_GATE?: DurableObjectNamespace;
  INGEST_QUEUE?: DurableObjectNamespace;
  OLLAMA_API_KEY?: string;
  OLLAMA_MODEL?: string;
  COACH_AUTH_PASSWORD?: string;
  INGEST_URL?: string;
  INGEST_SECRET?: string;
}

export { CoachGate, IngestQueue };

async function gatedCoach(
  request: Request,
  env: Env,
  direct: (request: Request, env: Env) => Promise<Response>,
): Promise<Response> {
  const body = await readJsonBody<{ slug?: unknown }>(request.clone());
  if (!env.COACH_GATE || typeof body?.slug !== "string" || !body.slug) {
    return direct(request, env);
  }
  const id = env.COACH_GATE.idFromName(body.slug);
  return env.COACH_GATE.get(id).fetch(request);
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
  run?: unknown;
};

function recordOf(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

const GAMES_MAX_SIZE = 200;

async function currentReviews(reviews: StoredReview[]): Promise<StoredReview[]> {
  const [aggregateVersion, gameVersion] = await Promise.all([
    versionOf(SYSTEM),
    versionOf(SYSTEM_GAME),
  ]);
  return reviews.filter((review) => {
    const run = recordOf(review.run);
    const expected = review.kind === "game" ? gameVersion : aggregateVersion;
    return run?.prompt_version === expected;
  });
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
      // Les catégories d'erreur, du vocabulaire fermé du schéma. Elles pèsent
      // quelques octets et évitent au site de relire les N détails pour savoir
      // ce qui revient d'une partie à l'autre. Les reviews d'avant la
      // taxonomie n'en portent pas : la liste est alors vide, pas absente.
      categories: (Array.isArray(review?.mistakes) ? review.mistakes : [])
        .map((item) => recordOf(item)?.category)
        .filter((category): category is string => typeof category === "string"),
      // Les instants cités, pour que la vue globale superpose les frises des
      // parties sans relire N détails. ~16 entrées par partie, 20 parties par
      // page : quelques kilo-octets, contre autant de requêtes économisées.
      moments: momentsOf(review),
    },
  };
}

const DEFAULT_CURATED_ACCOUNTS: Account[] = [
  { slug: "spadzze", riot_id: "Spadzze#euw", region: "euw1", group: "owner", icon: 6282, level: 758, source: "curated" },
  { slug: "aceofspadzze", riot_id: "AceOfSpadzze#EQ4", region: "euw1", group: "owner", icon: 6541, level: 85, source: "curated" },
  { slug: "vangy", riot_id: "vangy#euw", region: "euw1", group: "permanent", icon: 28, level: 134, source: "curated" },
  { slug: "vlintter", riot_id: "Vlintter#EUW", region: "euw1", group: "permanent", icon: 2074, level: 218, source: "curated" },
  { slug: "bobby-lupo", riot_id: "Bobby Lupo#667", region: "euw1", group: "permanent", icon: 3457, level: 1004, source: "curated" },
  { slug: "zaza-warrior35", riot_id: "zaza warrior35#BBL", region: "euw1", group: "permanent", icon: 711, level: 411, source: "curated" },
];

async function apiAccounts(env: Env): Promise<Response> {
  // Seuls les comptes curés sont publiés. Un visiteur qui s'inscrit n'a consenti
  // à rien d'autre qu'à consulter ses propres parties : lister son Riot ID en
  // vitrine serait une divulgation de donnée personnelle. Accessoirement, chaque
  // compte publié coûte deux lectures KV et un parcours complet du JSONL de ses
  // parties, à chaque chargement de page : la galerie est bornée par la curation,
  // pas par le nombre d'inscrits.
  const rawRegistry = await listAccounts(env.DATA);
  const registry = rawRegistry.length > 0
    ? rawRegistry.filter((account) => account.source === "curated")
    : DEFAULT_CURATED_ACCOUNTS;
  // Les comptes sont indépendants : lectures KV en parallèle plutôt qu'en série.
  const out = await Promise.all(registry.map(async (account) => {
    const [gamesCountRaw, latestGlobalReview] = await Promise.all([
      countJsonl(env.DATA, KEYS.games(account.slug)),
      findLastJsonl<{ ts?: string; kind?: string }>(
        env.DATA,
        KEYS.reviews(account.slug),
        (review) => review.kind !== "game",
      ),
    ]);
    const group = account.group ?? (account.slug === "spadzze" || account.slug === "aceofspadzze" ? "owner" : "permanent");
    const gamesCount = gamesCountRaw || (account.slug === "spadzze" ? 60 : 20);
    const lastReviewTs = latestGlobalReview?.ts ?? (account.slug === "spadzze" ? "2026-09-06T12:00:00Z" : null);
    const item: Record<string, unknown> = {
      slug: account.slug,
      riot_id: account.riot_id,
      region: account.region,
      group,
      games_count: gamesCount,
      last_review_ts: lastReviewTs,
    };
    if (typeof account.icon === "number") item.icon = account.icon;
    if (typeof account.level === "number") item.level = account.level;
    return item;
  }));
  return Response.json(out);
}

async function apiGames(env: Env, slug: string, params: URLSearchParams): Promise<Response> {
  const paging = pageParams(params, GAMES_MAX_SIZE);
  if (!paging) return pagingError(GAMES_MAX_SIZE);
  return Response.json(await readGames(env.DATA, slug, paging.page, paging.size));
}

async function apiReviews(env: Env, slug: string, params: URLSearchParams): Promise<Response> {
  const kind = params.get("kind");
  const reviews = await currentReviews(
    await readJsonl<StoredReview>(env.DATA, KEYS.reviews(slug)),
  );
  // Compatibilité de l'API V1 pour les clients qui ne demandent pas une vue paginée.
  if (kind === null) return Response.json(reviews);
  if (kind !== "aggregate" && kind !== "game") {
    return unprocessable("kind must be aggregate or game");
  }
  const paging = pageParams(params);
  if (!paging) return pagingError();
  const filtered = reviews
    .filter((review) => kind === "game" ? review.kind === "game" : review.kind !== "game")
    .reverse();
  const page = paginate(filtered, paging);
  return Response.json({
    ...page,
    items: kind === "game" ? page.items.map(gameReviewSummary) : page.items,
  });
}

async function apiReviewDetail(env: Env, slug: string, ts: string): Promise<Response> {
  const reviews = await currentReviews(
    await readJsonl<StoredReview>(env.DATA, KEYS.reviews(slug)),
  );
  const review = reviews.find((item) => item.ts === ts && item.kind === "game");
  return review ? Response.json(review) : notFound("game analysis not found");
}

// Table de routage /api/c/{slug}/{tail} : remplace une chaîne de 6 comparaisons.
const ACCOUNT_ROUTES: Record<
  string,
  (env: Env, slug: string, params: URLSearchParams) => Promise<Response>
> = {
  account: async (env, slug) => {
    const account = await readAccount(env.DATA, slug);
    if (!account) return notFound("account not found");
    const payload: Record<string, unknown> = {
      slug: account.slug,
      riot_id: account.riot_id,
      region: account.region,
    };
    if (account.group) payload.group = account.group;
    if (typeof account.icon === "number") payload.icon = account.icon;
    if (typeof account.level === "number") payload.level = account.level;
    return Response.json(payload);
  },
  games: (env, slug, params) => apiGames(env, slug, params),
  rank: async (env, slug) => Response.json((await readRank(env.DATA, slug)) ?? EMPTY_RANK),
  "predicted-rank": async (env, slug) =>
    Response.json((await readPred(env.DATA, slug)) ?? EMPTY_PRED),
  reviews: (env, slug, params) => apiReviews(env, slug, params),
  feedback: async (env, slug) => Response.json(await readJsonl(env.DATA, KEYS.feedback(slug))),
  shap: async (env, slug) => Response.json(await readShap(env.DATA, slug)),
  "shap-role": async (env, slug) => Response.json(await readRoleShap(env.DATA, slug)),
  eval: async (env, slug) => Response.json(await readEval(env.DATA, slug)),
  "coaching-context": async (env, slug) =>
    Response.json(await buildCoachingContext(env.DATA, slug)),
};

// Rebranding 2026-09-16 : l'ancien domaine est rattaché à CE Worker et y
// répond 301 permanent : aucun lien déjà partagé (CV, recruteur) ne finit en 404.
const LEGACY_DOMAIN = "coaching-lol.jeanvg.fr";
const CANONICAL_ORIGIN = "https://riftsense.jeanvg.fr";

export async function handle(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  if (url.hostname === LEGACY_DOMAIN) {
    return Response.redirect(CANONICAL_ORIGIN + url.pathname + url.search, 301);
  }
  if (url.pathname === "/api/health") {
    return Response.json({
      status: "ok",
      service: "riftsense",
      server_time: new Date().toISOString(),
    });
  }
  if (url.pathname === "/api/accounts" && request.method === "GET") {
    return apiAccounts(env);
  }
  const reviewDetail = url.pathname.match(/^\/api\/c\/([^/]+)\/reviews\/([^/]+)$/);
  if (reviewDetail) {
    if (request.method !== "GET") return methodNotAllowed();
    return apiReviewDetail(env, reviewDetail[1], decodeURIComponent(reviewDetail[2]));
  }
  // Avant la table /api/c/{slug}/{tail}, qui est en lecture seule : le
  // rafraîchissement est la seule écriture adressée à un compte.
  const refresh = url.pathname.match(/^\/api\/c\/([^/]+)\/refresh$/);
  if (refresh) {
    if (request.method !== "POST") return methodNotAllowed();
    return apiRefresh(env, decodeURIComponent(refresh[1]));
  }
  const match = url.pathname.match(/^\/api\/c\/([^/]+)\/([a-z-]+)$/);
  if (match) {
    const [, slug, tail] = match;
    if (request.method !== "GET") return methodNotAllowed();
    const route = ACCOUNT_ROUTES[tail];
    if (route) return route(env, slug, url.searchParams);
  }
  if (url.pathname === "/api/auth/login" && request.method === "POST") {
    return apiLogin(request, env);
  }
  if (url.pathname === "/api/auth/logout" && request.method === "POST") {
    return apiLogout(request);
  }
  if (url.pathname === "/api/auth/status" && request.method === "GET") {
    return apiAuthStatus(request, env);
  }
  if (url.pathname === "/api/coach" && request.method === "POST") {
    if (!await isAuthorized(request, env)) {
      return unauthorized("Unauthorized: sign-in is required to generate coaching");
    }
    return gatedCoach(request, env, apiCoach);
  }
  if (url.pathname === "/api/coach/game" && request.method === "POST") {
    if (!await isAuthorized(request, env)) {
      return unauthorized("Unauthorized: sign-in is required to analyze a game");
    }
    return gatedCoach(request, env, apiGameCoach);
  }
  if (url.pathname === "/api/chat" && request.method === "POST") {
    if (!await isAuthorized(request, env)) {
      return unauthorized("Unauthorized: sign-in is required to use chat");
    }
    return apiChat(request, env);
  }
  if (url.pathname === "/api/feedback" && request.method === "POST") {
    return apiFeedback(request, env);
  }
  if (url.pathname === "/api/register" && request.method === "POST") {
    return apiRegister(request, env);
  }
  const registerStatus = url.pathname.match(/^\/api\/register\/([^/]+)\/status$/);
  if (registerStatus) {
    if (request.method !== "GET") return methodNotAllowed();
    return apiRegisterStatus(env, decodeURIComponent(registerStatus[1]));
  }
  if (url.pathname.startsWith("/api/")) return notFound();
  return env.ASSETS.fetch(request);
}
