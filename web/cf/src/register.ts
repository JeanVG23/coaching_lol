/** Inscription d'un joueur : validation, slug, anti-doublon, routes HTTP.
 *
 * Le slug est calculé ICI et nulle part ailleurs : le service Python le reçoit
 * tel quel. Deux implémentations de la même normalisation dans deux runtimes
 * seraient un contrat de parité de plus, pour aucun gain.
 */
import { readAccount, type Account } from "./accounts";
import { jsonError, notFound, readJsonBody, unprocessable } from "./http";
import { REFRESH_COOLDOWN_MS, retryAfter, type JobStatus } from "./ingest_queue";
import type { Env } from "./index";

/** Miroir des clés de `PLATFORM_TO_REGIONAL` (src/core/riotlib.py).
 * Verrouillé par tests/test_platform_parity.py : proposer au visiteur une
 * plateforme que le service ne sait pas router produirait un `riot_id_not_found`
 * (le type levé par `riot_ingest.build_client`) une minute après l'inscription,
 * c'est-à-dire un doute sur son Riot ID, au lieu d'un refus immédiat et exact. */
export const PLATFORMS = [
  "euw1", "eun1", "tr1", "ru", "me1",
  "na1", "br1", "la1", "la2",
  "kr", "jp1",
  "oc1", "ph2", "sg2", "th2", "tw2", "vn2",
] as const;

/** Une inscription du même slug dans cette fenêtre rend les données existantes
 * sans rien collecter : c'est ce qui protège la clé Riot du visiteur qui
 * rafraîchit sa page. */
export const FRESH_WINDOW_MS = 6 * 60 * 60 * 1000;

/** Caractères refusés dans un pseudo : structure d'URL et points de contrôle.
 *
 * Une liste noire, et surtout pas une liste blanche latine : les pseudos Riot
 * acceptent le coréen, le cyrillique, le japonais et les espaces, qu'une liste
 * blanche ASCII refuserait. Ce qui est interdit ici est exactement ce qui
 * changerait la forme de l'URL construite côté service.
 *
 * La garde est doublée côté Python (`riotlib.puuid_from_riot_id` encode ses deux
 * segments) et c'est voulu : la validation refuse tôt et explique au visiteur,
 * l'encodage protège quel que soit l'appelant. */
const FORBIDDEN_IN_NAME = /[/?#%\\\u0000-\u001f\u007f]/;

export function parseRiotId(value: string): { gameName: string; tagLine: string } | null {
  const parts = value.trim().split("#");
  if (parts.length !== 2) return null;
  const gameName = parts[0].trim();
  const tagLine = parts[1].trim();
  if (!gameName || !tagLine) return null;
  if (gameName.length > 32 || !/^[A-Za-z0-9]{2,5}$/.test(tagLine)) return null;
  if (FORBIDDEN_IN_NAME.test(gameName)) return null;
  return { gameName, tagLine };
}

/** Hash FNV-1a, synchrone et pur : le slug se calcule uniquement côté
 * TypeScript, sans `crypto.subtle` ni `await`. Sert de repli quand un segment
 * ne contient aucun caractère ASCII alphanumérique (pseudo coréen, cyrillique,
 * purement ponctuation…) : sans ce repli, tous ces pseudos normaliseraient
 * vers le même segment vide et se disputeraient le même slug. */
function fnv1a(value: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export function slugFor(gameName: string, tagLine: string): string {
  const normalize = (s: string) =>
    s.normalize("NFD")
      .replace(/[̀-ͯ]/g, "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
  const segment = (raw: string) => normalize(raw) || `p${fnv1a(raw)}`;
  return `${segment(gameName)}-${segment(tagLine)}`;
}

function queueStub(env: Env) {
  // Une seule instance, nommée globalement : la sérialisation est globale parce
  // que la limite de débit Riot l'est.
  return env.INGEST_QUEUE!.get(env.INGEST_QUEUE!.idFromName("global"));
}

export async function apiRegister(request: Request, env: Env): Promise<Response> {
  if (!env.INGEST_QUEUE) return jsonError(503, "registration unavailable");
  const body = await readJsonBody<{ riot_id?: unknown; platform?: unknown }>(request);
  if (typeof body?.riot_id !== "string" || typeof body?.platform !== "string") {
    return unprocessable("riot_id and platform are required");
  }
  const platform = body.platform.toLowerCase();
  if (!(PLATFORMS as readonly string[]).includes(platform)) {
    return unprocessable("unknown platform");
  }
  const parsed = parseRiotId(body.riot_id);
  if (!parsed) return unprocessable("expected Riot ID format: Summoner#TAG");
  // Espaces internes réduits à un seul : "Le  Petit  Chat" et "Le Petit Chat"
  // sont le même Riot ID, sans quoi le même joueur obtiendrait un faux 409.
  const gameName = parsed.gameName.replace(/\s+/g, " ");

  const riotId = `${gameName}#${parsed.tagLine}`;
  const slug = slugFor(gameName, parsed.tagLine);

  const existing: Account | null = await readAccount(env.DATA, slug);
  if (existing && existing.riot_id.toLowerCase() !== riotId.toLowerCase()) {
    return jsonError(409, "this slug is already used by another account");
  }
  if (existing?.source === "curated") {
    // Une inscription n'est pas qu'une lecture : elle déclenche une collecte, donc
    // une ÉCRITURE sur `silver:{slug}:games` et `gold:{slug}:*`. Or ces clés ont
    // deux écrivains aux sémantiques opposées (le service fusionne,
    // sync_cloudflare.py remplace depuis le local). Sans preuve de propriété, un
    // étranger pourrait réécrire les agrégats d'un compte suivi. Le refus est un
    // 409 déjà existant : la liste des codes d'erreur du service reste fermée.
    return jsonError(409, "this account is already tracked and updates automatically; "
                          + "open it directly from the home page");
  }
  const last = existing?.last_ingest_ts ? Date.parse(existing.last_ingest_ts) : NaN;
  if (Number.isFinite(last) && Date.now() - last < FRESH_WINDOW_MS) {
    return Response.json({ slug, state: "done", fresh: true,
                           status_url: `/api/register/${slug}/status` });
  }

  const queued = await queueStub(env).fetch(new Request("http://do/enqueue", {
    method: "POST",
    body: JSON.stringify({ slug, riot_id: riotId, platform }),
  }));
  const status = await queued.json();
  return Response.json({ slug, status_url: `/api/register/${slug}/status`, ...status as object },
                       { status: 202 });
}

/** Réponse commune aux deux refus de cadence : le client n'a rien d'autre à
 * lire que le nombre de secondes, et `Retry-After` dit la même chose aux
 * intermédiaires HTTP. */
function tooEarly(seconds: number): Response {
  const minutes = Math.ceil(seconds / 60);
  return new Response(JSON.stringify({
    detail: `Data already up to date: you can refresh again in ${minutes} min`,
    retry_after: seconds,
  }), {
    status: 429,
    headers: { "content-type": "application/json", "retry-after": String(seconds) },
  });
}

/** Secondes restantes de la fenêtre de rafraîchissement, 0 si elle est passée. */
function cooldownLeft(lastIngestTs?: string): number {
  const last = lastIngestTs ? Date.parse(lastIngestTs) : NaN;
  if (!Number.isFinite(last)) return 0;
  return Date.now() - last < REFRESH_COOLDOWN_MS
    ? retryAfter(last, REFRESH_COOLDOWN_MS) : 0;
}

/** Recollecte un compte DÉJÀ enregistré, désigné par son slug.
 *
 * Distinct de `apiRegister`, et pas seulement par commodité : une inscription
 * calcule le slug depuis le Riot ID saisi, alors que les comptes curés portent
 * un slug écrit à la main dans `config/accounts.json` que `slugFor` ne
 * reproduira jamais (`Spadzze#euw` -> `spadzze-euw`, pas `spadzze`). Faire
 * passer le bouton « Actualiser » par l'inscription revenait donc à collecter un
 * compte fantôme à côté du vrai, aux frais de la clé Riot. Ici le slug de l'URL
 * EST la clé du compte : rien n'est deviné, et le Riot ID envoyé au service est
 * celui qui est stocké, pas celui que le navigateur a reconstitué. */
export async function apiRefresh(env: Env, slug: string): Promise<Response> {
  if (!env.INGEST_QUEUE) return jsonError(503, "data collection unavailable");
  const account = await readAccount(env.DATA, slug);
  if (!account) return notFound("account not found");

  const left = cooldownLeft(account.last_ingest_ts);
  if (left) return tooEarly(left);

  const queued = await queueStub(env).fetch(new Request("http://do/enqueue", {
    method: "POST",
    body: JSON.stringify({
      slug: account.slug, riot_id: account.riot_id, platform: account.region,
    }),
  }));
  const status = await queued.json() as JobStatus;
  // La file tranche en dernier : son horloge est fortement cohérente, celle de
  // `last_ingest_ts` ne l'est pas.
  if (status.retry_after) return tooEarly(status.retry_after);
  return Response.json({
    slug: account.slug,
    status_url: `/api/register/${account.slug}/status`,
    cooldown: Math.round(REFRESH_COOLDOWN_MS / 1000),
    ...status,
  }, { status: 202 });
}

export async function apiRegisterStatus(env: Env, slug: string): Promise<Response> {
  if (!env.INGEST_QUEUE) return jsonError(503, "registration unavailable");
  const response = await queueStub(env)
    .fetch(new Request(`http://do/status?slug=${encodeURIComponent(slug)}`));
  if (response.status === 404) {
    // Un job expiré mais un compte présent : l'inscription a bien abouti.
    const account = await readAccount(env.DATA, slug);
    return account
      ? Response.json({ state: "done", slug })
      : notFound("unknown registration");
  }
  return response;
}
