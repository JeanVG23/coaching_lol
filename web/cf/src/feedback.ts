import { accountFor } from "./accounts";
import { jsonError, notFound, unprocessable } from "./http";
import { KEYS, readJsonl, upsertJsonl } from "./readers";
import { NEG_TAGS, validateGameReview, validateReview, type Review } from "./schema";
import type { Env } from "./index";

interface ResponseItem {
  useful: boolean;
  tag: string | null;
  note: string | null;
}

const MAX_FEEDBACK_PER_HOUR = 30;

function foreignOrigin(request: Request): boolean {
  const origin = request.headers.get("Origin");
  return origin !== null && origin !== new URL(request.url).origin;
}

async function rateKey(request: Request): Promise<string> {
  const ip = request.headers.get("CF-Connecting-IP") ?? "anonymous";
  const bytes = new TextEncoder().encode(ip);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const fingerprint = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 20);
  const hour = Math.floor(Date.now() / 3_600_000);
  return `rate:feedback:${hour}:${fingerprint}`;
}

async function feedbackRateLimited(request: Request, env: Env): Promise<boolean> {
  const key = await rateKey(request);
  const count = Number(await env.DATA.get(key) ?? "0");
  if (Number.isFinite(count) && count >= MAX_FEEDBACK_PER_HOUR) return true;
  await env.DATA.put(key, String((Number.isFinite(count) ? count : 0) + 1), {
    expirationTtl: 7_200,
  });
  return false;
}

function badKey(key: string): Response {
  return unprocessable(`clé de réponse invalide : '${key}' (attendu 'kind,index')`);
}

/** Les 4 validations par réponse ne différaient que par le motif d'invalidité. */
function badResponse(key: string, why: string): Response {
  return unprocessable(`réponse invalide pour '${key}' : ${why}`);
}

export async function apiFeedback(request: Request, env: Env): Promise<Response> {
  if (foreignOrigin(request)) return jsonError(403, "origine non autorisée");
  const body = await request.json().catch(() => null) as {
    slug?: unknown;
    ts?: unknown;
    responses?: unknown;
  } | null;
  if (!body || typeof body.slug !== "string" || typeof body.ts !== "string"
      || typeof body.responses !== "object" || body.responses === null
      || Array.isArray(body.responses)) {
    return unprocessable("requête feedback invalide");
  }
  const slug = body.slug;
  if (!accountFor(slug)) return notFound("compte inconnu");

  const reviews = await readJsonl<{ ts: string; model: string; kind?: string; review: unknown }>(
    env.DATA,
    KEYS.reviews(slug),
  );
  const found = reviews.find((review) => review.ts === body.ts);
  if (!found) return notFound("review introuvable");
  const review = found.kind === "game"
    ? validateGameReview(found.review)
    : validateReview(found.review);
  if (!review) return jsonError(500, "review stockée non conforme");

  const sections: [string, unknown[]][] = [
    ["strength", review.strengths],
    ["mistake", review.mistakes],
  ];
  if (found.kind !== "game") sections.push(["habit", (review as Review).habits]);
  sections.push(["focus", [review.next_focus]]);
  const allowedSections = new Map(sections);

  const responses = new Map<string, ResponseItem>();
  for (const [key, raw] of Object.entries(body.responses as Record<string, unknown>)) {
    const match = /^([a-z]+),(\d+)$/.exec(key);
    if (!match) return badKey(key);
    const kind = match[1];
    const index = Number(match[2]);
    const section = allowedSections.get(kind);
    if (!section || index >= section.length) return badKey(key);
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
      return badResponse(key, "objet requis");
    }
    const value = raw as Record<string, unknown>;
    if (typeof value.useful !== "boolean") {
      return badResponse(key, "useful booléen requis");
    }
    const tag = value.tag ?? null;
    const note = value.note ?? null;
    if (tag !== null
        && (typeof tag !== "string" || !(NEG_TAGS as readonly string[]).includes(tag))) {
      return badResponse(key, "tag inconnu");
    }
    if (note !== null && typeof note !== "string") {
      return badResponse(key, "note doit être une chaîne");
    }
    // Clé normalisée (index numérique) : `key` brut peut porter des zéros non
    // significatifs, la lecture plus bas reconstruit `${kind},${index}`.
    responses.set(`${kind},${index}`, { useful: value.useful, tag, note });
  }

  if (await feedbackRateLimited(request, env)) {
    return jsonError(429, "trop de votes en peu de temps ; réessaie dans une heure");
  }

  const items: Array<{
    kind: string;
    index: number;
    useful: boolean;
    tag: string | null;
    note: string | null;
  }> = [];
  for (const [kind, section] of sections) {
    section.forEach((_, index) => {
      const response = responses.get(`${kind},${index}`);
      if (response) items.push({ kind, index, ...response });
    });
  }
  if (items.some((item) => !item.useful && item.tag === null)) {
    return unprocessable("feedback invalide (tag requis si useful=False)");
  }

  const feedback = {
    ts: body.ts,
    player: slug,
    rated_at: new Date().toISOString().slice(0, 19),
    model: found.model,
    overall_useful: null,
    items,
  };
  await upsertJsonl(env.DATA, KEYS.feedback(slug), feedback);
  return Response.json({ ok: true });
}
