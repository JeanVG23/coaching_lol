import { accountFor } from "./accounts";
import { KEYS, readJsonl } from "./readers";
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
  return Response.json(
    { detail: `clé de réponse invalide : '${key}' (attendu 'kind,index')` },
    { status: 422 },
  );
}

export async function apiFeedback(request: Request, env: Env): Promise<Response> {
  if (foreignOrigin(request)) {
    return Response.json({ detail: "origine non autorisée" }, { status: 403 });
  }
  const body = await request.json().catch(() => null) as {
    slug?: unknown;
    ts?: unknown;
    responses?: unknown;
  } | null;
  if (!body || typeof body.slug !== "string" || typeof body.ts !== "string"
      || typeof body.responses !== "object" || body.responses === null
      || Array.isArray(body.responses)) {
    return Response.json({ detail: "requête feedback invalide" }, { status: 422 });
  }
  const slug = body.slug;
  if (!accountFor(slug)) {
    return Response.json({ detail: "compte inconnu" }, { status: 404 });
  }

  const reviews = await readJsonl<{ ts: string; model: string; kind?: string; review: unknown }>(
    env.DATA,
    KEYS.reviews(slug),
  );
  const found = reviews.find((review) => review.ts === body.ts);
  if (!found) {
    return Response.json({ detail: "review introuvable" }, { status: 404 });
  }
  const review = found.kind === "game"
    ? validateGameReview(found.review)
    : validateReview(found.review);
  if (!review) {
    return Response.json({ detail: "review stockée non conforme" }, { status: 500 });
  }

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
      return Response.json(
        { detail: `réponse invalide pour '${key}' : objet requis` },
        { status: 422 },
      );
    }
    const value = raw as Record<string, unknown>;
    if (typeof value.useful !== "boolean") {
      return Response.json(
        { detail: `réponse invalide pour '${key}' : useful booléen requis` },
        { status: 422 },
      );
    }
    const tag = value.tag ?? null;
    const note = value.note ?? null;
    if (tag !== null
        && (typeof tag !== "string" || !(NEG_TAGS as readonly string[]).includes(tag))) {
      return Response.json(
        { detail: `réponse invalide pour '${key}' : tag inconnu` },
        { status: 422 },
      );
    }
    if (note !== null && typeof note !== "string") {
      return Response.json(
        { detail: `réponse invalide pour '${key}' : note doit être une chaîne` },
        { status: 422 },
      );
    }
    responses.set(`${kind},${index}`, {
      useful: value.useful,
      tag,
      note,
    });
  }

  if (await feedbackRateLimited(request, env)) {
    return Response.json(
      { detail: "trop de votes en peu de temps ; réessaie dans une heure" },
      { status: 429 },
    );
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
    return Response.json(
      { detail: "feedback invalide (tag requis si useful=False)" },
      { status: 422 },
    );
  }

  const feedback = {
    ts: body.ts,
    player: slug,
    rated_at: new Date().toISOString().slice(0, 19),
    model: found.model,
    overall_useful: null,
    items,
  };
  const existing = await readJsonl<Record<string, unknown>>(
    env.DATA,
    KEYS.feedback(slug),
  );
  const kept = existing.filter((line) => line.ts !== feedback.ts);
  kept.push(feedback);
  await env.DATA.put(
    KEYS.feedback(slug),
    kept.map((line) => JSON.stringify(line)).join("\n"),
  );
  return Response.json({ ok: true });
}
