import { accountFor } from "./accounts";
import { generateJson } from "./llm_client";
import { buildPayload, type Outcome } from "./payload";
import { render } from "./prompt";
import { appendJsonl, KEYS, readJson, type KVLike } from "./readers";
import { reviewJsonSchema, validateReview, type Review } from "./schema";
import type { Env } from "./index";

export interface CoachParams {
  slug: string;
  scope: string;
  outcome: string;
  target: string;
  model: string;
}

export type GenerateFn = (
  model: string,
  system: string,
  user: string,
  schema: unknown,
) => Promise<Record<string, unknown>>;

export interface SseEvent {
  event: "payload" | "llm" | "review" | "error";
  data: unknown;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function* coachFlow(
  deps: { kv: KVLike; generate: GenerateFn; now: () => string },
  params: CoachParams,
): AsyncGenerator<SseEvent> {
  const me = await readJson<Record<string, any>>(
    deps.kv,
    KEYS.gold(params.slug, params.scope),
  );
  const ref = await readJson<Record<string, any>>(
    deps.kv,
    KEYS.ref(params.target, params.scope),
  );
  if (!me || !ref) {
    const missing = !me
      ? `agrégat perso ${params.slug}/${params.scope}`
      : `référentiel ${params.target}/${params.scope}`;
    yield {
      event: "error",
      data: { error: `données manquantes (${missing}) — lance le sync local` },
    };
    return;
  }

  let payload: Record<string, any>;
  try {
    payload = buildPayload(me, ref, {
      player: params.slug,
      scope: params.scope,
      target: params.target,
      outcome: params.outcome as Outcome,
    });
  } catch (error) {
    yield { event: "error", data: { error: `payload invalide : ${errorMessage(error)}` } };
    return;
  }
  yield { event: "payload", data: { stage: "payload" } };

  const [system, user] = render(payload);
  const schema = reviewJsonSchema();
  yield { event: "llm", data: { stage: "llm", model: params.model } };

  let review: Review | null = null;
  try {
    for (let attempt = 0; attempt < 2 && review === null; attempt += 1) {
      review = validateReview(await deps.generate(params.model, system, user, schema));
    }
  } catch (error) {
    yield { event: "error", data: { error: `génération LLM : ${errorMessage(error)}` } };
    return;
  }
  if (review === null) {
    yield {
      event: "error",
      data: { error: "sortie LLM non conforme au schéma Review après 2 tentatives" },
    };
    return;
  }

  const record = {
    ts: deps.now(),
    model: params.model,
    scope: params.scope,
    target: params.target,
    payload,
    review,
    outcome_focus: params.outcome,
  };
  try {
    await appendJsonl(deps.kv, KEYS.reviews(params.slug), record);
  } catch (error) {
    yield { event: "error", data: { error: `persistance KV : ${errorMessage(error)}` } };
    return;
  }
  yield { event: "review", data: record };
}

export async function apiCoach(request: Request, env: Env): Promise<Response> {
  const body = await request.json().catch(() => null) as {
    slug?: string;
    scope?: string;
    outcome?: string;
    target?: string;
    model?: string;
  } | null;
  const slug = body?.slug ?? "";
  if (!accountFor(slug)) {
    return Response.json({ detail: "compte inconnu" }, { status: 404 });
  }
  if (!env.OLLAMA_API_KEY) {
    return Response.json({ detail: "OLLAMA_API_KEY non configuré" }, { status: 500 });
  }
  const params: CoachParams = {
    slug,
    scope: body?.scope ?? "adc",
    outcome: body?.outcome ?? "loss",
    target: body?.target ?? "challenger",
    model: body?.model || env.OLLAMA_MODEL || "kimi-k2.6",
  };
  const generate: GenerateFn = (model, system, user, schema) => generateJson(
    model,
    system,
    user,
    schema,
    { apiKey: env.OLLAMA_API_KEY! },
  );

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const event of coachFlow(
          { kv: env.DATA, generate, now: () => new Date().toISOString() },
          params,
        )) {
          controller.enqueue(encoder.encode(
            `event: ${event.event}\ndata: ${JSON.stringify(event.data)}\n\n`,
          ));
        }
      } catch (error) {
        controller.enqueue(encoder.encode(
          `event: error\ndata: ${JSON.stringify({ error: errorMessage(error) })}\n\n`,
        ));
      } finally {
        controller.close();
      }
    },
  });
  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
    },
  });
}
