import { readAccount } from "./accounts";
import { generateJson } from "./llm_client";
import { addGameReviewCauses, buildPayload, ROLE_SCOPES, type Outcome } from "./payload";
import { render, SYSTEM, SYSTEM_GAME, versionOf } from "./prompt";
import { jsonError, notFound, readJsonBody, unprocessable } from "./http";
import { mainRoleFromAnalysis } from "./main_role";
import {
  appendJsonl, KEYS, readJson, readJsonl, readRoleShap, type KVLike,
} from "./readers";
import { reviewJsonSchema, validateReview, type Review } from "./schema";
import { REVIEW_SCHEMA_VERSION } from "./generated/shared";
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
  if (!(params.scope.toLowerCase() in ROLE_SCOPES)) {
    yield { event: "error", data: { error: `unknown benchmark scope: ${params.scope}` } };
    return;
  }
  // Lectures indépendantes : en parallèle, un seul aller-retour KV avant le 1er event SSE.
  const [me, ref, previousReviews, gamePromptVersion] = await Promise.all([
    readJson<Record<string, any>>(deps.kv, KEYS.gold(params.slug, params.scope)),
    readJson<Record<string, any>>(deps.kv, KEYS.ref(params.target, params.scope)),
    readJsonl<Record<string, any>>(deps.kv, KEYS.reviews(params.slug)),
    versionOf(SYSTEM_GAME),
  ]);
  if (!me || !ref) {
    const missing = !me
      ? `player aggregate ${params.slug}/${params.scope}`
      : `${params.target} benchmark ${params.scope}`;
    yield {
      event: "error",
      data: { error: `missing data (${missing}) — refresh the account data` },
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
    payload = addGameReviewCauses(
      payload,
      previousReviews.filter((review) =>
        review.kind === "game" && review.run?.prompt_version === gamePromptVersion
      ),
      params.scope,
    );
  } catch (error) {
    yield { event: "error", data: { error: `invalid payload: ${errorMessage(error)}` } };
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
    yield { event: "error", data: { error: `LLM generation: ${errorMessage(error)}` } };
    return;
  }
  if (review === null) {
    yield {
      event: "error",
      data: { error: "LLM output did not match the Review schema after 2 attempts" },
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
    run: {
      prompt_version: await versionOf(SYSTEM),
      schema_version: REVIEW_SCHEMA_VERSION,
    },
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
  const body = await readJsonBody<{
    slug?: string;
    target?: string;
    model?: string;
  }>(request);
  const slug = body?.slug ?? "";
  if (!await readAccount(env.DATA, slug)) return notFound("unknown account");
  if (!env.OLLAMA_API_KEY) return jsonError(500, "OLLAMA_API_KEY is not configured");
  const mainRole = mainRoleFromAnalysis(await readRoleShap(env.DATA, slug));
  if (!mainRole) {
    return unprocessable("main role is unavailable — refresh the account data");
  }
  const params: CoachParams = {
    slug,
    scope: mainRole.scope,
    outcome: "overall",
    target: body?.target ?? "challenger",
    model: body?.model || env.OLLAMA_MODEL || "kimi-k2.6",
  };
  const generate: GenerateFn = (model, system, user, schema) => generateJson(
    model,
    system,
    user,
    schema,
    { apiKey: env.OLLAMA_API_KEY!, timeoutMs: 180_000, maxAttempts: 1 },
  );

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const event of coachFlow(
          { kv: env.DATA, generate, now: () => new Date().toISOString() },
          params,
        )) {
          if (event.event === "error") {
            console.error("aggregate coaching failed", event.data);
          }
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
