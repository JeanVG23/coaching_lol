import { readAccount } from "./accounts";
import { generateJson } from "./llm_client";
import { renderGame, SYSTEM_GAME, versionOf } from "./prompt";
import { appendJsonl, KEYS, readGamePayloadBundle, readJsonl, type KVLike } from "./readers";
import { gameReviewJsonSchema, validateGameReview, type GameReview } from "./schema";
import { jsonError, notFound, readJsonBody, unprocessable } from "./http";
import type { Env } from "./index";
import { GAME_REVIEW_SCHEMA_VERSION } from "./generated/shared";

type JsonRecord = Record<string, any>;
const SCHEMA_ATTEMPTS = 3;

export interface GameCoachParams {
  slug: string;
  matchId: string;
  model: string;
  force: boolean;
}

export type GameGenerateFn = (
  model: string, system: string, user: string, schema: unknown,
) => Promise<Record<string, unknown>>;

export interface GameSseEvent {
  event: "payload" | "llm" | "review" | "error";
  data: unknown;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function* gameCoachFlow(
  deps: { kv: KVLike; generate: GameGenerateFn; now: () => string },
  params: GameCoachParams,
): AsyncGenerator<GameSseEvent> {
  const [bundle, reviews, promptVersion] = await Promise.all([
    readGamePayloadBundle(deps.kv, params.slug),
    readJsonl<JsonRecord>(deps.kv, KEYS.reviews(params.slug)),
    versionOf(SYSTEM_GAME),
  ]);
  const cached = [...reviews].reverse().find((record) =>
    record.kind === "game" && (record.match_id ?? record.payload?.meta?.match_id) === params.matchId
    && record.run?.prompt_version === promptVersion
  );
  if (cached && !params.force) {
    yield { event: "review", data: { ...cached, cached: true } };
    return;
  }

  const entry = bundle.items[params.matchId];
  if (!entry) {
    yield { event: "error", data: { error: "this game cannot be analyzed — refresh the account data" } };
    return;
  }

  yield { event: "payload", data: { stage: "payload", match_id: params.matchId } };
  const [system, user] = renderGame(entry.payload);
  yield { event: "llm", data: { stage: "llm", model: params.model } };

  let review: GameReview | null = null;
  try {
    for (let attempt = 0; attempt < SCHEMA_ATTEMPTS && review === null; attempt += 1) {
      review = validateGameReview(await deps.generate(
        params.model, system, user, gameReviewJsonSchema(),
      ));
    }
  } catch (error) {
    yield { event: "error", data: { error: `LLM generation: ${errorMessage(error)}` } };
    return;
  }
  if (review === null) {
    yield { event: "error", data: { error: `LLM output did not match the schema after ${SCHEMA_ATTEMPTS} attempts` } };
    return;
  }

  const record = {
    ts: deps.now(),
    model: params.model,
    kind: "game",
    match_id: params.matchId,
    scope: entry.benchmark_scope,
    target: bundle.target,
    run: {
      prompt_version: await versionOf(SYSTEM_GAME),
      schema_version: GAME_REVIEW_SCHEMA_VERSION,
      payload_hash: entry.payload_hash,
    },
    payload: entry.payload,
    review,
  };
  try {
    await appendJsonl(deps.kv, KEYS.reviews(params.slug), record);
  } catch (error) {
    yield { event: "error", data: { error: `persistance KV : ${errorMessage(error)}` } };
    return;
  }
  yield { event: "review", data: record };
}

export async function apiGameCoach(request: Request, env: Env): Promise<Response> {
  const body = await readJsonBody<{
    slug?: unknown; match_id?: unknown; model?: unknown; force?: unknown;
  }>(request);
  if (!body || typeof body.slug !== "string" || typeof body.match_id !== "string") {
    return unprocessable("invalid slug or match_id");
  }
  if (!await readAccount(env.DATA, body.slug)) return notFound("unknown account");
  if (!env.OLLAMA_API_KEY) return jsonError(500, "OLLAMA_API_KEY is not configured");
  if (body.force !== undefined && typeof body.force !== "boolean") {
    return unprocessable("force must be a boolean");
  }
  const params: GameCoachParams = {
    slug: body.slug,
    matchId: body.match_id,
    model: typeof body.model === "string" && body.model
      ? body.model : env.OLLAMA_MODEL || "kimi-k2.6",
    force: body.force === true,
  };
  const generate: GameGenerateFn = (model, system, user, schema) => {
    return generateJson(model, system, user, schema, {
      apiKey: env.OLLAMA_API_KEY!, timeoutMs: 180_000, maxAttempts: 1,
    });
  };
  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const event of gameCoachFlow(
          { kv: env.DATA, generate, now: () => new Date().toISOString() }, params,
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
