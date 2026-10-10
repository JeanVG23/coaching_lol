import { readAccount } from "./accounts";
import { jsonError, notFound, readJsonBody, unprocessable } from "./http";
import { generateJson } from "./llm_client";
import { appendJsonl, KEYS, readJsonl, type KVLike } from "./readers";
import type { Env } from "./index";

type JsonRecord = Record<string, any>;

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  answer: string;
  refused_hidden_info: boolean;
}

export type ChatGenerateFn = (
  model: string,
  system: string,
  user: string,
  schema: unknown,
) => Promise<Record<string, unknown>>;

export const CHAT_SCHEMA = {
  type: "object",
  properties: {
    answer: { type: "string" },
    refused_hidden_info: { type: "boolean" },
  },
  required: ["answer", "refused_hidden_info"],
  additionalProperties: false,
};

export const SYSTEM_CHAT = [
  "You are the interactive coach for an already analyzed League of Legends game.",
  "Answer in concise, direct, actionable English using 'you'.",
  "The deterministic `payload` is the only factual source. `current_review` is previous LLM output the player may challenge; never treat it as evidence.",
  "ABSOLUTE INFORMATION ASYMMETRY: judge a decision only with information the player had at that moment (champ select, visible scoreboard, their own deaths/damage, purchases, gold, and HUD timers).",
  "If the player asks where an enemy was, what an enemy knew, about information hidden by fog of war, or about a position reconstructed afterward, explicitly refuse and set `refused_hidden_info` to true. NEVER use the complete timeline to answer.",
  "When the player explains their intent, make a clear judgment: acknowledge a valid special case or explain a better alternative grounded in the payload. Never invent an event or number.",
  "Respond only with the requested JSON.",
].join("\n");

function normalize(text: string): string {
  return text.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
}

function enemyNames(payload: JsonRecord): string[] {
  const comp = payload.context?.comp ?? {};
  return Object.entries(comp)
    .filter(([key, value]) => key.startsWith("enemy_") && typeof value === "string")
    .map(([, value]) => normalize(String(value)));
}

export function asksForHiddenEnemyPosition(messages: ChatMessage[], payload: JsonRecord): boolean {
  const latest = [...messages].reverse().find((message) => message.role === "user")?.content ?? "";
  const text = normalize(latest);
  const locationIntent = /\b(ou|where|position|localis|se trouv|etait)\b/.test(text);
  const hiddenActor = /jungl(er|e)? (ennemi|adverse)|enemy jungler|\bennemi\b|\badversaire\b/.test(text)
    || enemyNames(payload).some((name) => name && text.includes(name));
  return locationIntent && hiddenActor;
}

function validateMessages(value: unknown): ChatMessage[] | null {
  if (!Array.isArray(value) || value.length < 1 || value.length > 12) return null;
  const messages = value.filter((message): message is ChatMessage =>
    typeof message === "object" && message !== null
      && ((message as ChatMessage).role === "user" || (message as ChatMessage).role === "assistant")
      && typeof (message as ChatMessage).content === "string"
      && (message as ChatMessage).content.trim().length > 0
      && (message as ChatMessage).content.length <= 2_000);
  return messages.length === value.length && messages.at(-1)?.role === "user" ? messages : null;
}

function validateResponse(value: unknown): ChatResponse | null {
  if (typeof value !== "object" || value === null) return null;
  const response = value as ChatResponse;
  return typeof response.answer === "string" && response.answer.trim() !== ""
    && typeof response.refused_hidden_info === "boolean" ? response : null;
}

export async function chatTurn(
  deps: { kv: KVLike; generate: ChatGenerateFn; now: () => string },
  params: { slug: string; reviewTs: string; messages: ChatMessage[]; model: string },
): Promise<{ response: ChatResponse; record: JsonRecord }> {
  const reviews = await readJsonl<JsonRecord>(deps.kv, KEYS.reviews(params.slug));
  const review = reviews.find((item) => item.ts === params.reviewTs && item.kind === "game");
  if (!review) throw new Error("game analysis not found");
  const payload = review.payload ?? {};
  let response: ChatResponse;
  if (asksForHiddenEnemyPosition(params.messages, payload)) {
    response = {
      answer: "I can't use an enemy position that was hidden from you when you made the decision. I can analyze what your HUD, vision, and champ select reasonably allowed you to do.",
      refused_hidden_info: true,
    };
  } else {
    const user = JSON.stringify({
      payload,
      current_review: review.review ?? {},
      conversation: params.messages,
    });
    response = validateResponse(await deps.generate(
      params.model, SYSTEM_CHAT, user, CHAT_SCHEMA,
    )) ?? (() => { throw new Error("chat output does not match the schema"); })();
  }
  const record = {
    ts: deps.now(), review_ts: params.reviewTs, model: params.model,
    messages: params.messages, response,
  };
  await appendJsonl(deps.kv, KEYS.chats(params.slug), record);
  return { response, record };
}

export async function apiChat(request: Request, env: Env): Promise<Response> {
  const body = await readJsonBody<JsonRecord>(request);
  const slug = typeof body?.slug === "string" ? body.slug : "";
  if (!await readAccount(env.DATA, slug)) return notFound("unknown account");
  const reviewTs = typeof body?.review_ts === "string" ? body.review_ts : "";
  const messages = validateMessages(body?.messages);
  if (!reviewTs || !messages) return unprocessable("invalid review_ts or messages");
  if (!env.OLLAMA_API_KEY && !asksForHiddenEnemyPosition(messages, {})) {
    return jsonError(500, "OLLAMA_API_KEY is not configured");
  }
  const model = (typeof body?.model === "string" && body.model)
    || env.OLLAMA_MODEL || "kimi-k2.6";
  const generate: ChatGenerateFn = (chosen, system, user, schema) => generateJson(
    chosen, system, user, schema, { apiKey: env.OLLAMA_API_KEY ?? "" },
  );
  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      const emit = (event: string, data: unknown) => controller.enqueue(encoder.encode(
        `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`,
      ));
      try {
        emit("llm", { stage: "chat", model });
        const result = await chatTurn(
          { kv: env.DATA, generate, now: () => new Date().toISOString() },
          { slug, reviewTs, messages, model },
        );
        emit("message", result.response);
      } catch (error) {
        emit("error", { error: error instanceof Error ? error.message : String(error) });
      } finally {
        controller.close();
      }
    },
  });
  return new Response(stream, { headers: {
    "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
  } });
}
