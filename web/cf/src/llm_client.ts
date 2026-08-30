/** Client Ollama Cloud structured output — portage de src/04_coaching/llm_client.py. */

export class LLMError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LLMError";
  }
}

export interface GenerateOpts {
  apiKey: string;
  temperature?: number;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
  sleepImpl?: (milliseconds: number) => Promise<void>;
}

const MAX_ATTEMPTS = 4;

export async function generateJson(
  model: string,
  system: string,
  user: string,
  schema: unknown,
  opts: GenerateOpts,
): Promise<Record<string, unknown>> {
  const fetchImpl = opts.fetchImpl ?? fetch;
  const sleep = opts.sleepImpl
    ?? ((milliseconds: number) => new Promise<void>((resolve) => setTimeout(resolve, milliseconds)));
  const temperature = opts.temperature ?? 0.2;
  const timeoutMs = opts.timeoutMs ?? 180_000;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    try {
      const response = await fetchImpl("https://ollama.com/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${opts.apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: [
            { role: "system", content: system },
            { role: "user", content: user },
          ],
          format: schema,
          stream: false,
          options: { temperature },
        }),
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (response.status !== 429 && response.status < 500 && !response.ok) {
        throw new LLMError(`ollama HTTP ${response.status} (auth/requête invalide)`);
      }
      if (response.ok) {
        const body = await response.json() as { message?: { content?: string } };
        try {
          const parsed = JSON.parse(body.message?.content ?? "") as unknown;
          if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
            return parsed as Record<string, unknown>;
          }
        } catch {
          // Une réponse JSON Ollama dont le contenu n'est pas JSON est retentée.
        }
      }
    } catch (error) {
      if (error instanceof LLMError) throw error;
      // Timeout, erreur réseau et réponse Ollama non JSON sont retentés.
    }
    if (attempt < MAX_ATTEMPTS - 1) await sleep(2_000 * (attempt + 1));
  }
  throw new LLMError(`ollama : échec après ${MAX_ATTEMPTS} tentatives`);
}
