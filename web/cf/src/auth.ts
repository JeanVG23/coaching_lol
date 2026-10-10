import { jsonError, readJsonBody, unauthorized, unprocessable } from "./http";
import type { Env } from "./index";

export const COACH_COOKIE_NAME = "coach_token";
const DEFAULT_TTL_MS = 30 * 24 * 60 * 60 * 1000; // 30 jours

export function verifyPassword(input: string, expected: string): boolean {
  if (typeof input !== "string" || typeof expected !== "string") return false;
  const enc = new TextEncoder();
  const a = enc.encode(input);
  const b = enc.encode(expected);
  if (a.byteLength !== b.byteLength) return false;
  let diff = 0;
  for (let i = 0; i < a.byteLength; i++) {
    diff |= a[i] ^ b[i];
  }
  return diff === 0;
}

async function hmacSha256(message: string, secret: string): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sigBuffer = await crypto.subtle.sign("HMAC", key, enc.encode(message));
  return Array.from(new Uint8Array(sigBuffer), (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function createAuthToken(
  secret: string,
  ttlMs = DEFAULT_TTL_MS,
  now = Date.now(),
): Promise<string> {
  const exp = now + ttlMs;
  const payload = `coach:${exp}`;
  const sig = await hmacSha256(payload, secret);
  return `${payload}.${sig}`;
}

export async function verifyAuthToken(
  token: string,
  secret: string,
  now = Date.now(),
): Promise<boolean> {
  if (!token || !secret) return false;
  // Permet aussi d'utiliser directement le secret comme Bearer token (pratique pour tests/scripts)
  if (verifyPassword(token, secret)) return true;

  const dotIdx = token.lastIndexOf(".");
  if (dotIdx <= 0) return false;
  const payload = token.slice(0, dotIdx);
  const signature = token.slice(dotIdx + 1);

  const match = payload.match(/^coach:(\d+)$/);
  if (!match) return false;
  const exp = Number(match[1]);
  if (!Number.isFinite(exp) || exp < now) return false;

  const expectedSig = await hmacSha256(payload, secret);
  return verifyPassword(signature, expectedSig);
}

export function extractToken(request: Request): string | null {
  const authHeader = request.headers.get("Authorization");
  if (authHeader?.startsWith("Bearer ")) {
    return authHeader.slice(7).trim();
  }
  const customHeader = request.headers.get("X-Coach-Auth");
  if (customHeader) {
    return customHeader.trim();
  }
  const cookieHeader = request.headers.get("Cookie");
  if (cookieHeader) {
    const match = cookieHeader.match(new RegExp(`(?:^|;\\s*)${COACH_COOKIE_NAME}=([^;]+)`));
    if (match) return decodeURIComponent(match[1]);
  }
  return null;
}

export async function isAuthorized(request: Request, env: Env): Promise<boolean> {
  const secret = env.COACH_AUTH_PASSWORD;
  if (!secret) return false;
  const token = extractToken(request);
  if (!token) return false;
  return verifyAuthToken(token, secret);
}

export async function apiLogin(request: Request, env: Env): Promise<Response> {
  const secret = env.COACH_AUTH_PASSWORD;
  if (!secret) {
    return jsonError(500, "COACH_AUTH_PASSWORD is not configured on the server");
  }

  const body = await readJsonBody<{ password?: unknown }>(request);
  if (!body || typeof body.password !== "string" || !body.password) {
    return unprocessable("password is required");
  }

  if (!verifyPassword(body.password, secret)) {
    return unauthorized("Incorrect password");
  }

  const token = await createAuthToken(secret);
  const isHttps = new URL(request.url).protocol === "https:";
  const secureFlag = isHttps ? "; Secure" : "";
  const cookieVal = `${COACH_COOKIE_NAME}=${encodeURIComponent(token)}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000${secureFlag}`;

  return new Response(JSON.stringify({ ok: true, token }), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Set-Cookie": cookieVal,
    },
  });
}

export function apiLogout(request: Request): Response {
  const isHttps = new URL(request.url).protocol === "https:";
  const secureFlag = isHttps ? "; Secure" : "";
  const cookieVal = `${COACH_COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0${secureFlag}`;

  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Set-Cookie": cookieVal,
    },
  });
}

export async function apiAuthStatus(request: Request, env: Env): Promise<Response> {
  const configured = Boolean(env.COACH_AUTH_PASSWORD);
  const authenticated = await isAuthorized(request, env);
  return Response.json({ configured, authenticated });
}
