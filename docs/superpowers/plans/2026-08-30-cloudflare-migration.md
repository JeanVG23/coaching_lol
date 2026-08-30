# Migration Cloudflare — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrer le site coaching_lol de Fly.io vers un Worker Cloudflare unique (TS + KV), servira `coaching-lol.jean.vg`, avec le ML précalculé localement et poussé par un script de sync.

**Architecture:** Un Worker TypeScript sert les assets statiques (SPA) ET l'API `/api/*`, lit/écrit Workers KV (binding `DATA`). Le local Python reste le cerveau : fetch Riot, extraction silver/gold, entraînement, et un nouveau `sync_cloudflare.py` qui pousse données + prédictions ML précalculées dans KV via l'API REST Cloudflare. Le coaching passe par SSE (payload → Ollama → review → persiste KV).

**Tech Stack:** TypeScript (strict) sur Cloudflare Workers (wrangler v4, assets natifs, binding KV), vitest (fake KV par injection — pas de pool-workers), Python existant (riolib, ml_rank, payload/prompt/schema/coach/feedback comme source de portage), pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-cloudflare-migration-design.md` — le plan argumente depuis la spec ; l'exécuteur lit les deux.

## Global Constraints

- Plan Cloudflare **free uniquement** : Worker CPU ≤ 10 ms/req, KV 1 000 écritures/j, 25 Mo/valeur, 1 Go total. Tout travail CPU-lourd reste local (fetch Riot, ML).
- **Aucune clé Riot côté Worker** (le Worker ne parle jamais à Riot). Seul secret Worker : `OLLAMA_API_KEY`. `.env` (local) et `.dev.vars` (wrangler dev) sont gitignorés — ne JAMAIS les committer.
- Contrat API = spec §6 : routes lecture identiques ; `POST /api/coach` devient **SSE** ; `POST /api/fetch` et `GET /api/jobs/{id}` **supprimés** V1.
- Le pipeline Python local ne change PAS d'architecture — seule la couche serving est portée + un script de sync ajouté. `web/backend/` (Python) est conservé comme référence ; `web/backend/ml_rank.py` reste utilisé par le sync.
- Les clés KV `coaching:{slug}:reviews` / `coaching:{slug}:feedback` appartiennent au **Worker** : le sync ne les écrase jamais (amorce `--seed-reviews` uniquement si la clé est absente).
- TypeScript strict, zéro runtime npm (wrangler bundle), `type: module`.
- Commits conventionnels FR (`feat(cf): …`), un commit par tâche, tests verts avant commit.
- Précondition P0 (réputée faite) : worktree clean, 5 commits de cleanup + merge FF sur master + spec commitée.

---

## Phase P1 — Interim domaine (le domaine répond pendant la construction)

### Task 1: CNAME interim + certificat Fly

**Files:** aucun (opérationnel, 0 code).

**Interfaces:** aucune. Produit : `https://coaching-lol.jean.vg` répond (servi par Fly) — précondition implicite des tâches P2+ (le domaine reste vivant pendant la migration).

- [ ] **Step 1: CNAME dans le dashboard Cloudflare**

Dashboard Cloudflare → zone `jean.vg` → DNS → Records → Add record :

- Type : `CNAME`
- Name : `coaching-lol`
- Target : `coaching-lol.fly.dev`
- Proxy status : **DNS only** (nuage GRIS — indispensable : Fly doit pouvoir émettre son certificat Let's Encrypt ; proxied casserait l'émission)

- [ ] **Step 2: Auth Fly locale (interactif — taper `!`)**

Dans le prompt Claude Code, l'utilisateur tape :

```
! cd /Users/jeanvangysel/code/website/coaching_lol && fly auth login
```

- [ ] **Step 3: Certificat Fly pour le sous-domaine**

Run: `cd /Users/jeanvangysel/code/website/coaching_lol && fly certs add coaching-lol.jean.vg`
Expected: `Certificate added` puis (après ~1-2 min, `fly certs check coaching-lol.jean.vg`) `Certificate available`. Si "DNS problem" : vérifier que le CNAME est bien **DNS only** et attendre la propagation (TTL auto ~5 min).

- [ ] **Step 4: Vérification bout-en-bout**

Run: `curl -sS https://coaching-lol.jean.vg/api/health`
Expected: `{"status":"ok",...}` (le JSON health servi par FastAPI sur Fly, via le domaine). Vérifier aussi `curl -sSI https://coaching-lol.jean.vg/` → `200` et une redirection HTTP→HTTPS sur un appel `http://`.

---

## Phase P2 — Worker TS squelette + lecture KV + sync local

### Task 2: Scaffold `web/cf/` + health + assets SPA

**Files:**
- Create: `web/cf/package.json`, `web/cf/wrangler.toml`, `web/cf/tsconfig.json`, `web/cf/vitest.config.ts`
- Create: `web/cf/src/index.ts`
- Test: `web/cf/test/health.test.ts`
- Modify: `web/frontend/index.html` (toutes les refs `/static/…` → `/…`)
- Modify: `.gitignore` (node_modules, `.dev.vars`, `.wrangler/`)

**Interfaces:**
- Consumes: rien (premier fichier du Worker).
- Produces: `export interface Env { ASSETS: Fetcher }` et `export async function handle(request: Request, env: Env): Promise<Response>` dans `web/cf/src/index.ts` (les tâches 4/10/11 étendent `Env` et ajoutent des routes) ; `export default { fetch }` pour wrangler.

- [ ] **Step 1: Branche + scaffold config**

```bash
git checkout -b cloudflare-migration
```

`web/cf/package.json` :

```json
{
  "name": "coaching-lol-worker",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "wrangler dev",
    "deploy": "wrangler deploy",
    "test": "vitest run"
  },
  "devDependencies": {
    "@cloudflare/workers-types": "latest",
    "typescript": "^5.6.0",
    "vitest": "^3.0.0",
    "wrangler": "^4.30.0"
  }
}
```

`web/cf/wrangler.toml` (la section KV est ajoutée en Task 6, avec le vrai id) :

```toml
name = "coaching-lol"
main = "src/index.ts"
compatibility_date = "2026-08-30"

[assets]
directory = "../frontend"
not_found_handling = "single-page-application"
```

`web/cf/tsconfig.json` :

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "lib": ["ES2022"],
    "types": ["@cloudflare/workers-types"],
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true,
    "isolatedModules": true
  },
  "include": ["src", "test"]
}
```

`web/cf/vitest.config.ts` :

```ts
import { defineConfig } from "vitest/config";

export default defineConfig({ test: { include: ["test/**/*.test.ts"] } });
```

`.gitignore` — ajouter (si absents) :

```
web/cf/node_modules/
web/cf/.dev.vars
web/cf/.wrangler/
```

Puis :

```bash
cd web/cf && npm install
```

- [ ] **Step 2: Écrire le test en échec**

`web/cf/test/health.test.ts` :

```ts
import { describe, expect, it } from "vitest";
import { handle, type Env } from "../src/index";

const SPA_HTML = "<!doctype html><html><title>spa</title></html>";

function makeEnv(): Env {
  return {
    ASSETS: {
      fetch: async (req: Request) => {
        if (new URL(req.url).pathname === "/style.css") {
          return new Response("body{}", { headers: { "content-type": "text/css" } });
        }
        return new Response(SPA_HTML, { headers: { "content-type": "text/html" } });
      },
    },
  };
}

describe("handle", () => {
  it("GET /api/health répond ok", async () => {
    const r = await handle(new Request("http://x/api/health"), makeEnv());
    expect(r.status).toBe(200);
    const j = await r.json();
    expect(j.status).toBe("ok");
    expect(j.service).toBe("coaching-lol");
    expect(typeof j.server_time).toBe("string");
  });

  it("GET /api/inconnu répond 404 JSON", async () => {
    const r = await handle(new Request("http://x/api/inconnu"), makeEnv());
    expect(r.status).toBe(404);
    expect(await r.json()).toEqual({ detail: "Not Found" });
  });

  it("les chemins non-API délèguent aux assets (SPA fallback /c/{slug})", async () => {
    for (const p of ["/", "/c/spadzze", "/readme"]) {
      const r = await handle(new Request(`http://x${p}`), makeEnv());
      expect(r.status).toBe(200);
      expect(await r.text()).toBe(SPA_HTML);
    }
    const css = await handle(new Request("http://x/style.css"), makeEnv());
    expect(css.headers.get("content-type")).toBe("text/css");
  });
});
```

- [ ] **Step 3: Run — vérifier l'échec**

Run: `cd web/cf && npm test`
Expected: FAIL — `Cannot find module '../src/index'`.

- [ ] **Step 4: Implémentation minimale**

`web/cf/src/index.ts` :

```ts
export interface Env {
  ASSETS: Fetcher;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return handle(request, env);
  },
};

export async function handle(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  if (url.pathname === "/api/health") {
    return Response.json({
      status: "ok",
      service: "coaching-lol",
      server_time: new Date().toISOString(),
    });
  }
  if (url.pathname.startsWith("/api/")) {
    return Response.json({ detail: "Not Found" }, { status: 404 });
  }
  return env.ASSETS.fetch(request);
}
```

- [ ] **Step 5: Run — tests verts**

Run: `cd web/cf && npm test`
Expected: 3 passed.

- [ ] **Step 6: Corriger les refs d'assets dans index.html**

Les assets Workers servent `../frontend` **à la racine** — plus de préfixe `/static/`. Remplacer dans `web/frontend/index.html` chaque occurrence `href="/static/` → `href="/` et `src="/static/` → `src="/` (concerne `style.css`, `vendor/alpine.min.js`, `vendor/chart.umd.min.js`, `app.js` — env. 4 refs).

Vérifier qu'il ne reste rien : `grep -n "/static/" web/frontend/index.html web/frontend/app.js` → aucune ligne (les URLs Data Dragon de `app.js` pointent vers `ddragon.leagueoflegends.com`, elles ne bougent pas).

- [ ] **Step 7: Smoke wrangler dev**

Run: `cd web/cf && npx wrangler dev` puis dans un autre terminal :
`curl -sS http://localhost:8787/api/health` → JSON ok ;
`curl -sS http://localhost:8787/c/spadzze` → HTML du SPA ;
`curl -sS http://localhost:8787/style.css` → le CSS (200).
Arrêter wrangler (Ctrl+C).

- [ ] **Step 8: Commit**

```bash
git add web/cf/package.json web/cf/wrangler.toml web/cf/tsconfig.json web/cf/vitest.config.ts web/cf/src/index.ts web/cf/test/health.test.ts .gitignore web/frontend/index.html
git commit -m "feat(cf): squelette Worker (wrangler + assets SPA + /api/health) + refs /static/ servies a la racine"
```

(`web/cf/package-lock.json` aussi s'il est créé — le committer.)

---

### Task 3: `readers.ts` (KV) + `accounts.ts`

**Files:**
- Create: `web/cf/src/readers.ts`, `web/cf/src/accounts.ts`
- Test: `web/cf/test/readers.test.ts`

**Interfaces:**
- Consumes: rien hors stdlib.
- Produces (importés par les Tasks 4/10/11) :
  - `export interface KVLike { get(key: string): Promise<string | null>; put(key: string, value: string): Promise<void> }`
  - `export const KEYS: { games(slug), rank(slug), gold(slug, scope), ref(rank, scope), pred(slug), shap(slug), reviews(slug), feedback(slug) }` — fabriques de clés KV
  - `export function matchSeq(matchId: string): number`
  - `export async function readGames(kv, slug, page?, size?): Promise<{ items: Record<string, unknown>[]; page: number; size: number; total: number }>`
  - `export async function readRank(kv, slug): Promise<Record<string, unknown> | null>`
  - `export async function readPred(kv, slug): Promise<Record<string, unknown> | null>`
  - `export async function readShap(kv, slug): Promise<{ available: boolean; drivers: unknown[] }>`
  - `export async function readJsonl<T>(kv, key): Promise<T[]>`
  - `export async function readJson(kv, key): Promise<any>`
  - `export async function appendJsonl(kv, key, record): Promise<void>`
  - `accounts.ts` : `export interface Account { slug: string; riot_id: string; region: string }`, `export const ACCOUNTS: Account[]`, `export function accountFor(slug): Account | undefined`

Sémantique à respecter (parité avec `web/backend/readers.py`) :
- `matchSeq` : suffixe après le dernier `_` du `match_id`, `int`, fallback `0` si non numérique — le tri chronologique NE PEUT PAS se fier à l'ordre du fichier (append non trié).
- `readShap` : la valeur KV `shap:{slug}:drivers` EST le tableau des drivers (miroir du fichier `{slug}_drivers.json`) → `{ available: true, drivers: <valeur parsée> }` si c'est un tableau, sinon `{ available: false, drivers: [] }`.

- [ ] **Step 1: Test en échec**

`web/cf/test/readers.test.ts` :

```ts
import { describe, expect, it } from "vitest";
import { ACCOUNTS, accountFor } from "../src/accounts";
import { appendJsonl, KEYS, matchSeq, readGames, readJson, readJsonl, readPred, readRank, readShap } from "../src/readers";
import type { KVLike } from "../src/readers";

class MemoryKV implements KVLike {
  store = new Map<string, string>();
  async get(key: string) { return this.store.get(key) ?? null; }
  async put(key: string, value: string) { this.store.set(key, value); }
}

const GAME = (id: string, champ: string) => JSON.stringify({ match_id: id, champion: champ, win: true });

describe("KEYS", () => {
  it("fabrique les clés du layout spec §5", () => {
    expect(KEYS.games("spadzze")).toBe("silver:spadzze:games");
    expect(KEYS.rank("spadzze")).toBe("silver:spadzze:rank");
    expect(KEYS.gold("spadzze", "adc")).toBe("gold:spadzze:adc");
    expect(KEYS.ref("challenger", "adc")).toBe("ref:challenger:adc");
    expect(KEYS.pred("spadzze")).toBe("pred:spadzze");
    expect(KEYS.shap("spadzze")).toBe("shap:spadzze:drivers");
    expect(KEYS.reviews("spadzze")).toBe("coaching:spadzze:reviews");
    expect(KEYS.feedback("spadzze")).toBe("coaching:spadzze:feedback");
  });
});

describe("matchSeq", () => {
  it("extrait le suffixe numérique, fallback 0", () => {
    expect(matchSeq("EUW1_123456789")).toBe(123456789);
    expect(matchSeq("EUW1_1_42")).toBe(42);
    expect(matchSeq("pasunid")).toBe(0);
    expect(matchSeq("")).toBe(0);
  });
});

describe("readGames", () => {
  it("trie par séquence décroissante et pagine", async () => {
    const kv = new MemoryKV();
    await kv.put(KEYS.games("p"), ["EUW1_100", "EUW1_300", "EUW1_200"].map((id) => GAME(id, "Zeri")).join("\n"));
    const p1 = await readGames(kv, "p", 1, 2);
    expect(p1.total).toBe(3);
    expect(p1.items.map((g) => g.match_id)).toEqual(["EUW1_300", "EUW1_200"]);
    const p2 = await readGames(kv, "p", 2, 2);
    expect(p2.items.map((g) => g.match_id)).toEqual(["EUW1_100"]);
    const p3 = await readGames(kv, "p", 3, 2);
    expect(p3.items).toEqual([]);
  });

  it("clé absente → liste vide, pas d'erreur", async () => {
    const p = await readGames(new MemoryKV(), "inconnu", 1, 20);
    expect(p).toEqual({ items: [], page: 1, size: 20, total: 0 });
  });
});

describe("readRank / readPred / readShap", () => {
  it("rank absent → null, présent → objet parsé", async () => {
    const kv = new MemoryKV();
    expect(await readRank(kv, "p")).toBeNull();
    await kv.put(KEYS.rank("p"), JSON.stringify({ tier: "MASTER", league_points: 300 }));
    expect(await readRank(kv, "p")).toEqual({ tier: "MASTER", league_points: 300 });
  });

  it("pred absent → null", async () => {
    expect(await readPred(new MemoryKV(), "p")).toBeNull();
  });

  it("shap : la valeur KV EST le tableau de drivers", async () => {
    const kv = new MemoryKV();
    expect(await readShap(kv, "p")).toEqual({ available: false, drivers: [] });
    await kv.put(KEYS.shap("p"), JSON.stringify([{ feature: "gd10", sv: 0.3 }]));
    expect(await readShap(kv, "p")).toEqual({ available: true, drivers: [{ feature: "gd10", sv: 0.3 }] });
  });
});

describe("readJsonl / appendJsonl / readJson", () => {
  it("parse les lignes en ignorant les vides, append conserve l'existant", async () => {
    const kv = new MemoryKV();
    await kv.put("k", JSON.stringify({ a: 1 }) + "\n\n" + JSON.stringify({ a: 2 }) + "\n");
    expect(await readJsonl(kv, "k")).toEqual([{ a: 1 }, { a: 2 }]);
    await appendJsonl(kv, "k", { a: 3 });
    expect(await readJsonl(kv, "k")).toEqual([{ a: 1 }, { a: 2 }, { a: 3 }]);
    expect(await readJsonl(new MemoryKV(), "absent")).toEqual([]);
  });

  it("readJson : absent → null, JSON invalide → null", async () => {
    const kv = new MemoryKV();
    expect(await readJson(kv, "x")).toBeNull();
    await kv.put("x", "{pas du json");
    expect(await readJson(kv, "x")).toBeNull();
    await kv.put("x", "{\"ok\":true}");
    expect(await readJson(kv, "x")).toEqual({ ok: true });
  });
});

describe("accounts", () => {
  it("compte préconfiguré + lookup", () => {
    expect(ACCOUNTS).toContainEqual({ slug: "spadzze", riot_id: "Spadzze#euw", region: "euw1" });
    expect(accountFor("spadzze")?.riot_id).toBe("Spadzze#euw");
    expect(accountFor("inconnu")).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run — vérifier l'échec**

Run: `cd web/cf && npm test`
Expected: FAIL — modules `../src/readers` / `../src/accounts` introuvables.

- [ ] **Step 3: Implémentation**

`web/cf/src/accounts.ts` (miroir de `web/backend/accounts.json` — si le json évolue, mettre les deux à jour) :

```ts
export interface Account {
  slug: string;
  riot_id: string;
  region: string;
}

export const ACCOUNTS: Account[] = [{ slug: "spadzze", riot_id: "Spadzze#euw", region: "euw1" }];

export function accountFor(slug: string): Account | undefined {
  return ACCOUNTS.find((a) => a.slug === slug);
}
```

`web/cf/src/readers.ts` :

```ts
/** Lectures KV — portage de web/backend/readers.py (mêmes sémantiques). */
export interface KVLike {
  get(key: string): Promise<string | null>;
  put(key: string, value: string): Promise<void>;
}

export const KEYS = {
  games: (slug: string) => `silver:${slug}:games`,
  rank: (slug: string) => `silver:${slug}:rank`,
  gold: (slug: string, scope: string) => `gold:${slug}:${scope}`,
  ref: (rank: string, scope: string) => `ref:${rank}:${scope}`,
  pred: (slug: string) => `pred:${slug}`,
  shap: (slug: string) => `shap:${slug}:drivers`,
  reviews: (slug: string) => `coaching:${slug}:reviews`,
  feedback: (slug: string) => `coaching:${slug}:feedback`,
};

export function matchSeq(matchId: string): number {
  const tail = matchId.split("_").pop() ?? "";
  const n = Number(tail);
  return Number.isFinite(n) && tail !== "" ? n : 0;
}

export interface GamesPage {
  items: Record<string, unknown>[];
  page: number;
  size: number;
  total: number;
}

export async function readGames(kv: KVLike, slug: string, page = 1, size = 20): Promise<GamesPage> {
  const rows = await readJsonl(kv, KEYS.games(slug));
  const items = [...rows].sort(
    (a, b) => matchSeq(String(b.match_id ?? "")) - matchSeq(String(a.match_id ?? "")),
  );
  const start = (page - 1) * size;
  return { items: items.slice(start, start + size), page, size, total: items.length };
}

export async function readRank(kv: KVLike, slug: string): Promise<Record<string, unknown> | null> {
  return readJson(kv, KEYS.rank(slug));
}

export async function readPred(kv: KVLike, slug: string): Promise<Record<string, unknown> | null> {
  return readJson(kv, KEYS.pred(slug));
}

export async function readShap(kv: KVLike, slug: string): Promise<{ available: boolean; drivers: unknown[] }> {
  const v = await readJson(kv, KEYS.shap(slug));
  return Array.isArray(v) ? { available: true, drivers: v } : { available: false, drivers: [] };
}

export async function readJsonl<T = Record<string, unknown>>(kv: KVLike, key: string): Promise<T[]> {
  const raw = await kv.get(key);
  if (raw === null) return [];
  return raw
    .split("\n")
    .filter((l) => l.trim() !== "")
    .map((l) => JSON.parse(l) as T);
}

export async function readJson(kv: KVLike, key: string): Promise<any> {
  const raw = await kv.get(key);
  if (raw === null) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export async function appendJsonl(kv: KVLike, key: string, record: Record<string, unknown>): Promise<void> {
  const lines = await readJsonl(kv, key);
  lines.push(record as never);
  await kv.put(key, lines.map((l) => JSON.stringify(l)).join("\n"));
}
```

- [ ] **Step 4: Run — tests verts**

Run: `cd web/cf && npm test`
Expected: tous passés (Task 2 + Task 3).

- [ ] **Step 5: Commit**

```bash
git add web/cf/src/readers.ts web/cf/src/accounts.ts web/cf/test/readers.test.ts
git commit -m "feat(cf): readers KV (tri match_id, pagination, jsonl) + comptes preconfigures"
```

---

### Task 4: Routes lecture `/api/*`

**Files:**
- Modify: `web/cf/src/index.ts`
- Test: `web/cf/test/api.test.ts`

**Interfaces:**
- Consumes: `handle`/`Env` (Task 2) ; `KVLike`, `KEYS`, `readGames`, `readRank`, `readPred`, `readShap`, `readJsonl`, `readJson` (Task 3) ; `ACCOUNTS`, `accountFor` (Task 3).
- Produces: `Env` devient `{ DATA: KVLike; ASSETS: Fetcher; OLLAMA_API_KEY?: string; OLLAMA_MODEL?: string }` (OLLAMA_* consommés en Task 10). Routes servies : `GET /api/accounts`, `GET /api/c/{slug}/games|rank|predicted-rank|reviews|feedback|shap` — réponses **identiques en surface** aux routeurs FastAPI actuels.

- [ ] **Step 1: Test en échec**

`web/cf/test/api.test.ts` :

```ts
import { beforeEach, describe, expect, it } from "vitest";
import { handle, type Env } from "../src/index";
import { KEYS, type KVLike } from "../src/readers";

class MemoryKV implements KVLike {
  store = new Map<string, string>();
  async get(key: string) { return this.store.get(key) ?? null; }
  async put(key: string, value: string) { this.store.set(key, value); }
}

const SPA_HTML = "<!doctype html><title>spa</title>";

function makeEnv(): { env: Env; kv: MemoryKV } {
  const kv = new MemoryKV();
  const env = {
    DATA: kv,
    ASSETS: { fetch: async () => new Response(SPA_HTML, { headers: { "content-type": "text/html" } }) },
  } as unknown as Env;
  return { env, kv };
}

beforeEach(async () => {
  // appelé via la closure makeEnv() dans chaque test ; cf. seed() ci-dessous
});

async function seed(): Promise<{ env: Env; kv: MemoryKV }> {
  const { env, kv } = makeEnv();
  await kv.put(KEYS.games("spadzze"),
    [JSON.stringify({ match_id: "EUW1_10", champion: "Zeri", win: true }),
     JSON.stringify({ match_id: "EUW1_30", champion: "Jinx", win: false }),
     JSON.stringify({ match_id: "EUW1_20", champion: "Caitlyn", win: true })].join("\n"));
  await kv.put(KEYS.rank("spadzze"), JSON.stringify({ tier: "MASTER", league_points: 300, fetched_at: "2026-08-30T10:00:00" }));
  await kv.put(KEYS.pred("spadzze"), JSON.stringify({ predicted_rank: "master", proba: 0.61, n_games_used: 30, predicted_lp: 412 }));
  await kv.put(KEYS.shap("spadzze"), JSON.stringify([{ feature: "gd10", sv: 0.3 }]));
  await kv.put(KEYS.reviews("spadzze"), JSON.stringify({ ts: "2026-08-30T11:00:00", model: "kimi-k2.6", review: {} }));
  await kv.put(KEYS.feedback("spadzze"), JSON.stringify({ ts: "2026-08-30T11:00:00", items: [] }));
  return { env, kv };
}

describe("GET /api/accounts", () => {
  it("résume chaque compte : games_count + last_review_ts", async () => {
    const { env } = await seed();
    const r = await handle(new Request("http://x/api/accounts"), env);
    expect(r.status).toBe(200);
    expect(await r.json()).toEqual([{
      slug: "spadzze", riot_id: "Spadzze#euw", region: "euw1",
      games_count: 3, last_review_ts: "2026-08-30T11:00:00",
    }]);
  });
});

describe("GET /api/c/{slug}/games", () => {
  it("défauts page=1 size=20, tri séquence décroissante", async () => {
    const { env } = await seed();
    const r = await handle(new Request("http://x/api/c/spadzze/games"), env);
    const j = await r.json();
    expect(j).toEqual({ items: [
      { match_id: "EUW1_30", champion: "Jinx", win: false },
      { match_id: "EUW1_20", champion: "Caitlyn", win: true },
      { match_id: "EUW1_10", champion: "Zeri", win: true },
    ], page: 1, size: 20, total: 3 });
  });

  it("422 si page<1 ou size hors [1,200]", async () => {
    const { env } = await seed();
    for (const q of ["page=0", "size=0", "size=201", "page=abc", "size=abc"]) {
      const r = await handle(new Request(`http://x/api/c/spadzze/games?${q}`), env);
      expect(r.status).toBe(422);
      expect((await r.json()).detail).toBe("page>=1 et size in [1,200]");
    }
  });

  it("slug inconnu → liste vide (pas de 404, parité FastAPI)", async () => {
    const { env } = await seed();
    const r = await handle(new Request("http://x/api/c/inconnu/games"), env);
    expect(await r.json()).toEqual({ items: [], page: 1, size: 20, total: 0 });
  });
});

describe("GET /api/c/{slug}/rank + predicted-rank", () => {
  it("rank présent / vide structuré si absent", async () => {
    const { env } = await seed();
    expect(await (await handle(new Request("http://x/api/c/spadzze/rank"), env)).json())
      .toEqual({ tier: "MASTER", league_points: 300, fetched_at: "2026-08-30T10:00:00" });
    expect(await (await handle(new Request("http://x/api/c/inconnu/rank"), env)).json())
      .toEqual({ tier: null, division: null, league_points: null, wins: null, losses: null, fetched_at: null });
  });

  it("predicted-rank lit pred:{slug} (précalculé)", async () => {
    const { env } = await seed();
    expect(await (await handle(new Request("http://x/api/c/spadzze/predicted-rank"), env)).json())
      .toEqual({ predicted_rank: "master", proba: 0.61, n_games_used: 30, predicted_lp: 412 });
    expect(await (await handle(new Request("http://x/api/c/inconnu/predicted-rank"), env)).json())
      .toEqual({ predicted_rank: null, proba: null, n_games_used: 0 });
  });
});

describe("GET /api/c/{slug}/reviews|feedback|shap", () => {
  it("listes jsonl + shap structuré", async () => {
    const { env } = await seed();
    expect(await (await handle(new Request("http://x/api/c/spadzze/reviews"), env)).json()).toHaveLength(1);
    expect(await (await handle(new Request("http://x/api/c/spadzze/feedback"), env)).json()).toHaveLength(1);
    expect(await (await handle(new Request("http://x/api/c/spadzze/shap"), env)).json())
      .toEqual({ available: true, drivers: [{ feature: "gd10", sv: 0.3 }] });
    expect(await (await handle(new Request("http://x/api/c/inconnu/shap"), env)).json())
      .toEqual({ available: false, drivers: [] });
  });
});

describe("routes API inconnues", () => {
  it("POST /api/fetch et GET /api/jobs/{id} n'existent plus (V1)", async () => {
    const { env } = await seed();
    const r1 = await handle(new Request("http://x/api/fetch", { method: "POST", body: "{}" }), env);
    const r2 = await handle(new Request("http://x/api/jobs/abc"), env);
    expect(r1.status).toBe(404);
    expect(r2.status).toBe(404);
  });
});
```

- [ ] **Step 2: Run — vérifier l'échec**

Run: `cd web/cf && npm test`
Expected: FAIL — les nouvelles routes renvoient 404 (`/api/accounts` etc. non implémentés).

- [ ] **Step 3: Implémentation — étendre `index.ts`**

Remplacer l'intégralité de `web/cf/src/index.ts` par :

```ts
import { ACCOUNTS, accountFor } from "./accounts";
import { KEYS, readGames, readJsonl, readJson, readPred, readRank, readShap, type KVLike } from "./readers";

export interface Env {
  DATA: KVLike;
  ASSETS: Fetcher;
  OLLAMA_API_KEY?: string;
  OLLAMA_MODEL?: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return handle(request, env);
  },
};

const EMPTY_RANK = { tier: null, division: null, league_points: null, wins: null, losses: null, fetched_at: null };
const EMPTY_PRED = { predicted_rank: null, proba: null, n_games_used: 0 };

async function apiAccounts(env: Env): Promise<Response> {
  const out: unknown[] = [];
  for (const a of ACCOUNTS) {
    const games = await readGames(env.DATA, a.slug, 1, 1);
    const revs = await readJsonl<{ ts: string }>(env.DATA, KEYS.reviews(a.slug));
    out.push({
      slug: a.slug,
      riot_id: a.riot_id,
      region: a.region,
      games_count: games.total,
      last_review_ts: revs.length ? revs[revs.length - 1].ts : null,
    });
  }
  return Response.json(out);
}

function apiGames(env: Env, slug: string, params: URLSearchParams): Response {
  const page = Number(params.get("page") ?? 1);
  const size = Number(params.get("size") ?? 20);
  const okPage = Number.isInteger(page) && page >= 1;
  const okSize = Number.isInteger(size) && size >= 1 && size <= 200;
  if (!okPage || !okSize) {
    return Response.json({ detail: "page>=1 et size in [1,200]" }, { status: 422 });
  }
  return Response.json(readGames(env.DATA, slug, page, size)); // Response.json await les promesses
}

export async function handle(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  if (url.pathname === "/api/health") {
    return Response.json({ status: "ok", service: "coaching-lol", server_time: new Date().toISOString() });
  }
  if (url.pathname === "/api/accounts" && request.method === "GET") {
    return apiAccounts(env);
  }
  const m = url.pathname.match(/^\/api\/c\/([^/]+)\/([a-z-]+)$/);
  if (m) {
    const [, slug, tail] = m;
    if (request.method !== "GET") return Response.json({ detail: "Method Not Allowed" }, { status: 405 });
    if (tail === "games") return apiGames(env, slug, url.searchParams);
    if (tail === "rank") return Response.json((await readRank(env.DATA, slug)) ?? EMPTY_RANK);
    if (tail === "predicted-rank") return Response.json((await readPred(env.DATA, slug)) ?? EMPTY_PRED);
    if (tail === "reviews") return Response.json(await readJsonl(env.DATA, KEYS.reviews(slug)));
    if (tail === "feedback") return Response.json(await readJsonl(env.DATA, KEYS.feedback(slug)));
    if (tail === "shap") return Response.json(await readShap(env.DATA, slug));
  }
  if (url.pathname.startsWith("/api/")) {
    return Response.json({ detail: "Not Found" }, { status: 404 });
  }
  return env.ASSETS.fetch(request);
}
```

Note : `Response.json(promise)` accepte une Promise (le runtime l'await) — si la version de `@cloudflare/workers-types` résolue refuse le typage, écrire `return readGames(...).then((r) => Response.json(r))` à la place (comportement identique).

- [ ] **Step 4: Run — tests verts**

Run: `cd web/cf && npm test`
Expected: tous passés.

- [ ] **Step 5: Commit**

```bash
git add web/cf/src/index.ts web/cf/test/api.test.ts
git commit -m "feat(cf): routes lecture /api/* sur KV (accounts, games, rank, pred, reviews, feedback, shap)"
```

---

### Task 5: `sync_cloudflare.py` (push KV + précompute ML)

**Files:**
- Create: `src/collection/sync_cloudflare.py`
- Test: `tests/test_sync_cloudflare.py`

**Interfaces:**
- Consumes: `riotlib` (`rl.DATA`, `rl.load_env()` — riotlib.py:61), `web/backend/ml_rank.py` (`predict_rank(games: list[dict]) -> dict | None` — résout lui-même `src/core` + `src/01_data_engineering` dans sys.path, ml_rank.py:41-49 ; il faut seulement que `riotlib` soit importable, donc insérer `src/core`), `web/backend/accounts.json`.
- Produces: classe `KV` (REST PUT/GET, journal `puts`), `sync_account(kv, slug, *, seed_reviews=False)`, `sync_referential(kv)`, `load_accounts()`, `read_games(slug)`, `main()` (CLI `--slug`, `--skip-ref`, `--seed-reviews`, `--dry-run`). Consommé par la Task 6 (run réel) et la Task 15 (merge rapatriement — `from sync_cloudflare import KV`).

- [ ] **Step 1: Test en échec**

`tests/test_sync_cloudflare.py` :

```python
"""Tests du sync KV : dossiers temporaires, fake KV, ml_rank monkeypatché — 0 réseau."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src" / "core", ROOT / "src" / "collection", ROOT / "web" / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import riotlib as rl  # noqa: E402
import ml_rank  # noqa: E402
import sync_cloudflare as sc  # noqa: E402


class FakeKV(sc.KV):
    """KV en mémoire : mêmes méthodes, zéro requête."""

    def __init__(self):
        super().__init__("account", "namespace", "token")
        self.store: dict[str, str] = {}

    def put(self, key: str, value: str) -> None:
        self.puts.append(key)
        self.store[key] = value

    def get(self, key: str) -> str | None:
        return self.store.get(key)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "DATA", tmp_path)
    return tmp_path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_sync_account_pushes_keys(data_root, monkeypatch):
    _write(data_root / "02_silver" / "personal" / "p" / "games.jsonl",
           json.dumps({"match_id": "EUW1_10", "role": "BOTTOM"}) + "\n")
    _write(data_root / "02_silver" / "personal" / "p" / "rank.json",
           json.dumps({"tier": "MASTER"}))
    _write(data_root / "03_gold" / "personal" / "p" / "all" / "aggregate.json",
           json.dumps({"n_games": 10}))
    _write(data_root / "03_gold" / "personal" / "p" / "adc" / "aggregate.json",
           json.dumps({"n_games": 8}))
    _write(data_root / "06_shap" / "p_drivers.json", json.dumps([{"feature": "gd10"}]))

    monkeypatch.setattr(ml_rank, "predict_rank",
                        lambda games: {"predicted_rank": "master", "proba": 0.6, "n_games_used": 20})

    kv = FakeKV()
    sc.sync_account(kv, "p")

    assert kv.store["silver:p:games"] == (data_root / "02_silver" / "personal" / "p" / "games.jsonl").read_text()
    assert json.loads(kv.store["silver:p:rank"]) == {"tier": "MASTER"}
    assert json.loads(kv.store["gold:p:all"]) == {"n_games": 10}
    assert json.loads(kv.store["gold:p:adc"]) == {"n_games": 8}
    assert json.loads(kv.store["pred:p"]) == {"predicted_rank": "master", "proba": 0.6, "n_games_used": 20}
    assert json.loads(kv.store["shap:p:drivers"]) == [{"feature": "gd10"}]
    # les clés coaching:* ne sont JAMAIS touchées sans --seed-reviews
    assert not [k for k in kv.store if k.startswith("coaching:")]


def test_sync_account_slices_last_20_games(data_root, monkeypatch):
    seen: list[int] = []

    def fake_predict(games):
        seen.append(len(games))
        return None  # < MIN_ADC_GAMES -> pas de clé pred

    monkeypatch.setattr(ml_rank, "predict_rank", fake_predict)
    games = (data_root / "02_silver" / "personal" / "p" / "games.jsonl")
    games.parent.mkdir(parents=True, exist_ok=True)
    games.write_text("\n".join(json.dumps({"match_id": f"EUW1_{i}"}) for i in range(30)) + "\n")

    kv = FakeKV()
    sc.sync_account(kv, "p")
    assert seen == [20]
    assert "pred:p" not in kv.store  # predict None -> pas de clé


def test_seed_reviews_only_when_kv_key_absent(data_root):
    _write(data_root / "07_coaching" / "p" / "reviews.jsonl",
           json.dumps({"ts": "t1"}) + "\n")
    kv = FakeKV()
    sc.sync_account(kv, "p", seed_reviews=True)
    assert kv.store.get("coaching:p:reviews") == (data_root / "07_coaching" / "p" / "reviews.jsonl").read_text()
    # clé déjà présente (p.ex. reviews générées côté web) -> pas d'écrasement
    kv2 = FakeKV()
    kv2.store["coaching:p:reviews"] = json.dumps({"ts": "web"}) + "\n"
    sc.sync_account(kv2, "p", seed_reviews=True)
    assert json.loads(kv2.store["coaching:p:reviews"]) == {"ts": "web"}


def test_sync_referential(data_root):
    _write(data_root / "03_gold" / "referentiel" / "challenger" / "adc" / "aggregate.json",
           json.dumps({"n_games": 100}))
    kv = FakeKV()
    sc.sync_referential(kv)
    assert json.loads(kv.store["ref:challenger:adc"]) == {"n_games": 100}
```

- [ ] **Step 2: Run — vérifier l'échec**

Run: `poetry run pytest tests/test_sync_cloudflare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sync_cloudflare'`.

- [ ] **Step 3: Implémentation**

`src/collection/sync_cloudflare.py` :

```python
# src/collection/sync_cloudflare.py
"""Sync local -> Cloudflare Workers KV : données + prédictions ML précalculées.

0 appel Riot. Lit data/ (silver/gold/shap), PRÉCALCULE predicted-rank (+LP) via
web/backend/ml_rank.py — là où vivent les .pkl, qui ne quittent jamais la
machine — et pousse le tout dans KV via l'API REST Cloudflare (token dédié
dans .env : CF_API_TOKEN, CF_ACCOUNT_ID, CF_NAMESPACE_ID). Aucune dépendance node.

Propriété des clés coaching:* : le Worker est propriétaire des reviews/feedback
générés côté web. Ce sync ne les touche PAS par défaut ; --seed-reviews amorce
coaching:{slug}:reviews UNIQUEMENT si la clé KV est absente (premier déploiement).

Usage :
  poetry run python3 src/collection/sync_cloudflare.py                 # tous les comptes
  poetry run python3 src/collection/sync_cloudflare.py --slug spadzze  # un compte
  ... --skip-ref       # ne pas repousser le référentiel (rarement utile)
  ... --seed-reviews   # amorce coaching:{slug}:reviews si clé absente
  ... --dry-run        # affiche les clés sans rien écrire
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src" / "core", ROOT / "web" / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import riotlib as rl  # noqa: E402

ACCOUNTS_FILE = ROOT / "web" / "backend" / "accounts.json"
KV_URL = ("https://api.cloudflare.com/client/v4/accounts/{account}"
          "/storage/kv/namespaces/{ns}/values/{key}")


def load_accounts() -> list[dict]:
    return json.loads(ACCOUNTS_FILE.read_text())


def _match_seq(match_id: str) -> int:
    try:
        return int(match_id.rsplit("_", 1)[-1])
    except (ValueError, AttributeError):
        return 0


def read_games(slug: str) -> list[dict]:
    """Toutes les games silver du joueur, tri chronologique décroissant (match_seq)."""
    path = rl.DATA / "02_silver" / "personal" / slug / "games.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return sorted(rows, key=lambda r: _match_seq(r.get("match_id", "")), reverse=True)


class KV:
    """Client REST minimal Workers KV (PUT/GET) + journal des clés poussées."""

    def __init__(self, account: str, namespace: str, token: str):
        self.account, self.namespace, self.token = account, namespace, token
        self.puts: list[str] = []

    def _url(self, key: str) -> str:
        return KV_URL.format(account=self.account, ns=self.namespace,
                             key=quote(key, safe=""))

    def put(self, key: str, value: str) -> None:
        r = requests.put(self._url(key), data=value.encode("utf-8"),
                         headers={"Authorization": f"Bearer {self.token}",
                                  "Content-Type": "text/plain"}, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"KV PUT {key} -> HTTP {r.status_code} : {r.text[:200]}")
        self.puts.append(key)

    def get(self, key: str) -> str | None:
        r = requests.get(self._url(key),
                         headers={"Authorization": f"Bearer {self.token}"}, timeout=30)
        return r.text if r.status_code == 200 else None


def put_json(kv: KV, key: str, obj) -> None:
    kv.put(key, json.dumps(obj, ensure_ascii=False))


def sync_account(kv: KV, slug: str, *, seed_reviews: bool = False) -> None:
    games = read_games(slug)
    silver = rl.DATA / "02_silver" / "personal" / slug
    if (silver / "games.jsonl").exists():
        kv.put(f"silver:{slug}:games", (silver / "games.jsonl").read_text())
    if (silver / "rank.json").exists():
        kv.put(f"silver:{slug}:rank", (silver / "rank.json").read_text())

    gold = rl.DATA / "03_gold" / "personal" / slug
    if gold.is_dir():
        for scope_dir in sorted(p for p in gold.iterdir() if p.is_dir()):
            agg = scope_dir / "aggregate.json"
            if agg.exists():
                put_json(kv, f"gold:{slug}:{scope_dir.name}", json.loads(agg.read_text()))

    # predicted-rank : PRÉCALCUL local (miroir exact de l'ancien routeur Python,
    # qui passait les 20 dernières games à predict_rank).
    import ml_rank  # noqa: E402  (import tardif : pandas + pkl chargés seulement ici)
    pred = ml_rank.predict_rank(games[:20])
    if pred is not None:
        put_json(kv, f"pred:{slug}", pred)

    shap = rl.DATA / "06_shap" / f"{slug}_drivers.json"
    if shap.exists():
        kv.put(f"shap:{slug}:drivers", shap.read_text())

    if seed_reviews:
        reviews = rl.DATA / "07_coaching" / slug / "reviews.jsonl"
        if reviews.exists() and kv.get(f"coaching:{slug}:reviews") is None:
            kv.put(f"coaching:{slug}:reviews", reviews.read_text())


def sync_referential(kv: KV) -> None:
    ref = rl.DATA / "03_gold" / "referentiel"
    if not ref.is_dir():
        return
    for rank_dir in sorted(p for p in ref.iterdir() if p.is_dir()):
        for scope_dir in sorted(p for p in rank_dir.iterdir() if p.is_dir()):
            agg = scope_dir / "aggregate.json"
            if agg.exists():
                put_json(kv, f"ref:{rank_dir.name}:{scope_dir.name}",
                         json.loads(agg.read_text()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync local -> Cloudflare KV (cf. docstring)")
    ap.add_argument("--slug", help="limite le sync à un compte")
    ap.add_argument("--skip-ref", action="store_true", help="ne pas repousser le référentiel")
    ap.add_argument("--seed-reviews", action="store_true",
                    help="amorce coaching:{slug}:reviews si la clé KV est absente")
    ap.add_argument("--dry-run", action="store_true", help="affiche les clés sans écrire")
    args = ap.parse_args()

    env = rl.load_env()
    token, account, namespace = (env.get("CF_API_TOKEN", ""),
                                 env.get("CF_ACCOUNT_ID", ""),
                                 env.get("CF_NAMESPACE_ID", ""))
    if not (token and account and namespace):
        sys.exit("CF_API_TOKEN / CF_ACCOUNT_ID / CF_NAMESPACE_ID manquants dans .env "
                 "(cf. Task 6 du plan migration Cloudflare)")
    kv = KV(account, namespace, token)

    accounts = load_accounts()
    if args.slug:
        accounts = [a for a in accounts if a["slug"] == args.slug]
        if not accounts:
            sys.exit(f"compte inconnu : {args.slug}")
    for a in accounts:
        sync_account(kv, a["slug"], seed_reviews=args.seed_reviews)
    if not args.skip_ref:
        sync_referential(kv)

    print(f"{len(kv.puts)} clés poussées :")
    for k in kv.puts:
        print(f"  {k}")


if __name__ == "__main__":
    main()
```

NB dry-run : `--dry-run` n'appelle pas l'API → implémenter en court-circuitant `KV.put`/`KV.get` dans `main()` (p.ex. `class DryKV(KV)` avec `put` qui journalise sans requête et `get` qui renvoie `None`) ; les tests ne couvrent pas le dry-run (parcours trivial).

- [ ] **Step 4: Run — tests verts**

Run: `poetry run pytest tests/test_sync_cloudflare.py -v`
Expected: 4 passed. Puis `poetry run pytest tests/ -q` → la suite existante reste verte (le nouveau module n'importe rien de cassant au collect).

- [ ] **Step 5: Commit**

```bash
git add src/collection/sync_cloudflare.py tests/test_sync_cloudflare.py
git commit -m "feat(collection): sync_cloudflare — push KV (silver/gold/shap/ref) + precompute rang ML"
```

---

### Task 6: Run réel — namespace KV, token, .env, sync, binding

**Files:**
- Modify: `web/cf/wrangler.toml` (section `[[kv_namespaces]]` avec le vrai id)
- Create: `web/cf/.dev.vars` (gitignoré — `OLLAMA_API_KEY=...`, utilisé dès la Task 10)
- Modify: `.env` local (gitignoré — CF_API_TOKEN, CF_ACCOUNT_ID, CF_NAMESPACE_ID)

**Interfaces:**
- Consumes: `sync_cloudflare.py` (Task 5), `wrangler` (Task 2).
- Produces: binding `DATA` opérationnel dans wrangler.toml ; KV peuplé avec les vraies données ; `.dev.vars` pour le coach local (Task 10/12).

- [ ] **Step 1: Auth wrangler (interactif — taper `!`)**

L'utilisateur tape dans le prompt :

```
! cd /Users/jeanvangysel/code/website/coaching_lol/web/cf && npx wrangler login
```

(browser OAuth Cloudflare)

- [ ] **Step 2: Créer le namespace KV**

Run: `cd web/cf && npx wrangler kv namespace create coaching_lol_data`
Expected: sortie `[[kv_namespaces]]` avec `binding = "DATA"` et `id = "…"` — copier ce bloc.

- [ ] **Step 3: Coller le binding dans wrangler.toml**

Ajouter en fin de `web/cf/wrangler.toml` (avec l'id imprimé à l'étape précédente) :

```toml
[[kv_namespaces]]
binding = "DATA"
id = "<id imprimé par wrangler kv namespace create>"
```

- [ ] **Step 4: Token API pour le sync Python (dashboard)**

Dashboard Cloudflare → My Profile → API Tokens → Create Token → "Create Custom Token" :
- Permissions : **Account / Workers KV Storage / Edit** + **Account / Account Settings / Read**
- Account Resources : restriction au compte concerné (ou All accounts)
→ créer, copier le token (affiché une seule fois).

- [ ] **Step 5: Renseigner .env + .dev.vars**

`.env` (racine du repo — jamais commité) :

```
CF_API_TOKEN=<token étape 4>
CF_ACCOUNT_ID=<account id, visible dans le dashboard ou dans la sortie wrangler>
CF_NAMESPACE_ID=<id namespace étape 2>
```

`web/cf/.dev.vars` :

```
OLLAMA_API_KEY=<clé Ollama Cloud, valeur de OLLAMA_API_KEY du .env local>
```

Vérifier : `git status --short` ne montre NI `.env` NI `web/cf/.dev.vars` (couverts par .gitignore de la Task 2).

- [ ] **Step 6: Dry-run puis sync réel**

Run: `poetry run python3 src/collection/sync_cloudflare.py --dry-run`
Expected: la liste des clés (dry-run journalise sans écrire). Puis réel :
`poetry run python3 src/collection/sync_cloudflare.py`
Expected: `~13 clés poussées` (games, rank, gold×scopes, pred, shap) + `~36 clés référentiel` (4 rangs × scopes) ; zéro erreur HTTP. Vérifier : `cd web/cf && npx wrangler kv key list --namespace-id <id> | head -30`.

- [ ] **Step 7: Vérification bout-en-bout en local**

Run: `cd web/cf && npx wrangler dev` puis :

```bash
curl -sS http://localhost:8787/api/health
curl -sS http://localhost:8787/api/accounts          # games_count ≈ nombre de games silver locales
curl -sS "http://localhost:8787/api/c/spadzze/games?size=5"   # items triés séquence décroissante
curl -sS http://localhost:8787/api/c/spadzze/rank             # tier/league_points réels
curl -sS http://localhost:8787/api/c/spadzze/predicted-rank   # predicted_rank + proba + n_games_used (+ predicted_lp)
curl -sS http://localhost:8787/api/c/spadzze/shap             # available: true
curl -sS http://localhost:8787/api/c/spadzze/reviews          # [] (vide jusqu'au coach / seed)
```

Comparaison avec la prod Fly : chaque réponse doit avoir la même surface que `https://coaching-lol.fly.dev/api/...` (aux données près). Arrêter wrangler.

- [ ] **Step 8: Commit**

```bash
git add web/cf/wrangler.toml
git commit -m "feat(cf): binding KV DATA (namespace coaching_lol_data) + sync reel effectue"
```

---

## Phase P3 — Coach SSE + feedback (le Worker écrit)

### Task 7: `schema.ts` — validation Review (miroir Pydantic)

**Files:**
- Create: `web/cf/src/schema.ts`
- Test: `web/cf/test/schema.test.ts`

**Interfaces:**
- Consumes: rien (module pur).
- Produces (Tasks 10/11) :
  - `export interface Insight { point: string; evidence: string }`
  - `export interface Review { strengths: Insight[]; mistakes: Insight[]; habits: string[]; next_focus: string; confidence: number }`
  - `export function validateReview(raw: unknown): Review | null` — null = non conforme
  - `export function reviewJsonSchema(): Record<string, unknown>` — JSON-schema pour le `format` d'Ollama
  - `export const NEG_TAGS: readonly string[]` + `export type TagKind` (Task 11)
- Règles (parité stricte avec `src/04_coaching/schema.py:20-27`) : `strengths` 1-3 items, `mistakes` **exactement 3**, `habits` **exactement 2** chaînes, `next_focus` chaîne, `confidence` nombre ∈ [0,1]. Chaque Insight = `{point: string, evidence: string}` — Pydantic n'impose PAS de longueur minimale : une chaîne vide est ACCEPTÉE (parité, on ne resserre pas silencieusement). Les champs inconnus sont ignorés.

- [ ] **Step 1: Test en échec**

`web/cf/test/schema.test.ts` :

```ts
import { describe, expect, it } from "vitest";
import { NEG_TAGS, reviewJsonSchema, validateReview, type Review } from "../src/schema";

const INSIGHT = (p: string, e: string) => ({ point: p, evidence: e });

function validReview(): Record<string, unknown> {
  return {
    strengths: [INSIGHT("tu recalls tard", "recall 1450g vs 1100 challenger")],
    mistakes: [INSIGHT("m1", "e1"), INSIGHT("m2", "e2"), INSIGHT("m3", "e3")],
    habits: ["h1", "h2"],
    next_focus: "trader plus tôt",
    confidence: 0.7,
  };
}

describe("validateReview", () => {
  it("accepte une review conforme (champs extra ignorés, comme Pydantic)", () => {
    const raw = validReview();
    raw["champ_inconnu"] = 1;
    const r = validateReview(raw);
    expect(r).not.toBeNull();
    expect(r?.strengths).toHaveLength(1);
    expect(r?.mistakes).toHaveLength(3);
  });

  it("force 1-3 forces, exactement 3 erreurs, exactement 2 habitudes", () => {
    expect(validateReview({ ...validReview(), strengths: [] })).toBeNull();
    expect(validateReview({ ...validReview(), strengths: [1, 2, 3, 4].map(() => INSIGHT("a", "b")) })).toBeNull();
    expect(validateReview({ ...validReview(), mistakes: [INSIGHT("m", "e")] })).toBeNull();
    expect(validateReview({ ...validReview(), habits: ["h1"] })).toBeNull();
    expect(validateReview({ ...validReview(), habits: ["h1", 2] })).toBeNull();
  });

  it("point/evidence doivent être des chaînes (vide accepté — parité Pydantic)", () => {
    expect(validateReview({ ...validReview(), strengths: [{ point: "", evidence: "" }] })).not.toBeNull();
    expect(validateReview({ ...validReview(), strengths: [{ point: 1, evidence: "x" }] })).toBeNull();
    expect(validateReview({ ...validReview(), strengths: [{ point: "x" }] })).toBeNull();
  });

  it("confidence nombre dans [0,1] ; next_focus chaîne", () => {
    expect(validateReview({ ...validReview(), confidence: 1.2 })).toBeNull();
    expect(validateReview({ ...validReview(), confidence: "0.7" })).toBeNull();
    expect(validateReview({ ...validReview(), confidence: 0 })).not.toBeNull();
    expect(validateReview({ ...validReview(), confidence: 1 })).not.toBeNull();
    expect(validateReview({ ...validReview(), next_focus: 42 })).toBeNull();
    expect(validateReview({ ...validReview(), next_focus: undefined })).toBeNull();
  });

  it("non-objet / null -> null", () => {
    expect(validateReview(null)).toBeNull();
    expect(validateReview("review")).toBeNull();
    expect(validateReview([])).toBeNull();
  });
});

describe("reviewJsonSchema", () => {
  it("contraint la génération Ollama comme le fait Pydantic (minItems/maxItems)", () => {
    const s = reviewJsonSchema() as any;
    expect(s.type).toBe("object");
    expect(s.properties.strengths.minItems).toBe(1);
    expect(s.properties.strengths.maxItems).toBe(3);
    expect(s.properties.mistakes.minItems).toBe(3);
    expect(s.properties.mistakes.maxItems).toBe(3);
    expect(s.properties.habits.minItems).toBe(2);
    expect(s.properties.habits.maxItems).toBe(2);
    expect(s.properties.confidence.minimum).toBe(0);
    expect(s.properties.confidence.maximum).toBe(1);
    expect(s.required).toEqual(["strengths", "mistakes", "habits", "next_focus", "confidence"]);
  });
});

describe("NEG_TAGS", () => {
  it("reflète TagKind de schema.py (menu feedback)", () => {
    expect(NEG_TAGS).toEqual(["asymetrie", "stat-inventee", "profondeur-en-faute",
                              "trop-vague", "non-actionnable", "autre"]);
  });
});
```

- [ ] **Step 2: Run — vérifier l'échec**

Run: `cd web/cf && npm test`
Expected: FAIL — module `../src/schema` introuvable.

- [ ] **Step 3: Implémentation**

`web/cf/src/schema.ts` :

```ts
/** Validation de la Review LLM — miroir strict de src/04_coaching/schema.py (Review). */

export interface Insight {
  point: string;
  evidence: string;
}

export interface Review {
  strengths: Insight[]; // 1-3 (forcer 3 poussait au filler, cf. schema.py)
  mistakes: Insight[];  // exactement 3
  habits: string[];     // exactement 2
  next_focus: string;
  confidence: number;   // [0, 1]
}

export const NEG_TAGS = ["asymetrie", "stat-inventee", "profondeur-en-faute",
                         "trop-vague", "non-actionnable", "autre"] as const;
export type TagKind = (typeof NEG_TAGS)[number];

function isInsight(v: unknown): v is Insight {
  return (
    typeof v === "object" && v !== null &&
    typeof (v as Insight).point === "string" &&
    typeof (v as Insight).evidence === "string"
  );
}

function isStrArray(v: unknown, len: number): v is string[] {
  return Array.isArray(v) && v.length === len && v.every((x) => typeof x === "string");
}

export function validateReview(raw: unknown): Review | null {
  if (typeof raw !== "object" || raw === null) return null;
  const r = raw as Record<string, unknown>;
  if (!Array.isArray(r.strengths) || r.strengths.length < 1 || r.strengths.length > 3) return null;
  if (!Array.isArray(r.mistakes) || r.mistakes.length !== 3) return null;
  if (!r.strengths.every(isInsight) || !r.mistakes.every(isInsight)) return null;
  if (!isStrArray(r.habits, 2)) return null;
  if (typeof r.next_focus !== "string") return null;
  const c = r.confidence;
  if (typeof c !== "number" || !(c >= 0 && c <= 1)) return null;
  return { strengths: r.strengths, mistakes: r.mistakes, habits: r.habits, next_focus: r.next_focus, confidence: c };
}

/** JSON-schema passé à Ollama `format` — équivalent sémantique de Review.model_json_schema()
 * (variante inline sans $ref : mêmes contraintes minItems/maxItems, Ollama contraint pareil). */
export function reviewJsonSchema(): Record<string, unknown> {
  const insight = {
    type: "object",
    properties: { point: { type: "string" }, evidence: { type: "string" } },
    required: ["point", "evidence"],
    additionalProperties: false,
  };
  return {
    type: "object",
    properties: {
      strengths: { type: "array", minItems: 1, maxItems: 3, items: insight },
      mistakes: { type: "array", minItems: 3, maxItems: 3, items: insight },
      habits: { type: "array", minItems: 2, maxItems: 2, items: { type: "string" } },
      next_focus: { type: "string" },
      confidence: { type: "number", minimum: 0.0, maximum: 1.0 },
    },
    required: ["strengths", "mistakes", "habits", "next_focus", "confidence"],
    additionalProperties: false,
  };
}
```

- [ ] **Step 4: Run — tests verts**

Run: `cd web/cf && npm test`
Expected: tous passés.

- [ ] **Step 5: Commit**

```bash
git add web/cf/src/schema.ts web/cf/test/schema.test.ts
git commit -m "feat(cf): validation Review TS (miroir Pydantic) + JSON-schema pour Ollama format"
```

---

### Task 8: `payload.ts` + parité golden Python ↔ TS

**Files:**
- Create: `web/cf/src/payload.ts`
- Create: `tests/test_payload_parity.py` (regen/assert du golden côté Python)
- Create: `web/cf/test/payload.test.ts` (comparaison au golden côté TS)
- Create: `web/cf/test/golden/payload_{scope}_{outcome}.json` (fixtures générées, commitées)

**Interfaces:**
- Consumes: la source de vérité `src/04_coaching/payload.py` (à porter) et `compare.context_benchmark` (`src/reporting/compare.py:63-88`).
- Produces (Task 10) :
  - `export type Outcome = "overall" | "win" | "loss"`
  - `export interface BuildArgs { player: string; scope: string; target: string; outcome: Outcome }`
  - `export function buildPayload(me: Record<string, any>, ref: Record<string, any>, args: BuildArgs): Record<string, any>` — `me`/`ref` = CONTENU PARSE d'`aggregate.json` (ce que le sync pousse dans `gold:{slug}:{scope}` / `ref:{rank}:{scope}`)
  - `export const MIN_CONTEXT_N = 8` et `export function contextBenchmark(meAgg, refAgg, axis: "lane_pattern" | "gank_exposure")` — interne au module, testé via le golden
- Le golden EST le contrat : le test TS échoue si la moindre clé, valeur, arrondi ou ordre diffère de `payload.build`.

- [ ] **Step 1: Harness golden côté Python (regen + assert)**

`tests/test_payload_parity.py` :

```python
"""Parité payload Python <-> TS : génère/vérifie les fixtures golden.

Le Python est la source de vérité (payload.py inchangé). GOLDEN_REGEN=1 (ou
fichier absent) régénère la fixture avec {input: me/ref, args, expected}.
Le test vitest web/cf/test/payload.test.ts rejoue le même build en TS et
exige l'égalité profonde. Les fixtures sont commitées (contrat de parité).
Skip si les agrégats locaux n'existent pas (machine sans data/).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src" / "core", ROOT / "src" / "04_coaching", ROOT / "src" / "reporting"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import riotlib as rl  # noqa: E402
import payload as payload_mod  # noqa: E402

GOLDEN_DIR = ROOT / "web" / "cf" / "test" / "golden"
PLAYER, TARGET = "spadzze", "challenger"
CASES = [
    ("all", "overall"), ("all", "win"), ("all", "loss"),
    ("adc", "overall"), ("adc", "win"), ("adc", "loss"),
    ("zeri", "loss"),
]


def _agg(kind: str, name: str, scope: str):
    path = rl.GOLD_DIR / kind / name / scope / "aggregate.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@pytest.mark.parametrize("scope,outcome", CASES)
def test_payload_parity_golden(scope, outcome):
    me, ref = _agg("personal", PLAYER, scope), _agg("referentiel", TARGET, scope)
    if me is None or ref is None:
        pytest.skip(f"agregats absents en local : {scope}/{outcome}")
    expected_file = GOLDEN_DIR / f"payload_{scope}_{outcome}.json"
    built = payload_mod.build(PLAYER, scope, TARGET, outcome)
    if os.environ.get("GOLDEN_REGEN") == "1" or not expected_file.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        expected_file.write_text(json.dumps(
            {"args": {"player": PLAYER, "scope": scope, "target": TARGET, "outcome": outcome},
             "me": me, "ref": ref, "expected": built},
            ensure_ascii=False, indent=1) + "\n")
        return
    golden = json.loads(expected_file.read_text())
    assert built == golden["expected"], (
        f"payload.py a diverge de la fixture {expected_file.name} — "
        "regenerer avec GOLDEN_REGEN=1 si l'evolution est voulue")
```

- [ ] **Step 2: Générer les fixtures (elles valident au passage payload.py contre lui-même)**

Run: `GOLDEN_REGEN=1 poetry run pytest tests/test_payload_parity.py -v`
Expected: 7 passed (chaque cas écrit `web/cf/test/golden/payload_*.json`). Vérifier : `ls web/cf/test/golden/` → 7 fichiers.

- [ ] **Step 3: Test TS en échec**

`web/cf/test/payload.test.ts` :

```ts
import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { buildPayload } from "../src/payload";

const dir = new URL("./golden/", import.meta.url);

describe("parité golden payload (Python == TS)", () => {
  const files = readdirSync(dir).filter((f) => f.startsWith("payload_") && f.endsWith(".json"));
  it("a des fixtures à rejouer", () => expect(files.length).toBeGreaterThanOrEqual(7));
  for (const file of files) {
    const g = JSON.parse(readFileSync(new URL(`./golden/${file}`, import.meta.url), "utf8"));
    it(`rejoue ${file}`, () => {
      expect(
        buildPayload(g.me, g.ref, g.args),
      ).toEqual(g.expected);
    });
  }
});
```

Run: `cd web/cf && npm test`
Expected: FAIL — `Cannot find module '../src/payload'`.

- [ ] **Step 4: Portage de payload.py vers payload.ts**

Porter `src/04_coaching/payload.py` (fonction `build`, lignes ~90-200) + `compare.context_benchmark` (src/reporting/compare.py:63-88) dans `web/cf/src/payload.ts`, avec les signatures ci-dessus. Le portage est mécanique — les règles de traduction non évidentes :

1. **`round(x, 4)`** : Python arrondit *half-even* (bancaire). `Math.round(x * 10000) / 10000` suffit SAUF valeur exacte `x.xxxx5` (rare sur données réelles). Si le golden échoue sur exactement un 5 en 5e décimale, implémenter l'arrondi half-even au lieu de Math.round. Le golden est l'arbitre.
2. **`sorted(POS_META)`** (payload.py trie les clés du dict) : `Object.keys(POS_META).sort()` — tri lexicographique identique (clés ASCII).
3. **Union des clés `_zone_phase_signals`** (`set(me) | set(ref)`) : itérer `[...new Set([...Object.keys(me_zp), ...Object.keys(ref_zp)])].sort()` — l'ordre Python sur set est non déterministe en cas d'égalité PARFAITE de delta (jamais observé sur données réelles) ; le tri rend le TS déterministe et le golden confirme.
4. **Ordre des clés des objets** : Python (dict insertion-ordered) et JS (objet insertion-ordered) préservent l'ordre de construction — construire les clés dans le même ordre que payload.py.
5. **Seuils en dur** (à recopier à l'identique depuis payload.py) : notable lane si `|delta| >= 2` CS ou `>= 150` gold ; zone/phase si `delta >= 0.08` ; gold-state si `|delta| >= 0.10` ; notable positioning si `thr !== null && |delta| >= thr` ; `LOW_SAMPLE_THRESHOLD = 30` ; `MIN_CONTEXT_N = 8` (repli du context_benchmark).
6. `context_benchmark(me_agg, ref_agg, axis)` — le paramètre `outcome` de la version Python est inutilisé (by_lane_context n'est pas découpé par issue) : le port TS le retire ; le reste est un port direct (bucket dominant côté me hors `"unknown"` sauf si seul, repli global gd10 + `reason` si `n_ref < 8`).
7. `meta`, `low_sample`, et les blocs de signaux : port direct, clés et labels FR à l'identique (le LLM et le prompt dépendent du wording exact).

- [ ] **Step 5: Run — golden verts des deux côtés**

Run: `cd web/cf && npm test` (payload.test.ts rejoue les 7 fixtures → PASS)
puis `poetry run pytest tests/test_payload_parity.py -v` (mode assert → PASS)

- [ ] **Step 6: Commit**

```bash
git add web/cf/src/payload.ts web/cf/test/payload.test.ts tests/test_payload_parity.py web/cf/test/golden/
git commit -m "feat(cf): portage payload (signals + context benchmark) + parite golden Python/TS"
```

---

### Task 9: `prompt.ts` + `llm_client.ts`

**Files:**
- Create: `web/cf/src/prompt.ts`, `web/cf/src/llm_client.ts`
- Test: `web/cf/test/prompt.test.ts`, `web/cf/test/llm.test.ts`

**Interfaces:**
- Consumes: rien hors stdlib (chaînes pures + fetch).
- Produces (Task 10) :
  - `prompt.ts` : `export const SYSTEM: string` (copie VERBATIM de `src/04_coaching/prompt.py` — chaîne SYSTEM du coaching agrégé, PAS SYSTEM_GAME), `export function render(payload: Record<string, any>): [string, string]` → `[SYSTEM, user]`
  - `llm_client.ts` : `export class LLMError extends Error`, `export interface GenerateOpts { apiKey: string; temperature?: number; timeoutMs?: number; fetchImpl?: typeof fetch }`, `export async function generateJson(model: string, system: string, user: string, schema: unknown, opts: GenerateOpts): Promise<Record<string, unknown>>`
- Sémantique (parité `src/04_coaching/llm_client.py`) : POST `https://ollama.com/api/chat`, Bearer, body `{model, messages:[system,user], format: schema, stream: false, options: {temperature}}`, défaut temperature 0.2, timeout 180 s ; **retente** (max 4 tentatives, backoff `2*(tentative+1)` s) sur 429/5xx/timeout/réponse non-JSON ; **échec immédiat** (LLMError) sur 401 et autres 4xx.

- [ ] **Step 1: Tests en échec**

`web/cf/test/prompt.test.ts` :

```ts
import { describe, expect, it } from "vitest";
import { render, SYSTEM } from "../src/prompt";

describe("prompt", () => {
  it("SYSTEM : copie verbatim de prompt.py (règles asymétrie + benchmark)", () => {
    expect(SYSTEM).toContain("asymétrie");
    expect(SYSTEM.length).toBeGreaterThan(500); // le vrai texte fait ~2 ko
  });

  it("render(pl) -> [SYSTEM, user] avec le payload sérialisé", () => {
    const pl = { meta: { n_games_me: 18, scope: "adc", outcome_focus: "loss", target: "challenger" }, signals: [] };
    const [sys, user] = render(pl);
    expect(sys).toBe(SYSTEM);
    expect(user).toContain("Signaux de tes 18 dernières games");
    expect(user).toContain("adc");
    expect(user).toContain("challenger");
    expect(user).toContain(JSON.stringify(pl));
    expect(user.trimEnd().endsWith("Produis la review.")).toBe(true);
  });
});
```

NB : les assertions sur `user` reflètent le gabarit exact de `prompt.py:render` (« Signaux de tes {n} dernières games ({scope}, issue={outcome_focus}, vs {target}) :\n\n{json indent 2}\n\nProduis la review. ») — les recopier depuis la source si le wording diffère.

`web/cf/test/llm.test.ts` :

```ts
import { describe, expect, it, vi } from "vitest";
import { generateJson, LLMError } from "../src/llm_client";

const OK_BODY = JSON.stringify({
  message: { content: JSON.stringify({ strengths: [], ok: true }) },
});

function res(status: number, body: string = OK_BODY): Response {
  return new Response(body, { status });
}

describe("generateJson", () => {
  it("retente 429/5xx puis réussit (backoff 2s, 4s…)", async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(res(500))
      .mockResolvedValueOnce(res(429))
      .mockResolvedValueOnce(res(200));
    const out = await generateJson("kimi-k2.6", "s", "u", {}, { apiKey: "k", fetchImpl });
    expect(out).toEqual({ strengths: [], ok: true });
    expect(fetchImpl).toHaveBeenCalledTimes(3);
  });

  it("401 -> LLMError immédiate, sans retry", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(res(401));
    await expect(generateJson("m", "s", "u", {}, { apiKey: "k", fetchImpl })).rejects.toThrow(LLMError);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("contenu non-JSON -> retry puis LLMError après 4 tentatives", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      res(200, JSON.stringify({ message: { content: "{pas du json" } })));
    await expect(generateJson("m", "s", "u", {}, { apiKey: "k", fetchImpl })).rejects.toThrow(LLMError);
    expect(fetchImpl).toHaveBeenCalledTimes(4);
  });

  it("erreur réseau (fetch qui jette) -> retry, puis réussit", async () => {
    const fetchImpl = vi.fn()
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockResolvedValueOnce(res(200));
    const out = await generateJson("m", "s", "u", {}, { apiKey: "k", fetchImpl });
    expect(out).toEqual({ strengths: [], ok: true });
  });

  it("body de requête : format=schema, stream=false, temperature défaut 0.2", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(res(200));
    await generateJson("kimi-k2.6", "sys", "usr", { type: "object" }, { apiKey: "k", fetchImpl });
    const body = JSON.parse((fetchImpl.mock.calls[0][1] as RequestInit).body as string);
    expect(body.model).toBe("kimi-k2.6");
    expect(body.messages).toEqual([{ role: "system", content: "sys" }, { role: "user", content: "usr" }]);
    expect(body.format).toEqual({ type: "object" });
    expect(body.stream).toBe(false);
    expect(body.options.temperature).toBe(0.2);
  });
});
```

- [ ] **Step 2: Run — vérifier l'échec**

Run: `cd web/cf && npm test`
Expected: FAIL — modules introuvables.

- [ ] **Step 3: Implémentation**

`web/cf/src/prompt.ts` — copier la chaîne `SYSTEM` **verbatim** depuis `src/04_coaching/prompt.py` (le texte FR avec les 8 règles : asymétrie, preuve obligatoire, priorité notable/descriptive_only, concret benchmark-relatif, low_sample, forces sans remplissage, tutoiement, JSON strict) :

```ts
/** Prompts du coaching agrégé — portage de src/04_coaching/prompt.py (SYSTEM + render). */

export const SYSTEM = `<coller ici la chaîne SYSTEM de prompt.py, caractère pour caractère>`;

export function render(payload: Record<string, any>): [string, string] {
  const m = payload.meta;
  const user =
    `Signaux de tes ${m.n_games_me} dernières games (${m.scope}, ` +
    `issue=${m.outcome_focus}, vs ${m.target}) :\n\n` +
    JSON.stringify(payload, undefined, 2) +
    "\n\nProduis la review.";
  return [SYSTEM, user];
}
```

⚠️ La chaîne SYSTEM est trop longue pour être dupliquée dans ce plan sans risque d'altération : la copier depuis `src/04_coaching/prompt.py` (elle est délimitée par `SYSTEM = """…"""` dans la section coaching agrégé). Le test Step 1 (longueur > 500 + contient « asymétrie ») protège contre une copie tronquée.

`web/cf/src/llm_client.ts` :

```ts
/** Client Ollama Cloud structured output — portage de src/04_coaching/llm_client.py. */

export class LLMError extends Error {}

export interface GenerateOpts {
  apiKey: string;
  temperature?: number; // défaut 0.2
  timeoutMs?: number;   // défaut 180 000
  fetchImpl?: typeof fetch;
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
  const temperature = opts.temperature ?? 0.2;
  const timeoutMs = opts.timeoutMs ?? 180_000;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    let parsed: Record<string, unknown> | null = null;
    let retryable = false;
    try {
      const r = await fetchImpl("https://ollama.com/api/chat", {
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
      if (r.status === 429 || r.status >= 500) {
        retryable = true;
      } else if (!r.ok) {
        throw new LLMError(`ollama HTTP ${r.status} (auth/requete invalide)`);
      } else {
        const body = (await r.json()) as { message?: { content?: string } };
        try {
          parsed = JSON.parse(body.message?.content ?? "") as Record<string, unknown>;
        } catch {
          retryable = true; // contenu non parsable -> retente (comme llm_client.py)
        }
      }
    } catch (e) {
      if (e instanceof LLMError) throw e;
      retryable = true; // timeout (AbortSignal) ou erreur réseau
    }
    if (parsed !== null) return parsed;
    if (attempt < MAX_ATTEMPTS - 1 && retryable) {
      await new Promise((res) => setTimeout(res, 2000 * (attempt + 1)));
    } else if (!retryable) {
      throw new LLMError("reponse ollama inattendue");
    }
  }
  throw new LLMError(`ollama : echec apres ${MAX_ATTEMPTS} tentatives`);
}
```

- [ ] **Step 4: Run — tests verts**

Run: `cd web/cf && npm test`
Expected: tous passés. (Les tests LLM mockent `fetchImpl` — zéro réseau, backoff réel ~2 s acceptable.)

- [ ] **Step 5: Commit**

```bash
git add web/cf/src/prompt.ts web/cf/src/llm_client.ts web/cf/test/prompt.test.ts web/cf/test/llm.test.ts
git commit -m "feat(cf): prompt (SYSTEM verbatim + render) + client Ollama TS (retries 429/5xx/timeout)"
```

---

### Task 10: Coach SSE — `coach.ts` + route `POST /api/coach`

**Files:**
- Create: `web/cf/src/coach.ts`
- Modify: `web/cf/src/index.ts` (une route)
- Test: `web/cf/test/coach.test.ts`

**Interfaces:**
- Consumes: `buildPayload` (Task 8), `render`/`SYSTEM` (Task 9), `generateJson`/`GenerateOpts` (Task 9), `validateReview`/`reviewJsonSchema` (Task 7), `KEYS`/`readJson`/`appendJsonl`/`KVLike` (Task 3), `accountFor` (Task 3), `Env` (Task 4).
- Produces:
  - `export interface CoachParams { slug: string; scope: string; outcome: string; target: string; model: string }`
  - `export type GenerateFn = (model: string, system: string, user: string, schema: unknown) => Promise<Record<string, unknown>>`
  - `export async function* coachFlow(deps: { kv: KVLike; generate: GenerateFn; now: () => string }, p: CoachParams): AsyncGenerator<{ event: "payload" | "llm" | "review" | "error"; data: unknown }>`
  - `export async function apiCoach(request: Request, env: Env): Promise<Response>` — SSE `text/event-stream`
- Flux (parité `pipeline.run_coach` + `coach.py`) : construire le payload → **event `payload`** → **event `llm`** → `generate` → validation `Review` (jusqu'à **2 tentatives** comme `coach.py._generate`) → persister `{ts, model, scope, target, payload, review, outcome_focus}` en append `coaching:{slug}:reviews` → **event `review`** (data = l'enregistrement persisté). Toute défaillance → **event `error`** avec message, AUCUNE persistance partielle (spec §8).
- Body de la requête (parité `routers/jobs.py CoachReq`) : `{slug, scope="adc", outcome="loss", target="challenger", model?}` ; model défaut `env.OLLAMA_MODEL ?? "kimi-k2.6"` ; 404 `{detail: "compte inconnu"}` si slug inconnu ; 500 `{detail: "OLLAMA_API_KEY non configuré"}` si secret absent.

- [ ] **Step 1: Test en échec**

`web/cf/test/coach.test.ts` :

```ts
import { describe, expect, it } from "vitest";
import { coachFlow, type CoachParams } from "../src/coach";
import { KEYS, readJsonl, type KVLike } from "../src/readers";

class MemoryKV implements KVLike {
  store = new Map<string, string>();
  async get(key: string) { return this.store.get(key) ?? null; }
  async put(key: string, value: string) { this.store.set(key, value); }
}

const AGG = (n: number) => ({
  n_games: n, patch: "16.13", winrate: 0.5,
  overall: { deaths_per_game: 5 }, win: { deaths_per_game: 4 }, loss: { deaths_per_game: 6 },
});

const REVIEW = {
  strengths: [{ point: "s1", evidence: "e1" }],
  mistakes: [{ point: "m1", evidence: "e1" }, { point: "m2", evidence: "e2" }, { point: "m3", evidence: "e3" }],
  habits: ["h1", "h2"],
  next_focus: "focus",
  confidence: 0.6,
};

const PARAMS: CoachParams = { slug: "spadzze", scope: "adc", outcome: "loss", target: "challenger", model: "kimi-k2.6" };

async function seed(): Promise<MemoryKV> {
  const kv = new MemoryKV();
  await kv.put(KEYS.gold("spadzze", "adc"), JSON.stringify(AGG(18)));
  await kv.put(KEYS.ref("challenger", "adc"), JSON.stringify(AGG(400)));
  return kv;
}

async function collect(gen: AsyncGenerator<{ event: string; data: any }>) {
  const out: { event: string; data: any }[] = [];
  for await (const ev of gen) out.push(ev);
  return out;
}

describe("coachFlow", () => {
  it("flux payload -> llm -> review, persiste l'enregistrement complet", async () => {
    const kv = await seed();
    const events = await collect(coachFlow(
      { kv, now: () => "2026-08-30T12:00:00", generate: async () => REVIEW }, PARAMS));
    expect(events.map((e) => e.event)).toEqual(["payload", "llm", "review"]);
    const rec = events[2].data;
    expect(rec).toMatchObject({ ts: "2026-08-30T12:00:00", model: "kimi-k2.6", scope: "adc", target: "challenger", outcome_focus: "loss" });
    expect(rec.review).toEqual(REVIEW);
    expect(rec.payload.meta).toBeDefined();
    const persisted = await readJsonl(kv, KEYS.reviews("spadzze"));
    expect(persisted).toEqual([rec]);
  });

  it("sortie non conforme -> 2 tentatives puis event error, rien persisté", async () => {
    const kv = await seed();
    let calls = 0;
    const events = await collect(coachFlow(
      { kv, now: () => "t", generate: async () => { calls++; return { strengths: [] }; } }, PARAMS));
    expect(calls).toBe(2);
    expect(events.map((e) => e.event)).toEqual(["payload", "llm", "error"]);
    expect(events[2].data.error).toContain("schéma");
    expect(await readJsonl(kv, KEYS.reviews("spadzze"))).toEqual([]);
  });

  it("agregat absent (gold ou ref) -> event error explicite, pas d'appel LLM", async () => {
    const kv = new MemoryKV(); // vide
    let called = 0;
    const events = await collect(coachFlow(
      { kv, now: () => "t", generate: async () => { called++; return REVIEW; } }, PARAMS));
    expect(events.map((e) => e.event)).toEqual(["error"]);
    expect(events[0].data.error).toContain("sync");
    expect(called).toBe(0);
  });

  it("retry interne : 1re sortie invalide, 2e valide -> review persistée", async () => {
    const kv = await seed();
    let n = 0;
    const events = await collect(coachFlow(
      { kv, now: () => "t", generate: async () => (n++ === 0 ? { bad: 1 } : REVIEW) }, PARAMS));
    expect(events.map((e) => e.event)).toEqual(["payload", "llm", "review"]);
  });
});
```

- [ ] **Step 2: Run — vérifier l'échec**

Run: `cd web/cf && npm test`
Expected: FAIL — module `../src/coach` introuvable.

- [ ] **Step 3: Implémentation**

`web/cf/src/coach.ts` :

```ts
import { KEYS, appendJsonl, readJson, type KVLike } from "./readers";
import { buildPayload } from "./payload";
import { render } from "./prompt";
import { validateReview, reviewJsonSchema, type Review } from "./schema";
import { generateJson } from "./llm_client";
import type { Env } from "./index";

export interface CoachParams {
  slug: string;
  scope: string;
  outcome: string;
  target: string;
  model: string;
}

export type GenerateFn = (
  model: string, system: string, user: string, schema: unknown,
) => Promise<Record<string, unknown>>;

export interface SseEvent {
  event: "payload" | "llm" | "review" | "error";
  data: unknown;
}

export async function* coachFlow(
  deps: { kv: KVLike; generate: GenerateFn; now: () => string },
  p: CoachParams,
): AsyncGenerator<SseEvent> {
  const me = await readJson(deps.kv, KEYS.gold(p.slug, p.scope));
  const ref = await readJson(deps.kv, KEYS.ref(p.target, p.scope));
  if (!me || !ref) {
    const missing = !me ? `agregat perso ${p.slug}/${p.scope}` : `referentiel ${p.target}/${p.scope}`;
    yield { event: "error", data: { error: `données manquantes (${missing}) — lance le sync local` } };
    return;
  }
  const pl = buildPayload(me, ref, {
    player: p.slug, scope: p.scope, target: p.target,
    outcome: p.outcome as "overall" | "win" | "loss",
  });
  yield { event: "payload", data: { stage: "payload" } };

  const [system, user] = render(pl);
  const schema = reviewJsonSchema();
  yield { event: "llm", data: { stage: "llm", model: p.model } };

  let review: Review | null = null;
  for (let i = 0; i < 2 && !review; i++) {
    const raw = await deps.generate(p.model, system, user, schema);
    review = validateReview(raw);
  }
  if (!review) {
    yield { event: "error", data: { error: "sortie LLM non conforme au schéma Review après 2 tentatives" } };
    return;
  }

  const record = {
    ts: deps.now(),
    model: p.model,
    scope: p.scope,
    target: p.target,
    payload: pl,
    review,
    outcome_focus: p.outcome,
  };
  await appendJsonl(deps.kv, KEYS.reviews(p.slug), record);
  yield { event: "review", data: record };
}

export async function apiCoach(request: Request, env: Env): Promise<Response> {
  const body = await request.json().catch(() => null) as
    | { slug?: string; scope?: string; outcome?: string; target?: string; model?: string }
    | null;
  const slug = body?.slug ?? "";
  if (!(await import("./accounts")).accountFor(slug)) {
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
  const generate: GenerateFn = (m, s, u, sch) =>
    generateJson(m, s, u, sch, { apiKey: env.OLLAMA_API_KEY! });

  const stream = new ReadableStream({
    async start(controller) {
      const enc = new TextEncoder();
      const gen = coachFlow({ kv: env.DATA, generate, now: () => new Date().toISOString() }, params);
      try {
        for await (const ev of gen) {
          controller.enqueue(enc.encode(`event: ${ev.event}\ndata: ${JSON.stringify(ev.data)}\n\n`));
        }
      } catch (e) {
        controller.enqueue(enc.encode(`event: error\ndata: ${JSON.stringify({ error: String(e) })}\n\n`));
      } finally {
        controller.close();
      }
    },
  });
  return new Response(stream, {
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
  });
}
```

Dans `web/cf/src/index.ts`, ajouter avec les autres imports : `import { apiCoach } from "./coach";` et dans `handle`, AVANT le 404 générique `/api/*` :

```ts
  if (url.pathname === "/api/coach" && request.method === "POST") {
    return apiCoach(request, env);
  }
```

NB : l'import dynamique `await import("./accounts")` dans apiCoach évite un cycle d'imports index→coach→index — si le bundler le résout sans cycle (imports de TYPE uniquement), un import statique `accountFor` est preferred ; les deux compilent.

- [ ] **Step 4: Run — tests verts**

Run: `cd web/cf && npm test`
Expected: tous passés (coach.test.ts utilise `generate` injecté — zéro réseau).

- [ ] **Step 5: Commit**

```bash
git add web/cf/src/coach.ts web/cf/src/index.ts web/cf/test/coach.test.ts
git commit -m "feat(cf): coach SSE — payload -> Ollama -> validation Review -> persiste KV"
```

---

### Task 11: `POST /api/feedback` sur KV

**Files:**
- Create: `web/cf/src/feedback.ts`
- Modify: `web/cf/src/index.ts` (une route)
- Test: `web/cf/test/feedback.test.ts`

**Interfaces:**
- Consumes: `accountFor` (Task 3), `KEYS`/`readJsonl`/`appendJsonl` (Task 3), `validateReview` (Task 7), `NEG_TAGS`/`TagKind` (Task 7), `Env` (Task 4).
- Produces: `export async function apiFeedback(request: Request, env: Env): Promise<Response>`.
- Contrat (parité `web/backend/routers/feedback.py` + `src/04_coaching/feedback.py:55-99`) :
  - Body : `{slug, ts, responses: {"kind,index": {useful, tag?, note?}}}`.
  - 404 `{detail: "compte inconnu"}` / 404 `{detail: "review introuvable"}` (la review = ligne de `coaching:{slug}:reviews` avec `ts` égal).
  - 422 `{detail: "clé de réponse invalide : '<k>' (attendu 'kind,index')"}` si clé non parsable ; 422 `{detail: "réponse invalide pour '<k>' : …"}` si `{useful, tag?, note?}` mal formé (useful booléen obligatoire, tag ∈ NEG_TAGS ou absent, note chaîne ou absente).
  - Construction (feedback.py `build_feedback`) : items dans l'ordre strength (1-3), mistake (0-2… indices de section), habit (0-1), puis focus index 0 — SEULEMENT les clés présentes dans `responses` (skip = omis) ; **tag requis si `useful === false`** sinon 422 `{detail: "feedback invalide (tag requis si useful=False)"}`.
  - Enregistrement persisté (append/écrase par `ts` — un ts apparaît au plus une fois) : `{ts, player, rated_at, model, overall_useful: null, items}` avec `rated_at = new Date().toISOString()` (secondes, comme `datetime.now().isoformat(timespec="seconds")`).
  - Succès : `{"ok": true}`.

- [ ] **Step 1: Test en échec**

`web/cf/test/feedback.test.ts` :

```ts
import { describe, expect, it } from "vitest";
import { apiFeedback } from "../src/feedback";
import { KEYS, readJsonl, type KVLike } from "../src/readers";
import type { Env } from "../src/index";

class MemoryKV implements KVLike {
  store = new Map<string, string>();
  async get(key: string) { return this.store.get(key) ?? null; }
  async put(key: string, value: string) { this.store.set(key, value); }
}

const REVIEW = {
  strengths: [{ point: "s1", evidence: "e1" }],
  mistakes: [{ point: "m1", evidence: "e1" }, { point: "m2", evidence: "e2" }, { point: "m3", evidence: "e3" }],
  habits: ["h1", "h2"],
  next_focus: "focus",
  confidence: 0.6,
};

async function makeEnv(): Promise<Env> {
  const kv = new MemoryKV();
  await kv.put(KEYS.reviews("spadzze"), JSON.stringify({
    ts: "T1", model: "kimi-k2.6", scope: "adc", target: "challenger",
    payload: {}, review: REVIEW, outcome_focus: "loss",
  }));
  const env = { DATA: kv, ASSETS: { fetch: async () => new Response("x") } } as unknown as Env;
  return env;
}

const post = (env: Env, body: unknown) =>
  apiFeedback(new Request("http://x/api/feedback", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }), env);

describe("apiFeedback", () => {
  it("persist un feedback ordonné (strength, mistake, habit, focus) et répond ok", async () => {
    const env = await makeEnv();
    const r = await post(env, { slug: "spadzze", ts: "T1", responses: {
      "mistake,1": { useful: false, tag: "trop-vague" },
      "strength,0": { useful: true },
      "focus,0": { useful: true, note: "clair" },
    }});
    expect(r.status).toBe(200);
    expect(await r.json()).toEqual({ ok: true });
    const kv = (env as any).DATA as MemoryKV;
    const lines = await readJsonl(kv, KEYS.feedback("spadzze"));
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({ ts: "T1", player: "spadzze", model: "kimi-k2.6", overall_useful: null });
    expect(lines[0].items).toEqual([
      { kind: "strength", index: 0, useful: true, tag: null, note: null },
      { kind: "mistake", index: 1, useful: false, tag: "trop-vague", note: null },
      { kind: "focus", index: 0, useful: true, tag: null, note: "clair" },
    ]);
  });

  it("ré-annoter le même ts écrase (une ligne par ts)", async () => {
    const env = await makeEnv();
    await post(env, { slug: "spadzze", ts: "T1", responses: { "strength,0": { useful: true } } });
    await post(env, { slug: "spadzze", ts: "T1", responses: { "strength,0": { useful: false, tag: "autre" } } });
    const kv = (env as any).DATA as MemoryKV;
    const lines = await readJsonl(kv, KEYS.feedback("spadzze"));
    expect(lines).toHaveLength(1);
    expect(lines[0].items[0]).toMatchObject({ useful: false, tag: "autre" });
  });

  it("404 compte inconnu / review introuvable", async () => {
    const env = await makeEnv();
    expect((await post(env, { slug: "inconnu", ts: "T1", responses: {} })).status).toBe(404);
    expect((await post(env, { slug: "spadzze", ts: "PAS_LA", responses: {} })).status).toBe(404);
  });

  it("422 : clé non 'kind,index', useful non booléen, tag inconnu, tag requis si utile=false", async () => {
    const env = await makeEnv();
    expect((await post(env, { slug: "spadzze", ts: "T1", responses: { "mistake": { useful: true } } })).status).toBe(422);
    expect((await post(env, { slug: "spadzze", ts: "T1", responses: { "mistake,0": { useful: "oui" } } })).status).toBe(422);
    expect((await post(env, { slug: "spadzze", ts: "T1", responses: { "mistake,0": { useful: false, tag: "pas-un-tag" } } })).status).toBe(422);
    expect((await post(env, { slug: "spadzze", ts: "T1", responses: { "mistake,0": { useful: false } } })).status).toBe(422);
  });
});
```

- [ ] **Step 2: Run — vérifier l'échec**

Run: `cd web/cf && npm test`
Expected: FAIL — module `../src/feedback` introuvable.

- [ ] **Step 3: Implémentation**

`web/cf/src/feedback.ts` :

```ts
import { accountFor } from "./accounts";
import { KEYS, appendJsonl, readJsonl, type KVLike } from "./readers";
import { NEG_TAGS, validateReview, type Review } from "./schema";
import type { Env } from "./index";

interface ResponseItem { useful: boolean; tag: string | null; note: string | null; }

const BAD_KEY = (k: string) =>
  Response.json({ detail: `clé de réponse invalide : '${k}' (attendu 'kind,index')` }, { status: 422 });

export async function apiFeedback(request: Request, env: Env): Promise<Response> {
  const body = await request.json().catch(() => null) as
    | { slug?: string; ts?: string; responses?: Record<string, unknown> }
    | null;
  const slug = body?.slug ?? "";
  if (!accountFor(slug)) return Response.json({ detail: "compte inconnu" }, { status: 404 });

  const reviews = await readJsonl<{ ts: string; model: string; review: unknown }>(env.DATA, KEYS.reviews(slug));
  const found = reviews.find((r) => r.ts === body?.ts);
  if (!found) return Response.json({ detail: "review introuvable" }, { status: 404 });
  const review = validateReview(found.review);
  if (!review) return Response.json({ detail: "review stockée non conforme" }, { status: 500 });

  const responses = new Map<string, ResponseItem>();
  for (const [k, v] of Object.entries(body?.responses ?? {})) {
    const m = /^([a-z]+),(\d+)$/.exec(k);
    if (!m) return BAD_KEY(k);
    const useful = (v as ResponseItem)?.useful;
    if (typeof useful !== "boolean") {
      return Response.json({ detail: `réponse invalide pour '${k}' : useful booléen requis` }, { status: 422 });
    }
    let tag: string | null = (v as ResponseItem)?.tag ?? null;
    let note: string | null = (v as ResponseItem)?.note ?? null;
    if (tag !== null && !(NEG_TAGS as readonly string[]).includes(tag)) {
      return Response.json({ detail: `réponse invalide pour '${k}' : tag inconnu` }, { status: 422 });
    }
    if (note !== null && typeof note !== "string") {
      return Response.json({ detail: `réponse invalide pour '${k}' : note doit être une chaîne` }, { status: 422 });
    }
    responses.set(`${m[1]},${Number(m[2])}`, { useful, tag, note });
  }

  // build_feedback (feedback.py:55-78) : ordre strength/mistake/habit puis focus, skips omis
  const items: { kind: string; index: number; useful: boolean; tag: string | null; note: string | null }[] = [];
  const sections: [string, unknown[]][] = [
    ["strength", review.strengths],
    ["mistake", review.mistakes],
    ["habit", review.habits],
  ];
  for (const [kind, section] of sections) {
    section.forEach((_, i) => {
      const r = responses.get(`${kind},${i}`);
      if (r) items.push({ kind, index: i, ...r });
    });
  }
  const focus = responses.get("focus,0");
  if (focus) items.push({ kind: "focus", index: 0, ...focus });
  if (items.some((it) => !it.useful && it.tag === null)) {
    return Response.json({ detail: "feedback invalide (tag requis si useful=False)" }, { status: 422 });
  }

  const fb = {
    ts: body!.ts!,
    player: slug,
    rated_at: new Date().toISOString().slice(0, 19), // ISO secondes, comme timespec="seconds"
    model: found.model,
    overall_useful: null,
    items,
  };

  // persist_feedback (feedback.py:81-99) : append/écrase — un ts au plus une fois
  const lines = await readJsonl(env.DATA, KEYS.feedback(slug));
  const kept = lines.filter((l: { ts?: string }) => l.ts !== fb.ts);
  kept.push(fb);
  await env.DATA.put(KEYS.feedback(slug), kept.map((l: object) => JSON.stringify(l)).join("\n"));

  return Response.json({ ok: true });
}
```

NB : `rated_at` tronqué à la seconde — `new Date().toISOString()` donne des millisecondes ; `.slice(0, 19)` reproduit `isoformat(timespec="seconds")`. Le check « tag requis » s'applique aux items RETENUS (comme le validator Pydantic, qui ne voit que les items construits).

Dans `web/cf/src/index.ts` : `import { apiFeedback } from "./feedback";` et, avant le 404 générique :

```ts
  if (url.pathname === "/api/feedback" && request.method === "POST") {
    return apiFeedback(request, env);
  }
```

- [ ] **Step 4: Run — tests verts**

Run: `cd web/cf && npm test`
Expected: tous passés.

- [ ] **Step 5: Commit**

```bash
git add web/cf/src/feedback.ts web/cf/src/index.ts web/cf/test/feedback.test.ts
git commit -m "feat(cf): POST /api/feedback — annotation des reviews, append/ecrase par ts sur KV"
```

---

### Task 12: Frontend — SSE coaching, fetch retiré

**Files:**
- Modify: `web/frontend/app.js`
- Modify: `web/frontend/index.html`

**Interfaces:**
- Consumes: `POST /api/coach` (SSE : events `payload` → `llm` → `review` | `error`), `GET /api/c/{slug}/reviews` (existant, `loadReviews()` déjà en place).
- Produites: aucun contrat sortant — l'app appelle les routes ci-dessus.

- [ ] **Step 1: Retirer le flow fetch de app.js**

Dans `web/frontend/app.js` : supprimer les blocs `fetchGames()`, `startPoll()`, `pollJob()` et le champ `fetchN` de l'état `accountPage` (`grep -n "fetchGames\|startPoll\|pollJob\|fetchN" web/frontend/app.js` → 0 occurrence après suppression). La méthode utilitaire `api()` et `submitFb()` restent inchangées.

- [ ] **Step 2: Remplacer genCoach() par la consommation SSE**

Remplacer la méthode `genCoach()` par :

```js
async genCoach() {
  if (!this.slug) return;
  this.job = { type: "coach", status: "running" };
  try {
    const r = await fetch("/api/coach", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slug: this.slug, scope: this.scope, outcome: this.outcome, target: this.target }),
    });
    if (!r.ok || !r.body) throw new Error(`HTTP ${r.status} sur /api/coach`);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const ev = /^event: (.+)$/m.exec(frame)?.[1];
        const raw = /^data: (.+)$/m.exec(frame)?.[1];
        if (!ev || !raw) continue;
        const d = JSON.parse(raw);
        if (ev === "payload") this.job = { type: "coach", status: "running", progress: "payload construit" };
        else if (ev === "llm") this.job = { type: "coach", status: "running", progress: "génération LLM…" };
        else if (ev === "review") { this.job = { type: "coach", status: "done" }; this.loadReviews(); }
        else if (ev === "error") this.job = { type: "coach", status: "error", error: d.error || "erreur inconnue" };
      }
    }
  } catch (e) {
    this.job = { type: "coach", status: "error", error: String(e) };
  }
}
```

- [ ] **Step 3: index.html — remplacer la barre « Mettre à jour » (lignes ~76-83) par le repère sync locale**

Remplacer le bloc `<!-- barre d'actions -->` (input `fetchN` + bouton `@click="fetchGames()"`) par :

```html
  <!-- mise a jour des games : sync locale (le fetch Riot ne passe plus par le web) -->
  <div class="action-bar row wrap">
    <span class="faint" style="font-size:13px">Mise à jour des games (sync locale) :
      <code>poetry run python3 src/collection/sync_cloudflare.py</code></span>
    <div class="spacer"></div>
  </div>
```

Et dans le bandeau job (~lignes 85-98), supprimer le `<template x-if="job.type === 'fetch'">…</template>` (mort — plus de job fetch) ; le template `job.type === 'coach'` reste (états running/done/error déjà gérés, `job.error` affiché par le bloc `status === 'error'`).

- [ ] **Step 4: Vérification manuelle bout-en-bout (wrangler dev + .dev.vars)**

`cd web/cf && npx wrangler dev` (lit `web/cf/.dev.vars` → `OLLAMA_API_KEY` de la Task 6), ouvrir `http://localhost:8787/c/spadzze` :

1. Onglet Historique : games + rang + rang ML s'affichent (KV réel de la Task 6).
2. Onglet Coaching : « Générer le coaching » → bandeau passe par les états puis « ✅ Coaching prêt », la review apparaît et se note (feedback POST → recharger la page → l'annotation est là).
3. `curl -N -X POST http://localhost:8787/api/coach -H 'Content-Type: application/json' -d '{"slug":"spadzze"}'` → les events SSE défilent (`event: payload`… `event: review`).
4. Console navigateur : zéro erreur 404 (`/api/fetch` et `/api/jobs/*` ne sont plus appelés nulle part).

Arrêter wrangler.

- [ ] **Step 5: Commit**

```bash
git add web/frontend/app.js web/frontend/index.html
git commit -m "feat(web): coaching en SSE (payload->llm->review), fetch Riot local uniquement"
```

---

### Task 13: Deploy `*.workers.dev` + secret + e2e réel

**Files:** aucun commit attendu (opérations) — committer uniquement si l'e2e révèle un correctif.

**Interfaces:**
- Consumes: tout le Worker (Tasks 2-12), le KV peuplé (Task 6).
- Produces: le Worker déployé sur `coaching-lol.<sous-domaine>.workers.dev` avec secret `OLLAMA_API_KEY` actif — précondition du Task 14 (Custom Domain s'attache à ce Worker déployé).

- [ ] **Step 1: Premier deploy**

Run: `cd web/cf && npx wrangler deploy`
Expected: upload + sortie `https://coaching-lol.<subdomain>.workers.dev` (si le compte n'a pas encore de sous-domaine workers.dev, wrangler demande de le créer — suivre le prompt). Noter l'URL.

- [ ] **Step 2: Secret de production**

Run: `cd web/cf && npx wrangler secret put OLLAMA_API_KEY` (coller la clé Ollama Cloud quand demandé)
Expected: `Success! Uploaded 1 secret`. (`.dev.vars` ne s'applique qu'à `wrangler dev` — en prod c'est le secret.)

- [ ] **Step 3: E2E sur l'URL workers.dev**

```bash
BASE=https://coaching-lol.<subdomain>.workers.dev
curl -sS $BASE/api/health
curl -sS $BASE/api/accounts
curl -sS "$BASE/api/c/spadzze/games?size=5"
curl -sS $BASE/api/c/spadzze/predicted-rank
curl -sS $BASE/ | head -5          # SPA servie
curl -sSN -X POST $BASE/api/coach -H 'Content-Type: application/json' -d '{"slug":"spadzze","scope":"adc"}'
# -> events payload / llm / review (vrai appel Ollama, ~30-90 s)
curl -sS $BASE/api/c/spadzze/reviews | head -c 400   # la review générée est persistée en KV
```

Vérifier dans le dashboard (Workers > coaching-lol > Logs) : pas d'erreur CPU limit exceeded (marge spec §11 — le coaching = attente réseau, CPU quasi nul).

- [ ] **Step 4: Basculer l'usage personnel sur l'URL workers.dev** (le domaine jean.vg reste sur Fly jusqu'au Task 14) — optionnel, informative.

---

## Phase P4 — Bascule domaine + rapatriement + décommission

### Task 14: Custom Domain `coaching-lol.jean.vg`

**Files:** aucun (dashboard). Produit : `https://coaching-lol.jean.vg` servi par le Worker Cloudflare.

- [ ] **Step 1: Supprimer le CNAME interim**

Dashboard CF → zone `jean.vg` → DNS → supprimer l'enregistrement `CNAME coaching-lol → coaching-lol.fly.dev` (sinon le Custom Domain entre en conflit « record already exists »).

- [ ] **Step 2: Attacher le Custom Domain**

Dashboard CF → Workers & Pages → `coaching-lol` → Settings → Domains & Routes → Add → Custom Domain → `coaching-lol.jean.vg` → Add. CF crée l'enregistrement (proxied) et émet le certificat (~2 min, «Initializing» → «Active»).

- [ ] **Step 3: Vérification**

```bash
curl -sS https://coaching-lol.jean.vg/api/health
curl -sS https://coaching-lol.jean.vg/api/accounts
curl -sSN -X POST https://coaching-lol.jean.vg/api/coach -H 'Content-Type: application/json' -d '{"slug":"spadzze"}'
```
Expected: idem Task 13, sur le domaine. Vérifier le certificat (`curl -vI` → émis par Cloudflare, HTTP/2, pas d'avertissement). Le site Fly devient inutilisé mais reste up (sécurité pendant le Task 15).

---

### Task 15: Rapatriement du volume Fly (reviews/feedback) + merge + re-push

**Files:**
- Create: `data/fly_rapatriement/` (gitignoré — data/)
- Run: merge one-shot (heredoc python, ci-dessous) — pas de fichier committé

**Interfaces:**
- Consumes: `sync_cloudflare.KV` (Task 5), les fichiers volume Fly `07_coaching/{slug}/reviews.jsonl` + `feedback.jsonl`, le KV `coaching:{slug}:*`.
- Produces: KV `coaching:{slug}:reviews`/`feedback` = union locale ∪ Fly ∪ KV, dédupliquée par `ts` — aucune review web générée entre la Task 6 et ici n'est perdue (spec §7.3).

- [ ] **Step 1: Tirer le répertoire coaching du volume Fly**

```bash
mkdir -p data/fly_rapatriement
cd /Users/jeanvangysel/code/website/coaching_lol
fly ssh sftp pull -a coaching-lol /app/data/07_coaching data/fly_rapatriement
```

Si `sftp pull` refuse un répertoire : fallback par fichier —

```bash
fly ssh console -a coaching-lol -C "cat /app/data/07_coaching/spadzze/reviews.jsonl" > data/fly_rapatriement/spadzze_reviews.jsonl
fly ssh console -a coaching-lol -C "cat /app/data/07_coaching/spadzze/feedback.jsonl" > data/fly_rapatriement/spadzze_feedback.jsonl
```

Vérifier : `ls -la data/fly_rapatriement/` + `wc -l` — les fichiers ne sont pas vides (si vides : aucune review web n'a été générée sur Fly, passer directement au Step 3).

- [ ] **Step 2: Merge local ∪ Fly ∪ KV par ts, puis push**

One-shot (à lancer depuis la racine du repo) — fusionne les TROIS sources pour chaque slug du volume :

```bash
poetry run python3 - <<'EOF'
import json, sys
from pathlib import Path
ROOT = Path.cwd()
for p in (ROOT / "src" / "core", ROOT / "src" / "collection", ROOT / "web" / "backend"):
    sys.path.insert(0, str(p))
import riotlib as rl
from sync_cloudflare import KV

env = rl.load_env()
kv = KV(env["CF_ACCOUNT_ID"], env["CF_NAMESPACE_ID"], env["CF_API_TOKEN"])

def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []

def merge_union(parts):
    by_ts = {}
    for part in parts:            # l'ordre définit la priorité en cas de doublon de ts
        for rec in part:
            by_ts[rec["ts"]] = rec
    return [by_ts[ts] for ts in sorted(by_ts)]

fly_dir = ROOT / "data" / "fly_rapatriement"
slugs: set[str] = set()
for pat in ("*_reviews.jsonl", "*_feedback.jsonl"):
    slugs |= {p.name.split("_")[0] for p in fly_dir.glob(pat)}
sub = fly_dir / "07_coaching"
if sub.is_dir():
    slugs |= {d.name for d in sub.iterdir() if d.is_dir()}
if not slugs:
    slugs = {"spadzze"}
for slug in slugs:
    for kind in ("reviews", "feedback"):
        local = lines(rl.DATA / "07_coaching" / slug / f"{kind}.jsonl")
        fly_file = fly_dir / f"{slug}_{kind}.jsonl"
        fly = lines(fly_file) if fly_file.exists() else lines(fly_dir / "07_coaching" / slug / f"{kind}.jsonl")
        web = []
        raw = kv.get(f"coaching:{slug}:{kind}")
        if raw:
            web = [json.loads(l) for l in raw.splitlines() if l.strip()]
        merged = merge_union([local, fly, web])
        kv.put(f"coaching:{slug}:{kind}", "\n".join(json.dumps(r, ensure_ascii=False) for r in merged))
        # archive locale de la fusion (le local reste la source historique)
        out = rl.DATA / "07_coaching" / slug / f"{kind}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n")
        print(f"{slug}/{kind}: local={len(local)} fly={len(fly)} kv={len(web)} -> fusion={len(merged)}")
EOF
```

Expected: `spadzze/reviews: local=N fly=M kv=K -> fusion=L` avec L = union dédupliquée (K inclut les reviews générées sur le domaine depuis la Task 13/14).

- [ ] **Step 3: Vérification**

```bash
curl -sS https://coaching-lol.jean.vg/api/c/spadzze/reviews | python3 -c "import json,sys; print(len(json.load(sys.stdin)))"
curl -sS https://coaching-lol.jean.vg/api/c/spadzze/feedback | python3 -c "import json,sys; print(len(json.load(sys.stdin)))"
```
Expected: les compteurs = les fusions affichées ; l'onglet Coaching du site affiche l'historique complet (y compris reviews générées via Fly + via CF).

---

### Task 16: Décommission Fly + docs + merge final

**Files:**
- Modify: `web/README.md` (remplacer la doc de déploiement Fly par la doc Cloudflare)
- Modify: `CLAUDE.md` (architecture du code : ligne `web/`, état d'avancement)
- Modify: `.env` sample mentionné dans web/README.md si pertinent

**Interfaces:**
- Consumes: tout ce qui précède (le domaine CF est pleinement opérationnel, données rapatriées).
- Produces: repo documenté à jour, branche `cloudflare-migration` fusionnée dans `master`, app Fly détruite.

- [ ] **Step 1: Vérification finale AVANT destruction**

Checklist (toute case qui échoue = NE PAS détruire) :
- [ ] `https://coaching-lol.jean.vg/api/health` → ok
- [ ] reviews + feedback complets (compteurs du Task 15)
- [ ] un coaching SSE généré depuis le domaine (Task 14/15 l'ont fait — sinon le générer maintenant)
- [ ] `poetry run python3 src/collection/sync_cloudflare.py --dry-run` liste les clés sans erreur (le run réel a été fait)

- [ ] **Step 2: Détruire l'app Fly**

Run: `fly apps destroy coaching-lol` (confirmer — irréversible ; le volume part avec).
Expected: `Destroyed app coaching-lol`. Vérifier : `curl -sSI https://coaching-lol.fly.dev` → échec DNS/connexion (peut prendre quelques minutes).

- [ ] **Step 3: Réécrire la section déploiement de web/README.md**

Remplacer les instructions Fly (deploy, sftp push du référentiel, volume) par :

```markdown
## Déploiement (Cloudflare Workers)

Le site est servi par un Worker TypeScript unique : assets statiques (SPA) + API `/api/*`,
données dans Workers KV (namespace `coaching_lol_data`, binding `DATA`). Le local Python reste
le cerveau : fetch Riot, extraction silver/gold, ML. Le Worker ne parle jamais à Riot.

- Code : `web/cf/` (wrangler) — le backend Python `web/backend/` est conservé comme référence
  du contrat API et pour `ml_rank.py`, réutilisé par le sync.
- Déployer : `cd web/cf && npx wrangler deploy` (secret : `npx wrangler secret put OLLAMA_API_KEY`).
- Mettre à jour les données : `poetry run python3 src/collection/aggregate_games.py --riot-id …`
  puis `poetry run python3 src/collection/sync_cloudflare.py` (pousse silver/gold/shap/référentiel
  + predicted-rank/LP PRÉCALCULÉS ; ~50 écritures KV, plan free très large).
- Tests : `cd web/cf && npm test` (vitest, KV émulé) ; parité payload :
  `GOLDEN_REGEN=1 poetry run pytest tests/test_payload_parity.py` puis `npm test`.
- Local : `cd web/cf && npx wrangler dev` (lit `.dev.vars` pour OLLAMA_API_KEY).
- Domaine : `coaching-lol.jean.vg` (Custom Domain sur le Worker, zone CF jean.vg).
- Layout KV : `silver:{slug}:games|rank`, `gold:{slug}:{scope}`, `ref:{rank}:{scope}`,
  `pred:{slug}`, `shap:{slug}:drivers`, `coaching:{slug}:reviews|feedback` (clés coaching
  appartenant au Worker — le sync ne les écrase pas, amorce `--seed-reviews` au premier run).
- Historique : migré de Fly.io le 2026-08-30 (volume rapatrié et fusionné, cf.
  docs/superpowers/plans/2026-08-30-cloudflare-migration.md Task 15).
```

(Conserver les sections du README qui décrivent l'app elle-même ; retirer uniquement le Fly-specific.)

- [ ] **Step 4: CLAUDE.md — architecture + état**

Dans l'arborescence `src/`→`web/` de la section Architecture : mettre à jour l'entrée `web/` pour refléter `web/cf/` (Worker TS : assets + API + KV + coach SSE), `web/frontend/` (inchangé), `web/backend/` (référence + ml_rank pour le sync). Dans État d'avancement, ajouter :

```markdown
- **Migration Cloudflare** ✅ — 2026-08-30. Le site est un Worker TS unique (web/cf/) :
  assets + API sur Workers KV, coaching en SSE, ML précalculé au sync local
  (sync_cloudflare.py, aucune clé Riot côté Worker). Domaine coaching-lol.jean.vg.
  Fly.io décommissionné. Parité payload garantie par golden test Python/TS.
```

- [ ] **Step 5: Merge final**

```bash
git add web/README.md CLAUDE.md
git commit -m "docs: migration Cloudflare terminee — Worker TS + KV, Fly decommisionne"
git checkout master
git merge --ff-only cloudflare-migration
git push origin master
git branch -d cloudflare-migration
```

---

## Self-Review (fait à la rédaction)

1. **Couverture spec** : §3 architecture → Tasks 2-4 ; §4.1 Worker TS → Tasks 2,3,4,7,8,9,10,11 ; §4.2 sync → Tasks 5,6 ; §5 KV (layout + propriété coaching) → Tasks 3 (KEYS), 5 (seed si absent), 10/11 (écriture Worker) ; §6 contrat API → Task 4 (lecture), 10 (coach SSE), 11 (feedback), 12 (frontend), fetch/jobs supprimés (testé Task 4) ; §7 domaine → Tasks 1, 14, 15, 16 ; §8 erreurs → coachFlow error events (Task 10), 404/vides (Task 4), non-conformité schéma = pas de persistance (Tasks 7, 10) ; §9 tests → golden parité (Task 8), vitest KV émulé par injection (déviation assumée vs vitest-pool-workers : plus simple/stable, l'intégration réelle est couverte par `wrangler dev`), pytest sync (Task 5), e2e (Tasks 6, 12, 13) ; §10 phases P1-P4 → mapped 1:1 ; §11 risques → CPU surveillé au deploy (Task 13 Step 3), écritures KV comptées (~50/sync, Task 6).
2. **Placeholders** : le seul contenu non littéral est la chaîne `SYSTEM` (Task 9) — copie verbatim demandée depuis `prompt.py` avec test de garde (longueur + mot-clé) : c'est une référence de source exacte, pas un « TBD ». Le `<id imprimé>` de wrangler.toml est la sortie d'une commande nommée (Task 6 Step 3).
3. **Cohérence de types** : `KVLike`/`KEYS` définis Task 3, consommés 4/10/11 ; `Env.DATA` ajouté Task 4 et lu par 10/11 ; `buildPayload` (Task 8) consommé par coachFlow (Task 10) ; `validateReview`/`reviewJsonSchema` (Task 7) consommés par 10/11 ; `generateJson`/`render` (Task 9) consommés par 10 ; `accountFor` (Task 3) consommé par 10/11 ; `sync_cloudflare.KV` (Task 5) consommé par 6/15.