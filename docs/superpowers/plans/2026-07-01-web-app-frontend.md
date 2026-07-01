# Web App V1 — Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire le frontend statique (SPA Alpine) servi par le backend FastAPI déjà mergé : accueil (grille comptes), page compte (historique + coaching + feedback inline + SHAP), README.

**Architecture:** Un seul `index.html` + `style.css` (CSS tokens sur-mesure, pas de build) + `app.js` (composants Alpine) + deux libs vendored (Alpine.js, Chart.js). Le routeur SPA lit `location.pathname` ; le catch-all FastAPI sert déjà `index.html` pour `/`, `/c/{slug}`, `/readme`. Le frontend ne fait que consommer `/api/*` (aucune clé/secret n'est jamais référencée côté navigateur).

**Tech Stack:** Alpine.js 3.x (vendored), Chart.js 4.x (vendored), CSS hand-rolled (design tokens via variables CSS), FastAPI TestClient pour les tests de câblage. Pas de Node, pas de build.

## Global Constraints

- **Palette exacte (variables CSS, dans `:root` de `style.css`)** — recopiées verbatim de la spec :
  `--bg:#0e1116` · `--panel:#16181d` · `--panel-2:#1d2026` · `--border:#2a2d34` · `--text:#e8e9ec` · `--text-dim:#9a9da4` · `--text-faint:#6b6e75` · `--gold:#c8aa6e` · `--win:#3fb950` · `--loss:#f85149` · `--notable:#d29922`.
- **Esthétique : sombre raffinée, Apple pas Razer.** Zéro néon/glow/text-shadow colorée. Bordures 1px > ombres (ombre légère `0 1px 2px rgba(0,0,0,.3)` au besoin). Coins 6px. Transitions 150ms sur couleur uniquement, pas de transform/scale. Typo Inter, graisses 400/500/600 uniquement, échelle 12/13/14/16/20/28px. Chiffres tabulaires (`font-variant-numeric: tabular-nums`) pour les stats.
- **Densité équilibrée :** rangées games compactes/scannables (façon op.gg), espace généreux autour du narratif (header, cartes coaching, SHAP, README).
- **Pas de build step.** Alpine et Chart.js sont vendored dans `web/frontend/vendor/` et servis par `/static`. Aucun `npx`, aucun Vite.
- **Sécurité :** le frontend n'utilise QUE des appels `fetch('/api/...')`. `RIOT_API_ID` et `OLLAMA_API_KEY` ne sont JAMAIS référencés dans `index.html`/`app.js`/`style.css`. Le polling de jobs passe par `GET /api/jobs/{id}` (le pipeline bloquant tourne côté serveur dans le threadpool, jamais dans le navigateur).
- **Routes SPA déjà câblées côté backend** (`web/backend/main.py`) : `GET /` → `index.html`, `GET /c/{slug}` → `index.html`, `GET /readme` → `index.html`, `/static/*` monté sur `web/frontend/`. Ne pas modifier le backend dans ce plan.
- **Shape des données (lues via `/api/*`), recopiée verbatim** — le frontend s'y conforme :
  - `GET /api/accounts` → `[{"slug","riot_id","region","games_count","last_review_ts"}]`
  - `GET /api/c/{slug}/games?page=1&size=20` → `{"items":[<silver row>...],"page","size","total"}`. **Une silver row** : `{"match_id","puuid","rank"(null),"patch","champion","role"("MIDDLE"|"JUNGLE"|"BOTTOM"|"TOP"|"UTILITY"),"win"(bool),"queue","lane":{gd10,gd14,gd20,csd10,csd14,xpd10,csm10,csm14,gpm10,gpm14,xppm10,opponent},"comp":{...},"deaths":[{minute,phase,zone,killer_role,killer_champ,gold_state,is_solo,is_ganked_by_jungle,is_2v2}],"kills":[{minute,phase,is_solo,is_2v2}],"assists":[{minute,phase,is_2v2}],"position":{...}}`. **⚠️ Pas de champ `duration` ni `timestamp`** dans une silver row → l'historique affiche `patch` à la place du timestamp, et le KDA est calculé côté front via `kills.length / deaths.length / assists.length`.
  - `GET /api/c/{slug}/reviews` → `[{"ts","model","scope","target","outcome_focus","payload":{"meta":{player,scope,target,outcome_focus,patch,n_games_me,n_games_ref,winrate_me,low_sample,deaths_per_game:{overall|win|loss:{you,ref}}},"signals":[{group,key,label,you,ref,delta,unit,notable,descriptive_only?}],"context":{...}},"review":{"strengths":[{point,evidence}],"mistakes":[{point,evidence}],"habits":[str,str],"next_focus":str,"confidence":float}}]` (plus récent en dernier car append).
  - `GET /api/c/{slug}/feedback` → `[{"ts","player","rated_at","model","overall_useful"|"items":[{kind,index,useful,tag,note}]}]`.
  - `GET /api/c/{slug}/shap` → `{"available":bool,"drivers":[{"feature","mean_shap"}]}`.
  - `POST /api/fetch` body `{"slug","n":20}` → `{"job_id"}`. `POST /api/coach` body `{"slug","scope":"adc","outcome":"loss","target":"challenger","model":null}` → `{"job_id"}`. `GET /api/jobs/{id}` → `{"id","type","slug","status"("pending"|"running"|"done"|"error"),"progress","ts_start","ts_end","error","result_ref"}`.
- **Icône champion :** `https://ddragon.leagueofgraphs.com/cdn/${patch}.1/img/champion/${champion}.png` (le champ `champion` de la silver row = Riot `championName` = id image DDragon, ex. `Cassiopeia`, `Diana`, `MonkeyKing`). `onerror` → masquer l'`<img>` et afficher le nom texte en fallback (dégradation propre, pas d'icône cassée).
- **Evidence chip coloré par `kind`** (pas par `notable` : la review ne lie pas insight→signal). Forces → accent `--win`, Erreurs → accent `--loss`, Habitudes/Focus → accent `--gold`. Documenté comme écart vs spec (« couleur selon notable »).
- **Tags de feedback (menu sur ✗) = `NEG_TAGS`** verbatim : `asymetrie`, `stat-inventee`, `profondeur-en-faute`, `trop-vague`, `non-actionnable`, `autre`. Clé de réponse POST = `"kind,index"` avec `kind ∈ {strength,mistake,habit,focus}` et `index` = position dans la section (`focus` → 0).
- **Tests :** `tests/web/test_frontend.py` (TestClient FastAPI) = tests de **câblage/contrat** : assets vendored servis (200 + bon content-type), `index.html` référence les assets et contient les hooks Alpine clés, `app.js` contient les composants/appels `/api` attendus. L'interactivité Alpine n'est PAS testée unitairement (pas de Playwright dans le projet) — chaque tâche inclut une **vérification manuelle** (`uvicorn` + clic-through) en plus du test. Les assertions de câblage sont volontairement structurelles mais réelles (catchent un asset cassé, un hook retiré, un appel `/api` supprimé).
- **Lancer le serveur en local** : `.venv/bin/python -m uvicorn main:app --app-dir web/backend --reload` puis http://127.0.0.1:8000.
- **Lancer les tests web** : `.venv/bin/python -m pytest tests/web/ -v`.

---

## File Structure

```
web/frontend/
  index.html          # shell SPA : top-bar + <main> avec <template x-if> par page
  style.css           # tokens (:root) + base typo + composants nommés (.card/.row/.chip/.btn/...)
  app.js              # composants Alpine : app() routeur, homePage(), accountPage(slug), readmePage() + helpers (api/pollJob/fmtKDA/champIcon)
  vendor/
    alpine.min.js     # Alpine 3.x vendored (curl)
    chart.umd.min.js  # Chart.js 4.x vendored (curl)
tests/web/
  test_frontend.py    # tests de câblage (TestClient), grandit tâche par tâche
```

Responsabilités :
- `index.html` : structure statique + hooks Alpine (`x-data`, `x-init`, `x-if`, `x-for`, `x-text`, `@click`). Aucune logique imperative.
- `style.css` : un seul fichier, tokens en haut, composants nommés réutilisables. Pas de classes utilitaires.
- `app.js` : fonctions constructeurs de composants Alpine (retournent un objet d'état + méthodes `init()`). Un helper `api()` wrap `fetch` avec gestion d'erreur. `pollJob()` gère le setInterval.
- `test_frontend.py` : un test par préoccupation de câblage, étendu à chaque tâche.

---

### Task F1: Vendoring + shell SPA + design tokens

**Files:**
- Create: `web/frontend/vendor/alpine.min.js`
- Create: `web/frontend/vendor/chart.umd.min.js`
- Modify (replace entirely): `web/frontend/index.html`
- Modify (replace entirely): `web/frontend/style.css`
- Modify (replace entirely): `web/frontend/app.js`
- Create: `tests/web/test_frontend.py`

**Interfaces:**
- Consumes: backend routes `GET /`, `GET /c/{slug}`, `GET /readme`, `GET /static/*` (déjà en place dans `web/backend/main.py`).
- Produces: `app()` (routeur Alpine global, exposé sur `<body x-data="app()">`), helpers `api(path)` et `pollJob(jobId, onUpdate, onDone)` dans `app.js`. Les tâches suivantes ajoutent `homePage()`, `accountPage(slug)`, `readmePage()` au même `app.js` et des `<template x-if>` au même `index.html`.

- [ ] **Step 1: Vendor Alpine.js et Chart.js**

```bash
mkdir -p web/frontend/vendor
curl -fsSL -o web/frontend/vendor/alpine.min.js https://cdn.jsdelivr.net/npm/alpinejs@3.13.5/dist/cdn.min.js
curl -fsSL -o web/frontend/vendor/chart.umd.min.js https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js
```
Vérifier : `wc -c web/frontend/vendor/alpine.min.js web/frontend/vendor/chart.umd.min.js` — chaque fichier > 30 000 octets. Si un curl 404, retenter avec le tag `@3` (Alpine) ou `@4` (Chart.js) pour résoudre le latest : `https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js` et `https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js`.

- [ ] **Step 2: Écrire le test de câblage (échec d'abord)**

Créer `tests/web/test_frontend.py` :

```python
"""Tests de câblage du frontend statique (servi par FastAPI).

Pas de test d'interactivité Alpine (pas de Playwright) — on verrouille le câblage :
assets vendored servis, index.html référence les assets et porte les hooks clés,
app.js définit le routeur et les helpers. L'interactivité est vérifiée manuellement.
"""
from fastapi.testclient import TestClient

import main as main_mod


def _client():
    return TestClient(main_mod.app)


def test_index_served_at_root():
    r = _client().get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert 'x-data="app()"' in body
    assert '/static/vendor/alpine.min.js' in body
    assert '/static/vendor/chart.umd.min.js' in body
    assert '/static/style.css' in body
    assert '/static/app.js' in body


def test_spa_catch_all_serves_index():
    c = _client()
    for path in ("/c/spadzze", "/readme"):
        r = c.get(path)
        assert r.status_code == 200
        assert 'x-data="app()"' in r.text


def test_assets_served():
    c = _client()
    for path, ct in [
        ("/static/style.css", "text/css"),
        ("/static/app.js", "javascript"),
        ("/static/vendor/alpine.min.js", "javascript"),
        ("/static/vendor/chart.umd.min.js", "javascript"),
    ]:
        r = c.get(path)
        assert r.status_code == 200, path
        assert ct in r.headers["content-type"], path
        assert len(r.content) > 1000, path


def test_style_css_has_tokens():
    css = _client().get("/static/style.css").text
    for token in ("--bg:#0e1116", "--panel:#16181d", "--gold:#c8aa6e",
                  "--win:#3fb950", "--loss:#f85149", "tabular-nums"):
        assert token in css, token


def test_app_js_has_router_and_helpers():
    js = _client().get("/static/app.js").text
    assert "function app()" in js
    assert "function api(" in js
    assert "function pollJob(" in js
    assert "location.pathname" in js
```

- [ ] **Step 3: Lancer le test — il doit échouer**

Run: `.venv/bin/python -m pytest tests/web/test_frontend.py -v`
Expected: FAIL (les assets n'existent pas encore / `app()` non définie / index.html actuel ne porte pas `x-data="app()"`).

- [ ] **Step 4: Écrire `web/frontend/style.css` (tokens + base + primitives)**

Remplacer entièrement `web/frontend/style.css` par :

```css
:root {
  --bg:#0e1116; --panel:#16181d; --panel-2:#1d2026; --border:#2a2d34;
  --text:#e8e9ec; --text-dim:#9a9da4; --text-faint:#6b6e75;
  --gold:#c8aa6e; --win:#3fb950; --loss:#f85149; --notable:#d29922;
  --radius:6px;
  --maxw:1040px;
}

* { box-sizing:border-box; }

html, body { margin:0; padding:0; }

body {
  background:var(--bg); color:var(--text);
  font-family:Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  font-size:14px; line-height:1.55;
  font-variant-numeric:tabular-nums;
  -webkit-font-smoothing:antialiased;
}

a { color:var(--gold); text-decoration:none; }
a:hover { text-decoration:underline; }

h1,h2,h3 { font-weight:600; letter-spacing:-0.01em; margin:0; }
h1 { font-size:20px; }
h2 { font-size:16px; }
h3 { font-size:14px; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.06em; }

.muted { color:var(--text-dim); }
.faint { color:var(--text-faint); }
.num { font-variant-numeric:tabular-nums; }

/* --- top-bar --- */
.topbar {
  position:sticky; top:0; z-index:20;
  background:var(--bg);
  border-bottom:1px solid var(--border);
}
.topbar-inner {
  max-width:var(--maxw); margin:0 auto; padding:0 20px;
  height:52px; display:flex; align-items:center; gap:20px;
}
.topbar .brand { font-size:15px; font-weight:600; color:var(--gold); letter-spacing:0.02em; }
.topbar .brand:hover { text-decoration:none; }
.topbar .spacer { flex:1; }
.topbar a.nav { color:var(--text-dim); font-size:13px; }
.topbar a.nav:hover { color:var(--text); text-decoration:none; }

/* account switcher (Alpine dropdown) */
.switcher { position:relative; }
.switcher-btn {
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  color:var(--text); padding:6px 10px; font-size:13px; cursor:pointer; display:flex; gap:6px; align-items:center;
}
.switcher-btn:hover { border-color:var(--gold); }
.switcher-menu {
  position:absolute; top:calc(100% + 4px); left:0; min-width:180px;
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  padding:4px; box-shadow:0 4px 16px rgba(0,0,0,.4);
}
.switcher-menu a {
  display:block; padding:7px 10px; border-radius:4px; color:var(--text); font-size:13px;
}
.switcher-menu a:hover { background:var(--panel-2); text-decoration:none; }

/* --- layout --- */
.container { max-width:var(--maxw); margin:0 auto; padding:24px 20px 64px; }

.card {
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  padding:18px 20px;
}
.card + .card { margin-top:14px; }

.btn {
  background:var(--panel-2); border:1px solid var(--border); border-radius:var(--radius);
  color:var(--text); padding:7px 14px; font-size:13px; cursor:pointer; font-family:inherit;
  transition:border-color 150ms, color 150ms;
}
.btn:hover { border-color:var(--gold); color:var(--gold); }
.btn:disabled { opacity:.5; cursor:default; }
.btn-primary { background:var(--gold); color:#1a1208; border-color:var(--gold); font-weight:500; }
.btn-primary:hover { color:#1a1208; opacity:.9; }

.input, .select {
  background:var(--bg); border:1px solid var(--border); border-radius:var(--radius);
  color:var(--text); padding:6px 10px; font-size:13px; font-family:inherit;
}
.input:focus, .select:focus { outline:none; border-color:var(--gold); }
.input.narrow { width:72px; }

.row { display:flex; align-items:center; gap:10px; }
.row.wrap { flex-wrap:wrap; }
.spacer { flex:1; }

.badge { font-size:11px; padding:2px 7px; border-radius:4px; font-weight:500; }
.badge-win { background:rgba(63,185,80,.14); color:var(--win); }
.badge-loss { background:rgba(248,81,73,.14); color:var(--loss); }

/* tabs */
.tabs { display:flex; gap:0; border-bottom:1px solid var(--border); margin-bottom:18px; }
.tab {
  padding:10px 16px; font-size:13px; color:var(--text-dim); cursor:pointer;
  border-bottom:2px solid transparent; margin-bottom:-1px;
}
.tab:hover { color:var(--text); }
.tab.active { color:var(--gold); border-bottom-color:var(--gold); }

/* empty/error/loading states */
.state { padding:32px 20px; text-align:center; color:var(--text-dim); font-size:13px; }
.state.err { color:var(--loss); }

/* utilities cachées par Alpine s'affichent sans flash */
[x-cloak] { display:none !important; }
```

- [ ] **Step 5: Écrire `web/frontend/app.js` (routeur + helpers, pages en placeholder)**

Remplacer entièrement `web/frontend/app.js` par :

```javascript
// coaching_lol — frontend SPA (Alpine). Aucune clé/secret ici : tout passe par /api/*.

const NEG_TAGS = ["asymetrie", "stat-inventee", "profondeur-en-faute",
  "trop-vague", "non-actionnable", "autre"];
const DDRAGON = (patch, champ) =>
  `https://ddragon.leagueofgraphs.com/cdn/${patch}.1/img/champion/${champ}.png`;

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} on ${path}`);
  return r.json();
}

function pollJob(jobId, onUpdate, onDone) {
  const id = setInterval(async () => {
    try {
      const j = await api(`/api/jobs/${jobId}`);
      onUpdate(j);
      if (j.status === "done" || j.status === "error") {
        clearInterval(id);
        onDone(j);
      }
    } catch (e) {
      clearInterval(id);
      onDone({ status: "error", error: String(e) });
    }
  }, 1500);
  return id;
}

function fmtKDA(k, d, a) {
  const ka = k + a;
  const ratio = d === 0 ? "Perfect" : (ka / d).toFixed(2);
  return `${k}/${d}/${a} · ${ratio}`;
}

function routeOf(path) {
  if (path === "/" || path === "") return { name: "home" };
  const m = path.match(/^\/c\/([^/]+)$/);
  if (m) return { name: "account", slug: decodeURIComponent(m[1]) };
  if (path === "/readme") return { name: "readme" };
  return { name: "home" };
}

function app() {
  return {
    path: location.pathname,
    accounts: [],
    accountsLoading: true,
    get route() { return routeOf(this.path); },

    init() {
      window.addEventListener("popstate", () => { this.path = location.pathname; });
      api("/api/accounts").then(a => { this.accounts = a; this.accountsLoading = false; })
        .catch(() => { this.accountsLoading = false; });
    },

    go(p) {
      if (p === this.path) return;
      history.pushState({}, "", p);
      this.path = p;
    },

    switcherOpen: false,
    toggleSwitcher() { this.switcherOpen = !this.switcherOpen; },
    closeSwitcher() { this.switcherOpen = false; },
  };
}

// Placeholder components — remplis par les tâches suivantes.
function homePage()    { return { init() {} }; }
function accountPage() { return { init() {} }; }
function readmePage()  { return { init() {} }; }
```

- [ ] **Step 6: Écrire `web/frontend/index.html` (shell SPA)**

Remplacer entièrement `web/frontend/index.html` par :

```html
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>coaching_lol</title>
  <link rel="stylesheet" href="/static/style.css">
  <script defer src="/static/vendor/alpine.min.js"></script>
</head>
<body x-data="app()" x-init="init()" x-cloak>
  <nav class="topbar">
    <div class="topbar-inner">
      <a class="brand" href="/" @click.prevent="go('/')">coaching_lol</a>
      <div class="switcher" @click.outside="closeSwitcher()">
        <button class="switcher-btn" @click="toggleSwitcher()">
          <span x-text="route.name === 'account' ? route.slug : 'comptes'"></span>
          <span class="faint">▾</span>
        </button>
        <div class="switcher-menu" x-show="switcherOpen" x-transition>
          <template x-for="a in accounts" :key="a.slug">
            <a :href="'/c/' + a.slug" @click.prevent="go('/c/' + a.slug); closeSwitcher()"
               x-text="a.slug"></a>
          </template>
          <a href="/" @click.prevent="go('/'); closeSwitcher()" class="faint">— accueil</a>
        </div>
      </div>
      <div class="spacer"></div>
      <a class="nav" href="/readme" @click.prevent="go('/readme')">README</a>
    </div>
  </nav>

  <main class="container">
    <template x-if="route.name === 'home'">
      <div x-data="homePage()" x-init="init()"></div>
    </template>
    <template x-if="route.name === 'account'">
      <div x-data="accountPage(route.slug)" x-init="init()"></div>
    </template>
    <template x-if="route.name === 'readme'">
      <div x-data="readmePage()" x-init="init()"></div>
    </template>
  </main>

  <link rel="stylesheet" href="/static/style.css">
  <script src="/static/app.js"></script>
  <script src="/static/vendor/chart.umd.min.js"></script>
</body>
</html>
```

⚠️ **Correction à appliquer** : le `<link rel="stylesheet" href="/static/style.css">` est référencé deux fois (une dans `<head>`, une avant `app.js` en bas). C'est volontaire pour satisfaire le test `test_index_served_at_root` qui cherche la chaîne `/static/style.css` ET pour garantir le chargement — **mais la duplication est inutile**. Gardez **une seule** référence dans `<head>` (supprimez la ligne `<link>` en bas du `<body>`). Le test passe toujours car la référence `<head>` suffit. Le bloc scripts en bas du body reste : `app.js` + `chart.umd.min.js` (chargés après le DOM, Alpine est en `defer` dans le head).

`index.html` final (appliquez la correction) :

```html
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>coaching_lol</title>
  <link rel="stylesheet" href="/static/style.css">
  <script defer src="/static/vendor/alpine.min.js"></script>
</head>
<body x-data="app()" x-init="init()" x-cloak>
  <nav class="topbar">
    <div class="topbar-inner">
      <a class="brand" href="/" @click.prevent="go('/')">coaching_lol</a>
      <div class="switcher" @click.outside="closeSwitcher()">
        <button class="switcher-btn" @click="toggleSwitcher()">
          <span x-text="route.name === 'account' ? route.slug : 'comptes'"></span>
          <span class="faint">▾</span>
        </button>
        <div class="switcher-menu" x-show="switcherOpen" x-transition>
          <template x-for="a in accounts" :key="a.slug">
            <a :href="'/c/' + a.slug" @click.prevent="go('/c/' + a.slug); closeSwitcher()"
               x-text="a.slug"></a>
          </template>
          <a href="/" @click.prevent="go('/'); closeSwitcher()" class="faint">— accueil</a>
        </div>
      </div>
      <div class="spacer"></div>
      <a class="nav" href="/readme" @click.prevent="go('/readme')">README</a>
    </div>
  </nav>

  <main class="container">
    <template x-if="route.name === 'home'">
      <div x-data="homePage()" x-init="init()"></div>
    </template>
    <template x-if="route.name === 'account'">
      <div x-data="accountPage(route.slug)" x-init="init()"></div>
    </template>
    <template x-if="route.name === 'readme'">
      <div x-data="readmePage()" x-init="init()"></div>
    </template>
  </main>

  <script src="/static/app.js"></script>
  <script src="/static/vendor/chart.umd.min.js"></script>
</body>
</html>
```

Note : `app.js` doit être chargé **avant** qu'Alpine (en `defer`) n'initialise `x-data="app()"`. `defer` retarde Alpine jusqu'après le parsing du document mais les scripts en fin de body s'exécutent avant les `defer` ? Non — les scripts `defer` s'exécutent après les scripts normaux en fin de body, dans l'ordre. Donc `app.js` (normal, fin de body) s'exécute **avant** Alpine (`defer`), et `app()` sera définie quand Alpine s'initialise. ✅

- [ ] **Step 7: Lancer le test — il doit passer**

Run: `.venv/bin/python -m pytest tests/web/test_frontend.py -v`
Expected: 5 passed.

- [ ] **Step 8: Vérification manuelle**

```bash
.venv/bin/python -m uvicorn main:app --app-dir web/backend --port 8011
```
Ouvrir http://127.0.0.1:8011 — la top-bar s'affiche (brand, switcher « comptes », README), le `<main>` est vide (placeholders). Le switcher s'ouvre et liste les comptes de `/api/accounts`. Tuer le serveur (Ctrl-C).

- [ ] **Step 9: Commit**

```bash
git add web/frontend tests/web/test_frontend.py
git commit -m "feat(web): shell SPA + design tokens + vendoring Alpine/Chart.js"
```

---

### Task F2: Page accueil — grille de comptes

**Files:**
- Modify: `web/frontend/app.js` (remplacer `homePage()`)
- Modify: `web/frontend/index.html` (remplir le `<template x-if="home">`)
- Modify: `tests/web/test_frontend.py` (ajouter un test)

**Interfaces:**
- Consumes: `GET /api/accounts` → `[{slug,riot_id,region,games_count,last_review_ts}]` (déjà chargé dans `app().accounts`).
- Produces: `homePage()` avec `init()` qui utilise `this.$root` (le store `app()` parent est accessible via le composant englobant — mais Alpine n'expose pas le parent directement ; on passe les comptes via le DOM en lisant le scope parent). **Décision :** `homePage()` ne refetch pas — il reçoit les comptes du parent `app()`. Comme `<template x-if>` crée un nouveau scope qui hérite du scope parent en Alpine 3, `accounts` est accessible directement dans le template home. Donc `homePage()` n'a pas de `init()` utile ; on consomme `accounts` du scope parent dans le HTML.

- [ ] **Step 1: Ajouter le test (échec d'abord)**

Ajouter à `tests/web/test_frontend.py` :

```python
def test_home_page_wired():
    js = _client().get("/static/app.js").text
    assert "function homePage()" in js
    assert "/api/accounts" in js
```

- [ ] **Step 2: Lancer — échec**

Run: `.venv/bin/python -m pytest tests/web/test_frontend.py::test_home_page_wired -v`
Expected: FAIL (`homePage` est un placeholder sans logique, mais le test ne cherche que la présence — en fait `/api/accounts` est déjà dans `app()`. Le test passe déjà. **Ajustement :** rendre le test discriminant en vérifiant que `index.html` contient le bloc home avec `x-for` sur les comptes).

Remplacer le test par :

```python
def test_home_page_wired():
    body = _client().get("/").text
    js = _client().get("/static/app.js").text
    assert "function homePage()" in js
    # le template home itère sur les comptes du scope parent
    assert 'x-for="a in accounts"' in body
    assert "/api/accounts" in js
```

Run: `.venv/bin/python -m pytest tests/web/test_frontend.py::test_home_page_wired -v`
Expected: FAIL (`x-for="a in accounts"` absent du body — le template home est vide).

- [ ] **Step 3: Remplacer `homePage()` dans `app.js`**

Remplacer la ligne `function homePage()    { return { init() {} }; }` par :

```javascript
function homePage() {
  return {
    init() {
      // Les comptes sont chargés par le store app() parent (GET /api/accounts).
      // Pas de fetch ici — on consomme `accounts` via le scope hérité dans le HTML.
    },
  };
}
```

- [ ] **Step 4: Remplir le template home dans `index.html`**

Remplacer `<div x-data="homePage()" x-init="init()"></div>` (dans le template `home`) par :

```html
<div x-data="homePage()" x-init="init()">
  <h1 style="font-size:28px;margin-bottom:4px">Comptes de coaching</h1>
  <p class="muted" style="margin:0 0 22px">Sélectionnez un compte pour voir son historique, son coaching et son profil ML.</p>

  <template x-if="accountsLoading">
    <div class="state">Chargement des comptes…</div>
  </template>

  <template x-if="!accountsLoading && accounts.length === 0">
    <div class="state">Aucun compte configuré.</div>
  </template>

  <template x-if="!accountsLoading && accounts.length > 0">
    <div class="accounts-grid">
      <template x-for="a in accounts" :key="a.slug">
        <a class="account-card" :href="'/c/' + a.slug"
           @click.prevent="go('/c/' + a.slug)">
          <div class="ac-slug" x-text="a.slug"></div>
          <div class="ac-riot faint" x-text="a.riot_id"></div>
          <div class="ac-stats row">
            <span class="num"><span x-text="a.games_count"></span> games</span>
            <span class="faint" x-text="a.last_review_ts ? ('· review ' + a.last_review_ts.slice(0,10)) : '· pas de review'"></span>
          </div>
        </a>
      </template>
    </div>
  </template>
</div>
```

- [ ] **Step 5: Ajouter le CSS de la grille à `style.css`**

Ajouter à la fin de `web/frontend/style.css` :

```css
/* --- home: accounts grid --- */
.accounts-grid {
  display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:14px;
}
.account-card {
  display:block; background:var(--panel); border:1px solid var(--border);
  border-radius:var(--radius); padding:16px 18px; color:var(--text);
  transition:border-color 150ms;
}
.account-card:hover { border-color:var(--gold); text-decoration:none; }
.ac-slug { font-size:16px; font-weight:600; color:var(--text); }
.ac-riot { font-size:12px; margin:2px 0 10px; }
.ac-stats { font-size:13px; color:var(--text-dim); }
```

- [ ] **Step 6: Lancer les tests — passent**

Run: `.venv/bin/python -m pytest tests/web/test_frontend.py -v`
Expected: 6 passed.

- [ ] **Step 7: Vérification manuelle**

Boot uvicorn (port 8011), ouvrir `/` : grille de cartes comptes (slug, riot_id, games count, dernière review). Cliquer une carte → URL devient `/c/{slug}` (page vide pour l'instant, placeholder `accountPage`).

- [ ] **Step 8: Commit**

```bash
git add web/frontend/index.html web/frontend/style.css web/frontend/app.js tests/web/test_frontend.py
git commit -m "feat(web): page accueil — grille de comptes"
```

---

### Task F3: Page compte — header + fetch job + onglet Historique

**Files:**
- Modify: `web/frontend/app.js` (remplacer `accountPage()`)
- Modify: `web/frontend/index.html` (remplir le template `account`)
- Modify: `web/frontend/style.css` (composants game-row/job-banner/action-bar)
- Modify: `tests/web/test_frontend.py`

**Interfaces:**
- Consumes: `GET /api/c/{slug}/games?page=&size=` → `{items,page,size,total}` · `POST /api/fetch {slug,n}` → `{job_id}` · `GET /api/jobs/{id}` · `app().go(p)`.
- Produces: `accountPage(slug)` avec état `games, page, total, tab('history'), job{}` et méthodes `loadGames(), fetchGames(), startPoll(jobId), setTab(t)`. Les tâches F4/F5 ajoutent les onglets `coaching` et `shap` au même composant.

⚠️ **KDA calculé depuis les tableaux** : `k = g.kills.length, d = g.deaths.length, a = g.assists.length`. **Pas de duration/timestamp** dans la row → on affiche `g.patch`.

- [ ] **Step 1: Ajouter le test (échec d'abord)**

Ajouter à `tests/web/test_frontend.py` :

```python
def test_account_page_history_wired():
    body = _client().get("/c/spadzze").text
    js = _client().get("/static/app.js").text
    assert "function accountPage(" in js
    assert "/api/c/" in js and "/games" in js
    assert "/api/fetch" in js
    assert "/api/jobs/" in js
    assert 'class="game-row"' in body or "game-row" in body
    assert "job-banner" in body
```

Run: `.venv/bin/python -m pytest tests/web/test_frontend.py::test_account_page_history_wired -v`
Expected: FAIL (`accountPage` placeholder, pas de `game-row`/`job-banner`).

- [ ] **Step 2: Remplacer `accountPage()` dans `app.js`**

Remplacer `function accountPage() { return { init() {} }; }` par :

```javascript
function accountPage(slug) {
  return {
    slug,
    tab: "history",
    games: [], page: 1, size: 20, total: 0,
    gamesLoading: true, gamesError: null,
    fetchN: 20,
    job: null, // {type, status, progress, error}

    init() { this.loadGames(); },

    async loadGames() {
      this.gamesLoading = true; this.gamesError = null;
      try {
        const d = await api(`/api/c/${this.slug}/games?page=${this.page}&size=${this.size}`);
        this.games = d.items; this.total = d.total;
      } catch (e) { this.gamesError = String(e); }
      finally { this.gamesLoading = false; }
    },

    async prevPage() { if (this.page > 1) { this.page--; this.loadGames(); } },
    async nextPage() {
      if (this.page * this.size < this.total) { this.page++; this.loadGames(); }
    },

    async fetchGames() {
      this.job = { type: "fetch", status: "running", progress: "0/" + this.fetchN };
      try {
        const { job_id } = await api("/api/fetch", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slug: this.slug, n: Number(this.fetchN) }),
        });
        this.startPoll(job_id);
      } catch (e) { this.job = { type: "fetch", status: "error", error: String(e) }; }
    },

    startPoll(jobId) {
      pollJob(jobId,
        j => { this.job = j; },
        j => {
          this.job = j;
          if (j.status === "done" && j.type === "fetch") this.loadGames();
        },
      );
    },

    setTab(t) { this.tab = t; },

    kda(g) { return fmtKDA(g.kills.length, g.deaths.length, g.assists.length); },
    champIcon(g) { return DDRAGON(g.patch, g.champion); },
    iconFallback(e, champ) { e.target.style.display = "none"; e.target.nextElementSibling.style.display = ""; },
    roleLabel(r) {
      return ({ MIDDLE: "Mid", JUNGLE: "Jungle", BOTTOM: "Bot", TOP: "Top", UTILITY: "Support" })[r] || r;
    },
  };
}
```

- [ ] **Step 3: Remplir le template account dans `index.html`**

Remplacer `<div x-data="accountPage(route.slug)" x-init="init()"></div>` par :

```html
<div x-data="accountPage(route.slug)" x-init="init()">
  <!-- header de profil -->
  <div class="prof-header">
    <h1 style="font-size:28px" x-text="slug"></h1>
    <div class="row faint" style="margin-top:4px;font-size:13px">
      <span class="num"><span x-text="total"></span> games en cache</span>
    </div>
  </div>

  <!-- barre d'actions -->
  <div class="action-bar row wrap">
    <span class="muted" style="font-size:13px">Mettre à jour :</span>
    <input class="input narrow num" type="number" min="1" max="100" x-model.number="fetchN">
    <button class="btn btn-primary" @click="fetchGames()"
            :disabled="job && job.status === 'running'">Fetcher les games</button>
    <div class="spacer"></div>
  </div>

  <!-- bandeau job (sticky sous le header) -->
  <template x-if="job">
    <div class="job-banner" :class="job.status === 'error' ? 'err' : ''">
      <template x-if="job.type === 'fetch'">
        <span><span x-text="job.status === 'running' ? 'Pull Riot… ' + (job.progress || '') : (job.status === 'done' ? '✅ Games mises à jour' : '⛔ Erreur')"></span></span>
      </template>
      <template x-if="job.type === 'coach'">
        <span><span x-text="job.status === 'running' ? 'Coaching en cours…' : (job.status === 'done' ? '✅ Coaching prêt' : '⛔ Erreur')"></span></span>
      </template>
      <template x-if="job.status === 'error'">
        <span class="faint" x-text="job.error || ''"></span>
      </template>
    </div>
  </template>

  <!-- onglets -->
  <div class="tabs">
    <div class="tab" :class="tab === 'history' ? 'active' : ''" @click="setTab('history')">Historique</div>
    <div class="tab" :class="tab === 'coaching' ? 'active' : ''" @click="setTab('coaching')">Coaching</div>
    <div class="tab" :class="tab === 'shap' ? 'active' : ''" @click="setTab('shap')">Profil ML</div>
  </div>

  <!-- onglet Historique -->
  <template x-if="tab === 'history'">
    <div>
      <template x-if="gamesLoading"><div class="state">Chargement…</div></template>
      <template x-if="gamesError"><div class="state err" x-text="gamesError"></div></template>
      <template x-if="!gamesLoading && !gamesError && games.length === 0">
        <div class="state">Aucune game en cache. Lance un fetch.</div>
      </template>
      <template x-if="!gamesLoading && games.length > 0">
        <div>
          <template x-for="g in games" :key="g.match_id">
            <div class="game-row" :class="g.win ? 'w' : 'l'">
              <img class="champ-icon" :src="champIcon(g)" :alt="g.champion"
                   @error="iconFallback($event, g.champion)" loading="lazy">
              <span class="champ-fallback" style="display:none" x-text="g.champion"></span>
              <div class="gr-main">
                <div class="row" style="gap:8px">
                  <span class="badge" :class="g.win ? 'badge-win' : 'badge-loss'"
                        x-text="g.win ? 'W' : 'L'"></span>
                  <span class="gr-champ" x-text="g.champion"></span>
                  <span class="faint" x-text="roleLabel(g.role)"></span>
                </div>
              </div>
              <div class="gr-kda num" x-text="kda(g)"></div>
              <div class="gr-patch faint" x-text="g.patch"></div>
            </div>
          </template>
          <div class="row" style="justify-content:space-between;margin-top:16px">
            <button class="btn" @click="prevPage()" :disabled="page <= 1">← Précédent</button>
            <span class="faint" x-text="'page ' + page + ' / ' + Math.max(1, Math.ceil(total / size))"></span>
            <button class="btn" @click="nextPage()" :disabled="page * size >= total">Suivant →</button>
          </div>
        </div>
      </template>
    </div>
  </template>

  <!-- onglets coaching + shap : placeholders (tâches F4/F5) -->
  <template x-if="tab === 'coaching'"><div class="state">Coaching — à venir.</div></template>
  <template x-if="tab === 'shap'"><div class="state">Profil ML — à venir.</div></template>
</div>
```

- [ ] **Step 4: Ajouter le CSS à `style.css`**

Ajouter à la fin de `web/frontend/style.css` :

```css
/* --- account: header + actions + job --- */
.prof-header { margin-bottom:18px; }
.action-bar {
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  padding:12px 16px; margin-bottom:14px;
}
.job-banner {
  position:sticky; top:52px; z-index:10;
  background:var(--panel-2); border:1px solid var(--border); border-left:3px solid var(--gold);
  border-radius:var(--radius); padding:9px 14px; margin-bottom:16px; font-size:13px;
}
.job-banner.err { border-left-color:var(--loss); color:var(--loss); }

/* --- account: game rows (dense, scannable, op.gg-like) --- */
.game-row {
  display:grid; grid-template-columns:32px 1fr 150px 56px; gap:12px; align-items:center;
  padding:8px 12px; border:1px solid var(--border); border-left:3px solid var(--border);
  border-radius:var(--radius); background:var(--panel);
}
.game-row + .game-row { margin-top:6px; }
.game-row.w { border-left-color:var(--win); }
.game-row.l { border-left-color:var(--loss); }
.champ-icon { width:32px; height:32px; border-radius:4px; display:block; background:var(--bg); }
.champ-fallback { font-size:11px; color:var(--text-dim); width:32px; text-align:center; }
.gr-champ { font-weight:500; }
.gr-kda { font-size:13px; color:var(--text-dim); text-align:right; }
.gr-patch { font-size:12px; text-align:right; }
```

- [ ] **Step 5: Lancer les tests — passent**

Run: `.venv/bin/python -m pytest tests/web/test_frontend.py -v`
Expected: 7 passed.

- [ ] **Step 6: Vérification manuelle**

Boot uvicorn (8011), `/c/spadzze` : header (slug + games count), barre d'action (input N=20 + bouton fetcher), onglets (Historique actif), liste de games denses (icône champion, badge W/L vert/rouge, champion, rôle, KDA, patch). Pagination marche. **Ne pas lancer un vrai fetch** (consomme l'API Riot) — vérifier juste le rendu. Cliquer « Fetcher » déclencherait un vrai pull : à éviter en manual check ; on vérifie que le bouton est clicable et désactivé pendant un job simulé seulement si on a un job. Pour vérifier le bandeau job sans pull Riot : skip (le bandeau est vérifié visuellement via le template `x-if="job"`).

- [ ] **Step 7: Commit**

```bash
git add web/frontend/index.html web/frontend/style.css web/frontend/app.js tests/web/test_frontend.py
git commit -m "feat(web): page compte — header, fetch job (poll), onglet historique"
```

---

### Task F4: Page compte — onglet Coaching + feedback inline

**Files:**
- Modify: `web/frontend/app.js` (étendre `accountPage()` : état coaching + feedback + méthodes)
- Modify: `web/frontend/index.html` (remplacer le placeholder coaching)
- Modify: `web/frontend/style.css` (insight-card, evidence-chip, fb-*)
- Modify: `tests/web/test_frontend.py`

**Interfaces:**
- Consumes: `GET /api/c/{slug}/reviews` → liste (plus récent en dernier) · `POST /api/coach {slug,scope,outcome,target,model?}` → `{job_id}` · `GET /api/jobs/{id}` · `GET /api/c/{slug}/feedback` · `POST /api/feedback {slug,ts,responses}`.
- Produces: étends `accountPage` avec `scope,outcome,target,review,reviews,fbMap,fbBusy` et méthodes `loadReviews(), genCoach(), startPoll()` (déjà définie en F3 — réutiliser), `loadFeedback(), submitFb(kind,index), setFb(kind,index,useful)`. Le polling de coach réutilise `startPoll(jobId)` (F3) ; au `done` de type `coach`, appeler `loadReviews()`.

**Décisions de mapping :**
- `review = reviews[reviews.length - 1]` (dernière). `payload.meta` affiché en bandeau (n_games_me, winrate_me, low_sample). `review.review.{strengths[3],mistakes[3],habits[2],next_focus,confidence}`.
- Feedback inline : pour chaque insight, 3 boutons ✓ (useful=true) / ✗ (useful=false, ouvre menu tag) / skip (non envoyé). Clé réponse = `"kind,index"` où kind ∈ `{strength,mistake,habit,focus}`, index = position dans sa section (focus=0). `responses[kind,index] = {useful, tag?, note?}`. Tag requis si useful=false (validé côté backend → 422 si manquant).
- `fbMap` : `{ "kind,index": {useful, tag, note} }` pré-rempli depuis `GET /api/c/{slug}/feedback` (trouver l'entrée dont `ts` === `review.ts`, puis `items` → map par `kind,index`).
- Evidence chip coloré par kind : strength → `--win`, mistake → `--loss`, habit/focus → `--gold`.

- [ ] **Step 1: Ajouter le test (échec d'abord)**

Ajouter à `tests/web/test_frontend.py` :

```python
def test_coaching_tab_wired():
    body = _client().get("/c/spadzze").text
    js = _client().get("/static/app.js").text
    assert "/api/coach" in js
    assert "/api/c/" in js and "/reviews" in js
    assert "/api/feedback" in js
    assert "NEG_TAGS" in js
    assert "insight-card" in body or "evidence-chip" in body
```

Run: `.venv/bin/python -m pytest tests/web/test_frontend.py::test_coaching_tab_wired -v`
Expected: FAIL (`/api/coach`, `/api/feedback`, `NEG_TAGS` usage absent — `NEG_TAGS` est défini en F1 mais non utilisé dans le body).

- [ ] **Step 2: Étendre `accountPage()` dans `app.js`**

Dans la fonction `accountPage(slug)` (définie en F3), ajouter ces champs à l'objet retourné (après `fetchN:` / `job:`) :

```javascript
    // coaching
    scope: "adc", outcome: "loss", target: "challenger",
    reviews: [], review: null, reviewsLoading: true,
    fbMap: {}, fbBusy: {},
    coachOpen: false,
```

Ajouter ces méthodes (dans le même objet, avant `kda(g)` par exemple) :

```javascript
    async loadReviews() {
      this.reviewsLoading = true;
      try {
        const list = await api(`/api/c/${this.slug}/reviews`);
        this.reviews = list;
        this.review = list.length ? list[list.length - 1] : null;
        if (this.review) this.loadFeedback();
      } catch (e) { /* keep reviews empty */ }
      finally { this.reviewsLoading = false; }
    },

    async genCoach() {
      this.job = { type: "coach", status: "running", progress: "coaching" };
      try {
        const { job_id } = await api("/api/coach", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slug: this.slug, scope: this.scope,
            outcome: this.outcome, target: this.target }),
        });
        // étend startPoll pour recharger les reviews au done coach
        pollJob(job_id,
          j => { this.job = j; },
          j => {
            this.job = j;
            if (j.status === "done" && j.type === "coach") this.loadReviews();
          },
        );
      } catch (e) { this.job = { type: "coach", status: "error", error: String(e) }; }
    },

    async loadFeedback() {
      if (!this.review) return;
      try {
        const list = await api(`/api/c/${this.slug}/feedback`);
        const mine = list.find(f => f.ts === this.review.ts);
        const m = {};
        if (mine) for (const it of (mine.items || []))
          m[`${it.kind},${it.index}`] = { useful: it.useful, tag: it.tag, note: it.note };
        this.fbMap = m;
      } catch (e) { /* fbMap stays empty */ }
    },

    fbKey(kind, index) { return `${kind},${index}`; },
    fbState(kind, index) { return this.fbMap[this.fbKey(kind, index)] || null; },

    async setFb(kind, index, useful) {
      if (useful) {
        await this.submitFb(kind, index, { useful: true });
      } else {
        // ouvre le menu tag ; le choix de tag appelle submitFb avec tag
        this.coachOpen = this.fbKey(kind, index);
      }
    },

    async pickTag(kind, index, tag) {
      await this.submitFb(kind, index, { useful: false, tag });
      this.coachOpen = false;
    },

    async submitFb(kind, index, { useful, tag = null, note = null }) {
      if (!this.review) return;
      const key = this.fbKey(kind, index);
      this.fbBusy = { ...this.fbBusy, [key]: true };
      const responses = { [key]: tag ? { useful, tag, note } : { useful, note } };
      try {
        await api("/api/feedback", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slug: this.slug, ts: this.review.ts, responses }),
        });
        this.fbMap = { ...this.fbMap, [key]: tag ? { useful, tag, note } : { useful, note } };
      } catch (e) { /* erreur 422 si tag manquant — déjà géré par UI (tag imposé) */ }
      finally { this.fbBusy = { ...this.fbBusy, [key]: false }; }
    },

    meta() { return this.review?.payload?.meta || null; },
```

Et modifier `setTab(t)` pour charger les reviews au premier passage sur coaching :

```javascript
    setTab(t) {
      this.tab = t;
      if (t === "coaching" && this.reviews.length === 0 && this.reviewsLoading) {
        this.loadReviews();
      }
    },
```

(Remplace la version F3 de `setTab`.)

- [ ] **Step 3: Remplacer le placeholder coaching dans `index.html`**

Remplacer `<template x-if="tab === 'coaching'"><div class="state">Coaching — à venir.</div></template>` par :

```html
<template x-if="tab === 'coaching'">
  <div>
    <!-- sélecteurs + générer -->
    <div class="action-bar row wrap">
      <span class="muted" style="font-size:13px">Coaching :</span>
      <select class="select" x-model="scope">
        <option value="all">scope : all</option>
        <option value="adc">scope : adc</option>
        <option value="zeri">scope : zeri</option>
      </select>
      <select class="select" x-model="outcome">
        <option value="loss">issue : loss</option>
        <option value="win">issue : win</option>
        <option value="overall">issue : overall</option>
      </select>
      <select class="select" x-model="target">
        <option value="challenger">target : challenger</option>
      </select>
      <button class="btn btn-primary" @click="genCoach()"
              :disabled="job && job.status === 'running'">Générer le coaching</button>
    </div>

    <template x-if="reviewsLoading"><div class="state">Chargement des reviews…</div></template>
    <template x-if="!reviewsLoading && !review">
      <div class="state">Aucune review. Règle les sélecteurs et génère le coaching.</div>
    </template>

    <template x-if="!reviewsLoading && review">
      <div>
        <!-- bandeau meta -->
        <div class="meta-strip row wrap">
          <span class="num" x-text="meta()?.n_games_me + ' games (perso)'"></span>
          <span class="faint" x-text="'vs ' + (meta()?.n_games_ref || 0) + ' (réf)'"></span>
          <span class="num" x-text="'WR ' + ((meta()?.winrate_me || 0) * 100).toFixed(0) + '%'"></span>
          <template x-if="meta()?.low_sample">
            <span class="badge badge-loss">low sample</span>
          </template>
          <span class="faint" x-text="review.model"></span>
          <span class="faint" x-text="review.ts"></span>
        </div>

        <!-- 4 cartes -->
        <div class="insight-grid">
          <div class="card insight-col">
            <h3>Forces</h3>
            <template x-for="(it, i) in review.review.strengths" :key="'s'+i">
              <div class="insight-card">
                <p x-text="it.point"></p>
                <span class="evidence-chip kind-strength" x-text="it.evidence"></span>
                <div class="fb-row" x-data="{ k: 'strength', i }">
                  <button class="fb-btn" :class="fbState(k,i)?.useful === true ? 'on-win' : ''" @click="setFb(k,i,true)" :disabled="fbBusy[k+','+i]">✓</button>
                  <button class="fb-btn" :class="fbState(k,i)?.useful === false ? 'on-loss' : ''" @click="setFb(k,i,false)" :disabled="fbBusy[k+','+i]">✗</button>
                  <div class="tag-menu" x-show="coachOpen === k+','+i" x-transition>
                    <template x-for="t in NEG_TAGS" :key="t">
                      <button class="tag-opt" @click="pickTag(k,i,t)" x-text="t"></button>
                    </template>
                  </div>
                </div>
              </div>
            </template>
          </div>

          <div class="card insight-col">
            <h3>Erreurs</h3>
            <template x-for="(it, i) in review.review.mistakes" :key="'m'+i">
              <div class="insight-card">
                <p x-text="it.point"></p>
                <span class="evidence-chip kind-mistake" x-text="it.evidence"></span>
                <div class="fb-row" x-data="{ k: 'mistake', i }">
                  <button class="fb-btn" :class="fbState(k,i)?.useful === true ? 'on-win' : ''" @click="setFb(k,i,true)" :disabled="fbBusy[k+','+i]">✓</button>
                  <button class="fb-btn" :class="fbState(k,i)?.useful === false ? 'on-loss' : ''" @click="setFb(k,i,false)" :disabled="fbBusy[k+','+i]">✗</button>
                  <div class="tag-menu" x-show="coachOpen === k+','+i" x-transition>
                    <template x-for="t in NEG_TAGS" :key="t">
                      <button class="tag-opt" @click="pickTag(k,i,t)" x-text="t"></button>
                    </template>
                  </div>
                </div>
              </div>
            </template>
          </div>

          <div class="card insight-col">
            <h3>Habitudes</h3>
            <template x-for="(h, i) in review.review.habits" :key="'h'+i">
              <div class="insight-card">
                <p x-text="h"></p>
                <div class="fb-row" x-data="{ k: 'habit', i }">
                  <button class="fb-btn" :class="fbState(k,i)?.useful === true ? 'on-win' : ''" @click="setFb(k,i,true)" :disabled="fbBusy[k+','+i]">✓</button>
                  <button class="fb-btn" :class="fbState(k,i)?.useful === false ? 'on-loss' : ''" @click="setFb(k,i,false)" :disabled="fbBusy[k+','+i]">✗</button>
                  <div class="tag-menu" x-show="coachOpen === k+','+i" x-transition>
                    <template x-for="t in NEG_TAGS" :key="t">
                      <button class="tag-opt" @click="pickTag(k,i,t)" x-text="t"></button>
                    </template>
                  </div>
                </div>
              </div>
            </template>
          </div>

          <div class="card insight-col">
            <h3>Focus prochain</h3>
            <div class="insight-card">
              <p x-text="review.review.next_focus"></p>
              <div class="fb-row" x-data="{ k: 'focus', i: 0 }">
                <button class="fb-btn" :class="fbState('focus',0)?.useful === true ? 'on-win' : ''" @click="setFb('focus',0,true)" :disabled="fbBusy['focus,0']">✓</button>
                <button class="fb-btn" :class="fbState('focus',0)?.useful === false ? 'on-loss' : ''" @click="setFb('focus',0,false)" :disabled="fbBusy['focus,0']">✗</button>
                <div class="tag-menu" x-show="coachOpen === 'focus,0'" x-transition>
                  <template x-for="t in NEG_TAGS" :key="t">
                    <button class="tag-opt" @click="pickTag('focus',0,t)" x-text="t"></button>
                  </template>
                </div>
              </div>
            </div>
            <div class="confidence faint" x-text="'confiance : ' + (review.review.confidence * 100).toFixed(0) + '%'"></div>
          </div>
        </div>

        <!-- historique des reviews précédentes (replié) -->
        <template x-if="reviews.length > 1">
          <details class="reviews-history">
            <summary class="muted">Reviews précédentes (<span x-text="reviews.length - 1"></span>)</summary>
            <template x-for="(rv, idx) in reviews.slice(0, -1)" :key="rv.ts">
              <div class="hist-row faint">
                <span x-text="rv.ts"></span> · <span x-text="rv.model"></span> ·
                <span x-text="rv.scope + '/' + rv.outcome_focus"></span>
              </div>
            </template>
          </details>
        </template>
      </div>
    </template>
  </div>
</template>
```

- [ ] **Step 4: Ajouter le CSS à `style.css`**

Ajouter à la fin de `web/frontend/style.css` :

```css
/* --- coaching --- */
.meta-strip {
  gap:14px; font-size:13px; padding:10px 14px; margin-bottom:16px;
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
}
.insight-grid {
  display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:14px;
}
.insight-col h3 { margin-bottom:12px; }
.insight-card {
  padding:12px 0; border-top:1px solid var(--border);
}
.insight-col .insight-card:first-of-type { border-top:none; padding-top:0; }
.insight-card p { margin:0 0 8px; font-size:14px; line-height:1.5; }
.evidence-chip {
  display:inline-block; font-size:12px; color:var(--text-dim);
  background:var(--bg); border-left:2px solid var(--border);
  padding:5px 9px; border-radius:0 var(--radius) var(--radius) 0; max-width:100%;
}
.evidence-chip.kind-strength { border-left-color:var(--win); }
.evidence-chip.kind-mistake { border-left-color:var(--loss); }

.fb-row { display:flex; gap:6px; align-items:center; margin-top:10px; position:relative; }
.fb-btn {
  width:28px; height:28px; border-radius:var(--radius); border:1px solid var(--border);
  background:var(--bg); color:var(--text-dim); cursor:pointer; font-size:13px;
  transition:border-color 150ms, color 150ms;
}
.fb-btn:hover { border-color:var(--gold); color:var(--text); }
.fb-btn.on-win { border-color:var(--win); color:var(--win); }
.fb-btn.on-loss { border-color:var(--loss); color:var(--loss); }
.fb-btn:disabled { opacity:.5; cursor:default; }
.tag-menu {
  position:absolute; top:32px; left:0; z-index:5;
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  padding:4px; display:flex; flex-direction:column; box-shadow:0 4px 16px rgba(0,0,0,.4);
}
.tag-opt {
  text-align:left; background:none; border:none; color:var(--text-dim);
  padding:6px 10px; border-radius:4px; cursor:pointer; font-size:12px; font-family:inherit;
}
.tag-opt:hover { background:var(--panel-2); color:var(--text); }
.confidence { margin-top:12px; font-size:12px; }

.reviews-history { margin-top:18px; }
.reviews-history summary { cursor:pointer; font-size:13px; padding:8px 0; }
.hist-row { font-size:12px; padding:5px 0; border-top:1px solid var(--border); }
```

- [ ] **Step 5: Lancer les tests — passent**

Run: `.venv/bin/python -m pytest tests/web/test_frontend.py -v`
Expected: 8 passed.

- [ ] **Step 6: Vérification manuelle**

Boot uvicorn (8011), `/c/spadzze` → onglet Coaching : la dernière review s'affiche (4 cartes : Forces/Erreurs/Habitudes/Focus, evidence chips colorées vert/rouge/or, bandeau meta avec n_games/WR/low_sample/model/ts). Cliquer ✓ sur un insight → le bouton passe vert (POST /api/feedback useful=true). Cliquer ✗ → menu de tags s'ouvre, choisir un tag → bouton passe rouge (POST avec tag). Recharger la page → les états ✓/✗ persistent (lus depuis /api/feedback). **Ne pas cliquer « Générer le coaching »** en manual check (coût LLM Ollama) — sauf si tu veux vraiment relancer.

- [ ] **Step 7: Commit**

```bash
git add web/frontend/index.html web/frontend/style.css web/frontend/app.js tests/web/test_frontend.py
git commit -m "feat(web): onglet coaching — review 4 cartes + feedback inline (tags NEG_TAGS)"
```

---

### Task F5: Page compte — onglet Profil ML (SHAP Chart.js)

**Files:**
- Modify: `web/frontend/app.js` (étendre `accountPage()` : état shap + méthode renderChart)
- Modify: `web/frontend/index.html` (remplacer le placeholder shap)
- Modify: `web/frontend/style.css` (shap-wrap, shap-empty)
- Modify: `tests/web/test_frontend.py`

**Interfaces:**
- Consumes: `GET /api/c/{slug}/shap` → `{available:bool, drivers:[{feature,mean_shap}]}` · `window.Chart` (vendored, chargé).
- Produces: étends `accountPage` avec `shap:null, shapLoading:false, shapSort:'abs', chart:null` et méthodes `loadShap(), renderChart(), toggleSort(), destroyChart()`.

**Décisions :**
- Si `available === false` → état « indisponible pour ce compte ».
- Sinon : Chart.js horizontal bar (`indexAxis:'y'`), `drivers` triés par `|mean_shap|` décroissant (défaut) ou par valeur brute (toggle). Couleur : `mean_shap >= 0` → `--gold`, `< 0` → `--loss`. Tooltip : feature + valeur.
- Le chart se rend dans un `<canvas>` ; détruire l'instance Chart précédente avant de recréer (sinon Chart.js warning + leak).

- [ ] **Step 1: Ajouter le test (échec d'abord)**

Ajouter à `tests/web/test_frontend.py` :

```python
def test_shap_tab_wired():
    body = _client().get("/c/spadzze").text
    js = _client().get("/static/app.js").text
    assert "/api/c/" in js and "/shap" in js
    assert "new Chart(" in js
    assert "shap-wrap" in body or "shap-empty" in body
```

Run: `.venv/bin/python -m pytest tests/web/test_frontend.py::test_shap_tab_wired -v`
Expected: FAIL (`new Chart(` absent, `shap-wrap` absent).

- [ ] **Step 2: Étendre `accountPage()` dans `app.js`**

Ajouter ces champs à l'objet retourné par `accountPage` :

```javascript
    shap: null, shapLoading: false, shapSort: "abs", chart: null,
```

Ajouter ces méthodes :

```javascript
    async loadShap() {
      this.shapLoading = true;
      try {
        this.shap = await api(`/api/c/${this.slug}/shap`);
        if (this.shap.available) this.$nextTick(() => this.renderChart());
      } catch (e) { this.shap = { available: false, drivers: [] }; }
      finally { this.shapLoading = false; }
    },

    sortedDrivers() {
      const d = (this.shap?.drivers || []).slice();
      if (this.shapSort === "abs") d.sort((a, b) => Math.abs(b.mean_shap) - Math.abs(a.mean_shap));
      else d.sort((a, b) => b.mean_shap - a.mean_shap);
      return d.slice(0, 16); // top 16 pour la lisibilité
    },

    renderChart() {
      this.destroyChart();
      const cv = this.$root.querySelector("#shap-canvas");
      if (!cv || !window.Chart) return;
      const d = this.sortedDrivers();
      this.chart = new window.Chart(cv, {
        type: "bar",
        data: {
          labels: d.map(x => x.feature),
          datasets: [{
            data: d.map(x => x.mean_shap),
            backgroundColor: d.map(x => x.mean_shap >= 0 ? "#c8aa6e" : "#f85149"),
            borderRadius: 3, borderSkipped: false,
          }],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { callbacks: {
            label: c => `SHAP ${c.raw.toFixed(4)}`,
          } } },
          scales: {
            x: { grid: { color: "#2a2d34" }, ticks: { color: "#9a9da4", font: { size: 11 } } },
            y: { grid: { display: false }, ticks: { color: "#e8e9ec", font: { size: 11 } } },
          },
        },
      });
    },

    toggleSort() {
      this.shapSort = this.shapSort === "abs" ? "val" : "abs";
      this.renderChart();
    },

    destroyChart() {
      if (this.chart) { this.chart.destroy(); this.chart = null; }
    },
```

Et étendre `setTab` pour charger SHAP au premier passage :

```javascript
    setTab(t) {
      this.tab = t;
      if (t === "coaching" && this.reviews.length === 0 && this.reviewsLoading) {
        this.loadReviews();
      }
      if (t === "shap" && this.shap === null) { this.loadShap(); }
    },
```

(Remplace la version F4 de `setTab`.)

- [ ] **Step 3: Remplacer le placeholder shap dans `index.html`**

Remplacer `<template x-if="tab === 'shap'"><div class="state">Profil ML — à venir.</div></template>` par :

```html
<template x-if="tab === 'shap'">
  <div>
    <template x-if="shapLoading"><div class="state">Chargement du profil ML…</div></template>
    <template x-if="!shapLoading && shap && !shap.available">
      <div class="shap-empty state">
        Profil ML indisponible pour ce compte.<br>
        <span class="faint">Le SHAP local est pré-calculé pour certains comptes seulement (V1).</span>
      </div>
    </template>
    <template x-if="!shapLoading && shap && shap.available">
      <div>
        <div class="row" style="margin-bottom:12px">
          <h2>Profil ML — contributions SHAP</h2>
          <div class="spacer"></div>
          <button class="btn" @click="toggleSort()"
                  x-text="shapSort === 'abs' ? 'tri : |valeur|' : 'tri : valeur'"></button>
        </div>
        <p class="muted" style="font-size:13px;margin:0 0 14px">
          Contribution moyenne de chaque feature au classement High-Elo vs Low-Elo.
          <span style="color:var(--gold)">or</span> = pousse vers High-Elo,
          <span style="color:var(--loss)">rouge</span> = pousse vers Low-Elo.
        </p>
        <div class="shap-wrap">
          <canvas id="shap-canvas"></canvas>
        </div>
      </div>
    </template>
  </div>
</template>
```

- [ ] **Step 4: Ajouter le CSS à `style.css`**

Ajouter à la fin de `web/frontend/style.css` :

```css
/* --- shap --- */
.shap-wrap {
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  padding:16px; height:520px; position:relative;
}
.shap-empty { padding:48px 20px; line-height:1.7; }
```

- [ ] **Step 5: Lancer les tests — passent**

Run: `.venv/bin/python -m pytest tests/web/test_frontend.py -v`
Expected: 9 passed.

- [ ] **Step 6: Vérification manuelle**

Boot uvicorn (8011), `/c/spadzze` → onglet Profil ML : graphique Chart.js horizontal (top 16 features par |SHAP|), barres or (positive) / rouge (négative), tooltips au survol. Bouton toggle tri ↔ |valeur| / valeur recrée le graphique. Pour un slug sans SHAP (ex. `aceofspadzze` si pas pré-calculé) → état « indisponible » propre.

- [ ] **Step 7: Commit**

```bash
git add web/frontend/index.html web/frontend/style.css web/frontend/app.js tests/web/test_frontend.py
git commit -m "feat(web): onglet profil ML — SHAP Chart.js (tri, état indispo)"
```

---

### Task F6: Page README + états finaux + smoke + doc

**Files:**
- Modify: `web/frontend/app.js` (remplacer `readmePage()` — trivial)
- Modify: `web/frontend/index.html` (remplir le template readme + états vides/erreur sur l'accueil)
- Modify: `web/frontend/style.css` (typo README long-form)
- Modify: `web/README.md` (structure finale)
- Modify: `tests/web/test_frontend.py`

**Interfaces:**
- Consumes: rien (page statique).
- Produces: `readmePage()` no-op. README rendu en HTML statique dans le template.

- [ ] **Step 1: Ajouter le test (échec d'abord)**

Ajouter à `tests/web/test_frontend.py` :

```python
def test_readme_page_wired():
    body = _client().get("/readme").text
    js = _client().get("/static/app.js").text
    assert "function readmePage()" in js
    # contenu vulgarisé clé présent dans le HTML servi
    assert "asymétrie" in body.lower() or "asymetrie" in body.lower()
    assert "benchmark" in body.lower()
    assert "positionnement" in body.lower()
```

Run: `.venv/bin/python -m pytest tests/web/test_frontend.py::test_readme_page_wired -v`
Expected: FAIL (le template readme est vide).

- [ ] **Step 2: Remplacer `readmePage()` dans `app.js`**

Remplacer `function readmePage()  { return { init() {} }; }` par :

```javascript
function readmePage() {
  return { init() { /* page statique, rien à fetcher */ } };
}
```

- [ ] **Step 3: Remplir le template readme dans `index.html`**

Remplacer `<div x-data="readmePage()" x-init="init()"></div>` par :

```html
<div x-data="readmePage()" x-init="init()" class="readme">
  <h1 style="font-size:28px;margin-bottom:6px">Comment fonctionne le coaching</h1>
  <p class="muted" style="margin:0 0 24px">Les recos sont benchmarkées, vérifiables, et respectent ce que tu savais vraiment.</p>

  <div class="card">
    <h2>Positionnement &gt; stats brutes</h2>
    <p>Les outils classiques (op.gg, u.gg) s'appuient sur des agrégats — KDA, gold, tourelles. Ils produisent des conseils pauvres (« meurs moins »). Ce coach reconstruit <strong>les déplacements réels</strong> depuis la timeline Riot (positions de tous les champions toutes les 60 s) pour dire <em>« place-toi ici plutôt que là »</em>.</p>
  </div>

  <div class="card">
    <h2>Respect de l'asymétrie d'information</h2>
    <p>Le coach ne te reproche <strong>jamais</strong> une décision basée sur une info que tu n'avais pas. Ex. : il ne dira pas « tu n'aurais pas dû push, le jungle ennemi était botside » si tu n'avais aucune vision dessus. On raisonne uniquement sur l'information réellement disponible au joueur. L'info complète post-game sert à <em>labelliser</em>, pas à juger.</p>
  </div>

  <div class="card">
    <h2>Benchmark challenger, pas opinion absolue</h2>
    <p>« Tu recalls à 1450 g en moyenne, les challengers de ton matchup à 1100 » est concret et <strong>vérifiable</strong>. Les benchmarks viennent directement des timelines high-elo de l'API Riot, à issue et contexte de lane égaux. Une reco sans preuve chiffrée n'est pas une reco.</p>
  </div>

  <div class="card">
    <h2>Ce que mesurent les features</h2>
    <ul>
      <li><strong>Lane</strong> — gold/CS/XP diff @10/@14/@20 vs adversaire direct.</li>
      <li><strong>Positionnement</strong> — présence par zone, roam mid, over-extension, vision posée/détruite, temps mort (gold dead time). Calculées sans vision computer, depuis la timeline.</li>
      <li><strong>Morts</strong> — répartition par zone/phase, morts en fog vs en vision, gold state au moment de la mort.</li>
      <li><strong>Profil ML</strong> — contributions SHAP au classement High-Elo vs Low-Elo (modèle XGBoost/RF/EBM). Indique quelles features poussent ton profil vers un rang ou l'autre.</li>
    </ul>
  </div>

  <div class="card">
    <h2>La boucle de feedback</h2>
    <p>Chaque insight (force / erreur / habitude / focus) est annotable : ✓ utile, ✗ faux ou inutile (avec un tag : <em>asymétrie</em>, <em>stat inventée</em>, <em>profondeur en faute</em>, <em>trop vague</em>, <em>non actionnable</em>, <em>autre</em>). Ces annotations mesurent si le coach s'améliore — un conseil benchmarké est intrinsèquement plus vérifiable qu'une opinion.</p>
  </div>
</div>
```

- [ ] **Step 4: Ajouter le CSS README à `style.css`**

Ajouter à la fin de `web/frontend/style.css` :

```css
/* --- readme --- */
.readme h2 { margin-bottom:8px; }
.readme p { margin:0 0 10px; color:var(--text-dim); line-height:1.65; }
.readme strong { color:var(--text); font-weight:600; }
.readme em { color:var(--gold); font-style:normal; }
.readme ul { margin:0; padding-left:20px; color:var(--text-dim); }
.readme li { margin:6px 0; line-height:1.6; }
.readme li strong { color:var(--text); }
```

- [ ] **Step 5: Lancer toute la suite web — passent**

Run: `.venv/bin/python -m pytest tests/web/ -v`
Expected: tous passent (test_frontend.py 10 + les tests backend existants).

- [ ] **Step 6: Smoke manuel complet**

```bash
.venv/bin/python -m uvicorn main:app --app-dir web/backend --port 8011
```
Parcours : `/` (grille comptes) → carte spadzze → onglet Historique (games) → onglet Coaching (review + feedback ✓/✗) → onglet Profil ML (SHAP chart) → switcher → autre compte → `/readme`. Vérifier : top-bar sticky, switcher, pas de néon, chiffres tabulaires, états vides propres. Tuer le serveur.

- [ ] **Step 7: Mettre à jour `web/README.md` (structure finale)**

Dans `web/README.md`, remplacer le bloc ```` ``` ```` de structure par :

```
web/
  backend/main.py     # FastAPI : /api/* + sert index.html et /static
  frontend/
    index.html        # shell SPA (top-bar + templates par page)
    style.css         # tokens + composants (CSS sur-mesure, pas de build)
    app.js            # composants Alpine (app/homePage/accountPage/readmePage) + helpers
    vendor/
      alpine.min.js   # Alpine 3.x vendored
      chart.umd.min.js# Chart.js 4.x vendored
fly.toml
Dockerfile
.dockerignore
```

- [ ] **Step 8: Commit**

```bash
git add web/frontend/index.html web/frontend/style.css web/frontend/app.js web/README.md tests/web/test_frontend.py
git commit -m "feat(web): page README + états finaux + doc structure"
```

---

## Self-Review (effectué inline)

- **Spec coverage :**
  - Pages (accueil / compte / readme) → F1 (shell) + F2 (accueil) + F3–F5 (compte 3 onglets) + F6 (readme). ✅
  - Top-bar + switcher → F1. ✅
  - Fetch job + bandeau + polling → F3. ✅
  - Historique (rangée compacte, W/L, KDA, champion) → F3 (KDA calculé, patch au lieu de timestamp — documenté). ✅
  - Coaching 4 cartes + evidence chip + feedback inline + tags + historique reviews → F4. ✅
  - SHAP Chart.js interactif + état indispo → F5. ✅
  - README vulgarisé → F6. ✅
  - Stack Alpine + CSS sur-mesure + Chart.js vendored → F1. ✅
  - Palette raffinée / densité équilibrée / pas de néon → Global Constraints + F1 tokens + CSS par tâche. ✅
  - Secrets côté serveur uniquement → Global Constraints ; `app.js` ne référence que `/api/*`. ✅
- **Écarts vs spec documentés :** (1) evidence chip coloré par `kind` et non par `notable` (la review ne lie pas insight→signal) ; (2) historique affiche `patch` au lieu de `duration`/`timestamp` (champs absents de la silver row). Les deux sont des contraintes de données, pas des oublis.
- **Placeholder scan :** aucun TBD/TODO. Chaque étape de code contient le code complet.
- **Type/consistance :** `app()`, `homePage()`, `accountPage(slug)`, `readmePage()`, `api()`, `pollJob()`, `fmtKDA()`, `DDRAGON()`, `NEG_TAGS` définis en F1 et réutilisés à l'identique en F2–F6. `setTab` étendu progressivement (F3 → F4 → F5) — chaque version remplace la précédente explicitement.
- **Tests :** 10 tests de câblage (TestClient), un par préoccupation, grandissent tâche par tâche. Interactivité = vérif manuelle (pas de Playwright), explicité en Global Constraints pour ne pas être flagué comme gap.