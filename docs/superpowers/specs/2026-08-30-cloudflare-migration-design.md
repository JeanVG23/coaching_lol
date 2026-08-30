# Migration Cloudflare — design

> Date : 2026-08-30. Statut : design approuvé en brainstorming (approche 1 « précompute »).
> Contexte : pivot du 2026-08-30 — abandon de l'objectif parallèle « cours AOS », priorité
> au site web. Objectif : rattacher le site au domaine **jeanvg.fr** (sous-domaine
> **coaching-lol.jeanvg.fr**) et **quitter Fly.io** pour tout ramener sur Cloudflare.

## 1. Contexte et motivation

- **Un seul fournisseur.** jeanvg.fr est acheté chez OVH, mais DNS + mails sont déjà gérés
  sur Cloudflare (zone active). Fly.io est aujourd'hui le seul autre fournisseur de la
  stack : volume mono-machine, push sftp manuel du référentiel gold après chaque déploiement
  (cf. `web/README.md`), second dashboard. Le motif déclaré : **simplifier la stack**, quitte
  à réécrire le backend dans un autre langage.
- **État actuel** : app `coaching-lol` sur Fly.io — un process FastAPI sert l'API `/api/*`
  ET le frontend statique (`web/backend/main.py`), données sur volume `coaching_lol_data`,
  région cdg, auto-stop machines. URL : `coaching-lol.fly.dev`.
- **Contrainte structurante** : compte Cloudflare **free uniquement**. Workers free =
  10 ms CPU/req, Durable Objects limités, pas de Queues. Ça élimine d'office tout travail
  CPU-lourd côté Worker (le traitement d'un fetch Riot ×20 games dépasse largement 10 ms).

## 2. Décisions actées (brainstorming 2026-08-30)

| Question | Décision |
|---|---|
| Registrar / zone | OVH ; DNS + mails déjà sur Cloudflare → aucune étape nameservers |
| Sous-domaine | `coaching-lol.jeanvg.fr` (aligné sur le nom de l'app) |
| Fly.io | À quitter. **Interim** : le domaine pointe vers Fly pendant la construction (30 min, réversible), bascule + décommission à la parité |
| Plan Cloudflare | Free uniquement — architecture conçue pour tenir dedans |
| Périmètre V1 | Lecture + génération coaching **sur CF** ; fetch Riot = commande **locale** (la mise à jour des games ne passe plus par le site) |
| Approche ML | **Précompute local** : predicted-rank / predicted_lp calculés au sync (là où vivent les `.pkl`) et poussés dans KV comme JSON. Zéro inférence côté Worker |
| Langage serving | TypeScript (portage du chemin serving uniquement) |

**Approches écartées, pour mémoire :**

- **#2 m2cgen dès V1** (modèles xgb/rf exportés en JS pur, inférence côté Worker) :
  gain non exploitable tant que le fetch est local (les games ne changent QUE par sync
  local → la fraîcheur du précompute est équivalente) ; test de parité JS-vs-pkl à
  maintenir ; EBM non exportable → LP servi à 2 modèles sur 3 (dérive vs métriques
  documentées). **Réévaluée en V2** si on veut un jour le fetch-sur-Worker (plan payant).
- **#3 Python Workers (Pyodide, open beta au 2026-08)** : FastAPI/httpx officiels MAIS
  `requests`→httpx async obligatoire, `Path`→KV obligatoire, threads→? obligatoire,
  `.pkl` sklearn/xgboost hors liste supportée — presque autant de réécriture que le TS
  avec le risque beta en plus. Rejetée.
- **#C Frontend sur Cloudflare Pages + API séparée** : deux déployables, Pages en mode
  maintenance (CF recommande Workers static assets pour les nouveaux projets). Rejetée.

## 3. Architecture cible

```
[local Python — inchangé, le cerveau]         [Cloudflare — zone jeanvg.fr, la vitrine]
  collection Riot (build_referential,           ┌──────────────────────────────────┐
  aggregate_games, densify_*)                   │ Worker TS "coaching-lol"         │
  training ML (train_*, calibrate_*)            │  • assets statiques (SPA)        │
  extract/silver/gold, compare, CLI             │  • /api/* (routeur TS)           │
  ml_rank.py (précalcul rang/LP)                │  • KV binding "DATA"             │
        │                                       │  • SSE coach → Ollama Cloud      │
        │ sync_cloudflare.py                    └───────────────┬──────────────────┘
        │ (API REST CF, pousse données                 coaching-lol.jeanvg.fr
        │  + prédictions précalculées)                 (Custom Domain à la bascule)
        └────────────────────────────────────────────►
```

Un seul déployable côté CF. Pas de Durable Objects, pas de Queues. Le local reste
responsable des données et du ML ; le Worker lit KV, sert le frontend, génère le coaching
(appel Ollama = pure attente réseau, CPU quasi nul) et persiste reviews/feedback dans KV.

## 4. Composants

### 4.1 Worker TypeScript — nouveau `web/cf/`

```
web/cf/
  wrangler.toml        # worker "coaching-lol", assets = ../frontend, binding KV DATA
  src/
    index.ts           # routeur : /api/* + délégation assets (SPA fallback /c/{slug})
    readers.ts         # lectures KV (jeux, rang, agrégats, reviews, feedback, shap, pred)
    payload.ts         # portage de src/04_coaching/payload.py (constructeur du payload)
    prompt.ts          # portage de prompt.py (system prompts, chaînes quasi pures)
    schema.ts          # validation Review (équivalent Pydantic, types + gardes)
    llm_client.ts      # portage de llm_client.py (fetch Ollama, format JSON-schema)
    coach.ts           # orchestration SSE : payload → llm → validation → persiste KV
    feedback.ts        # sous-ensemble de feedback.py : validate + build + persiste
    accounts.ts        # comptes préconfigurés (miroir de web/backend/accounts.json)
  package.json         # wrangler, vitest-pool-workers (dev deps)
```

**Portage : ~1 000-1 200 lignes TS** depuis le chemin serving Python (~2 000 lignes :
`readers.py`, `payload.py`, `prompt.py`, `schema.py`, `llm_client.py`, sous-ensembles de
`coach.py`/`feedback.py`, routeurs). `game_journal.py` (coaching par-game) n'est **pas**
dans le périmètre web V1 — le site ne sert que le coaching agrégé, comme aujourd'hui.

**Restent Python local (inchangés)** : tout `src/` (riotlib, positioning, champion_profiles,
game_journal, training, compare…) + `web/backend/ml_rank.py`, réutilisé par le sync.

### 4.2 Sync local — nouveau `src/collection/sync_cloudflare.py`

0 appel Riot. Relit `data/`, **précalcule** `predicted-rank` + `predicted_lp` via
`web/backend/ml_rank.py`, pousse le tout dans KV via l'API REST Cloudflare
(`PUT .../storage/kv/namespaces/{ns}/values/{key}`, token dédié dans `.env` — aucune
dépendance node). Usage :

```bash
poetry run python3 src/collection/sync_cloudflare.py              # tout
poetry run python3 src/collection/sync_cloudflare.py --slug spadzze   # un compte
```

Le « Mettre à jour les games » d'aujourd'hui devient localement :
fetch local existant (`aggregate_games.py`) puis `sync_cloudflare.py`.

## 5. Couche données — Workers KV

1 clé = 1 fichier actuel (miroir du layout `data/`) :

| Clé KV | Contenu | Origine |
|---|---|---|
| `silver:{slug}:games` | games.jsonl (~128 Ko/joueur) | sync local |
| `silver:{slug}:rank` | rank.json (rang mis en cache au fetch) | sync local |
| `gold:{slug}:{scope}` | aggregate.json (all/adc/zeri) | sync local |
| `ref:{rank}:{scope}` | aggregate.json référentiel (~288 Ko/rang, 4 rangs) | sync local |
| `pred:{slug}` | predicted-rank + proba + predicted_lp **précalculés** | sync local |
| `shap:{slug}:drivers` | {slug}_drivers.json | sync local |
| `coaching:{slug}:reviews` | reviews.jsonl | **Worker** (coach persist) + seed sync |
| `coaching:{slug}:feedback` | feedback.jsonl | **Worker** (POST /api/feedback) |

Volumes : ~500 Ko/joueur + 1,1 Mo référentiel ≪ limites free (25 Mo/valeur, 1 Go total,
1 000 écritures/jour — sync ≈ 30-60 clés + usage perso, très large).

**Divergence reviews/feedback local ↔ KV : acceptée et documentée** (parité avec le
comportement actuel : le volume Fly et le local divergent déjà de la même façon — les
reviews générées côté web ne redescendent jamais au local). Pull KV→local = amélioration
V2 possible, pas bloquante.

## 6. Contrat API

| Endpoint | Statut V1 | Note |
|---|---|---|
| `GET /api/accounts` | identique | |
| `GET /api/c/{slug}/games` | identique | pagination + tri par n° de séquence match_id |
| `GET /api/c/{slug}/rank` | identique | |
| `GET /api/c/{slug}/predicted-rank` | identique en surface | lit `pred:{slug}` (précalculé) |
| `GET /api/c/{slug}/reviews` | identique | lit KV |
| `GET /api/c/{slug}/feedback` | identique | lit KV |
| `GET /api/c/{slug}/shap` | identique | |
| `POST /api/feedback` | identique | écrit KV (read-modify-write append) |
| `POST /api/coach` | **modifié → SSE** | événements `payload` → `llm` → `review` ; le frontend suit le flux au lieu de poller un job |
| `POST /api/fetch` | **supprimé V1** | remplacé par la commande locale fetch+sync |
| `GET /api/jobs/{id}` | **supprimé V1** | plus de système de jobs (coach = SSE direct) |

**Frontend** (`app.js`, `index.html`) : bouton « Mettre à jour » remplacé par un repère
« sync locale » (badge `fetched_at` déjà existant comme indicateur de fraîcheur) ;
« Générer le coaching » suit le flux SSE avec états. Le reste (Alpine, Chart.js vendored,
pages, SHAP local) ne bouge pas.

**Secrets** : seul `OLLAMA_API_KEY` devient un secret Worker (`wrangler secret put`).
**Aucune clé Riot côté serveur web en V1** (le Worker ne parle jamais à Riot) —
amélioration nette vs Fly aujourd'hui. `RIOT_API_ID` et le token CF restent dans le
`.env` local.

## 7. Domaine : interim puis bascule

1. **Interim (dès P1)** : dans le dashboard CF (zone jeanvg.fr), CNAME `coaching-lol` →
   `coaching-lol.fly.dev`, **DNS only** (nuage gris). Puis `fly certs add
   coaching-lol.jeanvg.fr` (Let's Encrypt côté Fly ; `force_https` déjà actif gère la
   redirection). Le domaine répond pendant toute la construction.
2. **Bascule (P4)** : supprimer le CNAME, ajouter **Custom Domain**
   `coaching-lol.jeanvg.fr` au Worker (cert géré par CF, ~2 min, sans interruption longue).
3. **Décommission** : **rapatrier d'abord le volume Fly** (reviews/feedback générés
   côté web — à ne pas perdre : `fly ssh sftp pull` ou équivalent, merge dans le local
   puis re-sync KV), puis `fly apps destroy coaching-lol`.

## 8. Erreurs et dégradation

- Clé KV absente → 404 / `null` propres, même sémantique que « fichier absent » aujourd'hui.
- Ollama KO → événement SSE `error` explicite, aucune review partielle persistée.
- Sync KO / données périmées → le site sert le dernier sync ; la fraîcheur reste visible
  via `fetched_at` (rang) et l'UI — dégradation douce, pas d'écran mort.
- LLM non conforme au schéma → rejet (validation `schema.ts`), pas de persistance —
  même discipline que la validation Pydantic actuelle.

## 9. Tests

- **Parité payload (le test critique du portage)** : payload TS vs payload Python sur les
  mêmes agrégats → JSON identique (golden test, relancé à chaque évolution du payload).
- Worker : tests routeur/readers sur KV émulé (`@cloudflare/vitest-pool-workers`).
- Sync : pytest comme le reste du pipeline local (dossiers temporaires, 0 réseau).
- Bout-en-bout : `wrangler dev` (health, games, un vrai coaching SSE), puis vérif sur
  `*.workers.dev` avant bascule domaine.

## 10. Phases (une session chacune)

| Phase | Contenu | État |
|---|---|---|
| P0 | Cleanup worktree + pivot AOS (commits, merge master, branches obsolètes) | ✅ 2026-08-30 |
| P1 | Interim : CNAME CF + `fly certs add` + vérif `https://coaching-lol.jeanvg.fr/api/health` | à faire |
| P2 | Worker TS squelette + assets + readers KV + `sync_cloudflare.py` → parité LECTURE sur `*.workers.dev` | à faire |
| P3 | Coach SSE + feedback écriture KV + ajustements frontend | à faire |
| P4 | Bascule Custom Domain + rapatriement volume Fly + décommission + doc README | à faire |

## 11. Risques connus et non-buts

- **CPU 10 ms free** : le coach SSE est CPU-léger (payload build ≈ quelques ms), mais
  c'est LA marge à surveiller en P3 ; si elle craque, lever vers Workers Paid (5 $/mois)
  est le plan de secours clean — pas une re-archi.
- **Écritures KV 1 000/jour** : usage perso → large ; à revoir si le site s'ouvre.
- **Non-buts V1** : auth (décision actée V1 web, inchangée), fetch-sur-Worker, inférence
  ML côté Worker (m2cgen), pull KV→local, coaching par-game sur le web.
- Le pipeline Python local ne change PAS d'architecture — cette migration ne touche que
  la couche serving.