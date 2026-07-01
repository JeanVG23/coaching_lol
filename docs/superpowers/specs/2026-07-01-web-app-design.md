# Web App V1 — Interface Fly.io (FastAPI + front statique)

> Spec de l'interface web de coaching_lol. Rend le pipeline local utilisable depuis
> le navigateur : gérer plusieurs comptes de coaching, fetcher les games, générer le
> coaching, annoter le feedback, consulter son profil ML.
>
> Date : 2026-07-01. État : design validé, en attente de plan d'implémentation.

## Contexte et décision

Le projet coaching_lol existe en CLI Python (pipeline médaillon `src/`, données `data/`).
On veut l'exposer via un site perso hébergé sur **Fly.io**, en un **seul process FastAPI**
qui sert à la fois l'API `/api/*` et le frontend statique. L'usage est **personnel** :
toi + ~5-10 comptes (smurfs + amis). L'auth est reportée (URL publique acceptée, faible
proba de découverte).

Choix d'infrastructure déjà actés (cf. `web/README.md`, `Dockerfile`, `fly.toml`) :
FastAPI (réutilise les schémas Pydantic existants, async-native pour les jobs longs),
Fly.io (front + back dans une image, volume persistant pour `data/`).

## Périmètre V1

### Pages

1. **`/` — Accueil = liste des comptes.** Grille de cartes par slug préconfiguré
   (`spadzze`, `aceofspadzze`…). Chaque carte : nom, rang, winrate, nb de games en
   cache, timestamp de la dernière review. **Pas de création de compte depuis l'UI**
   (liste fixée côté serveur). Pattern : tuiles lolalytics.

2. **`/c/{slug}` — Page compte.** Le cœur du site. Sections en onglets (pattern
   op.gg/u.gg) :
   - **Header de profil** (sticky) : rang, winrate, games, KDA sommaire, patch.
   - **Bandeau job** (apparaît si un job tourne, sticky sous le header) : barre de
     progression fetch (« pull 12/20 ») ou coach (« coaching en cours »).
   - **Barre d'actions** : « Mettre à jour les games » (champ N, défaut 20) ;
     « Générer le coaching » + sélecteurs scope (`all`/`adc`/`zeri`), issue
     (`loss`/`win`/`overall`), target (`challenger`).
   - **Onglet Historique** (défaut) : game list type op.gg — rangée compacte par game
     (icône champion Data Dragon, bordure gauche verte/rouge W/L, KDA, durée, CS,
     rôle, champion, timestamp), pagination légère.
   - **Onglet Coaching** : review courante en 4 cartes (Forces / Erreurs / Habitudes /
     Focus), chaque point avec son **evidence chip** chiffré (couleur selon `notable`).
     Feedback **inline** sous chaque insight (boutons ✓/✗/skip + menu de tag sur ✗).
     Historique des reviews précédentes en replié en bas.
   - **Onglet Profil ML** : graphique SHAP local Chart.js (barres horizontales triables,
     contribution par feature, ce compte vs global, tooltip au survol). Palette LoL.
     État « indisponible » propre pour les slugs non pré-calculés.

3. **`/readme` — README.** Page statique vulgarisée : asymétrie d'information, benchmark
   challenger, pourquoi positionnement > stats brutes, ce que mesurent les features.

### Top-bar (commune)

Sticky : nom du site à gauche, **sélecteur de compte** (dropdown Alpine) au centre pour
basculer d'un slug à l'autre, lien **README** à droite. Pattern : switcher op.gg.

### Parcours type

Accueil → carte compte → (job fetch si besoin) → onglet Coaching → générer → lire la
review → annoter le feedback → onglet Profil ML. Le sélecteur de top-bar permet de
changer de compte à tout moment.

### Décisions clés

- **Compte de coaching = un slug** (`spadzze`, `aceofspadzze`, smurfs…).
- **Auth reportée** — URL publique acceptée. À revisiter si visibilité croît.
- **Fetch = pull Riot → silver → gold en un seul job** (déterministe, 1 clic).
- **Coaching LLM = job séparé** (lent/coûteux, relançable sans re-fetcher — A/B modèles,
  changer les sélecteurs).
- **Coaching V1 = agrégé N games** (miroir du CLI `coach.py`), pas par-game.
- **SHAP local V1 = slugs pré-calculés uniquement** (spadzze). Les autres voient
  « indisponible ». Calcul par-slug = V2 (réutilise le travail ML).

### Embarqué en V1

Fetch agrégé (1 job) · coaching LLM agrégé (job séparé) · historique games · feedback
par-insight inline · SHAP local interactif (slugs pré-calculés) · README · multi-compte
préconfiguré · sélecteur de compte en top-bar.

### Repoussé (V2+)

Compte-rendu par-game · draft coaching live · UI d'ajout de compte · auth · SHAP
explorateur global (rang/scope filtrable) · calcul SHAP local par-slug à la demande.

### Hors scope

Computer vision (Phase 2 du projet, gated sur la démonstration de valeur du coach).

## Modèle de données + persistance

**Approche A — lire les fichiers existants directement, pas de DB.** La donnée est petite
(silver 127 KB / 40 games, reviews 32 KB, ~930 M tout inclus) et déjà structurée en
JSONL/JSON. Le pipeline écrit, le web lit. Aucune duplication, aucune logique de sync.

Entités V1 :

| Entité | Source | Lecture/écriture |
|---|---|---|
| `accounts` | **config serveur** `web/backend/accounts.json` : `slug → riotId (gameName#tagLine) → puuid` (puuid résolu via `account-v1`, caché) | lecture (liste fixée côté serveur) |
| `games` | `data/02_silver/personal/<slug>/games.jsonl` | lecture |
| `aggregates` | `data/03_gold/personal/<slug>/<scope>/aggregate.json` | lecture |
| `reviews` | `data/07_coaching/<slug>/reviews.jsonl` (append, comme `coach.py`) | lecture/écriture |
| `feedback` | `data/07_coaching/<slug>/feedback.jsonl` | lecture/écriture |
| `shap` | `data/06_shap/<slug>_drivers.json` + `<slug>_sv_ensemble.npy` + global `ranking.json`, `gam_crosscheck.json`, `ebm_interactions.json` | lecture |
| `jobs` | **nouveau** `data/08_jobs/jobs.jsonl` | lecture/écriture |

Note sur `jobs.jsonl` : nouvel objet, pas de stockage existant aujourd'hui. Schéma :
`id, type(fetch|coach), slug, status(pending|running|done|error), progress, ts_start,
ts_end, error, result_ref`. Append pour création, update par `id` (réécriture du fichier
ou index en mémoire). Emplacement retenu : `data/08_jobs/jobs.jsonl` (nouvelle couche
médaillon dédiée à l'état des jobs web, hors pipeline de features).

### Gap SHAP signalé

Le SHAP local n'existe que pour **spadzze** (`spadzze_drivers.json`, `spadzze_sv_ensemble.npy`).
V1 se limite aux slugs pré-calculés (cf. décision ci-dessus). Le calcul par-slug est un
prérequis ML pour V2, hors scope de cette spec.

## Stack front

**Alpine.js + Tailwind + Chart.js**, CSS Tailwind **précompilé** (CLI standalone,
`style.css` généré commité, pas de build Node à l'exécution).

- **Alpine.js** (~15 KB, pas de build) : réactivité légère saupoudrée dans le HTML —
  gère l'état (job en cours, insights d'une review, sélecteurs, dropdown de compte).
- **Tailwind** (CSS précompilé) : styling. Palette LoL (or `#c8aa6e`, fond `#0e1014`,
  panel `#161a22`).
- **Chart.js** : graphique SHAP local (barres horizontales triables + tooltip).

Raison : on garde le modèle « front statique servi par FastAPI » (pas de Vite, pas de
build Node, Docker reste simple), tout en évitant le JS impératif verbeux. C'est le
point doux pour ~3 pages avec 3-4 endroits réactifs.

## Architecture async

**A — background task (threadpool) + polling.**

⚠️ Le pipeline existant (`aggregate_games.py`, `coach.py`) est **synchrone, bloquant**
(requêtes `requests` à Riot/Ollama). Lancé directement dans un endpoint FastAPI async,
il bloquerait l'event loop. Donc le job **doit tourner dans un threadpool** via
`asyncio.to_thread` / `fastapi.concurrency.run_in_threadpool`.

- `POST /api/fetch` démarre le job en thread, renvoie `{job_id}` immédiatement.
- Le front rafraîchit `GET /api/jobs/{id}` toutes les ~2 s (Alpine `setInterval`).
- État dans `jobs.jsonl` (persistant — survit aux redémarrages/auto-stop Fly).
- **uvicorn `--workers 1`** (mono-machine perso, 512 Mo) — état job trivial à partager,
  évite la complexité multi-process. `jobs.jsonl` reste source de vérité si on monte.

**Progression :**
- Job **fetch** : fine — `progress = "12/20"`, mis à jour dans `jobs.jsonl` à chaque game
  pullée. Demande un **petit refactor du pipeline** pour accepter un *callback de
  progression* sur la boucle de pull (ou wrapper la boucle per-game).
- Job **coach** : coarse — `pending → running → done|error` (appel LLM monobloc).

**Auto-stop Fly** : un job en cours garde la machine éveillée (pas idle). Le polling du
front réveille la machine au prochain check. Le volume persiste à l'arrêt.

## Infra Fly réelle

- **Volume persistant** 3 Go monté à `/app/data` :
  ```toml
  [[mounts]]
  source = "coaching_data"
  destination = "/app/data"
  ```
  `fly volumes create coaching_data --size 3 --region cdg`.
- **Seed du volume** : au 1er déploiement, le volume est vide. On pousse **tout le
  `data/` local** (~930 M, miroir complet — y compris `01_raw` 859 M) via
  `fly ssh sftp put` (ou `tar | fly ssh console`). Garde reextract/densify fonctionnels
  sans repull.
- **Secrets** : `fly secrets set RIOT_API_ID=... OLLAMA_API_KEY=... OLLAMA_MODEL=kimi-k2.6`.
  Disponibles en env var dans le process — le code les lit via `.env` en local, via l'env
  Fly en prod.
- **`.dockerignore`** : déjà OK (`data/` exclu de l'image → tout vient du volume, pas de
  conflit fichier-vs-volume au mount).

## Endpoints prévus

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/api/health` | sonde (existe) |
| GET | `/api/accounts` | liste des slugs + indicateur par compte (games count, dernière review) |
| GET | `/api/c/{slug}/games` | historique games (paginé) |
| POST | `/api/fetch` | mettre à jour les games d'un slug (pull Riot → silver → gold, job async) → `{job_id}` |
| POST | `/api/coach` | générer le coaching (LLM, job async) → `{job_id}` |
| GET | `/api/jobs/{id}` | suivi d'un job |
| GET | `/api/c/{slug}/reviews` | historique des reviews |
| POST | `/api/feedback` | annoter un insight (boucle d'éval) |
| GET | `/api/c/{slug}/shap` | SHAP local d'un slug (ou `null` si non pré-calculé) |

## Composants à isoler

- **`accounts`** — config serveur (liste de slugs + résolution puuid). Pas éditable
  depuis l'UI.
- **Job runner** — threadpool + écriture `jobs.jsonl` + callback de progression. Unité
  isolée, testable sans FastAPI.
- **Pipeline wrappers** — adaptateurs autour de `aggregate_games` (fetch) et `coach`
  (coaching) acceptant un callback de progression. Réutilisent le pipeline existant,
  ne le réécrivent pas.
- **Lecteurs de données** — fonctions pures qui lisent silver/gold/reviews/feedback/shap
  et renvoient des structures typées (Pydantic). Isolées du transport HTTP.
- **Vues Alpine** — un module Alpine par page (accueil, compte, readme), état local +
  appels `/api/*`.

## Tests

- **Lecteurs de données** : tests purs sur les fichiers silver/reviews/feedback (fixtures
  réduites) — pagination, parsing, cas vide.
- **Job runner** : cycle de vie d'un job (pending → running → done/error), update par id,
  persistance dans `jobs.jsonl`.
- **Pipeline wrappers** : callback de progression appelé au bon rythme (mock du pipeline).
- **API** : smoke test des endpoints (`/api/health`, `/api/accounts`, cycle fetch→poll).

## Risques / points d'attention

- **Threadpool + event loop** : tout appel au pipeline bloquant doit passer par
  `to_thread`. Oubli = freeze du serveur pendant un job.
- **`jobs.jsonl` concurrent** : un seul writer à la fois (un worker), mais l'update par
  `id` réécrit le fichier — garder une section critique ou un verrou en mémoire.
- **Seed volume** : ~930 M à pousser au 1er setup — vérifier la bande passante Fly / ne
  pas le refaire à chaque déploiement.
- **Tailwind précompilé** : le CLI standalone doit tourner en local avant commit ;
  oubli = CSS périmé. Documenter dans `web/README.md`.
- **SHAP V1 limité à spadzze** : gérer proprement l'état indispo pour ne pas casser
  l'UX des autres comptes.