# TODO — Coaching LoL

> Suite logique après la **Phase 2 narration LLM** (merge `feat/llm-coaching`).
> Source : spec `docs/superpowers/specs/2026-06-30-llm-coaching-narration-design.md` (§ Hors scope / Critères de succès) + `CLAUDE.md` (§ Prochaines étapes).

## ⚡ Immédiat — valider l'incrément narration (critères de succès non atteints)

- [x] **Premier run réel** — `deepseek-v4-pro` (commit `5ced84b`). A révélé que deepseek-v4-pro n'honore pas la contrainte `format` → prompt durci (règle 7 + règle 3 étendue à toutes les `descriptive_only`). Review conforme, asymétrie tenue, profondeur en observation neutre.
- [x] **A/B modèles manuel** — même payload (spadzze/adc/loss) rejoué sur 4 modèles, tous conformes au schéma (3 forces / 3 erreurs / 2 habitudes) :
  | Modèle | Conf. | Format | Profondeur (règle 3) | Benchmark contextuel | Note |
  |---|---|---|---|---|---|
  | `deepseek-v4-pro` | 0.60 | OK (après durcissement) | neutre « écart typique de rang » | non | baseline, plate |
  | `glm-5.2` | 0.55 | OK natif | non mentionnée (sûr) | non | la plus concise, deltas bien mis en forme |
  | `minimax-m3` | 0.50 | OK natif | non mentionnée (sûr) | **oui** (`all-in: -429g @10 vs +24g`) | narration la plus riche, exploite `context_benchmark` |
  | `kimi-k2.6` | 0.60 | OK natif | **la meilleure** (« marqueurs descriptifs de ton rang, pas des fautes ») | non | respecte le plus fidèlement l'asymétrie/règle 3 ; plus lent (timeout) |
  - **Verdict** : `kimi-k2.6` ≥ `minimax-m3` > `glm-5.2` > `deepseek-v4-pro`. Le durcissement du prompt est **model-agnostic** (les 4 honorent le schéma) → confirme la thèse projet « la qualité dépend du prompt+features, pas du modèle ». `minimax-m3` à privilégier si on veut exploiter le benchmark contextuel ; `kimi-k2.6` pour le respect maximal de l'asymétrie.
  - Catalogue Ollama Cloud récupéré via `GET https://ollama.com/api/tags` (35 modèles) ; pas de `kimi-k2.7` général (existe en `-code`), `kimi-k2.6` retenu.
  - Reste à faire : vérifier `OLLAMA_MODEL` du `.env` (correctif `1ca1973`) — poser `OLLAMA_MODEL=kimi-k2.6` et lancer sans `--model`.

## 🚧 Court terme — fermer la boucle d'évaluation (le vrai goulot)

- [x] **Scoring d'utilité** — boucle de feedback « ce conseil était-il juste / utile ? » ✅
  - Implémenté : `src/04_coaching/feedback.py` (CLI `annotate`/`summary`), schéma
    `Feedback`/`FeedbackItem` dans `schema.py`. CLI interactive par-insight (9 items,
    `y/n/s` + tag fixe `NEG_TAGS` + note sur faux), persiste
    `data/07_coaching/<player>/feedback.jsonl` (1 ligne/review, réannotation écrase par `ts`).
  - `summary` agrège : taux global, par section, top tags (signal actionnable pour durcir
    le prompt), par modèle, tendance (5 dernières vs précédentes, low_sample `<10`).
  - Spéc : `docs/superpowers/specs/2026-06-30-utility-scoring-design.md`.
  - Objectif atteint : pouvoir dire si le coach s'améliore (intrinsèquement vérifiable
    grâce au benchmark challenger, contrairement aux opinions absolues).
  - **Reste à faire** : nourrir la boucle (annoter les vraies reviews Spadzze au fil de
    l'usage), puis itérer le prompt sur les top tags dominants.
- [ ] **Compte-rendu par-game** (fin de partie) — incrément payload par-game, pas seulement agrégé.
  - Nécessite un payload par game (1 game → 1 review) en plus du payload agrégé N games.
  - Réutiliser `payload.build` en mode « single game » (ou un `build_one`).

## 🔧 Consolidation technique

- [ ] **Refactor `compare.py`** — exposer une fonction de données partagée plutôt que dupliquer la logique « delta saillant » entre `compare` et `payload._*_signals`. Recouvrement assumé aujourd'hui (consommateurs distincts) ; à mutualiser maintenant que les deux existent.
- [ ] **Industrialisation** — poursuivre la migration vers Pydantic + Parquet/DuckDB (cf. Notes de développement CLAUDE.md) : `04_dataset` Parquet, `05_model`, flux consolidé.
- [ ] **Robustesse ML/SHAP** — valider la qualité des **prescriptions SHAP vs heuristiques** (les features sont là, mais la pertinence des prescriptions reste à valider).

## 📊 Données — densifier si besoin

- [ ] **Benchmark Zeri** densifié (sampling champion ciblé) si la slice `zeri` reste trop fine pour des conseils fiables.

## 🌐 Dev web — interface Fly.io (FastAPI + front statique)

> Infrastructure scaffolding en place (`web/`, `Dockerfile`, `fly.toml`) et déployable.
> Voir `web/README.md` et spec à venir `docs/superpowers/specs/2026-07-01-web-app-design.md`.

- [x] **Infra scaffolding** — FastAPI (API `/api/*` + sert front statique) sur Fly.io.
  Smoke test local validé (`/api/health`, `/`, `/static`).
- [ ] **1. Cadrage fonctionnel V1** — périmètre : ce qu'on embarque maintenant vs plus tard.
  Décision prise : un compte de coaching = un **slug** (`spadzze`, `aceofspadzze`, smurfs…).
  Auth **reportée** (URL publique acceptée pour l'instant, faible proba de découverte).
- [ ] **2. Modèle de données + persistance** — entités (compte/slug, game, review, feedback),
  SQLite vs JSONL, mapping des dossiers `data/` actuels vers un schéma requêtable.
- [ ] **3. Stack front** — choix du framework d'interactivité (vanilla / HTMX / Alpine /
  Svelte/React) + Tailwind (styling). Décision avant les vues.
- [ ] **4. Architecture async** — jobs longs (fetch N games + coaching = minutes) :
  background task + polling, ou SSE, ou file. Shape l'UX (« en cours… »).
- [ ] **5. UX / vues** — wireframes + parcours utilisateur.
- [ ] **6. Infra Fly réelle** — volume persistant pour `data/` (exclu de l'image via
  `.dockerignore`) + secrets (`fly secrets set RIOT_API_ID OLLAMA_API_KEY`).
- [ ] **7. Code** — endpoints (`/api/accounts`, `/api/fetch`, `/api/coach`, `/api/feedback`)
  + vues frontend.

## 🟦 Phase 2 — CV / Live Client (gated)

- [ ] **Computer vision pour les trous** — uniquement si le coach basé timeline **démontre sa valeur** (boucle d'éval positive). Sinon le problème vient des features, pas de la vision.
  - Cibles : cooldowns exacts, skillshots loupés/touchés, micro-position entre frames 60 s, zone de caméra.
  - Piste maligne : rejouer depuis les **replays `.rofl`** en mode spectateur (Live Client API + caméra disponibles, sans impacter la game live).
  - ToS Riot : overlay en lecture seule, **aucune automatisation d'input**.

## ✅ Déjà fait (référence)

- Phase 1 validée (positionnement reconstruit sans vision, AUC dia_chall 0.655 → 0.724).
- Référentiels multi-rangs (~4454 games / patch 16.13, Diamond→Challenger).
- ML/SHAP industrialisé (Ensemble XGBoost/RF/EBM, dataset densifié ~7 873 rows).
- Macro-positionnement (17 features, manifeste COACHING_SAFE/ML_ONLY).
- **Narration LLM** (Ollama Cloud, `deepseek-v4-pro`, structured output, Review typée Pydantic, persistance `data/07_coaching/`). Asymétrie gardée bout-en-bout.