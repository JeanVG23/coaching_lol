# RiftSense — Coach IA personnalisé pour League of Legends

> Projet en phase d'industrialisation (pipeline ML). Ce document fixe vision, stack,
> schéma de données et état courant. Rien n'est figé.

## Vision

**Coach IA centré sur le joueur** (pas l'équipe entière), produisant un compte-rendu de
fin de game qui dépasse les stats classiques. Les outils existants (op.gg, u.gg…) s'appuient
sur des agrégats (KDA, gold, tourelles) → conseils pauvres et souvent faux (« meurs moins »
au lieu de « place-toi ici »). On capture le **positionnement et les déplacements réels**.

### Principes directeurs

1. **Positionnement > stats brutes.** Une carte des déplacements des 10 joueurs >> un KDA.
2. **Respect de l'asymétrie de l'information.** Le coach ne JAMAIS reprocher une décision
   basée sur une info que le joueur n'avait pas. On ne raisonne que sur l'info réellement
   disponible au joueur.
3. **Centré sur soi.** Une frame/minute suffit pour un avis sur l'équipe ; pour soi on veut
   une granularité plus fine (cooldowns, sorts loupés…).
4. **Le LLM ne voit pas la vidéo brute.** Il reçoit un **journal structuré** d'événements
   et d'états déjà extraits. L'extraction (API Riot) fait le travail ; le LLM raconte.
5. **D'abord le journal fiable, ensuite le coaching.**
6. **Riot-first.** API Riot post-game = source active ; capture live / CV = hors périmètre actuel.

## Source de données : Riot-first

Le projet actuel utilise une seule source : l'API Riot **post-game**, sans OCR ni capture locale.

- **Match-V5 + Timeline** (post-game) : positions x/y de tous les champions toutes les 60 s,
  gold/XP/items par joueur, et tous les events discrets (kills, objectifs, wards, level-ups,
  ordre de skill, achats). La **timeline** est le joyau.

La capture live et la CV sont un projet séparé, hors périmètre actuel.

### Accès API (clé prod « Coach_LoL_LLMs », 39 méthodes)

Clé de production (limites généreuses, pas d'expiration 24 h) → backoff poli sur 429 suffit.

| API | Endpoint clé | Rôle | Routing |
|-----|-------------|------|---------|
| account-v1 | `accounts/by-riot-id/{gameName}/{tagLine}` | Riot ID → `puuid` (porte d'entrée) | **régional** |
| match-v5 | `matches/by-puuid/{puuid}/ids`, `matches/{id}`, **`matches/{id}/timeline`** | Cœur du MVP | **régional** |
| league-v4 | `entries/by-puuid/{puuid}` | Elo/LP (rend summoner-v4 inutile) | **plateforme** |
| lol-challenges-v1 | `player-data/{puuid}`, `challenges/percentiles` | Bonus benchmarking profil | plateforme |
| champion-mastery-v4 | `champion-masteries/by-puuid/{puuid}` | Profilage (main vs pick), Phase 2 | plateforme |

**Piège routing** : account-v1 + match-v5 = régional (`europe`/`americas`/`asia`) ;
league-v4 / mastery = plateforme (`euw1`…).
**Hors scope** : summoner-v4, clash-v1, tournament-*, lol-status-v4, champion-v3.

### Les deux niveaux d'information (CRITIQUE pour l'asymétrie)

- **« Ce qui peut être prescrit »** — uniquement les features exactes et vérifiables depuis
  la timeline (`COACHING_SAFE`), jamais un proxy statistique présenté comme un reproche.
- **« Ce qui reste statistique »** — `ML_ONLY` (proxys flous) sert le modèle, mais n'est jamais
  injecté comme conseil actionnable.
- **« Ce qui sert à labelliser »** — la timeline complète et les benchmarks servent à comparer
  et labelliser après coup ; ils ne sont jamais présentés au LLM comme une connaissance
  que le joueur avait au moment T.

## Stack cible

Langage principal : **Python** (écosystème ML). Lancer depuis la racine, dans
l'environnement Poetry (`poetry shell` ou préfixer `poetry run`) :
`python3 src/<dossier>/<script>.py` — chaque script insère `src/core/` dans `sys.path` avant
`import riotlib` (convention flat-import, pas de package Python dans `src/`).

| Brique | Techno |
|--------|--------|
| Données de jeu | API Riot : Match-V5 + Timeline |
| Validation schémas | Pydantic |
| Stockage | Parquet/JSONL en local ; Cloudflare KV pour les données web publiées |
| Analytics local | DuckDB |
| Extraction structurée | Ollama local ou Cloud, structured output JSON (schéma imposé, T° basse) |
| Synthèse narration | Petit modèle local pour l'extraction ; modèle plus gros (API) pour la narration nuancée |

**À éviter (sur-ingénierie)** : capture live / CV ; scraping
YouTube ; gros fine-tuning de LLM ; modèle supervisé d'erreurs sans dataset labellisé
(heuristiques déterministes d'abord) ; Postgres (DuckDB suffit) ; full video understanding.

## Séquencement par phases

- **Phase 1 — Coach 100 % API, zéro vision (MVP)** ✅ : N dernières games via Match-V5 →
  features macro/positionnement → Ollama → compte-rendu. Valider « le coach est-il utile ? »
  avant d'investir dans un autre projet. S'il est mauvais, le problème vient des features,
  pas de la source de données.
- **Phase 2 — capture live / CV** : **hors périmètre actuel** (projet séparé ; aucune
  intégration, aucun code de capture).
- **Phase 3 — ML / spécialisation (si justifié)** : heuristiques → ML supervisé seulement
  une fois des labels accumulés. Pas de fine-tuning avant d'en prouver le besoin.

## Le levier de qualité : les features, pas le LLM

La qualité du rendu final dépend à ~90 % de la **couche de features**, pas du modèle. Le LLM
ne fait que raconter ce que les features ont conclu. Investir là.

- **Coaching relatif à un benchmark challenger, pas absolu.** « Tu recall à 1450 g en moyenne,
  les challengers à 1100 » est concret et vérifiable ; « recall plus tôt » est creux. Benchmarks
  = timelines high-elo de l'API Riot.
- Features macro à fort signal : proximité aux objectifs avant spawn, timing de recall vs
  état de wave, **morts en fog vs morts en vision**, indice d'overextension (distance à la
  tour la plus proche × ennemis non visibles), gold dead time, diff CS/XP par minute.

## Schéma de dataset

Principe : **pas « une ligne = une game »**. Au moins 4 tables/fichiers logiques.

- **`games`** — métadonnées globales : patch, champion, rôle, durée, résultat, side, elo.
- **`states_timeline`** — états échantillonnés depuis la timeline Riot : `game_id`,
  `timestamp_ms`, positions x/y, gold/XP/level/items, wards et événements discrets.
  Aucun champ HP/mana/cooldown/caméra : ce besoin appartient au projet séparé capture live / CV.
- **`events`** — événements discrets : `game_id`, `timestamp_ms`, `event_type`,
  `actor` (self/ally/enemy_visible), `zone`, `confidence`, `payload_json`.
- **`reviews`** — labels et résumés : erreurs détectées, bons moves, scores
  lane/macro/vision, résumé final généré.

## Pipeline de résumé

1. Récupération via l'API Riot post-game.
2. Conversion en événements et états structurés.
3. Agrégation en **features haut niveau** benchmarkées challenger, ex. : « 3 pushes sans
   vision en side lane », « 2 recalls tardifs avant drake (challengers : -350 g plus tôt) »,
   « forte discipline de reset après crash », « sorts majeurs souvent lancés sans setup ».
4. Envoi du résumé structuré au LLM.
5. Sortie LLM imposée par schéma JSON : `strengths[]` (3) / `mistakes[]` (3) / habitudes à
   corriger (2) / `next_focus[]` (1) / `confidence` ; chaque `strength`/`mistake` porte sa
   **preuve chiffrée** (`evidence` par point, fusionné).

## Pistes de coach envisagées

- **Coach champ select** — picks/bans et matchups.
- **Coach in-game / fin de game** — axe principal : journal agrégé + compte-rendu.

## Évaluation

Prévoir **dès le départ** une boucle de feedback (« ce conseil était-il juste / utile ? »).
Le coaching benchmarké challenger est intrinsèquement plus vérifiable que les opinions
absolues du LLM.

## Architecture du code (médaillon, numérotée pour l'ordre du pipeline)

Code dans `src/`, données dans `data/` (couches numérotées). `src/` rangé par rôle (pas de
vrac à la racine) : `core/` (libs partagées), `collection/` (appels API Riot),
`pipeline_ops/` (maintenance médaillon, 0 appel API), `reporting/` (livrable heuristique
pré-ML), `experiments/` (spikes historiques), puis `01_data_engineering` → `04_coaching`
(pipeline ML). `tests/` (pytest) couvrent la dérivation déterministe + extraction comp +
agrégation contextuelle. Lancer : `poetry run pytest tests/`.

```
src/
  core/           riotlib.py, positioning.py, champion_profiles.py, game_journal.py,
                  journal_signals.py, turrets.py, ml_features.py,
                  ranks.py, role_features.py, cli.py, kv_keys.py, dataset_split.py, ml_rank.py,
                  ebm_explain.py, settings.py
  collection/     build_referential.py, aggregate_games.py, sync_cloudflare.py,
                  refresh_cloudflare.py, pipeline.py, densify_targets.py, densify_sweet_spot.py,
                  densify_players.py, fetch_apex_lp.py, fetch_ladder.py,
                  backfill_summoner_profiles.py
  pipeline_ops/   reextract_silver.py, rebuild_gold.py, build_turret_map.py, compress_raw.py,
                  archive_patch.py, list_unknown_champions.py, dataset_report.py,
                  role_readiness.py
  reporting/      compare.py
  experiments/    phase1_pull.py
  01_data_engineering/  build_dataset.py, build_player_dataset.py, build_player_lp_dataset.py,
                        build_role_player_dataset.py, build_split.py, build_sequence_dataset.py
  02_data_science/      train_ensemble.py, calibrate_rank.py, train_player_ensemble.py,
                        train_role_ensemble.py, train_player_lp.py, calibrate_player_rank.py,
                        analyze_auc_vs_ngames.py,
                        lp_metrics.py, audit_leakage.py, poc/per_player_hypothesis.py,
                        sequence_model.py, sequence_data.py, train_sequence_model.py,
                        pretrain_sequence_model.py, cv_common.py
  03_data_analyse/      shap_analysis.py, plotter.py
  04_coaching/          payload.py, prompt.py, schema.py, llm_client.py, coach.py, feedback.py,
                        grounding.py, counterfactual.py
data/
  00_static/      champion_traits.json, sr_turrets.json (force-add : config source),
                  ddragon/<version>/ (ignoré)
  01_raw/         JSON API brut compressé .json.zst (~10 Go -> ~750 Mo, ×13). Lecture/écriture
                  transparentes via riotlib._read_raw/_write_raw (tolérante .json.zst->.json.gz->.json)
                  rank_snapshots/<plateforme>/<jour>.jsonl.zst + manifest.json : ladder daté
  02_silver/{referentiel/<rank>,personal/<player>}/games.jsonl   # 1 ligne = 1 game nettoyée (+ comp)
  03_gold/{referentiel/<rank>,personal/<player>}/<scope>/aggregate.json   # agrégats benchmarks
  04_dataset/     adc_dataset.parquet, densify_targets.json, datasets per-player/LP,
                  <role>_dataset.parquet + <role>_player_dataset.parquet (+ .meta.json de provenance)
  05_model/       modèles ML + metrics (xgb_highelo.pkl, player_metrics.json, player_lp_metrics.json,
                  rank_calibration.json, auc_vs_ngames.{json,png}) ; par rôle :
                  <role>_player_ebm.pkl, <role>_player_metrics.json, role_readiness.json
  06_shap/        player/high_elo/ + game/dia_chall/ : analyse EBM glass-box unifiée
                  (ebm_shape_functions.json = seuils de bascule, ebm_ranking, cross-check
                  SHAP-arbres vs EBM, diagnostics LOWESS, visuels)
  07_coaching/<player>/reviews.jsonl + feedback.jsonl
web/
  cf/             Worker TypeScript de production : API, assets, KV, Ollama SSE
                  (src/http.ts = réponses d'erreur + pagination partagées)
                  src/auth.ts = mot de passe COACH_AUTH_PASSWORD + cookie HMAC (30 j)
                  src/coach_gate.ts = Durable Object de verrou, une instance par joueur
                  src/game_coach.ts = coaching d'UNE partie (POST /api/coach/game)
                  src/coaching_context.ts = scopes globaux/par rôle, fraîcheur des bilans
                  src/curation.ts = désignation de la partie pédagogiquement utile
  cf/client/      SPA Vite/Vue Router et composants Vue TypeScript
                  ingest-sync.ts = composable unique de rafraîchissement (bouton hero +
                  onglet SHAP : même job, même cooldown, une instance dans AccountPage) ;
                  role-analysis.ts = contrat shap:{slug}:role (union discriminée available,
                  un message typé par motif, logit affiché en proximité à l'apex, jamais
                  converti en probabilité ni rang) ; shap-view.ts = regroupe les drivers EBM
                  par feature (`groupByBase`) et par thème (`groupByTheme`), thème porté par
                  la clé `theme` de `shared/feature_catalog.json` (vocabulaire fermé miroir
                  de `THEMES` dans tests/test_feature_catalog.py : economy/positioning/
                  vision/fights/objectives/context, un thème inconnu ou vidé casse le test).
                  L'onglet ML rend ces groupes en DOM natif (jauge, verdict, lignes de
                  thème) : Chart.js est retiré du bundle (`web/cf/package.json`) et
                  ShapProfile.ce.vue n'a plus de `<canvas>`.
                  Icônes : aucun emoji dans le client (chaque OS les dessine à sa façon,
                  hors palette). `icons.ts` recopie les seuls tracés Lucide utilisés,
                  rendus par `<AppIcon>` en `currentColor` ; les objectifs (drakes,
                  Baron, Atakhan…) sont des masques monochromes CSS sur
                  `public/images/objectives/*.png`. `championKey` (game-history.ts) est
                  l'unique table des noms Data Dragon. `tests/web/test_frontend.py`
                  échoue si un emoji réapparaît (glyphes typographiques ✦✓✕✗✎ permis).
                  game-timeline.ts = frises, côté vue. Celle d'UNE partie place un repère
                  par instant cité (jamais un insight planté à zéro faute de date) ;
                  `PHASE_BANDS` découpe en minutes ABSOLUES (0-5 / 5-15 / 15-25 / 25+) parce
                  qu'agréger N parties de durées différentes interdit le tiers relatif.
                  `dominantCategory`/`dominantPhase` ne désignent rien à égalité (une review
                  de 5 erreurs porte souvent 5 catégories distinctes ; « X domine : 1 sur 5 »
                  se lirait comme une conclusion), et la vue globale publie pour la même
                  raison une CONCENTRATION (`denseSpan` : la plus courte fenêtre contiguë
                  tenant 70 % des erreurs, « 113 sur 139 entre 5 et 25 minutes ») plutôt
                  qu'un classement, laning (55) et mid (58) étant à égalité de fait ;
                  coaching-recurrence.ts = grille de présence de la vue globale, une colonne
                  par partie dans l'ordre du temps, alimentée par `summary.categories` que
                  le Worker publie dans la liste légère des reviews.
                  ⚠️ `src/game_moments.ts` est le SEUL endroit où la regex d'horodatage est
                  écrite : le Worker en dérive `summary.moments` (la liste légère porte les
                  instants, donc la vue globale superpose N frises sans relire N détails) et
                  le client la réexporte via `game-timeline.ts`. `tests/web/test_frontend.py`
                  échoue si `[0-5]\d` réapparaît ailleurs.
                  ⚠️ Le coaching est un COMPLÉMENT de l'onglet ML, pas son doublon : il ne
                  reprend AUCUN de ses idiomes visuels (barres de contribution, jauge,
                  colonnes forces/leviers). Servir un classement à barres des deux côtés
                  ferait passer un comptage d'occurrences pour un poids de modèle. Le
                  coaching a l'axe que l'EBM n'a pas, le TEMPS : frise par partie, grille
                  chronologique en global. `tests/web/test_frontend.py` échoue si un de ces
                  idiomes revient dans GameReviews.vue ou GlobalCoaching.ce.vue.
                  Les insights sont repliés au niveau 1 (catégorie + titre + horodatages) :
                  déplié, un lot de 6 posait ~2 100 caractères d'un bloc. Le repli et l'axe
                  de temps vivent dans `styles/coaching-insights.css` (`.shared-insights
                  .insight-toggle`, `.coach-timeline .timeline-axis`), pas en scoped dans un
                  seul composant : deux gestes différents sur deux onglets de la même
                  fonctionnalité coûteraient plus cher que le texte économisé.
  cf/public/      assets statiques copiés tels quels par Vite
service/          Service d'ingestion Cloud Run (Flask + gunicorn, 1 worker/1 thread) :
                  app.py (routes + secret partagé), riot_ingest.py (métier : profil,
                  préflight, fenêtre du rôle), role_scoring.py (modèles par rôle),
                  storage.py (R2), errors.py (codes stables dérivés du TYPE).
                  L'image copie src/core/, service/, data/00_static/ et les cinq
                  exports EBM : tout code appelé depuis le service DOIT vivre dans
                  src/core/, sinon il est introuvable en production.
shared/         prompts/*.txt (source de vérité, lue par prompt.py ET par le Worker)
                + schemas/*.json (générés depuis Pydantic)
config/           accounts.json (ignoré, données perso) + accounts.example.json (gabarit)
                  role_corpus.json (VERSIONNÉ : certification d'ouverture par rôle)
```
⚠️ `web/backend/` (FastAPI, ère Fly.io) a été SUPPRIMÉ le 2026-09-04. Les trois modules
qui n'étaient pas du serving et que la collecte locale utilise toujours ont migré :
`ml_rank.py` et `settings.py` → `src/core/`, `pipeline.py` → `src/collection/`. Le reste
(main/jobs/readers/routers) ne servait que l'ancienne app ; l'historique git le garde.
⚠️ `data/` est gitignoré SAUF `data/00_static/champion_traits.json` (force-add : config source).
⚠️ **`COACHING_DATA_DIR`** déplace toute la pile médaillon (`riotlib.DATA` ET
`champion_profiles.STATIC_DIR`, sans quoi la démo lirait les catalogues réels). C'est ce qui
permet à `make demo` de rejouer les scripts de production sur `tests/fixtures/demo/` sans
toucher aux données réelles. Lue à l'import : côté tests, la fixture `demo_data` substitue
les attributs de module (portée fonction — une redirection qui survit à son test contamine
les suivants).
⚠️ Les chemins des couches sont définis dans `src/core/riotlib.py` (`RAW_DIR`/`SILVER_DIR`/`GOLD_DIR`).
Renommer un dossier data SANS mettre à jour le code → le code recrée l'ancien.

### Modules `core/`

Détail par fichier (`riotlib.py`, `positioning.py`, `game_journal.py`,
`champion_profiles.py`, `ml_features.py`, `ranks.py`, `cli.py`, `kv_keys.py`,
`ebm_explain.py`) : **`src/core/CLAUDE.md`**. À retenir au niveau projet :
`ranks.py` est la source unique des rangs/cibles (déplacer la frontière de rang
se fait ICI, pas dans les ~11 scripts qui la consommaient avant) ; `positioning.py`
sépare mécaniquement `COACHING_SAFE` (14 features → ML + coaching) de `ML_ONLY`
(3 proxys vision, jamais prescrits) ; `game_journal.py` ne restitue que l'info
que le joueur avait (aucun proxy `ML_ONLY`).

### Modules `collection/`

- **`build_referential.py`** — collecte les benchmarks par rang (league-v4/-exp-v4).
  `python3 src/collection/build_referential.py --region euw1 [--rank R] [--players N]`.
- **`aggregate_games.py`** — pipeline perso : N games → silver + gold (all/adc).
- **`sync_cloudflare.py`** — publication des agrégats, rangs, prédictions ML, SHAP, reviews et
  feedbacks locaux vers Cloudflare KV. Fusionne l'historique distant et supporte `--dry-run`.
  `--seed-reviews` n'amorce les reviews que si la clé est absente ; `--push-coaching` fusionne
  reviews + annotations locales dans KV par `ts` (`merge_jsonl` : la ligne distante écrite
  depuis le site survit, la ligne locale gagne sur un `ts` commun) — sans lui, les annotations
  CLI resteraient invisibles du taux publié. Publie aussi le **bundle de payloads unitaires**
  (`payload.build_game_bundle`, 50 parties les plus récentes, garde-fou 20 Mio,
  `--skip-game-payloads` pour sauter l'étape) : le Worker n'a pas accès au raw Riot, donc le
  coaching par-game du site ne peut exister que si ce bundle est construit localement.
- **`densify_targets.py`** — sélection **chirurgicale** des joueurs à densifier vers le sweet
  spot ~30 games/joueur (cf. `analyze_auc_vs_ngames.py`). 0 API : relit `adc_dataset.parquet`
  (comptage par joueur sur le référentiel double-ADC), cible la bande `[--min-games, --threshold[`,
  trie par écart croissant, `--exclude-ranks` pour écarter diamond (frontière apprise =
  challenger vs master ; densifier diamond pousse la classe low loin du boundary = bruit). Écrit
  `data/04_dataset/densify_targets.json`, consommé par `densify_players.py --target-list`.
- **`densify_sweet_spot.py`** — orchestrateur one-command : bake-in `[15,30[` hors diamond, tri
  par gap croissant → `densify_targets.json` → chaîne vers `densify_players.py --target-list`.
  Dry-run par défaut ; `--run` lance le scraping. Usage :
  `poetry run python3 src/collection/densify_sweet_spot.py --run --history 60`.
- **`densify_players.py`** — reprend une liste de joueurs (tous ceux d'un rang, ou `--target-list`)
  et va chercher leur historique de matchs supplémentaire (`--history`, `--days`) ; dédup par
  match_id connu, arrêt anticipé par joueur dès que `gap` games ADC neuves trouvées, checkpoint
  silver+gold périodique.
- **`fetch_apex_lp.py`** — LP courant horodaté (3 appels API) pour la régression LP.

### Orchestration (`Makefile`)

Le pipeline n'est pas une suite de commandes à lancer de tête : c'est un graphe de
dépendances déclaré dans le `Makefile`. Chaque étape dépend des artefacts qu'elle lit ET du
code qui la produit (`src/core/*.py` en bloc), donc toucher `positioning.py` périme silver →
gold → datasets → modèles. `make plan` répond « qu'est-ce qui est périmé ? » en déléguant à
`make -n` (la seule réponse honnête). `make graph` affiche le DAG.
`make analyse` (inclus dans `make pipeline`) régénère l'analyse EBM glass-box des deux
niveaux via `core/ebm_explain.py` : le niveau player explique le modèle servi, le niveau
game re-entraîne d'abord le modèle d'explication du jeu-type
(`train_ensemble.py --target dia_chall`, jamais servi pour le rang).
`make roles` déroule la chaîne par rôle (datasets per-game, fenêtres à N fixe,
modèles, table d'ouverture) et fait partie de `make pipeline`, donc de `make plan` :
sans ce câblage, toucher `role_features.py` ne périmait rien et les cinq modèles se
relançaient de tête. Le snapshot du ladder est une dépendance FICHIER (jour le plus
récent marqué complet au manifeste, `make print-SNAPSHOT_DAY` pour le lire) : une
nouvelle capture périme mécaniquement les cinq datasets, et un snapshot absent renvoie
vers `make ladder` au lieu d'un « No rule to make target ».
⚠️ **Les étapes réseau ne sont JAMAIS des dépendances** : `collect` (Riot), `lp-label`
(fetch_apex_lp), `ladder` (fetch_ladder) et `sync`/`sync-push` (Cloudflare) sont des
cibles explicites, et `tests/test_pipeline_graph.py` échoue si l'une d'elles devient un
prérequis de `pipeline`.
⚠️ `01_raw` grossit hors du graphe : le témoin `data/.stamps/raw` est réévalué à chaque
invocation par `find -newer … -print -quit`. C'est le seul nœud dont l'amont n'est pas
produit par make.
`make collect` remplace `run_scraping.sh`/`run_uniform_scraping.sh` (supprimés le
2026-09-04) : mêmes appels, paramètres en variables (`RANKS`/`PLAYERS`/`GAMES`/`ROUNDS`).
CI : `ci.yml` (push/PR, lock gelé) + `weekly.yml` (cron lundi, dépendances résolues SANS le
lock puis `make demo`) : sur lock gelé, un cron ne vérifierait rien de plus que la CI.

### Modules `pipeline_ops/` (0 API)

- **`reextract_silver.py`** — ré-extrait le silver depuis le raw caché ; à relancer après toute
  évolution d'`extract_game`.
- **`rebuild_gold.py`** — régénère tout le gold depuis le silver.
- **`compress_raw.py`** — migration one-shot : `01_raw/*.json` → `.json.zst` (vérification
  roundtrip avant suppression). Idempotent, `--dry-run`.
- **`list_unknown_champions.py`** — scanner : champions du silver absents de la table curée,
  triés par fréquence (pour compléter `champion_traits.json` au fil de l'eau).
- **`dataset_report.py`** — état des lieux des datasets ML en une commande (`--json` pour
  comparer entre densifications) : volumes per-game, profondeur games/joueur (seuils
  ≥5/10/15/20/30, qualifiés per-player par rang résolu au mode), fenêtre temporelle (patch +
  âge newest/médian/oldest), composition per-player (dominance intra-classe : high ≈ 81 %
  challenger, low ≈ 73 % master → frontière réelle ≈ master vs challenger), cross-check
  `player_metrics.json` (⚠ DÉRIVE si le modèle servi n'est plus entraîné sur l'effectif courant).
  À relancer après chaque densification/rebuild.
- **`archive_patch.py`** — archive raw/silver/gold/dataset du patch courant avant de passer au suivant.
- **`build_demo_fixtures.py`** — fabrique `tests/fixtures/demo/` (cible `make demo`) : 49 parties
  réelles **pseudonymisées** + amorce silver + Data Dragon élagué aux seuls champs lus.
  Deux natures d'identifiants, deux traitements, et c'est le point à ne pas simplifier :
  les **opaques** (`puuid`, `summonerId`, `matchId`) sont remplacés partout par balayage
  récursif (un champ oublié = une fuite) ; les **pseudonymes** uniquement aux clés qui les
  portent, parce qu'un joueur nommé « Aatrox » ferait réécrire le `championName` de toutes
  les games par un remplacement global. `audit()` relit ce qui a été écrit et échoue sur
  survivance ; `tests/test_demo.py` rejoue ce contrôle à chaque run.
- **`generate_shared.py`** : sérialise les artefacts partagés entre les deux runtimes,
  `shared/prompts/*.txt` + les schémas Pydantic -> `shared/schemas/*.json` et
  `web/cf/src/generated/shared.ts` (committé, pour que `wrangler deploy` ne dépende
  d'aucun runtime Python). `make generate-shared`. La parité est vérifiée par
  `tests/test_shared_contract.py`, qui compare le fichier généré sur disque à ce que
  le générateur produit : un prompt modifié sans régénération échoue en CI.

### Modules `reporting/` et `experiments/`

- **`compare.py`** — livrable coaching : slice perso vs référentiels, à issue égale. Section
  **benchmark conditionné** par contexte de lane (`context_benchmark`, repli `MIN_CONTEXT_N=8`
  loggué, `unknown` exclu du bucket dominant). Section **benchmark positionnement** (`POS_ROWS`,
  14 features COACHING_SAFE, médianes à issue égale) avec garde-fou asymétrie en `assert` au
  chargement (toute feature ML_ONLY → crash). Note prescriptive sur le sens contre-intuitif de
  la profondeur.
- **`phase1_pull.py`** — spike : détail visuel d'UNE game (déplacements/minute + morts).

### Pipeline ML

- **`01_data_engineering/`** : `build_dataset.py` consolide en table tabulaire ML-ready (Parquet).
  **1 ligne = 1 ADC d'une game.** Le référentiel ré-extrait **les DEUX ADC de chaque game depuis
  le raw** (0 API) — le silver ne stocke qu'un joueur ciblé par game, s'y limiter ne récupérait
  l'ADC que des games où le ciblé était ADC (~3 088 rows) ; en relisant le raw on densifie à
  ~games×2 (≈ 7 873 rows). Colonnes méta temporelles `patch`/`game_ts` pour `dataset_report.py`.
  > ⚠️ **FLAW ASSUMÉ — transfert de rang.** Le rang d'une game = rang de collecte du joueur
  > ciblé, transféré **aux deux ADC** en supposant un **MMR égal dans le lobby** (vrai en solo
  > queue high-elo). L'ADC ennemi n'a donc pas son rang réel mesuré. Acceptable pour un classif
  > high/low ; à revoir si on descend en elo. Games multi-rangs : rang résolu au **mode**,
  > tie-break sur le rang le plus bas.
  > ➜ La chaîne PAR RÔLE (plus bas) ne porte plus ce flaw : elle labellise chaque joueur
  > par un snapshot daté du ladder, pas par le rang de collecte de son lobby.
  `build_player_dataset.py` — 1 ligne = 1 joueur ≥`MIN_PLAYER_GAMES` games ADC référentiel,
  agrégées sur la totalité de l'historique disponible. `build_player_lp_dataset.py` — per-player
  SANS balance-cap, apex seulement (diamond exclu — LP non comparable).
- **`02_data_science/`** : `train_ensemble.py` — classif High-Elo vs Low-Elo via **ensemble à 3
  biais inductifs** (XGBoost=GBDT, Random Forest=bagging, EBM=GA²M glass-box). SHAP moyen sur
  les 2 arbres ; EBM = validateur indépendant + interactions par paires.
  `calibrate_rank.py` — calibration proba→rang (`src/core/ml_rank.py`) : modèle `high_elo`
  binaire (M/D vs GM/C), on calibre la proba moyenne ensemble (xgb+rf) par rang réel sur le
  référentiel (`data/05_model/rank_calibration.json`), puis place le joueur au rang calibré le
  plus proche de sa proba moyenne sur ses dernières games ADC.
  `train_player_ensemble.py` — ensemble xgb/rf/ebm, **purged CV** (folds joueurs StratifiedKFold
  + agrégats de train recalculés en excluant les matchs joués par un joueur de val ; ~37 % des
  games des qualifiés opposent 2 joueurs du dataset — features en miroir — et le graphe des games
  partagées est une composante géante à 98.7 %, donc group-CV par composantes impossible ; une
  passe contrôle isole la fuite pure). `auc_cv` = purgée (headline honnête), `auc_cv_naive`/
  `auc_cv_control` dans `player_metrics.json`. `analyze_auc_vs_ngames.py` — sweep du cap N
  games/joueur (label fixe sur l'historique complet, purged CV) pour calibrer `MIN_PLAYER_GAMES`.
  `train_player_lp.py` — régression LP per-player (apex tiers, ensemble xgb/rf/ebm REGRESSORS,
  random search graine fixe en purged CV précalculée par fold, sélection au Spearman pooled OOF).
  `calibrate_player_rank.py`, `lp_metrics.py` (Spearman pooled/by-tier + RMSE, garde anti-NaN
  `_safe_spearman`), `audit_leakage.py` (diagnostic OOF/AUC), `poc/per_player_hypothesis.py`
  (hypothèse constance, repris en prod par `src/core/ml_rank.py`).
- **`03_data_analyse/`** : `shap_analysis.py` (CLI fine `--level {player,game}` sur le moteur
  `core/ebm_explain.py` : ranking + shape functions exactes avec seuils de bascule,
  interactions par paires au niveau game, cross-check SHAP-sur-arbres, visuels et
  diagnostics LOWESS via `plotter.py` ; drivers Spadzze au niveau game uniquement).
- **`04_coaching/`** : narration LLM (Ollama Cloud, structured output). Pipeline,
  modules (`payload`/`prompt`/`schema`/`llm_client`/`coach`/`feedback`), boucle
  d'éval (`grounding`/`counterfactual`/`model_ab`) et historique A/B modèles :
  **`src/04_coaching/README.md`** (documentation détaillée du dossier). Ce qui
  suit est ce que le README ne couvre pas.
  - **Prompts/schémas partagés** : le texte des prompts vit dans
    `shared/prompts/*.txt` (`prompt.py` le lit et strip le newline final,
    exactement comme le Worker via `web/cf/src/generated/shared.ts` : un seul
    texte pour les deux runtimes). `review_json_schema()`/`game_review_json_schema()`
    renvoient le schéma JSON **strict** inliné (`$defs` résolus, pas de
    `title`/`description`, `additionalProperties: false`) consommé aussi par le
    Worker. `schema_version_of`/`REVIEW_SCHEMA_VERSION`/`GAME_REVIEW_SCHEMA_VERSION`
    = empreinte sha256 du schéma, comme `prompt.version_of` pour le prompt.
    ⚠️ `PROMPT_VERSION`/`GAME_PROMPT_VERSION` sont figées par
    `tests/test_shared_contract.py` : `web/cf/src/coaching_context.ts` compare
    le `prompt_version` d'une review à la version courante pour classer
    `ready`/`stale` côté web — un hash qui bougerait sans le vouloir périmerait
    silencieusement cette classification.
  - **Chemin par-game** : `payload.build_game` (journal `game_journal` +
    repères référentiel à issue égale ; recalls enrichis d'items résolus,
    coût CS, spike fini/adverse et mort consécutive ; morts enrichies des derniers
    signaux publics du jungler, du support et des tourelles encore debout ; bloc `context` = comp
    botlane/jungle/mid + `lane_pattern`/`gank_exposure` via `derive_context`),
    `prompt.SYSTEM_GAME` (règle matchup basée sur ce `context` + règle de gold
    relatif au prochain achat de chaque recall), `coach.py --game
    [latest|MATCH_ID]` (records `kind: "game"` + `match_id`), `coach.py
    --game-batch [N]` (défaut 10 : reviews des N dernières games ADC pas
    encore reviewées, dédup par `match_id`, poursuit sur échec).
    La sortie par-game porte une catégorie fermée et un titre court par insight,
    accepte 1 à 5 erreurs et tente au plus 3 validations de schéma. Le feedback est
    ventilé par catégorie et le contrefactuel `no_opponent_spike` vérifie que le
    coach cesse de citer un spike retiré.
  - **Chemin par-game côté web** : `payload.build_game_bundle` sérialise ces
    payloads (clé KV `riftsense:{slug}:game-payloads`, `payload_hash` +
    `benchmark_scope` = rôle sinon global) ; `web/cf/src/game_coach.ts`
    relit l'entrée demandée, réutilise une review existante sans appel LLM et
    ne régénère que sur `force`. Les motifs d'indisponibilité publiés
    (`raw_missing`/`benchmark_missing`/`not_eligible`) sont dérivés du **type**
    d'exception (`RawMissing`/`BenchmarkMissing`/`GameNotEligible`), jamais du
    texte du message. ⚠️ Les trois chemins LLM du site (`/api/coach`,
    `/api/coach/game`, `/api/chat`) sont derrière `auth.ts`
    (`COACH_AUTH_PASSWORD`) et sérialisés par joueur par le Durable Object
    `CoachGate` (verrou tenu jusqu'à la fermeture réelle du flux SSE). Le
    client Ollama du Worker streame (`stream: true`) : sans streaming, une
    génération > 125 s finit en HTTP 524 (Ollama Cloud est lui-même derrière
    Cloudflare).
  - **Publication du taux d'éval** : recalculé À LA LECTURE côté site
    (`web/cf/src/evaluation.ts`, `GET /api/c/<slug>/eval`), jamais poussé
    précalculé (un blob figé au sync serait périmé dès la première annotation
    laissée depuis le web). Seuils (`_OBJECTIVE_N=10`, `_OBJECTIVE_RATE=0.70`)
    verrouillés entre les deux runtimes par `tests/test_eval_parity.py`.
  Lancer : `python3 src/04_coaching/coach.py --player spadzze --scope adc
  [--game|--game-batch N]`. Aucun réseau côté `grounding`/`feedback`.

### Chaîne par rôle (le modèle public explicable)

Chaîne distincte du per-player ADC ci-dessus, et réponse directe au FLAW ASSUMÉ du
transfert de rang : le label ne vient plus du dossier silver (rang de COLLECTE du
joueur ciblé, recopié sur tout le lobby) mais d'un **snapshot daté du ladder**.
Entrées : `make ladder` (réseau), puis `make roles` (hors-ligne).

- **`collection/fetch_ladder.py`** (réseau) : un fichier par jour dans
  `data/01_raw/rank_snapshots/<plateforme>/`, lisible seulement après inscription
  `complete: true` au manifeste. Une pagination interrompue ne peut donc pas passer
  pour un ladder entier ; `load_snapshot` refuse un jour incomplet et recompte les
  lignes avant de rendre la main.
- **`core/role_features.py`** : manifeste de features PAR RÔLE. Plusieurs entrées de
  `ml_features` n'ont pas de sens hors botlane (les 2v2 reposent sur la paire bot,
  `support_deaths_early` est structurellement nul pour un support) : les servir
  publierait des explications fausses. Trois catégories de publication :
  `HIDDEN_ML_ONLY` (proxys, jamais affichés), `PUBLIC_DESCRIPTIVE` (affiché, jamais
  formulé en levier), `PUBLIC_ACTIONABLE`.
  ⚠️ L'EBM public est **entraîné** sur `public_features`, pas filtré à l'affichage :
  retirer un terme après coup casserait l'identité « somme des contributions = score ».
- **`01_data_engineering/build_dataset.py --role X`** : per-game NON labellisé du rôle
  (les deux joueurs du rôle relus depuis le raw, 0 API).
- **`01_data_engineering/build_role_player_dataset.py --role X --snapshot-day J`** :
  fenêtres à profondeur FIXE `(puuid, role, as_of)`, N=20, jointes au snapshot. La
  profondeur fixe est le point à ne pas assouplir : à N variable, le volume
  d'historique collecté devient un proxy du rang et passe dans les statistiques de
  dispersion. Écrit aussi un sidecar `.meta.json` (jour du snapshot, effectifs par
  tier, `label_age_days` min/p50/max, empan temporel des parties agrégées).
- **`02_data_science/train_role_ensemble.py --role X`** : purged CV (agrégats de train
  recalculés en excluant les matchs joués avec un joueur de validation) + held-out
  stratifié **répété sur 10 graines**, EBM `interactions=0`.
  ⚠️ `DEFAULT_BOUNDARY = "diamond"` (DIAMOND vs GM+CHALLENGER) : MASTER est
  inséparable de DIAMOND dans cet espace de features (sonde du 2026-09-18, cinq
  rôles : DIAMOND vs MASTER seul rend 0.53 à 0.61 d'AUC, soit le hasard), et la
  classe limitante de `_balance` reste GM+CHALLENGER dans tous les cas. Collecter
  des Diamants ou des Masters n'augmente donc PAS l'effectif d'entraînement.
  ⚠️ Le held-out ne se lit JAMAIS sur un tirage : à 28 joueurs, le même modèle sur
  les mêmes données rend 0.566 ou 0.918 selon les personnes tirées. Les métriques
  publient `auc_heldout_median` + `auc_heldout_seeds` (un tirage par graine), et
  PAS `auc_heldout` : garder le nom en y écrivant la médiane servirait une
  statistique sous le nom d'une autre. Le sens d'un driver vient de la shape function
  (décile p90 moins décile p10), jamais de la moyenne des contributions rendues par
  `eval_terms` : elles sont CENTRÉES, donc leur moyenne vaut zéro et son signe est du
  bruit numérique. Reporte le sidecar de provenance dans les métriques.
- **`pipeline_ops/role_readiness.py`** : décide l'ouverture publique.
  `marge = médiane des tirages - bruit - 0.70`, ET `corpus == "production"`, où
  `bruit = max(erreur-type de Hanley-McNeil, dispersion observée sur les graines)`.
  Retenir la plus grande des deux n'est pas un double comptage : elles estiment le
  même bruit d'échantillonnage par deux chemins, et chacune peut sous-estimer
  l'autre à faible effectif. `pass_rate` (part des tirages qui passeraient seuls)
  est un diagnostic publié, pas un critère. Des métriques d'avant le held-out
  répété font sauter le rôle avec un message, elles ne sont pas relues de travers.
  Aucun repli sur les anciennes métriques `utility_*` (servir l'AUC d'un modèle
  sous le nom d'un autre).
  ⚠️ **La certification vit dans `config/role_corpus.json`, versionné**, pas dans
  `role_readiness.json`, que `make roles` réécrit en entier : posée là, elle
  disparaîtrait au premier réentraînement, sans bruit et sans trace. Chaque entrée
  porte le `model_id` qu'elle certifie, et ce `model_id` est l'empreinte de la
  FONCTION DE SCORE (`ebm_lookup.compute_model_id` : sha256 de role/boundary/
  features/intercept/terms), pas un identifiant de run. La certification survit
  donc à une reconstruction identique (`make roles` sur les mêmes données rend les
  mêmes `model_id`, mesuré le 2026-09-20) et meurt dès que le modèle calcule autre
  chose, ce qui referme le rôle de lui-même. Sans ce lien, certifier ouvrirait au
  public le modèle SUIVANT, que personne n'a lu. Il n'y a plus de
  drapeau `--corpus` : un drapeau certifie sans laisser de trace dans un diff, et
  ferait une seconde source de vérité. Le fichier est une dépendance make de la
  table (`ROLE_CORPUS`), sinon éditer la certification ne périmerait rien et
  `make roles` répondrait « rien à faire » juste après la décision qui compte.
- **`service/role_scoring.py`** : sert l'EBM du rôle à un visiteur inscrit.
  ⚠️ Le modèle n'est PAS embarqué en pickle mais en table de correspondance JSON
  (`<role>_ebm_export.json`, écrit par `train_role_ensemble.py`) : un pickle
  d'`interpret` ne se recharge que sur la version qui l'a écrit, et les dépendances du
  service sont bornées en `>=`. `tests/test_ebm_lookup.py` prouve l'évaluateur sur un
  EBM synthétique (les artefacts sont gitignorés, la CI n'en a aucun) et
  `make verify-exports` le prouve sur les cinq artefacts réels, avant chaque build.
  ⚠️ L'éligibilité est décidée par un PRÉFLIGHT, après les 20 parties de profil et
  avant toute collecte supplémentaire : avec les cinq rôles fermés, un visiteur ne
  coûte pas un appel Riot de plus qu'avant.
  ⚠️ `shap:{slug}:role` est réécrite à CHAQUE ingestion, en union discriminée par
  `available` : sans cela une analyse publiée survivrait à sa péremption (rang qui
  baisse, rôle qui change, rôle qui referme).
  Le score publié est un `logit`, jamais une `probability` : MASTER n'appartient à
  aucune des deux classes d'entraînement et aucune calibration proba vers rang
  n'existe pour ces modèles. C'est une proximité à l'apex, pas un rang prédit.

⚠️ **Plus aucune borne d'âge du label** (retirée le 2026-09-18) : la règle des 14 jours
suivait le rythme des patchs LoL, pas une propriété de la donnée, et la tenir imposait
de recollecter tout le corpus à chaque capture. L'écart n'est plus refusé, il est
mesuré par ligne (`label_age_days`) et remonte jusqu'à la table d'ouverture, en face
de la décision.

Pipeline contexte (0 API) : `champion_profiles` (fetch DDragon one-shot) → `reextract_silver`
(silver + comp) → compléter `champion_traits.json` via `list_unknown_champions` → `rebuild_gold`
(+ `by_lane_context`) → `compare`. **Principe asymétrie** : le comp (info post-game complète)
sert UNIQUEMENT de contexte de benchmark, jamais à reprocher une décision sur une info cachée.

Features clés : **facettes win/loss** (neutralise le biais d'issue), **benchmark de lane**
(gold/CS/XP diff @10/@14/@20 vs adversaire), **gold-state des morts** (avance/retard),
**contexte de matchup botlane** (lane_pattern + gank_exposure, benchmarkés à contexte égal),
**benchmark positionnement** (présence/roam, over-extension, vision — timeline, 0 CV,
COACHING_SAFE uniquement).
Scopes : `all` · `adc` (BOTTOM). Filtre patch courant, SR (mapId 11),
ranked solo (queue 420). Spec : `docs/superpowers/specs/`.
⚠️ **`docs/superpowers/` (specs et plans) n'est plus versionné** (2026-09-07) : ces documents
vivent en local, seuls `docs/MODEL_CARD.md` et `docs/PROGRESS.md` restent suivis. Toute
référence à une spec dans ce fichier pointe donc vers un document absent d'un clone frais ;
l'historique git garde les versions antérieures à cette date.

## État d'avancement

Historique complet des runs, métriques et decisions (dates, chiffres, specs) :
`docs/PROGRESS.md`. Résumé de l'état actif ci-dessous.

- **Rebranding "Coaching LoL" → "RiftSense"** ✅ (2026-09-16). Un recruteur a fait
  remarquer que "Coaching LoL" ne rendait pas justice à la profondeur technique du
  projet (pipeline médaillon, ML, EBM/SHAP, LLM grounded). Renommage du repo GitHub,
  du domaine (`coaching-lol.jeanvg.fr` → `riftsense.jeanvg.fr`, ancien domaine
  redirigé), du Worker Cloudflare et du préfixe des clés KV liées au coaching
  (`coaching:{slug}:*` → `riftsense:{slug}:*`, migration additive). Les entrées
  datées ci-dessous mentionnant `coaching-lol.jeanvg.fr` décrivent un état passé
  réel et ne sont pas réécrites.
- **Chaîne par rôle** 🚧 (2026-09-18) : cinq modèles per-player explicables
  (un par rôle), labellisés par snapshot DATÉ du ladder et non plus par le rang de
  collecte ; fenêtres à profondeur fixe N=20. Frontière servie DIAMOND vs
  GM+CHALLENGER depuis le 2026-09-18 : médiane held-out EBM 0.77 à 0.87 selon le
  rôle sur 10 tirages, marge positive sur 4 rôles (TOP négatif, un tirage à 0.566
  gonflant la dispersion).
  Trois rôles sont OUVERTS au public depuis leur certification manuelle du 2026-09-20
  (`config/role_corpus.json`, `corpus: "production"`) : JUNGLE (marge +0.097), BOTTOM
  (+0.048), SUPPORT (+0.052). TOP (marge négative) et MIDDLE (+0.004, positive mais non
  certifiée) restent en `research`. L'ouverture demande une certification manuelle, pas
  une marge positive, et elle tombe d'elle-même au prochain `model_id` : une recapture
  du ladder (labels à 82 jours au 2026-09-24) referme donc les trois rôles jusqu'à
  nouvelle lecture. Détail et chiffres : `docs/PROGRESS.md`.
- **Rang servi = per-player** (ensemble xgb+rf+ebm, hypothèse constance sur tout
  l'historique, `MIN_PLAYER_GAMES=15`), test held-out AUC 0.688. **Per-game rang
  ABANDONNÉ** (2026-09-08, acté) : frontière master/GM illisible à N=1
  (`high_elo` per-game auc_cv 0.609 ; transformer séquentiel 0.546 ≈ MLP 0.504).
  `calibrate_rank.py` reste déprécié.
- **`train_ensemble.py --target dia_chall`** reste vivant, mais seulement comme
  modèle d'explication du jeu-type pour l'analyse EBM glass-box (jamais servi
  pour le rang). ⚠️ Son AUC de référence est **0.6312** (rebuild 2026-09-09,
  43 000 rows) — **pas 0.724** (chiffre de juillet sur ~7 873 rows, population
  non comparable, protocole CV inchangé). Ce qui légitime encore ce nœud n'est
  plus l'AUC mais `ebm_gap_vs_ensemble = 0.0004` (l'EBM seul égale l'ensemble) :
  les seuils de bascule des shape functions sont **descriptifs**, jamais des
  cibles.
- **Analyse ML unifiée (EBM glass-box)** ✅ 2026-09-08 : `core/ebm_explain.py`
  (registre `LEVELS` player/game) sert les deux niveaux + le sync KV
  (`shap:{slug}:drivers`). Payload publié = top-20 sur 125 features, proxys
  `ML_ONLY` retirés par `assert` (asymétrie), `map_depth` en lecture descriptive.
- **Coaching par-game servi par le site** ✅ 2026-09-07 : `POST /api/coach/game`,
  verrouillé par le Durable Object `CoachGate`, protégé par mot de passe
  (`auth.ts`). **Métrique produit atteinte** ✅ 2026-09-07 : ≥70 % de
  `mistakes` utiles sur ≥10 reviews par-game (100 % obtenu sur la cohorte
  `350f7c404b5b`, `kimi-k2.6`).
- **Coaching enrichi recalls/tracking/catégories** 🚧 2026-09-21 : code P0 terminé.
  Après rejet de la cohorte mixte `e1f3a4cb13da` à 81,9 %, le prompt `07692894c681`
  impose l'unité ou le sens de chaque chiffre. Les mêmes 10 games rejouées avec
  `deepseek-v4.1-flash` passent le gate : 91,8 % d'ancrage strict (669/729), 308/308
  horodatages exacts, 0 violation d'asymétrie, 0 retry, médiane 93,2 s. Les 12
  contrefactuels passent à 91,7 % de sensibilité et 95,7 % de grounding ; seul
  `zone_to_top` échoue sur 1/3 games. L'annotation humaine de ce lot a été explicitement
  sautée : ne pas le présenter comme validé en utilité. Les 10 reviews ont été fusionnées
  dans KV via `sync_cloudflare.py --slug spadzze --coaching-only`, sans toucher aux clés
  pipeline. Worker `b0f1bdf7-1c35-42c3-a24a-bfa3c8456c27` déployé et vérifié : l'API
  publique sert les 10 reviews `07692894c681` (5 wins/5 losses), toutes `ready`.
- **Migration Cloudflare** ✅ 2026-08-31 : `web/cf/` en prod sur
  `coaching-lol.jeanvg.fr` ; Fly.io inactif (facturation non réactivée).
- **Dépôt exécutable après clone** ✅ 2026-09-04 : `make demo` (0 réseau/clé,
  fixtures pseudonymisées, chaîne de prod rejouée avec `mock_llm`).
- **Régression LP hybride (apex)** ✅ 2026-07-07 : Spearman pooled test 0.5373,
  servi via `ml_rank.predict_rank`.
- **Recherche transformer séquentiel + SSL** ✅ 2026-07-18 (branche
  `research/sequence-transformer`, non servi) : gain net sur `dia_chall`
  (AUC 0.645 vs tabulaire 0.633), bruit sur `high_elo`. SSL mask-and-reconstruct
  ≈ 0 gain (prétexte prédictif future-event pas encore testé).
- **Protocole d'éval gold standard (per-player)** ✅ 2026-07-18 : split
  canonique unique (`data/04_dataset/split.json`, 70/15/15 stratifié), sélection
  HP en CV sur train, headline sur test held-out.
- Phases 1 → 1.8 validées : positionnement reconstruit sans vision, agrégation
  multi-games (pattern ~37 % morts ADC = BOT early), référentiels multi-rangs
  (~4454 games / patch 16.13), ML/SHAP industrialisé, macro-positionnement
  (17 features `positioning`, 14 COACHING_SAFE câblées en coaching).
- Clé **dev** Riot (throttle ~100 req/2min) → rate-limiter intégré ; clé prod
  utilisée en pratique.

### Prochaines étapes

1. **Coaching (axe prioritaire)** : annoter la cohorte `07692894c681`, déjà servie par
   le site sans annotation humaine. Les gates automatiques passent (91,8 % d'ancrage,
   91,7 % de sensibilité), mais l'utilité n'a pas été remesurée ; la preuve humaine
   disponible reste la cohorte Kimi précédente. Ensuite
   **coacher le plancher** — cibler les games du
   pire décile p10 (insight ML per-player : le rang = le plancher, pas la
   moyenne) et boucle de focus inter-games (adhérence au `next_focus`).
2. Stabiliser et valider la **robustesse ML/SHAP** (qualité des prescriptions SHAP vs
   heuristiques reste à valider).
3. Poursuivre l'industrialisation : modèles Pydantic et flux consolidé.

## Notes de développement

- Scripts encore au stade prototype ; migration Pydantic + Parquet/DuckDB prévue à
  l'industrialisation.
- Régénérer le gold après un changement de features : `python3 src/pipeline_ops/rebuild_gold.py`.
- Verdict : `python3 src/reporting/compare.py --scope adc --outcome {loss,win,overall}`.
- Garder la sortie LLM strictement typée (schéma JSON + validation Pydantic) pour éviter les
  résumés qui partent en vrille.
- Heuristiques déterministes (explicables, debuggables) avant tout ML supervisé.
