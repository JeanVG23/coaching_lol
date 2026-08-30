# Coaching LoL — Coach IA personnalisé pour League of Legends

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
   et d'états déjà extraits. L'extraction (API Riot + vision) fait le travail ; le LLM raconte.
5. **D'abord le journal fiable, ensuite le coaching.**
6. **Riot-first, CV-for-the-gaps.** API Riot = source principale ; CV = combler les trous.

## Source de données : Riot-first, CV-for-the-gaps

La vision n'est PAS la source principale. L'API Riot donne gratuitement, sans OCR :

- **Match-V5 + Timeline** (post-game) : positions x/y de tous les champions toutes les 60 s,
  gold/XP/items par joueur, et tous les events discrets (kills, objectifs, wards, level-ups,
  ordre de skill, achats). La **timeline** est le joyau.
- **Live Client Data API** (`https://127.0.0.1:2999`, pendant la game) : abilities, runes,
  items, gold, level, scoreboard, feed d'events en temps réel.

### Accès API (clé prod « Coach_LoL_LLMs », 39 méthodes)

Clé de production (limites généreuses, pas d'expiration 24 h) → backoff poli sur 429 suffit.

| API | Endpoint clé | Rôle | Routing |
|-----|-------------|------|---------|
| account-v1 | `accounts/by-riot-id/{gameName}/{tagLine}` | Riot ID → `puuid` (porte d'entrée) | **régional** |
| match-v5 | `matches/by-puuid/{puuid}/ids`, `matches/{id}`, **`matches/{id}/timeline`** | Cœur du MVP | **régional** |
| match-v5 | `matches/by-puuid/{puuid}/replays` | Bonus Phase 2 (pont .rofl) | régional |
| league-v4 | `entries/by-puuid/{puuid}` | Elo/LP (rend summoner-v4 inutile) | **plateforme** |
| lol-challenges-v1 | `player-data/{puuid}`, `challenges/percentiles` | Bonus benchmarking profil | plateforme |
| champion-mastery-v4 | `champion-masteries/by-puuid/{puuid}` | Profilage (main vs pick), Phase 2 | plateforme |
| spectator-v5 | `active-games/by-summoner/{puuid}` | Live uniquement, Phase 2 | plateforme |

**Piège routing** : account-v1 + match-v5 = régional (`europe`/`americas`/`asia`) ;
league-v4 / mastery / spectator = plateforme (`euw1`…).
**Hors scope** : summoner-v4, clash-v1, tournament-*, lol-status-v4, champion-v3.

### Ce que la CV doit combler (phase 2 seulement)

Uniquement ce que Riot ne donne pas : cooldowns exacts, skillshots loupés/touchés,
micro-positionnement entre frames 60 s, zone de caméra.

### Les deux niveaux d'information (CRITIQUE pour l'asymétrie)

- **« Ce que je savais »** — Live Client Data API : ne renvoie **que ce que le joueur voit**
  (fog of war respecté natif). → **Seule** base autorisée pour reprocher/juger une décision.
- **« Ce qui s'est réellement passé »** — Match Timeline post-game : info complète (positions
  ennemies même invisibles). → Sert uniquement à **labelliser a posteriori**. Ne JAMAIS le
  présenter au LLM comme une connaissance qu'avait le joueur.

## Stack cible

Langage principal : **Python** (écosystème vision/ML). Lancer depuis la racine, dans
l'environnement Poetry (`poetry shell` ou préfixer `poetry run`) :
`python3 src/<dossier>/<script>.py` — chaque script insère `src/core/` dans `sys.path` avant
`import riotlib` (convention flat-import, pas de package Python dans `src/`).

| Brique | Techno |
|--------|--------|
| Données de jeu | API Riot : Match-V5 + Timeline, Live Client Data API |
| Extraction frames/vidéo (phase 2) | FFmpeg, OpenCV, NumPy, Tesseract/EasyOCR |
| Détection minimap (tardif) | YOLO-like (Ultralytics/supervision) ou template matching + règles |
| Validation schémas | Pydantic |
| Stockage | Parquet (compact, colonne) — JSONL acceptable en ingestion |
| Analytics local | DuckDB |
| Extraction structurée locale | Ollama, structured output JSON (schéma imposé, T° basse) |
| Synthèse narration | Petit modèle local pour l'extraction ; modèle plus gros (API) pour la narration nuancée |

**À éviter (sur-ingénierie)** : CV tant que le MVP timeline n'est pas validé ; scraping
YouTube ; gros fine-tuning de LLM ; modèle supervisé d'erreurs sans dataset labellisé
(heuristiques déterministes d'abord) ; Postgres (DuckDB suffit) ; full video understanding.

## Séquencement par phases

- **Phase 1 — Coach 100 % API, zéro vision (MVP)** ✅ : N dernières games via Match-V5 →
  features macro/positionnement → Ollama → compte-rendu. Valider « le coach est-il utile ? »
  avant d'investir dans le pipeline CV. Si le coach timeline est bon, on sait où mettre la
  CV. S'il est mauvais, le problème vient des features, pas de la vision.
- **Phase 2 — CV pour les trous** : vision ciblée sur ce qui manque (cooldowns, skillshots,
  micro-position, caméra). Piste : rejouer la game depuis les **fichiers replay (.rofl)** en
  mode spectateur (Live Client API + caméra disponibles, sans impacter la game live).
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
- **`states_timeline`** — états échantillonnés : `game_id`, `timestamp_ms`, `my_hp_pct`,
  `my_mana_pct`, `q_cd`…`r_cd`, `flash_cd`, `tp_cd`, `visible_enemy_count`,
  `visible_on_minimap_*`, `camera_zone`, `gold_unspent`, `wave_state_estimate`.
  (Positions/gold/level viennent de la timeline Riot ; HP/mana/cooldowns/caméra = CV phase 2.)
- **`events`** — événements discrets : `game_id`, `timestamp_ms`, `event_type`,
  `actor` (self/ally/enemy_visible), `zone`, `confidence`, `payload_json`.
- **`reviews`** — labels et résumés : erreurs détectées, bons moves, scores
  lane/macro/vision, résumé final généré.

## Pipeline de résumé

1. Récupération (Riot API phase 1 ; + extraction CV phase 2).
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
- **Overlay de lecture d'écran** — lit l'écran → données → LLM, garantissant l'asymétrie.
  (ToS Riot : overlay en lecture seule, aucune automatisation d'input.)

## Évaluation

Prévoir **dès le départ** une boucle de feedback (« ce conseil était-il juste / utile ? »).
Le coaching benchmarké challenger est intrinsèquement plus vérifiable que les opinions
absolues du LLM.

## Architecture du code (médaillon, numérotée pour l'ordre du pipeline)

Code dans `src/`, données dans `data/` (couches numérotées). `src/` rangé par rôle (pas de
vrac à la racine) : `core/` (libs partagées), `collection/` (appels API Riot / Live Client),
`pipeline_ops/` (maintenance médaillon, 0 appel API), `reporting/` (livrable heuristique
pré-ML), `experiments/` (spikes historiques), puis `01_data_engineering` → `04_coaching`
(pipeline ML). `tests/` (pytest) couvrent la dérivation déterministe + extraction comp +
agrégation contextuelle. Lancer : `poetry run pytest tests/`.

```
src/
  core/           riotlib.py, positioning.py, champion_profiles.py, game_journal.py, ml_features.py
  collection/     build_referential.py, aggregate_games.py, live_capture.py,
                  densify_targets.py, densify_sweet_spot.py, densify_players.py, fetch_apex_lp.py
  pipeline_ops/   reextract_silver.py, rebuild_gold.py, compress_raw.py,
                  archive_patch.py, list_unknown_champions.py, dataset_report.py
  reporting/      compare.py
  experiments/    phase1_pull.py
  01_data_engineering/  build_dataset.py, build_player_dataset.py, build_player_lp_dataset.py,
                        build_sequence_dataset.py
  02_data_science/      train_ensemble.py, calibrate_rank.py, train_player_ensemble.py,
                        train_player_lp.py, calibrate_player_rank.py, analyze_auc_vs_ngames.py,
                        lp_metrics.py, audit_leakage.py, poc/per_player_hypothesis.py,
                        sequence_model.py, sequence_data.py, train_sequence_model.py,
                        pretrain_sequence_model.py
  03_data_analyse/      shap_analysis.py, plot_custom_shap.py
  04_coaching/          payload.py, prompt.py, schema.py, llm_client.py, coach.py, feedback.py
data/
  00_static/      champion_traits.json (force-add : config source), ddragon/<version>/ (ignoré)
  01_raw/         JSON API brut compressé .json.zst (~10 Go -> ~750 Mo, ×13). Lecture/écriture
                  transparentes via riotlib._read_raw/_write_raw (tolérante .json.zst->.json.gz->.json)
  02_silver/{referentiel/<rank>,personal/<player>}/games.jsonl   # 1 ligne = 1 game nettoyée (+ comp)
  03_gold/{referentiel/<rank>,personal/<player>}/<scope>/aggregate.json   # agrégats benchmarks
  04_dataset/     adc_dataset.parquet, densify_targets.json, datasets per-player/LP
  05_model/       modèles ML + metrics (xgb_highelo.pkl, player_metrics.json, player_lp_metrics.json,
                  rank_calibration.json, auc_vs_ngames.{json,png})
  06_shap/        SHAP/EBM outputs
  07_coaching/<player>/reviews.jsonl + feedback.jsonl
```
⚠️ `data/` est gitignoré SAUF `data/00_static/champion_traits.json` (force-add : config source).
⚠️ Les chemins des couches sont définis dans `src/core/riotlib.py` (`RAW_DIR`/`SILVER_DIR`/`GOLD_DIR`).
Renommer un dossier data SANS mettre à jour le code → le code recrée l'ancien.

### Modules `core/`

- **`riotlib.py`** — socle : `RiotClient` (routing régional account/match vs plateforme league ;
  rate-limiter ~1.3s/appel ; `entries_by_puuid` = rang courant via league-v4, frais pour le
  coach web), helpers (`approx_zone`, `phase_of`, `patch_of`), `get_match_timeline` (cache raw
  compressé zstd), `extract_game` (silver + benchmark de lane + sous-objet `comp` des 6 champions
  botlane + sous-objet `position` via `positioning`), `aggregate`/`write_gold` (gold, facettes
  win/loss + dimension `by_lane_context` + bloc `positioning` = médianes des 14 features
  COACHING_SAFE via `_fmedian`, sans arrondi entier), chemins médaillon. Importe `champion_profiles`.
- **`positioning.py`** — features macro-positionnement depuis la timeline (0 CV, module pur).
  `positioning_features` → 17 scalaires nichés sous `record["position"]`. Manifeste d'asymétrie
  mécanique : `COACHING_SAFE` (14 exactes → ML + coaching) vs `ML_ONLY` (3 proxys vision →
  jamais prescrits). ⚠️ **Profondeur** (`avg/max_map_depth`) : sens contre-intuitif (valeur haute
  → diamond, rang INFÉRIEUR) → marqueur de risque, jamais à prescrire.
- **`game_journal.py`** — journal structuré d'UNE game depuis match+timeline raw (0 CV). Morts
  et recalls **horodatés** (clock mm:ss) avec contexte : zone/phase, gold-state vs adversaire,
  **gold non dépensé**, killer/gank, **objectif up/imminent** (timers v1 en tête de module :
  drake 5:00/+5:00, baron 25:00/+6:00 — ajuster par patch ; Elder/Atakhan ignorés). Recalls =
  clusters d'`ITEM_PURCHASED` (inclut resets après mort, `gold_before` = plancher frame précédente),
  **`item_ids`** par recall (achats bruts, `ITEM_UNDO` honoré = retiré, `ITEM_SOLD` ignoré v1).
  **Asymétrie** : uniquement de l'info que le joueur avait — aucun proxy ML_ONLY.
  **Conséquences post-mort** (chaîne causale, 2026-07-18) : chaque mort porte un bloc
  `consequences` calculé mécaniquement — objectifs (`ELITE_MONSTER_KILL` ennemi) et
  bâtiments (`BUILDING_KILL`, ⚠️ `teamId` = équipe qui PERD) pris dans les
  `CONSEQUENCE_WINDOW_S=60` s post-mort, + `team_gold_swing_90s` (écart de gold
  d'équipe avant vs ~90 s après). Clé omise si fenêtre vide. `SYSTEM_GAME` impose de
  restituer la chaîne (« mort → Baron perdu → -1 840 g ») en formulation corrélationnelle.
- **`champion_profiles.py`** — identité champion : `champion_vector` (Data Dragon + table curée,
  résolution casse-insensible), `derive_context(comp)` → `lane_pattern`
  (poke/all_in/scaling/mixed/unknown) et `gank_exposure` (low/med/high/unknown). `fetch_ddragon`
  (one-shot, idempotent). Champion inconnu → `unknown`, jamais d'erreur. `fetch_ddragon_items`/
  `load_items` — même pattern one-shot pour le catalogue d'items Data Dragon (`item.json` →
  `data/00_static/ddragon/<version>/`, `{id: {name, cost}}`).
- **`ml_features.py`** — FEATURES canonique (partagé train/serve) + `aggregate_player_features`
  (mean/std/p10/p50/p90 + `win_rate`) + `resolve_rank` (mode, tie-break rang le plus bas).

### Modules `collection/`

- **`build_referential.py`** — collecte les benchmarks par rang (league-v4/-exp-v4).
  `python3 src/collection/build_referential.py --region euw1 [--rank R] [--players N]`.
- **`aggregate_games.py`** — pipeline perso : N games → silver + gold (all/adc/zeri).
- **`live_capture.py`** — capture Live Client Data API pendant une game ; zéro dépendance hors
  stdlib (copiable seul sur une machine sans le reste du repo).
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
  `build_player_dataset.py` — 1 ligne = 1 joueur ≥`MIN_PLAYER_GAMES` games ADC référentiel,
  agrégées sur la totalité de l'historique disponible. `build_player_lp_dataset.py` — per-player
  SANS balance-cap, apex seulement (diamond exclu — LP non comparable).
- **`02_data_science/`** : `train_ensemble.py` — classif High-Elo vs Low-Elo via **ensemble à 3
  biais inductifs** (XGBoost=GBDT, Random Forest=bagging, EBM=GA²M glass-box). SHAP moyen sur
  les 2 arbres ; EBM = validateur indépendant + interactions par paires.
  `calibrate_rank.py` — calibration proba→rang (`web/backend/ml_rank.py`) : modèle `high_elo`
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
  (hypothèse constance, repris en prod par `web/backend/ml_rank.py`).
- **`03_data_analyse/`** : `shap_analysis.py` (SHAP global + Spadzze + cross-check EBM :
  direction par feature via `explain_local`, interactions par paires via `explain_global`) et
  `plot_custom_shap.py`.
- **`04_coaching/`** : narration LLM (Ollama Cloud, structured output).
  `payload.py` (gold perso+réf → payload déterministe, **safe-only** : positioning ⊂
  COACHING_SAFE, profondeur `descriptive_only`), `prompt.py` (system asymétrie + benchmark-relatif,
  FR), `schema.py` (Pydantic : `Review` 1-3 forces / 3 erreurs / 2 habitudes / 1 focus / confidence,
  **preuve chiffrée par point** — forcer exactement 3 forces poussait au remplissage, cause du tag
  feedback « trop-vague » ; `GameReview` 0-2 forces / 1-3 erreurs `GameInsight` — **`cause`
  obligatoire (POURQUOI : mécanisme de mort / comportement) + horodatage mm:ss dans l'evidence,
  sur forces ET erreurs** (réponse feedback « je sais pas pourquoi je suis mort ») ;
  pas de habits sur 1 game ; `Feedback`/`FeedbackItem`),
  `llm_client.py` (client `https://ollama.com/api/chat`, `OLLAMA_API_KEY`, `format`=JSON-schema,
  défaut `kimi-k2.6`), `coach.py` (CLI : payload→prompt→client→validation→affiche+persiste),
  `feedback.py` (CLI `annotate`/`summary` : boucle d'éval par-insight).
  **Chemin par-game** : `payload.build_game` (journal `game_journal` + repères référentiel à issue
  égale ; recalls enrichis d'items résolus {nom, coût} via `champion_profiles.load_items` — plus
  d'`item_ids` bruts côté LLM ; bloc `context` = comp botlane/jungle/mid + `lane_pattern`/
  `gank_exposure` via `derive_context`), `prompt.SYSTEM_GAME` (règle matchup basée sur ce
  `context` + règle de gold relatif au prochain achat de chaque recall), `coach.py --game [latest|MATCH_ID]` (records `kind: "game"` +
  `match_id`), `coach.py --game-batch [N]` (défaut 10 : reviews par-game des N dernières games ADC
  pas encore reviewées, dédup par `match_id`, poursuit sur échec, bilan final).
  `feedback.py annotate --pending` : itère en série toutes les reviews sans feedback. `summary` :
  taux par section + top tags + par modèle + tendance + verbatims `tag_notes` + bloc `Objectif
  par-game` (`objective_stats` : % mistakes utiles sur les reviews `kind: "game"`). Le champ
  `note` est aussi exposé côté web (`web/frontend/`, textarea sous chaque item noté, `POST /api/feedback`).
  Lancer : `python3 src/04_coaching/coach.py --player spadzze --scope adc [--game|--game-batch N]`,
  `python3 src/04_coaching/feedback.py annotate --player spadzze [--last|--ts|--pending]`. Aucun réseau.

Pipeline contexte (0 API) : `champion_profiles` (fetch DDragon one-shot) → `reextract_silver`
(silver + comp) → compléter `champion_traits.json` via `list_unknown_champions` → `rebuild_gold`
(+ `by_lane_context`) → `compare`. **Principe asymétrie** : le comp (info post-game complète)
sert UNIQUEMENT de contexte de benchmark, jamais à reprocher une décision sur une info cachée.

Features clés : **facettes win/loss** (neutralise le biais d'issue), **benchmark de lane**
(gold/CS/XP diff @10/@14/@20 vs adversaire), **gold-state des morts** (avance/retard),
**contexte de matchup botlane** (lane_pattern + gank_exposure, benchmarkés à contexte égal),
**benchmark positionnement** (présence/roam, over-extension, vision — timeline, 0 CV,
COACHING_SAFE uniquement).
Scopes : `all` · `adc` (BOTTOM) · `zeri` (champion). Filtre patch courant, SR (mapId 11),
ranked solo (queue 420). Spec : `docs/superpowers/specs/`.

## État d'avancement

- **Phase 1 VALIDÉE** ✅ — positionnement reconstruit sans vision. Insight type :
  « 1 ennemi = 5/8 de tes morts » >> « meurs moins ».
- **Phase 1.5 — agrégation multi-games** ✅ — pattern récurrent sur 14-20 games : ~37 % des
  morts ADC = BOT en early ; l'ennemi ADC signe ~45 % des morts.
- **Phase 1.6 — référentiels multi-rangs** ✅ — collecte globale ~4454 games / patch 16.13
  (Diamond, Master, GM, Challenger). `compare.py` et benchmarks contextuels intégrés.
- **Phase 1.7 — ML & SHAP** 🚧 — pipeline médaillon industrialisé. Classif High-Elo vs Low-Elo
  (ensemble XGB/RF/EBM). Dataset densifié : 2 ADC/game depuis le raw → ~7 873 rows.
- **Phase 1.8 — macro-positionnement (timeline, 0 CV)** ✅ — module `positioning` (17 features).
  ML : AUC **dia_chall 0.655 → 0.724 (+0.069)**, top-3 discriminants EBM tous positionnels.
  Coaching : 14 features câblées dans `aggregate`/`compare`. ⚠️ `xgb/rf/ebm_highelo.pkl` à
  ré-entraîner avant de servir en inférence web — per-game déprécié 2026-07-18, non servi ; le
  serving utilise les `*_player_highelo.pkl`. **AUC high_elo = 0.589** (frontière Master|GM
  peu séparable sur features macro, contrairement à dia_chall).
- **Rang ML estimé (web)** ✅ — onglet Historique de `/c/{slug}` : rang placé par l'ensemble
  xgb+rf sur les dernières games ADC, calibré par `calibrate_rank.py`. Confiance affichée
  explicitement (signal faible) — pas de fausse certitude.
- **Rang ML per-player (constance)** ✅ — `web/backend/ml_rank.py` utilise le modèle per-player
  (features mean/std/p10/p50/p90 + `win_rate`, seuil `MIN_ADC_GAMES=15`), reprenant l'hypothèse
  validée par `poc/per_player_hypothesis.py` (dispersion/plancher > tendance centrale).
  **`MIN_PLAYER_GAMES=15`** (relevé 5→15 : à 5 games l'AUC s'effondrait à 0.531, bruit de
  matchmaking > signal de dispersion ; agrégation sur tout l'historique + `win_rate` corrige).
  **AUC_cv purgée 0.635** (982 joueurs 491/491, après densification sweet-spot `[15,30[` hors
  diamond, 2026-07-06 ; dispersion 57.9 % du signal SHAP). Fuite par games partagées ≈ +0.005
  d'AUC (purge = 8.7 % des games de train, 0 joueur droppé) — l'hypothèse constance tient.
  **AUC vs N** (`analyze_auc_vs_ngames.py`) : pool fixe ≥50, N=15→0.588, 20→0.619, 25→0.628,
  **30→0.635 (peak)**, 40→0.624, 50→0.599. Sweet spot ~30 ; au-delà = bruit CV. La config prod
  (qualify=15, cap=tout l'historique) ≈ le peak — déjà au plateau ; monter le seuil ne gagnerait
  que ~+0.01-0.02 pour un pool divisé par 2-3. **Plafond ~0.65** sur la frontière master/GM —
  les leviers sont le pool (densifier), les features, ou la frontière de rang (dia_chall 0.72),
  pas plus de N ni de joueurs sur cette bande. Historique complet des runs dans
  `data/05_model/player_metrics.json`.
- **Régression LP (hybride, apex tiers)** ✅ — 2026-07-07. Pipeline : `fetch_apex_lp.py` →
  `build_player_lp_dataset.py` (per-player SANS balance-cap, apex seulement, diamond exclu) →
  `train_player_lp.py` (ensemble xgb/rf/ebm REGRESSORS, random search graine fixe en purged CV,
  sélection au Spearman pooled OOF, SHAP). Serving hybride : `ml_rank.predict_rank` ajoute
  `predicted_lp` (moyenne ensemble, ≥0) quand le rang placé est apex et que les `.pkl` LP
  existent (dégradation propre sinon) ; le placement binaire 4 rangs est inchangé. Drift
  temporel du label LP (fetch au train vs games jusqu'à ~13 j) = limite connue actée. Spec :
  `docs/superpowers/specs/2026-07-07-lp-production-design.md`.
  **Métriques run 2026-07-07** (`data/05_model/player_lp_metrics.json`) : 1148 joueurs
  (master 703 / challenger 367 / grandmaster 78, 130 droppés sans LP courant). Ensemble OOF
  purgé : **Spearman pooled = 0.5186** (baseline POC 0.5028, gate passée) ; by_tier challenger
  0.5995, grandmaster 0.5979, master 0.4103 ; RMSE 517.2 LP. Dispersion = 56.1 % du signal SHAP.
- **Recherche — transformer séquentiel + SSL** ✅ — 2026-07-18. Branche parallèle
  `research/sequence-transformer` (0 perturbation du pipeline existant). Transformer à la main
  (4 couches, d_model=64, masked-mean-pool) sur les séquences d'états par-minute (20-d : ADC ciblé
  + adverse + diffs gold/cs/xp/level), CV purgé identique au baseline tabulaire (folds
  joueur-groupés + purge miroir, standardisation per-feature **train-only par fold non
  négociable**). Étape 1 supervisée vs ensemble tabulaire (RF+EBM) + MLP contrôle ; Étape 2 SSL
  mask-and-reconstruct (delta mesuré, pretrain par-fold train-only → delta propre, pas d'avantage
  transductif). Verdict sur `dia_chall` ; `high_elo` (master/GM) null = bruit de label non
  interprétable (plafond ~0.589 connu). Spec : `docs/superpowers/specs/2026-07-18-sequence-transformer-design.md`.
  Métriques : `data/05_model/sequence_metrics.json`.
  **Métriques run 2026-07-18** (`data/05_model/sequence_metrics.json`) : `dia_chall` séquence
  **AUC 0.645** (±0.008, 42 996 rows) BAT tabular 0.633 / MLP 0.530 → la représentation séquentielle
  capte un signal que l'agrégat rate sur la frontière séparable (thèse renforcée). `high_elo`
  séquence 0.546 ≈ tabular 0.554 ≈ bruit (95 378 rows, master/GM peu séparable). SSL
  `delta_ssl = -0.0195` (≈0) : le prétexte MSE mask-and-reconstruct est faible sur signaux lisses
  (gold monotone, position continue → quasi-interpolation) — **pas un verdict sur le SSL en
  général**, un prétexte prédictif (future-event) reste à tester en étape 3. ⚠ **Caveat env Mac** :
  torch+xgboost ne cohabitent pas (double-load libomp → SIGSEGV) → baseline tabulaire RF+EBM
  (xgb exclu) ; MPS n'implémente pas le nested-tensor de `src_key_padding_mask` → run CPU
  (`--device cpu`). 0 API (relit `_read_raw`).
- **Protocole d'éval gold standard (per-player)** ✅ — 2026-07-18. Split canonique unique
  `data/04_dataset/split.json` (par joueur, stratifié, graine fixe, 70/15/15, cf.
  `src/core/dataset_split.py` + `src/01_data_engineering/build_split.py`). Sélection des
  hyperparamètres en k-fold SUR LE TRAIN, headline sur le TEST held-out ; calibration + test
  hors du modèle servi (réservée à une future couche calibration/conformal, non encore
  implémentée).
  Purge étendue via `purged_train_features` (fold-val ∪ holdout). ⚠ Le headline test est
  volontairement plus bas que les anciens OOF-à-plat (fin de l'optimisme de sélection + modèle
  sur ~70 % des joueurs) : c'est la mesure honnête. **FLAW ASSUMÉ (GM)** : ~78 GM au total →
  calib/test GM petits (~12 chacun), métriques GM bruitées ; remédiation renvoyée à un script
  ultérieur. Spec : `docs/superpowers/specs/2026-07-18-gold-standard-eval-protocol-design.md`.
  **Métriques run 2026-07-18** — rang (`player_metrics.json`) : cv_train.auc=0.5912 (n=687,
  343/344) / test.auc=0.677 (n=147, 73/74) ; split 1345 joueurs (train 942 / calib 202 / test
  201, 70/15/15 stratifié). LP (`player_lp_metrics.json`) : cv_train.spearman_pooled=0.4931
  (rmse 535.4, n=805) / test.spearman_pooled=0.5373 (rmse 555.1, n=170) ; by_tier test :
  challenger 0.6601 (n=55), grandmaster 0.7545 (n=11, bruité), master 0.3634 (n=104).
- **Per-game DÉPRÉCIÉ** — 2026-07-18. `train_ensemble.py` / `calibrate_rank.py` arrêtés (non
  servis, AUC ~0.63/0.59 trop aléatoire) ; code et artefacts conservés pour l'historique.
- **Compte-rendu par-game (axe prioritaire coaching)** ✅ — 2026-07-05. Diagnostic feedback :
  les tags « trop-vague »/« non-actionnable » venaient du **payload agrégé** (le LLM ne peut pas
  être plus précis que des médianes) + du schéma forçant 3 forces. Fix : `game_journal` (morts/
  recalls horodatés + contexte) → `coach.py --game` → `GameReview` (horodatage obligatoire par
  erreur au niveau schéma). Vérifié bout en bout (kimi-k2.6). **Itération cause (2026-07-08)** :
  2e signal feedback (« je sais pas pourquoi je suis mort », « aucune idée de pourquoi » sur les
  forces) → `GameInsight` ajoute un champ `cause` obligatoire (POURQUOI = mécanisme de mort /
  comportement) sur forces ET erreurs, + `SYSTEM_GAME` restitue le contexte de mort du journal
  (killer/gank/zone/objectif). Validé : « tu prolonges en lane sans gold à dépenser → exposé aux
  all-ins 2v2 et ganks (morts à 4:42 par Karma…) ». **Métrique de succès : ≥70 % de mistakes
  utiles sur ≥10 reviews par-game annotées, 0 rejet « trop-vague ».**
- **Premier verdict ADC** : laning = LE levier (≈ -10 à -16 CS @14 vs challenger, *toutes*
  issues) ; mauvaise gestion du retard (gold@20 en lose -1252 vs -322) ; morts = symptôme.
- Clé **dev** (throttle ~100 req/2min, attentes 429 si saturé) → rate-limiter intégré.
  `.env` (clé `RIOT_API_ID` ; pas de `RIOT_REGION` → passer `--region euw1`), `data/` ignoré.

### Prochaines étapes

1. ✅ **Ollama branché** — `src/04_coaching/` génère un compte-rendu agrégé (Ollama Cloud,
   persisté dans `data/07_coaching/`). Modèle défaut `kimi-k2.6` (retenu après A/B, cf.
   `src/04_coaching/README.md` ; surclassable via `--model`/`OLLAMA_MODEL`).
   ✅ **Boucle d'éval** — `feedback.py annotate/summary`.
   ✅ **Compte-rendu par-game** — `coach.py --game`.
   ✅ **Boucle batch+pending** (2026-07-06) — outillage en place. Il reste à **annoter
   effectivement ≥10 reviews par-game** (métrique ≥70 % de mistakes utiles). L'approche C
   (génération auto post-game) reste à suivre. Ensuite **coacher le plancher** — cibler les
   games du pire décile p10 (insight ML per-player : le rang = le plancher, pas la moyenne) et
   boucle de focus inter-games (adhérence au `next_focus` mesurée par les features).
2. **Benchmark Zeri** densifié (sampling champion ciblé) si la slice reste trop fine.
3. Stabiliser et valider la **robustesse ML/SHAP** (qualité des prescriptions SHAP vs
   heuristiques reste à valider).
4. Poursuivre l'industrialisation : modèles Pydantic et flux consolidé.
5. Phase 2 (CV / Live Client) seulement si le coach démontre sa valeur.

## Notes de développement

- Scripts encore au stade prototype ; migration Pydantic + Parquet/DuckDB prévue à
  l'industrialisation.
- Régénérer le gold après un changement de features : `python3 src/pipeline_ops/rebuild_gold.py`.
- Verdict : `python3 src/reporting/compare.py --scope adc --outcome {loss,win,overall}`.
- Garder la sortie LLM strictement typée (schéma JSON + validation Pydantic) pour éviter les
  résumés qui partent en vrille.
- Heuristiques déterministes (explicables, debuggables) avant tout ML supervisé.