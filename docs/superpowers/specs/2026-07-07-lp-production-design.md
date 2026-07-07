# Design — Modèle LP en production (hybride binaire + régression LP)

> Date : 2026-07-07. Statut : validé en brainstorming, prêt pour le plan d'implémentation.
> Suite directe du POC concluant `2026-07-07-lp-regression-poc-design.md`
> (spearman within-tier : master 0.38 n=704, GM 0.55 n=78, challenger 0.60 n=367 ;
> pooled 0.5028 avec un XGBRegressor unique non tuné).

## Contexte & motivation

Le POC a montré que le LP réel (échelle continue partagée par Master→GM→Challenger,
sans divisions ni reset) porte un signal prédictif net **à l'intérieur de chaque
tier**, pas seulement à la frontière — donc plus fin que le label binaire high/low
servi aujourd'hui par `web/backend/ml_rank.py`. Objectif : industrialiser ce modèle
dans les dossiers principaux (`src/`, `web/`) et l'**optimiser** (ensemble complet
xgb+rf+ebm + tuning d'hyperparamètres, là où le POC reprenait sans tuning les
hyperparamètres du binaire).

## Décisions actées en brainstorming

1. **Rôle prod : hybride LP + binaire.** Le placement 4 rangs existant (binaire +
   calibration) reste LE chemin de placement, inchangé. Si le rang estimé est
   master, grandmaster ou challenger, le modèle LP **affine** avec un LP prédit
   (« ~Master · 250 LP estimés »). Diamond reste servi exactement comme aujourd'hui
   (hors échelle LP, pas de `predicted_lp`).
2. **Drift temporel : refetch au train.** `fetch_apex_lp.py` (3 appels API) est
   promu du POC vers `src/collection/` et relancé juste avant chaque entraînement.
   Le label LP est fetché « maintenant » alors que les games ont jusqu'à ~13 jours :
   drift borné par la fraîcheur du dataset, documenté comme limite connue (identique
   au POC). Pas de persistance du LP à la collecte (silver inchangé).
3. **Optimisation : ensemble + tuning léger.** Ensemble xgb+rf+ebm **regressors**
   (mêmes 3 biais inductifs que le binaire) + random search à graine fixe
   (~40 configs XGB, ~20 RF, ~10 EBM) en purged CV, critère de sélection =
   **Spearman pooled out-of-fold**. Pas d'Optuna (overfit de la CV probable sur
   ~1150 joueurs).
4. **Dataset LP séparé, binaire intact.** Nouveau
   `data/04_dataset/adc_player_lp_dataset.parquet` **sans balance-cap** (c'est une
   régression, pas de classes à équilibrer) : tous les joueurs qualifiés ≥15 games
   ADC, rank ∈ {master, grandmaster, challenger}, joints au LP courant.
   `build_player_dataset.py` (binaire, avec cap) n'est pas modifié.
5. **`poc/` reste tel quel** (référence historique). Le code prod ne doit **pas
   importer depuis `poc/`** — la logique de pool qualifié est réimplémentée dans le
   module prod (elle réutilise `ml_features.resolve_rank` /
   `aggregate_player_features`, déjà partagés).

## Architecture (5 briques)

```
src/collection/fetch_apex_lp.py            (promu du POC, 3 appels API)
    apex_league(challenger|grandmaster|master)
    → data/04_dataset/apex_lp.json  {puuid: {tier, leaguePoints}} + fetched_at (ISO)

src/01_data_engineering/build_player_lp_dataset.py     (0 API)
    adc_dataset.parquet (référentiel) → pool qualifié ≥15 games ADC,
    rang résolu au mode ∈ {master, grandmaster, challenger},
    agrégats mean/std/p10/p50/p90 + win_rate (ml_features),
    join LP par puuid depuis apex_lp.json (drop + comptage si absent = tier changé)
    → data/04_dataset/adc_player_lp_dataset.parquet (+ .csv), SANS balance-cap

src/02_data_science/train_player_lp.py                 (0 API)
    purged CV 5 folds (réutilise purged_train_features de train_player_ensemble),
    StratifiedKFold stratifié sur le tier, random search par modèle,
    ensemble xgb+rf+ebm regressors (moyenne simple), SHAP
    → data/05_model/{xgb,rf,ebm}_player_lp.pkl
    → data/05_model/player_lp_features.json
    → data/05_model/player_lp_metrics.json

web/backend/ml_rank.py                                 (hybride)
    predict_rank() inchangé dans sa logique de placement ; si predicted_rank ∈
    {master, grandmaster, challenger} ET modèles LP présents → "predicted_lp": int

web/frontend                               affichage « · ~N LP estimés » à côté du rang
```

## Détail des composants

### `src/collection/fetch_apex_lp.py`

Reprise du script POC (`poc/script/fetch_apex_lp.py`), adaptée aux conventions de
`src/collection/` : `.env` via `riotlib.load_env()`, `--region`/`RIOT_REGION`,
`RiotClient.apex_league(tier)` pour les 3 tiers. Deux changements vs POC :
- sortie dans `data/04_dataset/apex_lp.json` (données, plus `poc/output/`) —
  `data/` est déjà gitignoré, aucune fuite de puuids ;
- ajout d'un champ top-level `fetched_at` (ISO 8601 UTC) pour tracer la fraîcheur du
  label ; les entrées joueurs passent sous une clé `players`.
Log : nombre d'entrées par tier + total. Relançable à volonté (idempotent, 3 appels).
Note connue (POC) : `apex_league("master")` a renvoyé exactement 10 000 entrées —
possible cap de l'API, à logger en warning si le compte retombe pile sur 10 000.

### `src/01_data_engineering/build_player_lp_dataset.py`

- Lit `adc_dataset.parquet`, filtre `source == "referentiel"`.
- Pool qualifié : par puuid, ≥15 games (`MIN_PLAYER_GAMES = 15`, aligné binaire),
  rang résolu au mode (`ml_features.resolve_rank`) ∈ {master, grandmaster,
  challenger} (diamond exclu).
- Agrégats `ml_features.aggregate_player_features(g, mf.FEATURES)` sur la totalité
  de l'historique (même sémantique que le binaire).
- Join LP : lit `apex_lp.json`, colonne `lp` = leaguePoints courant ; joueurs absents
  droppés et comptés (`n_dropped_no_lp`, tier changé depuis la collecte). Erreur
  claire si `apex_lp.json` manque (« lancer fetch_apex_lp.py d'abord »).
- **Pas de balance-cap.** Colonnes : `puuid`, `rank`, `lp`, agrégats, `win_rate`,
  `n_games`.
- Sortie : `adc_player_lp_dataset.parquet` + `.csv`. Log : joueurs par tier
  avant/après join, `n_dropped_no_lp`, `fetched_at` du fichier LP utilisé.

### `src/02_data_science/train_player_lp.py`

- **Purged CV réutilisée** : importe `purged_train_features` de
  `train_player_ensemble.py` (flat-import validé au POC) — ~37 % des games opposent
  2 ADC du dataset (features en miroir), la purge est obligatoire pour des métriques
  honnêtes.
- **Folds** : `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` stratifié
  sur le **tier** (`y` continu, on stratifie sur la catégorie pour équilibrer les
  proportions master/GM/chall par fold).
- **Optimisation de coût structurante** : la purge ne dépend que du découpage en
  folds, pas des hyperparamètres → les features de train purgées sont
  **précalculées une fois par fold** puis réutilisées pour toutes les configs du
  random search. (Sans ça, chaque config repayerait le recalcul des agrégats.)
- **Random search** (graine fixe `random_state=42`) :
  - XGBRegressor : ~40 configs tirées de grilles sur `max_depth` {2,3,4},
    `n_estimators` {200,300,500}, `learning_rate` {0.03,0.05,0.1},
    `min_child_weight` {3,5,10}, `subsample` {0.7,0.8,1.0},
    `colsample_bytree` {0.7,0.8,1.0}, `reg_lambda` {0.5,1.0,3.0} ;
  - RandomForestRegressor : ~20 configs sur `n_estimators` {300,500},
    `max_depth` {None,8,12}, `min_samples_leaf` {2,5,10},
    `max_features` {"sqrt",0.3,0.5} ;
  - ExplainableBoostingRegressor : ~10 configs sur `max_bins` {128,256},
    `interactions` {0,10,20}, `learning_rate` {0.01,0.02}.
  - Critère de sélection par modèle : **Spearman pooled** des prédictions
    out-of-fold en purged CV.
- **Ensemble** : moyenne simple des prédictions des 3 modèles retenus ; métriques
  ensemble = Spearman pooled + by_tier + RMSE sur les prédictions OOF moyennées.
- **Modèles finaux** : refit de chaque meilleur modèle sur 100 % du dataset →
  `{xgb,rf,ebm}_player_lp.pkl` + `player_lp_features.json` (ordre des colonnes).
- **SHAP** : sur le XGB final (TreeExplainer), top-20 features + part de dispersion
  (std/p10/p90 vs mean/p50), même format que `player_metrics.json` — permet de
  vérifier si l'hypothèse constance tient aussi sur la cible LP.
- **`player_lp_metrics.json`** : spearman pooled/by_tier (avec n), RMSE, meilleure
  config par modèle, spearman par modèle individuel, `n_players_by_tier`,
  `n_dropped_no_lp`, `fetched_at` du label, baseline POC (0.5028) rappelée en champ
  `poc_baseline_spearman_pooled` pour mesurer le gain du tuning/ensemble.
- Garde-fous métriques : réutiliser la sémantique `_safe_spearman` du POC (None si
  <10 points ou entrée constante — jamais de NaN silencieux).

### `web/backend/ml_rank.py` (hybride)

- `predict_rank(games)` : signature et logique de placement **inchangées**.
- Après le placement, si `predicted_rank ∈ {"master", "grandmaster", "challenger"}` :
  charge les 3 regressors LP (lazy, `functools.lru_cache`, même pattern que
  `_load_models`), réutilise l'agrégat `agg` déjà calculé (aucun recalcul), prédit
  `lp = mean(xgb, rf, ebm)`, arrondi entier, borné à `max(0, lp)` → champ
  `"predicted_lp": int` dans le dict retourné.
- **Dégradation propre** : si un des `.pkl` LP ou `player_lp_features.json` est
  absent (modèle pas encore entraîné sur cette machine), pas de champ
  `predicted_lp`, pas de crash, pas de log d'erreur bruyant (le binaire suffit).
- Diamond ou rang non-apex : jamais de `predicted_lp`.

### `web/frontend`

Là où le rang ML estimé est affiché (onglet Historique de `/c/{slug}`), si la
réponse API porte `predicted_lp`, afficher « · ~{lp} LP estimés » à côté du rang.
Champ absent → affichage actuel inchangé. Même prudence de wording que l'existant
(signal faible — le Spearman within-tier master est ~0.4, c'est une estimation,
pas une mesure).

## Ce qui ne change pas

- `build_player_dataset.py` (balance-cap inclus), `train_player_ensemble.py`,
  `calibrate_player_rank.py`, `calibrate_rank.py`, tous les `.pkl` binaires,
  `player_metrics.json`, `player_rank_calibration.json` : intacts.
- Le placement 4 rangs de `ml_rank.py` (calibration proba→rang) : intact.
- `poc/` : conservé tel quel, non importé par le code prod.
- Silver/collecte : aucun changement (pas de persistance LP à la collecte).

## Hors scope (différé)

- Persistance du snapshot LP à la collecte (si le drift s'avère coûteux, itération
  future).
- Diamond / normalisation LP-within-division.
- Suppression du balance-cap du pipeline binaire.
- Remplacement du placement binaire par le modèle LP.
- Multi-région.

## Tests

- `tests/test_build_player_lp_dataset.py` : pool qualifié (min_games, diamond
  exclu, rang au mode), join LP (drop + comptage des absents), pas de cap
  (fixtures synthétiques, patterns repris de `test_poc_lp_regression.py`).
- `tests/test_train_player_lp.py` : fonctions pures — échantillonnage des configs
  (déterministe à graine fixe), sélection de la meilleure config au Spearman,
  garde `_safe_spearman` (None sur entrée dégénérée).
- `tests/test_ml_rank.py` (existant, étendu) : `predicted_lp` présent pour un rang
  apex, absent pour diamond, absent si les modèles LP manquent, borné ≥0.

## Séquencement d'exécution réelle (après le code)

1. `fetch_apex_lp.py` → `apex_lp.json` frais.
2. `build_player_lp_dataset.py` → dataset LP (~1150 joueurs attendus).
3. `train_player_lp.py` → modèles + métriques ; **lecture des résultats** : le
   Spearman pooled OOF doit ≥ la baseline POC 0.5028 (sinon, investiguer avant de
   servir).
4. Vérification web locale : `predicted_lp` visible pour un joueur apex.

## Critères de succès

- Pipeline complet reproductible en 3 commandes (fetch → build → train), 0 code
  importé depuis `poc/`.
- Spearman pooled OOF ≥ baseline POC (0.5028) ; métriques within-tier reportées.
- Le web affiche un LP estimé pour les joueurs placés master+ sans régression du
  placement existant (tests verts, dont les tests web existants).
