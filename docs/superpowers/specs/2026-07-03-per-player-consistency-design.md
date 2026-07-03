# Design — Pipeline per-player (features de constance) pour le rang ML web

> Date : 2026-07-03. Statut : validé en brainstorming, prêt pour le plan d'implémentation.

## Contexte & motivation

Le POC `poc/per_player_hypothesis.py` a testé l'hypothèse « challenger = consistance, pas
pic » : au lieu d'1 ligne = 1 game, on agrège par joueur (≥N games) des stats
`mean/std/p10/p50/p90` par feature, puis on entraîne un ensemble XGB/RF/EBM sur ces lignes
per-player. Résultat, sur données densifiées (`dia_chall`, extrêmes Diamond vs Challenger) :

- **+0.12 AUC** vs une baseline per-game entraînée sur les *mêmes joueurs* (isole l'effet
  agrégation de l'effet subset) : 0.707 → **0.852** (min_games=3, 307 joueurs).
- La **dispersion** (`std`/`p10`/`p90`) porte davantage de signal SHAP que la **tendance
  centrale** (`mean`/`p50`) : 56-58 % vs 42-44 %, confirmé par deux biais inductifs
  distincts (SHAP arbres + EBM glass-box).
- Le signal le plus fort est un **plancher** (`pos_avg_dist_to_ally__p10`, #1 dans les deux
  runs) : un challenger ne se distingue pas par son pic (`__p90` = feature la plus faible,
  8-19 %) mais par sa pire game qui reste acceptable.

Objectif de ce spec : **implémenter ce POC en production**, côté rang ML affiché sur
`/c/{slug}` (onglet Historique). Aujourd'hui `web/backend/ml_rank.py::predict_rank` fait un
agrégat naïf — moyenne simple des probabilités **per-game** sur ≥3 games ADC — ce qui ne
capture ni dispersion ni plancher. On remplace ce chemin par un vrai modèle per-player
entraîné sur les agrégats, au seuil validé par le POC (`min_games=5`).

Dataset référentiel actuel (bien plus dense qu'au moment du POC) : 11290 rows ADC,
**404 joueurs ≥5 games** — largement suffisant pour entraîner en prod.

## Décisions actées en brainstorming

1. **Approche** : nouveau pipeline per-player complet (dataset + modèle + calibration),
   *en parallèle* du pipeline per-game existant — pas de fusion des deux approches, pas de
   features roulantes greffées sur le modèle per-game actuel.
2. **Seuil** : `MIN_ADC_GAMES` passe de 3 à **5** (seuil validé par le POC). En dessous de
   5 games ADC, `predict_rank` renvoie `None` — **pas de fallback** sur l'ancien chemin
   per-game pour 3-4 games (un seul chemin de code à maintenir, cohérent avec le seuil
   scientifiquement validé plutôt qu'un compromis UX).
3. **Non-régression** : les artefacts per-game existants (`xgb_highelo.pkl`,
   `rf_highelo.pkl`, `ebm_highelo.pkl`, `features.json`, `metrics.json`,
   `rank_calibration.json`) ne sont **jamais écrasés** — ils restent consommés tels quels
   par `shap_analysis.py` (onglet profil ML SHAP), `calibrate_rank.py`, `audit_leakage.py`,
   `train_ensemble.py`. Le payload de coaching LLM (`src/04_coaching/`) n'est pas concerné.
   Les nouveaux artefacts portent tous un nom distinct (`player_*` / `*_player_highelo.pkl`).

## Architecture

```
src/core/ml_features.py                          (NOUVEAU — partagé)
  FEATURES                     : liste canonique (déduplique train_ensemble.py + POC)
  resolve_rank(group)          : mode + tie-break rang le plus bas
  aggregate_player_features(df, features) -> dict : mean/std/p10/p50/p90 + n_games
        │
        ├─ src/01_data_engineering/build_player_dataset.py   (NOUVEAU, 0 API)
        │     adc_dataset.parquet (référentiel) ──group by puuid, ≥5 games──>
        │     data/04_dataset/adc_player_dataset.parquet
        │
        ├─ src/02_data_science/train_player_ensemble.py      (NOUVEAU, miroir train_ensemble.py)
        │     adc_player_dataset.parquet ──ensemble xgb/rf/ebm, StratifiedKFold──>
        │     data/05_model/{xgb,rf,ebm}_player_highelo.pkl
        │     data/05_model/player_features.json
        │     data/05_model/player_metrics.json  (+ test d'hypothèse dispersion vs centrale)
        │
        ├─ src/02_data_science/calibrate_player_rank.py      (NOUVEAU, miroir calibrate_rank.py)
        │     adc_player_dataset.parquet + modèles player ──proba moy. par rang──>
        │     data/05_model/player_rank_calibration.json
        │
        └─ web/backend/ml_rank.py                            (RÉÉCRIT)
              games ADC du joueur (≥5) ──game_to_row × N──> aggregate_player_features
              ──ensemble xgb+rf──> proba ──calibration──> predicted_rank
```

### Pourquoi un module `src/core/ml_features.py`

`FEATURES` est aujourd'hui dupliqué mot pour mot dans `train_ensemble.py` et
`poc/per_player_hypothesis.py` ; `resolve_rank` n'existe que dans le POC. Le nouveau
pipeline a besoin de la **même** logique d'agrégation à l'entraînement (offline, sur un
DataFrame de plusieurs joueurs) et à l'inférence (online, sur les games d'1 seul joueur) —
divergence train/serve si dupliquée. Un module `core` unique (cohérent avec la convention
CLAUDE.md `core/` = « libs partagées ») élimine les deux problèmes à la fois. Le
per-game `FEATURES` de `train_ensemble.py` est remplacé par un import depuis ce module
(pas de duplication de la liste elle-même, aucun changement de comportement).

### `aggregate_player_features` — contrat

```python
def aggregate_player_features(df: pd.DataFrame, features: list[str]) -> dict:
    """1 groupe de games (index = games d'UN joueur) -> dict plat
    {f"{feature}__{stat}": float for stat in (mean,std,p10,p50,p90)} + {"n_games": int}.
    std ddof=1 (0.0 si 1 seule game). NaN propagées si la feature est NaN sur toutes les
    games du groupe (colonne manquante ou non calculable)."""
```

Reprend exactement la logique de `build_per_player` dans le POC, extraite en fonction pure
réutilisable (actuellement inline dans une boucle `for puuid, g in df.groupby(...)`).
Utilisée par `build_player_dataset.py` (un appel par groupe puuid) et par
`ml_rank.py` (un seul appel, sur les games ADC d'un joueur).

## Détail par fichier

### `src/01_data_engineering/build_player_dataset.py`

- Lit `data/04_dataset/adc_dataset.parquet`, filtre `source == "referentiel"`.
- Groupe par `puuid`, garde les groupes de taille ≥ `MIN_PLAYER_GAMES = 5`.
- Par groupe : `rank = resolve_rank(group)`, `**aggregate_player_features(group, FEATURES)`.
- Sort `data/04_dataset/adc_player_dataset.parquet` (+ `.csv` pour inspection, même
  convention que `build_dataset.py`). Colonnes : `puuid`, `rank`, `high_elo`, `n_games`,
  puis `{feature}__{stat}` pour chaque feature × 5 stats.
- Log : nombre de joueurs avant/après filtre, répartition par rang, positifs/négatifs pour
  `high_elo`.

### `src/02_data_science/train_player_ensemble.py`

- Mêmes 3 modèles (xgb/rf/ebm), mêmes rôles inductifs que `train_ensemble.py`, mais
  hyperparamètres repris du POC (adaptés au n plus faible et à l'espace de features ~5×
  plus large) : `xgb` inchangé (max_depth=3, n_estimators=300, min_child_weight=5),
  `rf` inchangé, `ebm(interactions=0)` (vs 10 en per-game — pas assez de rows pour des
  paires fiables à l'échelle per-player).
- CV : `StratifiedKFold` (pas de group CV — 1 ligne = 1 joueur, pas de fuite joueur→fold
  possible par construction).
- Cible par défaut `high_elo` (GM+Chall vs M+D) — c'est la cible consommée par le rang web
  (`predicted_rank` couvre les 4 rangs via calibration), pas `dia_chall` qui n'a de sens
  qu'en recherche (extrêmes). Le flag `--target` reste disponible pour ré-exécuter le POC
  dia_chall en interne si besoin de re-valider l'hypothèse sur données fraîches.
- **Test d'hypothèse conservé** : reprend `shap_group_analysis` du POC (masse `|SHAP|`
  xgb+rf groupée par stat, cross-check EBM) et l'écrit dans `player_metrics.json` sous une
  clé `dispersion_analysis` — garde l'observabilité "constance" en prod (pas seulement un
  AUC), conformément à la demande de rajouter les features de constance de façon visible.
- Sorties : `data/05_model/{xgb,rf,ebm}_player_highelo.pkl`, `data/05_model/player_features.json`,
  `data/05_model/player_metrics.json`.

### `src/02_data_science/calibrate_player_rank.py`

- Même logique que `calibrate_rank.py` mais sur `adc_player_dataset.parquet` et les
  modèles `*_player_highelo.pkl` : proba moyenne (xgb+rf) par rang réel (mode résolu du
  joueur), écrite dans `data/05_model/player_rank_calibration.json`.

### `web/backend/ml_rank.py` (réécriture ciblée)

- `MIN_ADC_GAMES = 5` (renommage conceptuel non nécessaire, la constante existe déjà).
- Chargeurs `_load_models`/`_load_features`/`_load_calibration` pointent vers les fichiers
  `player_*` / `*_player_highelo.pkl`.
- `predict_rank(games)` :
  1. Filtre `role == "BOTTOM"` ; si `< MIN_ADC_GAMES` → `None` (inchangé).
  2. `rows = [build_dataset.game_to_row(g, rank=None, source="inference") for g in adc_games]`
     → `pd.DataFrame(rows)`.
  3. `agg = ml_features.aggregate_player_features(rows_df, FEATURES)` (même fonction
     qu'à l'entraînement — garantie train/serve).
  4. `X = pd.DataFrame([agg]).reindex(columns=player_features).astype(float)`.
  5. Moyenne des probas xgb+rf (comme aujourd'hui), calibration au rang le plus proche.
  6. Retour `{"predicted_rank", "proba", "n_games_used": len(adc_games)}` — même forme de
     sortie qu'aujourd'hui, donc **aucun changement côté frontend/`routers/predicted_rank.py`**.
- Suppression de `_game_proba` (moyenne per-game), plus utilisée.

### Tests

- `tests/web/test_ml_rank.py` : adapter les fixtures à 5 games ADC (au lieu de 3), mocker
  `_load_features` avec des noms `__mean/__std/...`, mocker `_load_calibration` avec la
  nouvelle table. Cas à couvrir : `<5 games ADC` → `None` ; agrégation + calibration
  correcte ; filtrage des games non-ADC (inchangé).
- Nouveau test `tests/test_ml_features.py` : `aggregate_player_features` sur un petit
  DataFrame synthétique (2-3 games, valeurs connues) → vérifie mean/std/p10/p50/p90 et le
  cas `n=1` (`std=0.0`) et le cas colonne entièrement NaN.
- `resolve_rank` : test de tie-break (déjà couvert implicitement par le POC, à formaliser).

## Ce qui ne change pas

- Le pipeline per-game (`build_dataset.py`, `train_ensemble.py`, `calibrate_rank.py`,
  `shap_analysis.py`, `audit_leakage.py`) et ses artefacts `*_highelo.pkl`/`features.json`/
  `rank_calibration.json` restent strictement inchangés.
- Le payload de coaching LLM (`src/04_coaching/`), le SHAP profile web (`routers/shap.py`),
  le format de réponse de `GET /api/c/{slug}/predicted-rank` (même clés JSON).
- `poc/` reste tel quel (script + résultats historiques) ; son README est mis à jour en fin
  d'implémentation pour pointer vers l'implémentation prod.

## Séquencement

1. `src/core/ml_features.py` (+ tests) — extraction pure, TDD.
2. `build_player_dataset.py` → génère `adc_player_dataset.parquet`, vérifier les comptes
   joueurs/rangs en sortie (attendu proche de 404 joueurs ≥5 games, cf. exploration).
3. `train_player_ensemble.py` → entraîne, inspecte `player_metrics.json` (AUC + part
   dispersion/centrale) avant de continuer.
4. `calibrate_player_rank.py` → calibration.
5. `ml_rank.py` réécrit + tests mis à jour.
6. Vérification manuelle end-to-end : lancer le serveur web, appeler
   `/api/c/{slug}/predicted-rank` sur un compte avec ≥5 games ADC connues, comparer au
   comportement actuel.
7. Mise à jour `poc/README.md` (pointeur vers la prod) et `CLAUDE.md` (nouveaux scripts,
   nouvel état d'avancement).

## Critères de succès

- `player_metrics.json` : AUC out-of-fold per-player > AUC per-game `high_elo` actuel
  (0.589, cf. CLAUDE.md) — sinon le pipeline per-player n'apporte rien en prod et le
  design doit être reconsidéré avant câblage web.
- Tests verts (`ml_features`, `ml_rank` mis à jour).
- `/api/c/{slug}/predicted-rank` répond avec la même forme JSON, sur le nouveau modèle,
  seuil 5 games.
- `player_metrics.json` expose la part de signal dispersion vs tendance centrale
  (observabilité de l'hypothèse "constance" conservée en prod).

## Hors scope (différé)

- Fallback per-game pour 3-4 games ADC (tranché : non).
- Fusion des agrégats roulants dans le modèle per-game existant (option écartée en
  brainstorming).
- Ré-entraînement du modèle per-game `dia_chall`/`high_elo` ou de `shap_analysis.py`.
- Densification supplémentaire du référentiel (404 joueurs ≥5 games jugé suffisant pour
  ce spec ; à revisiter si `player_metrics.json` montre un signal trop bruité).
- Feature `floor_score` explicite (p10 normalisé, mentionnée comme piste dans le POC) —
  au-delà du périmètre "implémenter le POC tel quel".
