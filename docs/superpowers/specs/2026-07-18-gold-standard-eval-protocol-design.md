# Spec — Protocole d'évaluation « gold standard » per-player + arrêt du per-game

> Créé le 2026-07-18. Prépare le pipeline per-player à accueillir la couche AOS4
> (conformal prediction, spec **séparée et ultérieure**) en posant d'abord un socle
> d'évaluation propre. **Cette spec ne contient aucun conformal** : elle réserve
> seulement le terrain.

## 1. Contexte & problème

Le pipeline ML per-player servi en prod (`web/backend/ml_rank.py`) repose sur deux
modèles :

- **rang high/low** — `train_player_ensemble.py` → `xgb/rf_player_highelo.pkl` +
  `player_rank_calibration.json` (via `calibrate_player_rank.py`) ;
- **régression LP** — `train_player_lp.py` → `{xgb,rf,ebm}_player_lp.pkl`.

Les deux s'entraînent aujourd'hui en **k-fold purgée à plat sur tout le dataset**, avec
un défaut méthodologique unique mais réel : **les hyperparamètres sont sélectionnés sur
le même OOF qui sert de rapport** (`search_best` prend la meilleure config au
Spearman/AUC OOF). Sélection et évaluation sont confondues → les métriques publiées
(AUC 0.635, Spearman 0.5186) sont **légèrement optimistes**. Il n'existe **aucun jeu de
test vierge**.

Deux conséquences :
1. On ne peut pas affirmer honnêtement la performance du modèle servi.
2. La future couche conformal (AOS4) exige un jeu **vierge de tout fit/sélection** pour
   calibrer ses intervalles et un jeu **jamais touché** pour vérifier la couverture. Le
   pipeline actuel n'offre ni l'un ni l'autre.

## 2. Objectif & non-objectifs

**Objectif.** Adopter le protocole d'évaluation textbook — **held-out test + k-fold CV
*dans* le train** — sur les modèles per-player servis, via un **split canonique unique**
partagé, et **documenter l'arrêt** du pipeline per-game.

**Non-objectifs (explicites).**
- **Aucun conformal / intervalle de confiance ici.** Le set de calibration est créé et
  *réservé*, mais aucune logique conformale n'est écrite. → spec AOS4 ultérieure.
- Pas de migration du per-game vers le nouveau protocole (il est arrêté, cf. §8).
- Pas de suppression de code ni d'artefacts existants.
- Pas de refonte des features ni du dataset (`adc_player_dataset.parquet` inchangé).

## 3. Décisions verrouillées

| Décision | Choix | Raison |
|----------|-------|--------|
| Protocole | held-out test + CV-in-train | le plus lisible/textbook (préféré à la nested CV) |
| Découpage | **train / calibration / test** disjoints, au niveau **joueur** | 3 rôles nets ; calibration réservée à AOS4 |
| Granularité du split | **par `puuid`**, stratifié par rang, graine fixe | reproductible ; cohérent entre modèles |
| Modèle servi | refit sur **train uniquement** | calibration + test restent vierges pour AOS4 |
| Flaw GM | **assumé et documenté** | 78 GM au total → calib/test GM petits ; remédiation renvoyée à un script ultérieur |
| Per-game | **arrêté + documenté**, non supprimé | AUC ~0.63/0.59, non servi, remplacé par le per-player |

## 4. Le split canonique

### 4.1 Artefact
Nouveau script **`src/01_data_engineering/build_split.py`** (0 API) produisant
**`data/04_dataset/split.json`** :

```json
{
  "seed": 42,
  "proportions": {"train": 0.70, "calibration": 0.15, "test": 0.15},
  "created_from": ["adc_player_dataset.parquet", "adc_player_lp_dataset.parquet"],
  "n_by_bucket_by_rank": {"train": {"master": 492, "challenger": 257, "grandmaster": 55, "diamond": 340}, "calibration": {"...": 0}, "test": {"...": 0}},
  "assignment": {"<puuid>": "train|calibration|test", ...}
}
```

### 4.2 Règles de construction
- **Population** : l'**union** des `puuid` de `adc_player_dataset.parquet` (dataset rang,
  inclut diamond + apex) et de `adc_player_lp_dataset.parquet` (dataset LP, apex). L'union
  garantit qu'aucun joueur consommé par un modèle n'échappe au split (un joueur apex présent
  seulement dans le dataset LP est couvert). Chaque modèle filtre ensuite l'union à son propre
  dataset par lookup `puuid` → **même bucket partout**, cohérence garantie entre modèles.
- **Stratification par rang résolu** (mode, tie-break rang le plus bas — même règle que
  `ml_features.resolve_rank`) : proportions respectées tier par tier.
- **Graine fixe** (`SEED = 42`) → déterministe.
- **Proportions 70/15/15** : constantes en tête de module, ajustables. 70 % de train
  protège la qualité du modèle servi ; 15/15 laisse calibration et test utilisables sur
  master/challenger (GM sous-doté = flaw assumé, cf. §9).

### 4.3 Consommation
`train_player_ensemble.py`, `train_player_lp.py` et `calibrate_player_rank.py` lisent
`split.json` et filtrent leur dataset par bucket. Si `split.json` est absent → erreur
explicite (« lance `build_split.py` d'abord »), pas de repli silencieux.

## 5. Purge étendue aux trois sets

La purge existante (`train_player_ensemble.purged_train_features`) recalcule les agrégats
de train en excluant les matchs partagés avec un ensemble de joueurs cible. On la réutilise
telle quelle, appliquée à **deux niveaux** :

- **Niveau externe (nouveau)** : les features **train** sont purgées des matchs partagés
  avec **(calibration ∪ test)** ; les features **calibration** sont purgées des matchs
  partagés avec **test**. Garantit que test n'a influencé ni le modèle ni (plus tard) la
  calibration conformale. Les features **test** sont calculées telles quelles.
- **Niveau interne (inchangé)** : à l'intérieur du train, la k-fold de sélection continue
  de purger chaque fold-train des matchs partagés avec son fold-val.

Direction de la purge = « ce qu'on évalue ne doit jamais avoir façonné ce qui prédit » :
train ← purgé de calib∪test ; calib ← purgé de test ; test ← intact.

## 6. Refactor des scripts per-player

Comportement cible **identique** pour `train_player_ensemble.py` (classif AUC) et
`train_player_lp.py` (régression Spearman), en réutilisant leur `GRIDS`/`search_best`
existants :

1. Charger `split.json`, découper le dataset en train / calibration / test par `puuid`.
2. **Sélection** : k-fold CV **sur le train uniquement** (folds internes purgés). Choisir
   la meilleure config à la métrique OOF-train (AUC pour le rang, Spearman pooled pour le
   LP) — la calibration et le test ne sont **jamais** vus ici.
3. **Refit** de la meilleure config sur **tout le train** (purgé de calib∪test).
4. **Verdict** : évaluer sur le **test** (held-out). Ces chiffres deviennent le *headline*.
5. **Persistance** : sauvegarder les `.pkl` refités sur le train. La calibration reste sur
   disque via `split.json` (non consommée). Le test n'entre dans aucun modèle.

`calibrate_player_rank.py` : la table proba→rang (`player_rank_calibration.json`) est
calculée sur les **probabilités OOF du train** (pas en in-sample, pour ne pas réintroduire
d'optimisme) ; sa qualité est reportée sur le test.

## 7. Format des métriques (attente de baisse)

`player_metrics.json` et `player_lp_metrics.json` gagnent une structure explicite :

```json
{
  "cv_train": { "...métriques de sélection sur l'OOF du train..." },
  "test":     { "...headline honnête, held-out..." },
  "split": {"proportions": ..., "n_by_bucket": ...}
}
```

⚠️ **Attente à acter dès maintenant** : le *headline* test sera **plus bas** que les
chiffres OOF-à-plat actuels — double effet (1) fin de l'optimisme de sélection, (2) modèle
entraîné sur ~70 % des joueurs. Ce n'est pas une régression, c'est la mesure honnête.
On documentera l'ancien et le nouveau côte à côte dans `player_metrics.json` et CLAUDE.md.

## 8. Arrêt du per-game (documenté)

- En-tête de docstring **« DÉPRÉCIÉ — arrêté le 2026-07-18 »** sur `train_ensemble.py` et
  `calibrate_rank.py`, avec la raison : AUC trop basse (~0.63 dia_chall / ~0.59 high_elo),
  **non servi en prod**, remplacé par le per-player « constance/plancher ». Trop de variance
  intrinsèque (1 game = signal quasi aléatoire) pour une valeur prédictive utile.
- **Ne rien supprimer** : code et artefacts (`*_highelo.pkl`, `rank_calibration.json`)
  conservés pour l'historique et la reproductibilité.
- Entrée dans **CLAUDE.md** (section pipeline ML + état d'avancement) actant l'arrêt.
- `analyze_auc_vs_ngames.py` est **per-player** → conservé, hors périmètre d'arrêt.

## 9. Conséquences, risques, alternatives

- **Flaw GM (assumé)** : 78 GM → ~55 train / ~12 calibration / ~12 test. Métriques et
  (plus tard) couverture conformale GM bruitées. Documenté façon « FLAW ASSUMÉ » ;
  remédiation future = script dédié (densification GM ou protocole GM spécifique).
- **Coût en données du modèle servi** : entraîné sur ~70 % des joueurs (calib+test tenus
  hors modèle pour garder AOS4 valide). Conséquence acceptée ; proportions ajustables.
- **Alternative écartée — nested CV** : plus rigoureuse et sans coût en données, mais moins
  compréhensible et ~5× le compute (EBM lourd). Écartée au profit de la lisibilité.
- **Alternative différée — refit final sur train+test** : possible une fois le headline test
  figé (modèle servi plus fort), mais consommerait le test. Reporté ; pas dans cette spec.

## 10. Tests (pytest)

- `split.json` : déterministe (même seed → même assignation) ; buckets disjoints ; union =
  toute la population ; proportions par tier respectées à ε près.
- Cohérence inter-modèles : un `puuid` présent dans le dataset LP a le **même** bucket que
  dans le dataset rang.
- Purge externe : aucun `match_id` partagé entre les joueurs de train et ceux de
  (calibration ∪ test) ne subsiste dans le calcul des features de train.
- Garde d'absence : appel d'un script per-player sans `split.json` → erreur explicite.

## 11. Critère de succès

1. `build_split.py` produit un `split.json` déterministe, stratifié, disjoint, couvrant
   toute la population per-player.
2. Les deux modèles per-player servis s'entraînent en CV-sur-train et publient un
   **headline test** (attendu plus bas, documenté).
3. Le set **calibration** existe, est réservé, et **aucune ligne de conformal** n'est écrite.
4. Le per-game est marqué déprécié (docstrings + CLAUDE.md), sans suppression.
5. Tests verts ; `ml_rank.py` continue de servir sans changement d'interface.
