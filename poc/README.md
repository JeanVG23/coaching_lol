# POC — hypothèse « challenger = consistance, pas pic »

> **Implémenté en prod** (2026-07) : cf.
> `docs/superpowers/specs/2026-07-03-per-player-consistency-design.md` et
> `docs/superpowers/plans/2026-07-03-per-player-consistency.md`. Pipeline :
> `src/core/ml_features.py` (agrégation partagée) → `src/01_data_engineering/build_player_dataset.py`
> → `src/02_data_science/train_player_ensemble.py` → `calibrate_player_rank.py` →
> `web/backend/ml_rank.py` (seuil relevé à 5 games ADC, cf. `player_metrics.json`
> pour les métriques mesurées en prod).

## Question

En agrégeant ≥3 games par joueur (au lieu d'1 ligne = 1 game), on réduit le bruit
d'une mauvaise partie **et** on rend testable l'idée que le rang se distingue par la
**dispersion** du joueur (std / plancher p10 / plafond p90) plutôt que par sa
tendance centrale (mean / p50). Hypothèse : un challenger brille par sa **constance**
(ou son **plancher**), pas par son pic.

## Méthode (`per_player_hypothesis.py`)

1. Subset **dia_chall référentiel** (diamond vs challenger — extrêmes, signal propre).
2. Pour chaque joueur à ≥ `--min-games` games : par feature → `mean, std, p10, p50, p90`
   + `n_games`. Rang résolu au **mode** (tie-break rang le plus bas, cf. CLAUDE.md).
3. **Baseline apples-to-apples** : modèle per-game sur les *mêmes* joueurs (≥3 games),
   split groupé par puuid → isole l'effet *agrégation* de l'effet *subset*.
4. Modèle per-player : ensemble **XGB + RF + EBM**, CV stratifiée (1 row = 1 joueur).
5. **Test d'hypothèse** : masse `|SHAP|` (xgb+rf) groupée par type d'agrégat.
   - dispersion = `{__std, __p10, __p90}` vs centrale = `{__mean, __p50}`
   - cross-check **EBM** (main effects, biais inductif ≠ arbres).

Lancement :
```bash
.venv/bin/python poc/per_player_hypothesis.py                 # dia_chall, min-games=3
.venv/bin/python poc/per_player_hypothesis.py --min-games 5
.venv/bin/python poc/per_player_hypothesis.py --target high_elo
```
Sortie : console + `poc/results.json`.

## Résultats observés (juil. 2026)

### Avant densification (dataset original)
| min_games | joueurs | chall / dia | AUC base per-game | AUC per-player | ΔAUC | disp. SHAP | EBM disp. |
|---|---|---|---|---|---|---|---|
| 3 | 166 | 141 / 25 | 0.643 | 0.768 | +0.126 | 56.9 % | 56.8 % |
| 5 | 88  | 75 / 13  | 0.644 | 0.760 | +0.116 | 40.4 % | 55.1 % |

→ diamond critique (25 puis 13 joueurs) ; xgb s'effondrait à 0.46 à min_games=5.
  Verdict non robuste au seuil.

### Après densification (scraping de 200 joueurs diamond, voir ci-dessous)
| min_games | joueurs | chall / dia | AUC base per-game | AUC per-player | ΔAUC | disp. SHAP | EBM disp. |
|---|---|---|---|---|---|---|---|
| 3 | 204 | 142 / **62** | 0.691 | **0.811** | **+0.120** | **58.1 %** | **56.9 %** |
| 5 | ~120 | ~95 / ~30  | ~0.69 | ~0.80 | ~+0.11 | 51.4 % | 54.3 % |

### Après densification poussée (batch 1 + batch 2, 400 primaires diamond)
| min_games | joueurs | chall / dia | AUC base per-game | AUC per-player | ΔAUC | disp. SHAP | EBM disp. |
|---|---|---|---|---|---|---|---|
| 3 | 307 | 143 / **164** | 0.707 | **0.852** | **+0.145** | **56.0 %** | **57.4 %** |

Classes enfin **équilibrées** (ratio 47/53). ADC diamond ≥3 games : 25 → 62 → **164**.
AUC per-player 0.768 → 0.811 → **0.852**. Hypothèse confirmée par deux biais inductifs
distincts (SHAP arbres 56 % + EBM glass-box 57 %). Feature #1 : `pos_avg_dist_to_ally__p10`
(plancher positionnel) — le **plancher** distingue le challenger, pas le plafond.

Top features SHAP (min_games=3, densifié) : `pos_avg_dist_to_ally__p10` (#1),
`pos_frac_roam_mid__mean`, `pos_wards_placed_early__mean`, `pos_max_map_depth__p50`,
`gpm14__p50`, `plates_diff_early__p10`, `pos_frac_overextended__std`, `gpm14__mean`.

`pos_avg_dist_to_ally__p10` (plancher de la distance aux alliés) sort **#1 dans les
deux runs** — le plancher positionnel distingue le challenger. `__p90` (plafond) reste
la stat la plus faible (18.6 % à ≥3, **8.6 %** à ≥5) : un challenger ne se distingue
**pas par son pic**.

### Densification effectuée (2026-07-02)
`build_referential.py --rank diamond --players 200 --games 25 --max-pages 20 --patch 16.13`
(+ ajout `startTime` à `riotlib.match_ids` pour ne pas fetcher de vieux patches).
Silver diamond : 2050 → **10012 rows**, puuids uniques 1487 → **7586**, joueurs ≥3
games (tous rôles) 64 → **255**. ADC diamond ≥3 games : 25 → **62**.

## Lecture — ce qui est solide vs ce qui reste fragile

**Solide (confirmé sur données densifiées) :**
- **L'agrégation aide** : +0.12 AUC vs baseline per-game sur les *mêmes* joueurs,
  stable sur les deux seuils. AUC per-player **0.811** (ensemble), xgb/rf/ebm
  convergent (0.81 / 0.81 / 0.78) — fini l'effondrement xgb.
- **« Pas le pic »** : `__p90` porte le moins de signal dans tous les runs
  (8-19 %). Un challenger ne se distingue pas par sa meilleure game.
- **Dispersion > tendance centrale** : 58 % (SHAP) / 57 % (EBM) à min_games=3 —
  deux biais inductifs distincts convergent. L'hypothèse « consistance/plancher »
  se confirme au seuil 55 %.
- **Le plancher positionnel** (`pos_avg_dist_to_ally__p10`) est la feature #1 —
  raffinement de l'hypothèse : le challenger se distingue par son **plancher**,
  pas seulement par sa basse volatilité.

**Encore fragile :**
- À `min_games=5`, le SHAP tree retombe à 51 % (sous le seuil strict), l'EBM tient
  à 54 %. La diamond classe reste plus fine que challenger (62 vs 142 à ≥3).
- Le flaw de transfert de rang (ADC ennemi hérite du rang de la game) persiste —
  orthogonal à ce POC.

## Verdict POC

**Hypothèse confirmée** sur données densifiées (min_games=3) : le rang se distingue
**davantage par la dispersion/plancher du joueur que par sa tendance centrale**,
et **pas du tout par son plafond**. Raffinement : le signal le plus fort est le
**plancher positionnel** (distance aux alliés en p10), pas la simple volatilité.

**Suites possibles :**
- Densifier encore diamond (objectif ~150 joueurs ≥3 ADC games) pour sécuriser
  min_games=5 sous le seuil strict.
- Rapatrier l'insight dans le modèle per-game : ajouter des agrégats roulants par
  joueur (std/p10 sur les N dernières games) comme features — calculés sur le fold
  d'entraînement pour éviter la fuite.
- Trancher plancher vs consistance : ajouter explicitement une feature
  `floor_score` (p10 normalisé par feature) et laisser SHAP la classer.

## Fichiers
- `per_player_hypothesis.py` — pipeline complet (agrégation, baseline, CV, SHAP, EBM).
- `results.json` — dernier run persisté.

## À noter
- `pos_avg_map_depth` / `pos_max_map_depth` : sens contre-intuitif (haut = rang
  inférieur), jamais à prescrire — mais ici on classifie, pas on prescribe.
- Le flaw de transfert de rang (ADC ennemi hérite du rang de la game) reste
  orthogonal à ce POC ; l'agrégation ne le corrige pas.