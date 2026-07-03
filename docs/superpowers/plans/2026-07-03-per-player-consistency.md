# Pipeline per-player (features de constance) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer l'estimation naïve du rang ML web (moyenne per-game) par le
pipeline per-player validé dans `poc/per_player_hypothesis.py` — features agrégées
(mean/std/p10/p50/p90) par joueur sur ≥5 games ADC, qui capture la dispersion/plancher
au lieu de la seule tendance centrale.

**Architecture:** Nouveau module partagé `src/core/ml_features.py` (FEATURES
canonique + agrégation, réutilisé offline et online). Nouveau pipeline parallèle
`build_player_dataset.py` → `train_player_ensemble.py` → `calibrate_player_rank.py`,
strictement additif (n'écrase jamais les artefacts per-game existants). `web/backend/ml_rank.py`
est réécrit pour consommer ce nouveau pipeline.

**Tech Stack:** Python, pandas/numpy, scikit-learn (RandomForest, StratifiedKFold),
xgboost, interpret (EBM), shap, FastAPI (web/backend), pytest.

## Global Constraints

- Seuil de games ADC requis : **5** (validé par le POC), pas de fallback per-game
  pour 3-4 games — `predict_rank` renvoie `None` en dessous.
- Ne jamais écraser les artefacts per-game existants : `xgb_highelo.pkl`,
  `rf_highelo.pkl`, `ebm_highelo.pkl`, `features.json`, `metrics.json`,
  `rank_calibration.json` (consommés par `shap_analysis.py`, `calibrate_rank.py`,
  `audit_leakage.py`, `train_ensemble.py`).
- Tous les nouveaux artefacts portent le marqueur `player` : `adc_player_dataset.parquet`,
  `{xgb,rf,ebm}_player_highelo.pkl`, `player_features.json`, `player_metrics.json`,
  `player_rank_calibration.json`.
- Le format JSON de `GET /api/c/{slug}/predicted-rank` ne change pas :
  `{"predicted_rank": str|None, "proba": float|None, "n_games_used": int}`.
- Convention flat-import du repo : chaque script insère lui-même `src/core` dans
  `sys.path` avant d'importer un module partagé (pas de package Python dans `src/`).
- Tests : `.venv/bin/python -m pytest tests/ -v`. `tests/conftest.py` ajoute déjà
  `src/core` (et d'autres dossiers `src/`) à `sys.path` pour tous les tests, y compris
  `tests/web/` — donc `import ml_features` et `import riotlib` fonctionnent directement
  dans n'importe quel fichier de test sans setup supplémentaire.
- Critère de succès mesurable (Task 4) : AUC out-of-fold per-player > 0.589 (AUC
  per-game `high_elo` actuel, cf. CLAUDE.md) — sinon s'arrêter avant de câbler le web
  (Task 6) et remonter le résultat.

---

## Task 1: Module partagé `src/core/ml_features.py`

**Files:**
- Create: `src/core/ml_features.py`
- Test: `tests/test_ml_features.py`

**Interfaces:**
- Produces: `FEATURES: list[str]` (41 noms de features, canonique) ; `RANK_ORD: dict[str,int]` ;
  `AGG_STATS: list[str]` (`["mean","std","p10","p50","p90"]`) ; `DISPERSION_STATS: set[str]`
  (`{"std","p10","p90"}`) ; `CENTRAL_STATS: set[str]` (`{"mean","p50"}`) ;
  `resolve_rank(group: pd.DataFrame) -> str` ; `player_feature_names(features: list[str] = FEATURES) -> list[str]` ;
  `aggregate_player_features(df: pd.DataFrame, features: list[str] = FEATURES) -> dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ml_features.py`:

```python
"""Tests de ml_features — agrégation per-player (mean/std/p10/p50/p90) et résolution
de rang, extraites de poc/per_player_hypothesis.py en module partagé (train + serve)."""
import numpy as np
import pandas as pd
import pytest

import ml_features as mf


def test_aggregate_player_features_basic_stats():
    df = pd.DataFrame({"csm10": [4.0, 6.0, 8.0]})
    agg = mf.aggregate_player_features(df, features=["csm10"])
    assert agg["csm10__mean"] == pytest.approx(6.0)
    assert agg["csm10__std"] == pytest.approx(2.0)
    assert agg["csm10__p50"] == pytest.approx(6.0)
    assert agg["n_games"] == 3


def test_aggregate_player_features_single_game_std_zero():
    df = pd.DataFrame({"csm10": [5.0]})
    agg = mf.aggregate_player_features(df, features=["csm10"])
    assert agg["csm10__std"] == 0.0
    assert agg["n_games"] == 1


def test_aggregate_player_features_missing_column_is_nan():
    df = pd.DataFrame({"csm10": [5.0, 6.0]})
    agg = mf.aggregate_player_features(df, features=["csm10", "gpm10"])
    assert np.isnan(agg["gpm10__mean"])
    assert np.isnan(agg["gpm10__std"])
    assert np.isnan(agg["gpm10__p10"])


def test_aggregate_player_features_all_nan_column_stays_nan():
    df = pd.DataFrame({"csm10": [np.nan, np.nan]})
    agg = mf.aggregate_player_features(df, features=["csm10"])
    assert np.isnan(agg["csm10__mean"])


def test_resolve_rank_mode_with_tie_break_lowest():
    group = pd.DataFrame({"rank": ["diamond", "master", "master", "diamond"]})
    assert mf.resolve_rank(group) == "diamond"


def test_resolve_rank_mode_no_tie():
    group = pd.DataFrame({"rank": ["challenger", "challenger", "diamond"]})
    assert mf.resolve_rank(group) == "challenger"


def test_player_feature_names_order():
    names = mf.player_feature_names(["csm10", "gpm10"])
    assert names == [
        "csm10__mean", "csm10__std", "csm10__p10", "csm10__p50", "csm10__p90",
        "gpm10__mean", "gpm10__std", "gpm10__p10", "gpm10__p50", "gpm10__p90",
        "n_games",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ml_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_features'`

- [ ] **Step 3: Implement `src/core/ml_features.py`**

```python
"""src/core/ml_features.py — features ADC canoniques + agrégation per-player.

Source unique de vérité pour la liste de features ML (déduplique
train_ensemble.py / poc/per_player_hypothesis.py) et pour l'agrégation per-player
(mean/std/p10/p50/p90), partagée entre l'entraînement offline
(build_player_dataset.py) et l'inférence online (web/backend/ml_rank.py) — même
code des deux côtés, pas de divergence train/serve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "csm10", "csm14", "gpm10", "gpm14", "xppm10",
    "n_deaths", "deaths_early", "deaths_mid", "deaths_late",
    "deaths_solo", "deaths_teamfight", "deaths_early_jungle", "deaths_early_2v2",
    "kills_solo", "kills_2v2", "assists_2v2", "kda_1v1", "kda_2v2",
    "support_deaths_early", "plates_diff_early", "frames_in_base_early",
    "frac_behind", "frac_ahead",
    "avg_dragon_prox",
    "pos_frac_own_lane_early", "pos_frac_river_early", "pos_frac_roam_mid",
    "pos_frac_enemy_half", "pos_frac_base",
    "pos_avg_map_depth", "pos_max_map_depth", "pos_frac_overextended",
    "pos_avg_dist_to_ally", "pos_gold_dead_time",
    "pos_wards_placed", "pos_wards_placed_early", "pos_control_wards_placed",
    "pos_wards_killed",
    "pos_frac_deaths_in_fog", "pos_avg_unaccounted_enemies", "pos_overext_x_unaccounted",
]

RANK_ORD = {"diamond": 0, "master": 1, "grandmaster": 2, "challenger": 3}
AGG_STATS = ["mean", "std", "p10", "p50", "p90"]
DISPERSION_STATS = {"std", "p10", "p90"}
CENTRAL_STATS = {"mean", "p50"}


def resolve_rank(group: pd.DataFrame) -> str:
    """Rang du joueur = mode de ses games ; tie-break sur le rang le plus bas
    (ne pas gonfler high_elo aux frontières — cf. CLAUDE.md)."""
    counts = group["rank"].value_counts()
    top = counts[counts == counts.max()]
    return sorted(top.index, key=lambda r: RANK_ORD[r])[0]


def player_feature_names(features: list[str] = FEATURES) -> list[str]:
    """Ordre canonique des colonnes agrégées : {feature}__{stat} puis n_games."""
    return [f"{f}__{s}" for f in features for s in AGG_STATS] + ["n_games"]


def aggregate_player_features(df: pd.DataFrame, features: list[str] = FEATURES) -> dict:
    """Games d'UN joueur (1 ligne par game) -> dict plat {feature}__{stat} + n_games.
    std ddof=1 (0.0 si une seule game). NaN propagée si la feature est absente ou
    vide sur tout le groupe (XGBoost/EBM gèrent le NaN nativement, pas d'imputation)."""
    rec: dict = {}
    for f in features:
        vals = df[f].dropna() if f in df.columns else pd.Series([], dtype=float)
        if vals.empty:
            rec[f"{f}__mean"] = np.nan
            rec[f"{f}__std"] = np.nan
            rec[f"{f}__p10"] = np.nan
            rec[f"{f}__p50"] = np.nan
            rec[f"{f}__p90"] = np.nan
        else:
            rec[f"{f}__mean"] = vals.mean()
            rec[f"{f}__std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            rec[f"{f}__p10"] = np.percentile(vals, 10)
            rec[f"{f}__p50"] = np.percentile(vals, 50)
            rec[f"{f}__p90"] = np.percentile(vals, 90)
    rec["n_games"] = len(df)
    return rec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ml_features.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/core/ml_features.py tests/test_ml_features.py
git commit -m "feat(core): add ml_features — canonical FEATURES + per-player aggregation"
```

---

## Task 2: Dédupliquer `FEATURES` dans `train_ensemble.py`

**Files:**
- Modify: `src/02_data_science/train_ensemble.py:41-73`

**Interfaces:**
- Consumes: `ml_features.FEATURES` (Task 1).
- Produces: aucun changement d'interface — `train_ensemble.FEATURES` garde le même
  nom et le même contenu, seule la source change.

- [ ] **Step 1: Remplacer la liste locale par un import**

Dans `src/02_data_science/train_ensemble.py`, remplacer :

```python
import riotlib as rl
from sklearn.model_selection import StratifiedGroupKFold
```

par :

```python
import riotlib as rl
import ml_features
from sklearn.model_selection import StratifiedGroupKFold
```

Puis remplacer tout le bloc (lignes 57-73) :

```python
FEATURES = [
    "csm10", "csm14", "gpm10", "gpm14", "xppm10",
    "n_deaths", "deaths_early", "deaths_mid", "deaths_late",
    "deaths_solo", "deaths_teamfight", "deaths_early_jungle", "deaths_early_2v2",
    "kills_solo", "kills_2v2", "assists_2v2", "kda_1v1", "kda_2v2",
    "support_deaths_early", "plates_diff_early", "frames_in_base_early",
    "frac_behind", "frac_ahead",
    "avg_dragon_prox",
    # positionnement macro (timeline, 0 CV)
    "pos_frac_own_lane_early", "pos_frac_river_early", "pos_frac_roam_mid",
    "pos_frac_enemy_half", "pos_frac_base",
    "pos_avg_map_depth", "pos_max_map_depth", "pos_frac_overextended",
    "pos_avg_dist_to_ally", "pos_gold_dead_time",
    "pos_wards_placed", "pos_wards_placed_early", "pos_control_wards_placed",
    "pos_wards_killed",
    "pos_frac_deaths_in_fog", "pos_avg_unaccounted_enemies", "pos_overext_x_unaccounted",
]
```

par :

```python
FEATURES = ml_features.FEATURES  # canonique, cf. src/core/ml_features.py
```

- [ ] **Step 2: Vérifier que le fichier s'importe et que la liste est inchangée**

Run:
```bash
.venv/bin/python -c "
import sys
sys.path.insert(0, 'src/core')
sys.path.insert(0, 'src/02_data_science')
import train_ensemble as te
assert len(te.FEATURES) == 41
assert te.FEATURES[0] == 'csm10' and te.FEATURES[-1] == 'pos_overext_x_unaccounted'
print('OK', len(te.FEATURES), 'features')
"
```
Expected: `OK 41 features`

- [ ] **Step 3: Commit**

```bash
git add src/02_data_science/train_ensemble.py
git commit -m "refactor(data-science): dedupe FEATURES via ml_features (no behavior change)"
```

---

## Task 3: `src/01_data_engineering/build_player_dataset.py`

**Files:**
- Create: `src/01_data_engineering/build_player_dataset.py`
- Test: `tests/test_build_player_dataset.py`

**Interfaces:**
- Consumes: `ml_features.FEATURES`, `ml_features.resolve_rank`,
  `ml_features.aggregate_player_features` (Task 1) ; lit `data/04_dataset/adc_dataset.parquet`
  (colonnes `puuid`, `rank`, `source`, + `ml_features.FEATURES`).
- Produces: fonction `build_player_rows(df: pd.DataFrame, min_games: int = 5) -> pd.DataFrame`
  (colonnes `puuid`, `rank`, `high_elo`, `n_games`, + `{feature}__{stat}` pour chaque
  feature × 5 stats) ; fichier `data/04_dataset/adc_player_dataset.parquet` (consommé
  par Task 4 et Task 5).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_player_dataset.py`:

```python
"""build_player_dataset vit dans src/01_data_engineering/ (dossier non importable
tel quel, cf. tests/test_build_dataset_flatten.py pour le même pattern de chargement)."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "core"))
_spec = importlib.util.spec_from_file_location(
    "build_player_dataset", _SRC / "01_data_engineering" / "build_player_dataset.py")
build_player_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_player_dataset)


def _rows(puuid, rank, values):
    return pd.DataFrame({
        "puuid": [puuid] * len(values),
        "rank": [rank] * len(values),
        "csm10": values,
    })


def test_build_player_rows_filters_below_min_games():
    df = pd.concat([
        _rows("p1", "diamond", [4.0, 5.0]),                       # 2 games < min
        _rows("p2", "challenger", [6.0, 7.0, 8.0, 9.0, 10.0]),    # 5 games >= min
    ], ignore_index=True)
    out = build_player_dataset.build_player_rows(df, min_games=5)
    assert list(out["puuid"]) == ["p2"]
    assert out.iloc[0]["n_games"] == 5


def test_build_player_rows_computes_high_elo_label():
    df = _rows("p1", "challenger", [6.0] * 5)
    out = build_player_dataset.build_player_rows(df, min_games=5)
    assert out.iloc[0]["high_elo"] == 1

    df2 = _rows("p2", "diamond", [6.0] * 5)
    out2 = build_player_dataset.build_player_rows(df2, min_games=5)
    assert out2.iloc[0]["high_elo"] == 0


def test_build_player_rows_aggregates_csm10():
    df = _rows("p1", "diamond", [4.0, 6.0, 8.0, 10.0, 12.0])
    out = build_player_dataset.build_player_rows(df, min_games=5)
    assert out.iloc[0]["csm10__mean"] == 8.0


def test_build_player_rows_empty_when_no_player_meets_threshold():
    df = _rows("p1", "diamond", [4.0, 5.0])
    out = build_player_dataset.build_player_rows(df, min_games=5)
    assert out.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_build_player_dataset.py -v`
Expected: FAIL — le fichier `src/01_data_engineering/build_player_dataset.py` n'existe
pas encore (`FileNotFoundError` ou `spec` None dans `importlib`).

- [ ] **Step 3: Implement `src/01_data_engineering/build_player_dataset.py`**

```python
#!/usr/bin/env python3
"""
01_data_engineering — dataset per-player (1 ligne = 1 joueur >= MIN_PLAYER_GAMES
games ADC référentiel).

Agrège adc_dataset.parquet (référentiel, 1 ligne = 1 ADC d'une game, déjà construit
par build_dataset.py) par puuid : pour chaque joueur à >= MIN_PLAYER_GAMES games,
mean/std/p10/p50/p90 par feature (cf. poc/per_player_hypothesis.py — hypothèse
"constance/plancher" validée sur données densifiées, +0.12 AUC vs per-game). Rang
résolu au mode (tie-break rang le plus bas, cf. ml_features.resolve_rank).

0 appel API (relit un dataset déjà construit).

Sortie : data/04_dataset/adc_player_dataset.parquet (+ .csv pour inspection).
Usage : .venv/bin/python src/01_data_engineering/build_player_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # riotlib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core")) # ml_features
import pandas as pd
import riotlib as rl
import ml_features as mf

DATASET_DIR = rl.DATA / "04_dataset"
MIN_PLAYER_GAMES = 5
HIGH_ELO = {"grandmaster", "challenger"}


def build_player_rows(df: pd.DataFrame, min_games: int = MIN_PLAYER_GAMES) -> pd.DataFrame:
    """df : rows per-game référentiel (colonnes puuid, rank, + mf.FEATURES).
    Retourne 1 ligne par joueur ayant >= min_games games, colonnes agrégées
    {feature}__{stat} + n_games, rank, high_elo. Vide si aucun joueur ne qualifie."""
    rows = []
    for puuid, g in df.groupby("puuid"):
        if len(g) < min_games:
            continue
        rec = {"puuid": puuid, "rank": mf.resolve_rank(g)}
        rec.update(mf.aggregate_player_features(g, mf.FEATURES))
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["high_elo"] = out["rank"].isin(HIGH_ELO).astype(int)
    return out


def main() -> int:
    df = pd.read_parquet(DATASET_DIR / "adc_dataset.parquet")
    ref = df[df["source"] == "referentiel"].copy()
    print(f"  {len(ref)} games référentiel | {ref['puuid'].nunique()} joueurs uniques")

    out = build_player_rows(ref)
    print(f"  >= {MIN_PLAYER_GAMES} games : {len(out)} joueurs")
    if out.empty:
        print("  ⚠ aucun joueur ne qualifie -> rien à écrire")
        return 1
    print(f"  répartition rangs : {dict(out['rank'].value_counts())}")
    print(f"  high_elo (GM+Chall=1) : {dict(out['high_elo'].value_counts())}")

    out.to_parquet(DATASET_DIR / "adc_player_dataset.parquet", index=False)
    out.to_csv(DATASET_DIR / "adc_player_dataset.csv", index=False)
    print(f"\n✓ Dataset per-player écrit dans {DATASET_DIR}/adc_player_dataset.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_build_player_dataset.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run against real data and inspect output**

Run: `.venv/bin/python src/01_data_engineering/build_player_dataset.py`
Expected output (approximatif, cf. exploration : 404 joueurs référentiel >=5 games ADC
sur le dataset actuel — le nombre exact peut varier légèrement) :
```
  11290 games référentiel | ... joueurs uniques
  >= 5 games : ~400 joueurs
  répartition rangs : {'master': ..., 'grandmaster': ..., 'diamond': ..., 'challenger': ...}
  high_elo (GM+Chall=1) : {0: ..., 1: ...}

✓ Dataset per-player écrit dans .../data/04_dataset/adc_player_dataset.parquet
```
Vérifier que `high_elo` a bien les deux classes représentées avec >=10 rows chacune
(sinon la CV du Task 4 sera dégénérée — si ce n'est pas le cas, s'arrêter et remonter
le résultat avant de continuer).

- [ ] **Step 6: Commit**

```bash
git add src/01_data_engineering/build_player_dataset.py tests/test_build_player_dataset.py
git commit -m "feat(data-engineering): add per-player dataset builder (mean/std/p10/p50/p90)"
```

---

## Task 4: `src/02_data_science/train_player_ensemble.py`

**Files:**
- Create: `src/02_data_science/train_player_ensemble.py`
- Test: `tests/test_train_player_ensemble.py`

**Interfaces:**
- Consumes: `ml_features.AGG_STATS`, `ml_features.DISPERSION_STATS`,
  `ml_features.CENTRAL_STATS`, `ml_features.FEATURES`, `ml_features.player_feature_names`
  (Task 1) ; lit `data/04_dataset/adc_player_dataset.parquet` (Task 3).
- Produces: fonction pure `dispersion_share_analysis(per_feature: dict[str, float]) -> dict`
  (clés `share_by_stat: dict[str,float]`, `dispersion_share_of_signal: float`) ; fichiers
  `data/05_model/{xgb,rf,ebm}_player_highelo.pkl`, `data/05_model/player_features.json`
  (liste de colonnes, même ordre que `ml_features.player_feature_names()`),
  `data/05_model/player_metrics.json` (clé `dispersion_analysis`) — consommés par
  Task 5 et Task 6.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_player_ensemble.py`:

```python
"""train_player_ensemble vit dans src/02_data_science/ (dossier non importable tel
quel, même pattern de chargement que tests/test_build_dataset_flatten.py). Seule la
fonction pure dispersion_share_analysis est testée ici — le reste (CV, fit des
modèles) est vérifié par exécution réelle (cf. plan, Task 4 Step 6)."""
import importlib.util
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC / "core"))
_spec = importlib.util.spec_from_file_location(
    "train_player_ensemble", _SRC / "02_data_science" / "train_player_ensemble.py")
train_player_ensemble = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_player_ensemble)


def test_dispersion_share_analysis_groups_by_stat_suffix():
    per_feature = {
        "csm10__mean": 1.0, "csm10__p50": 1.0,                        # central = 2.0
        "csm10__std": 3.0, "csm10__p10": 1.0, "csm10__p90": 0.0,      # dispersion = 4.0
        "n_games": 0.5,
    }
    result = train_player_ensemble.dispersion_share_analysis(per_feature)
    assert result["dispersion_share_of_signal"] == pytest.approx(4.0 / 6.0, abs=1e-4)
    assert result["share_by_stat"]["std"] == pytest.approx(3.0 / 6.5, abs=1e-4)
    assert result["share_by_stat"]["n_games"] == pytest.approx(0.5 / 6.5, abs=1e-4)


def test_dispersion_share_analysis_ignores_unknown_suffix():
    per_feature = {"weird__unknownstat": 5.0, "csm10__mean": 1.0}
    result = train_player_ensemble.dispersion_share_analysis(per_feature)
    # "unknownstat" n'est dans aucun bucket -> exclu du total (mean=1.0 seul compte)
    assert result["share_by_stat"]["mean"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_player_ensemble.py -v`
Expected: FAIL — le fichier `src/02_data_science/train_player_ensemble.py` n'existe
pas encore.

- [ ] **Step 3: Implement `src/02_data_science/train_player_ensemble.py`**

```python
#!/usr/bin/env python3
"""
02_data_science — entraîne l'ensemble per-player (XGBoost, Random Forest, EBM) pour
séparer high-elo (GM+Chall) de low (master+diamond) à partir des features agrégées
par joueur (mean/std/p10/p50/p90), cf. poc/per_player_hypothesis.py.

Reprend l'architecture de train_ensemble.py (mêmes 3 biais inductifs : GBDT / bagging
/ GA²M glass-box), appliquée au dataset per-player
(data/04_dataset/adc_player_dataset.parquet, 1 ligne = 1 joueur >= 5 games ADC).
EBM interactions=0 (vs 10 en per-game) : pas assez de rows pour des paires fiables sur
un espace de features ~5x plus large.

N'écrase JAMAIS les artefacts du pipeline per-game (xgb_highelo.pkl, features.json,
etc., cf. docs/superpowers/specs/2026-07-03-per-player-consistency-design.md) : tous
les fichiers de sortie portent le marqueur "player".

CV : StratifiedKFold (pas de group CV — 1 ligne = 1 joueur, aucune fuite joueur->fold
possible par construction, contrairement au per-game qui groupe par puuid).

Conserve le test d'hypothèse "constance" du POC (masse |SHAP| groupée par type
d'agrégat, dispersion vs tendance centrale) dans player_metrics.json, pour garder
l'observabilité en prod.

Sorties : data/05_model/{xgb,rf,ebm}_player_highelo.pkl, player_features.json,
player_metrics.json
Usage : .venv/bin/python src/02_data_science/train_player_ensemble.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # riotlib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core")) # ml_features
import numpy as np
import pandas as pd
import riotlib as rl
import ml_features as mf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import shap

DATASET = rl.DATA / "04_dataset" / "adc_player_dataset.parquet"
MODEL_DIR = rl.DATA / "05_model"
HIGH_ELO = {"grandmaster", "challenger"}


def make_models() -> dict:
    return {
        "xgb": xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.0, eval_metric="logloss", tree_method="hist",
            random_state=42,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=5,
            max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=42,
        ),
        "ebm": ExplainableBoostingClassifier(
            interactions=0, random_state=42,
        ),
    }


def dispersion_share_analysis(per_feature: dict[str, float]) -> dict:
    """per_feature : {"{feature}__{stat}": importance} (+ 'n_games' optionnel).
    Masse groupée par type d'agrégat -> test direct de l'hypothèse dispersion
    (std/p10/p90) vs tendance centrale (mean/p50). Suffixes hors AGG_STATS ignorés."""
    by_stat = {s: 0.0 for s in mf.AGG_STATS}
    by_stat["n_games"] = 0.0
    for fn, val in per_feature.items():
        if fn == "n_games":
            by_stat["n_games"] += val
            continue
        _, _, stat = fn.rpartition("__")
        if stat in by_stat:
            by_stat[stat] += val
    total = sum(by_stat.values()) or 1.0
    share = {k: round(v / total, 4) for k, v in by_stat.items()}
    disp_mass = sum(by_stat[s] for s in mf.DISPERSION_STATS)
    cent_mass = sum(by_stat[s] for s in mf.CENTRAL_STATS)
    disp_share = round(disp_mass / (disp_mass + cent_mass or 1), 4)
    return {"share_by_stat": share, "dispersion_share_of_signal": disp_share}


def shap_dispersion_analysis(X: pd.DataFrame, y: pd.Series, models: dict) -> dict:
    """SHAP (xgb+rf) sur les modèles finaux, résumé par type d'agrégat. Cross-check
    EBM (main effects, biais inductif différent des arbres)."""
    shap_xgb = np.abs(shap.TreeExplainer(models["xgb"]).shap_values(X))
    shap_rf = np.abs(shap.TreeExplainer(models["rf"]).shap_values(X))
    if shap_rf.ndim == 3:
        shap_rf = shap_rf[:, :, 1]
    mean_abs = (shap_xgb.mean(axis=0) + shap_rf.mean(axis=0)) / 2.0
    per_feat = dict(zip(X.columns, mean_abs.tolist()))
    tree_result = dispersion_share_analysis(per_feat)

    ebm_scores = {}
    data = models["ebm"].explain_global().data()
    for nm, sc in zip(data["names"], data["scores"]):
        arr = np.asarray(sc, dtype=float)
        ebm_scores[str(nm)] = float(np.mean(np.abs(arr))) if arr.size else 0.0
    ebm_result = dispersion_share_analysis(ebm_scores)

    top20 = sorted(per_feat.items(), key=lambda kv: kv[1], reverse=True)[:20]
    return {
        "shap_share_by_stat": tree_result["share_by_stat"],
        "shap_dispersion_share": tree_result["dispersion_share_of_signal"],
        "ebm_dispersion_share": ebm_result["dispersion_share_of_signal"],
        "top20_shap": [{"feature": k, "mean_abs_shap": round(v, 5)} for k, v in top20],
    }


def main() -> int:
    df = pd.read_parquet(DATASET)
    features = mf.player_feature_names(mf.FEATURES)
    X = df.reindex(columns=features)
    y = df["rank"].isin(HIGH_ELO).astype(int)
    print(f"  {len(df)} joueurs >=5 games | pos={int(y.sum())} / neg={int((1-y).sum())}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = {name: np.zeros(len(X)) for name in make_models().keys()}
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        models = make_models()
        for name, model in models.items():
            model.fit(X_train, y_train)
            oof_preds[name][val_idx] = model.predict_proba(X_val)[:, 1]

    per_model = {}
    print("\n  Perf par modèle (CV out-of-fold) :")
    for name, preds in oof_preds.items():
        m_auc = roc_auc_score(y, preds)
        m_acc = accuracy_score(y, (preds >= 0.5).astype(int))
        per_model[name] = {"auc": round(m_auc, 4), "acc": round(m_acc, 4)}
        print(f"    {name:<4} AUC={m_auc:.3f}  accuracy={m_acc:.3f}")

    ensemble_proba = np.mean(list(oof_preds.values()), axis=0)
    auc = roc_auc_score(y, ensemble_proba)
    acc = accuracy_score(y, (ensemble_proba >= 0.5).astype(int))
    print(f"\n  Ensemble CV out-of-fold : AUC={auc:.3f}  accuracy={acc:.3f}")
    print(classification_report(y, (ensemble_proba >= 0.5).astype(int),
                                target_names=["low(M/D)", "high(GM/C)"], digits=3))

    final_models = make_models()
    for name, model in final_models.items():
        model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in final_models.items():
        with open(MODEL_DIR / f"{name}_player_highelo.pkl", "wb") as f:
            pickle.dump(model, f)

    dispersion = shap_dispersion_analysis(X.fillna(X.median()), y, final_models)
    print(f"\n  -> dispersion (std/p10/p90) = {dispersion['shap_dispersion_share']:.1%} du "
          f"signal (SHAP) | EBM cross-check = {dispersion['ebm_dispersion_share']:.1%}")

    (MODEL_DIR / "player_features.json").write_text(json.dumps(features, indent=2))
    (MODEL_DIR / "player_metrics.json").write_text(json.dumps({
        "auc_cv": round(auc, 4), "acc_cv": round(acc, 4),
        "per_model_cv": per_model,
        "n_players": len(df), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "features": features,
        "dispersion_analysis": dispersion,
    }, indent=2))

    print(f"\n✓ Modèles per-player écrits dans {MODEL_DIR}/ (marqueur 'player_highelo')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_player_ensemble.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/02_data_science/train_player_ensemble.py tests/test_train_player_ensemble.py
git commit -m "feat(data-science): add per-player ensemble training (xgb/rf/ebm)"
```

- [ ] **Step 6: Run against real data and check the success criterion**

Run: `.venv/bin/python src/02_data_science/train_player_ensemble.py`

Puis :
```bash
.venv/bin/python -c "
import json
m = json.load(open('data/05_model/player_metrics.json'))
print('AUC per-player  :', m['auc_cv'])
print('AUC per-game (référence, CLAUDE.md) : 0.589')
print('dispersion share (SHAP) :', m['dispersion_analysis']['shap_dispersion_share'])
print('dispersion share (EBM)  :', m['dispersion_analysis']['ebm_dispersion_share'])
"
```

**Décision** : si `auc_cv > 0.589`, continuer vers Task 5/6. Si ce n'est pas le cas,
**s'arrêter ici** et remonter le résultat avant de câbler le web (cf. Global
Constraints) — ne pas broncher silencieusement en cas de régression.

---

## Task 5: `src/02_data_science/calibrate_player_rank.py`

**Files:**
- Create: `src/02_data_science/calibrate_player_rank.py`

**Interfaces:**
- Consumes: `data/05_model/player_features.json`, `data/05_model/{xgb,rf}_player_highelo.pkl`
  (Task 4), `data/04_dataset/adc_player_dataset.parquet` (Task 3).
- Produces: `data/05_model/player_rank_calibration.json` — liste de
  `{"rank": str, "mean_proba": float, "median_proba": float, "n": int}`, consommée par
  Task 6.

- [ ] **Step 1: Implement `src/02_data_science/calibrate_player_rank.py`**

```python
#!/usr/bin/env python3
"""
02_data_science — calibration proba -> rang pour le modèle per-player.

Miroir de calibrate_rank.py, appliqué aux modèles/dataset per-player. Le modèle
player_highelo est binaire (low M/D vs high GM/C) : on calibre la proba moyenne
ensemble (xgb+rf) par rang réel sur le dataset per-player, pour placer un joueur sur
les 4 rangs dans web/backend/ml_rank.py.

Sortie : data/05_model/player_rank_calibration.json
Usage : .venv/bin/python src/02_data_science/calibrate_player_rank.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # riotlib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core")) # (cohérent avec les autres scripts du pipeline, cf. Task 3/4)
import numpy as np
import pandas as pd
import riotlib as rl

DATASET = rl.DATA / "04_dataset" / "adc_player_dataset.parquet"
MODEL_DIR = rl.DATA / "05_model"
RANKS = ["diamond", "master", "grandmaster", "challenger"]


def main() -> int:
    features = json.loads((MODEL_DIR / "player_features.json").read_text())
    models = {}
    for name in ("xgb", "rf"):
        with open(MODEL_DIR / f"{name}_player_highelo.pkl", "rb") as f:
            models[name] = pickle.load(f)

    df = pd.read_parquet(DATASET)
    X = df.reindex(columns=features)
    proba = np.mean([m.predict_proba(X)[:, 1] for m in models.values()], axis=0)
    df["ensemble_proba"] = proba

    calibration = []
    print("  Calibration proba -> rang (per-player) :")
    for rank in RANKS:
        sub = df[df["rank"] == rank]["ensemble_proba"]
        if not len(sub):
            continue
        row = {"rank": rank, "mean_proba": round(float(sub.mean()), 4),
               "median_proba": round(float(sub.median()), 4), "n": int(len(sub))}
        calibration.append(row)
        print(f"    {rank:<12} mean={row['mean_proba']:.3f} "
              f"median={row['median_proba']:.3f}  n={row['n']}")

    (MODEL_DIR / "player_rank_calibration.json").write_text(json.dumps(calibration, indent=2))
    print(f"\n✓ Calibration écrite dans {MODEL_DIR}/player_rank_calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run against real data**

Run: `.venv/bin/python src/02_data_science/calibrate_player_rank.py`
Expected: 4 lignes affichées (diamond/master/grandmaster/challenger), `mean_proba`
strictement croissant de diamond à challenger (sinon la calibration est incohérente —
si ce n'est pas le cas, inspecter `player_metrics.json` avant de continuer).

Vérifier le fichier produit :
```bash
cat data/05_model/player_rank_calibration.json
```

- [ ] **Step 3: Commit**

```bash
git add src/02_data_science/calibrate_player_rank.py
git commit -m "feat(data-science): add per-player rank calibration"
```

---

## Task 6: Réécrire `web/backend/ml_rank.py`

**Files:**
- Modify: `web/backend/ml_rank.py` (réécriture complète)
- Modify: `tests/web/test_ml_rank.py` (réécriture complète)

**Interfaces:**
- Consumes: `ml_features.aggregate_player_features`, `ml_features.FEATURES` (Task 1) ;
  `build_dataset.game_to_row(g: dict, rank: str|None, source: str) -> dict` (inchangé,
  `src/01_data_engineering/build_dataset.py`) ; `data/05_model/{xgb,rf}_player_highelo.pkl`,
  `player_features.json`, `player_rank_calibration.json` (Tasks 4-5).
- Produces: `predict_rank(games: list[dict]) -> dict | None` — même forme de sortie
  qu'avant (`{"predicted_rank": str, "proba": float, "n_games_used": int}` ou `None`),
  consommée sans changement par `web/backend/routers/predicted_rank.py`.

- [ ] **Step 1: Write the failing tests (remplace le fichier existant)**

Remplacer entièrement `tests/web/test_ml_rank.py` par :

```python
"""Tests de ml_rank.predict_rank — agrégation per-player (mean/std/p10/p50/p90) +
mapping calibration. Modèles/calibration mockés (pas de dépendance aux .pkl réels)."""
import numpy as np
import pytest

import ml_rank


class FakeModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]])


ADC_GAME = {
    "match_id": "EUW1_1", "role": "BOTTOM", "champion": "Zeri", "win": True,
    "lane": {"csm10": 8.0, "csm14": 7.5, "gpm10": 400, "gpm14": 420, "xppm10": 500},
    "deaths": [], "kills": [], "assists": [],
    "avg_dragon_prox": 0.5, "support_deaths_early": 0,
    "plates_diff_early": 0, "frames_in_base_early": 0,
    "position": {},
}
NON_ADC_GAME = {**ADC_GAME, "role": "MIDDLE"}

CALIBRATION = [
    {"rank": "diamond", "mean_proba": 0.2},
    {"rank": "master", "mean_proba": 0.4},
    {"rank": "grandmaster", "mean_proba": 0.65},
    {"rank": "challenger", "mean_proba": 0.85},
]


def _patch_loaders(monkeypatch, proba):
    monkeypatch.setattr(ml_rank, "_load_models",
                        lambda: {"xgb": FakeModel(proba), "rf": FakeModel(proba)})
    monkeypatch.setattr(ml_rank, "_load_features", lambda: ["csm10__mean"])
    monkeypatch.setattr(ml_rank, "_load_calibration", lambda: CALIBRATION)


def test_predict_rank_none_when_not_enough_adc_games(monkeypatch):
    _patch_loaders(monkeypatch, 0.5)
    result = ml_rank.predict_rank([ADC_GAME] * 4)  # 4 < MIN_ADC_GAMES (5)
    assert result is None


def test_predict_rank_none_when_no_adc_games(monkeypatch):
    _patch_loaders(monkeypatch, 0.5)
    result = ml_rank.predict_rank([NON_ADC_GAME] * 5)
    assert result is None


def test_predict_rank_maps_to_closest_calibrated_rank(monkeypatch):
    _patch_loaders(monkeypatch, 0.8)
    result = ml_rank.predict_rank([ADC_GAME] * 5)
    assert result["predicted_rank"] == "challenger"
    assert result["n_games_used"] == 5
    assert result["proba"] == pytest.approx(0.8)


def test_predict_rank_filters_non_adc_games(monkeypatch):
    _patch_loaders(monkeypatch, 0.3)
    result = ml_rank.predict_rank([ADC_GAME] * 5 + [NON_ADC_GAME] * 2)
    assert result["n_games_used"] == 5
    assert result["predicted_rank"] == "diamond"
```

- [ ] **Step 2: Run tests to verify they fail against the current implementation**

Run: `.venv/bin/python -m pytest tests/web/test_ml_rank.py -v`
Expected: FAIL — `test_predict_rank_none_when_not_enough_adc_games` échoue (l'ancien
code accepte 4 games, `MIN_ADC_GAMES` vaut encore 3) ; les autres échouent aussi car
`_load_features`/`_load_calibration` mockés renvoient des noms/valeurs orientés
per-player que l'ancien code (moyenne per-game) n'utilise pas de la même façon.

- [ ] **Step 3: Rewrite `web/backend/ml_rank.py`**

```python
"""Estime le rang du joueur via l'ensemble ML per-player (xgb+rf) entraîné sur le
référentiel high-elo, appliqué aux features agrégées (mean/std/p10/p50/p90) des
dernières games ADC (BOTTOM) du joueur — cf.
docs/superpowers/specs/2026-07-03-per-player-consistency-design.md et
poc/per_player_hypothesis.py (hypothèse "constance/plancher" validée, +0.12 AUC vs
un modèle per-game moyenné).

Le modèle est binaire (low M/D vs high GM/C) : on place le joueur sur les 4 rangs en
comparant sa probabilité à une calibration par rang (mean_proba par rang sur le
dataset per-player), précalculée hors-ligne par
`src/02_data_science/calibrate_player_rank.py` et écrite dans
`data/05_model/player_rank_calibration.json`.

MIN_ADC_GAMES = 5 (seuil validé par le POC) : en dessous, pas de rang (pas de
fallback sur un autre chemin de code — décision actée en brainstorming).
"""
from __future__ import annotations

import functools
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import riotlib as rl

# riotlib est déjà résolu par le sys.path applicatif (main.py) ou par
# tests/conftest.py — mais ml_features (src/core/) n'y est pas garanti (dépend
# d'un état de réorg non commité), donc on l'ajoute nous-mêmes, défensivement,
# comme le fait déjà ce fichier pour DATA_ENG ci-dessous.
CORE = Path(__file__).resolve().parents[2] / "src" / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
import ml_features as mf

DATA_ENG = Path(__file__).resolve().parents[2] / "src" / "01_data_engineering"
if str(DATA_ENG) not in sys.path:
    sys.path.insert(0, str(DATA_ENG))
import build_dataset  # noqa: E402

MODEL_DIR = rl.DATA / "05_model"
MIN_ADC_GAMES = 5


@functools.lru_cache(maxsize=1)
def _load_models() -> dict:
    models = {}
    for name in ("xgb", "rf"):
        with open(MODEL_DIR / f"{name}_player_highelo.pkl", "rb") as f:
            models[name] = pickle.load(f)
    return models


@functools.lru_cache(maxsize=1)
def _load_features() -> list[str]:
    return json.loads((MODEL_DIR / "player_features.json").read_text())


@functools.lru_cache(maxsize=1)
def _load_calibration() -> list[dict]:
    return json.loads((MODEL_DIR / "player_rank_calibration.json").read_text())


def predict_rank(games: list[dict]) -> dict | None:
    """None si moins de MIN_ADC_GAMES games ADC (BOTTOM) dans l'historique fourni."""
    adc_games = [g for g in games if g.get("role") == "BOTTOM"]
    if len(adc_games) < MIN_ADC_GAMES:
        return None
    rows = pd.DataFrame([
        build_dataset.game_to_row(g, rank=None, source="inference") for g in adc_games
    ])
    agg = mf.aggregate_player_features(rows, mf.FEATURES)

    models = _load_models()
    features = _load_features()
    # astype(float) : une seule ligne agrégée avec un NaN isolé reste en dtype object
    # sans valeur de comparaison pour upcaster en float -> XGBoost refuse (même
    # raisonnement que l'ancien _game_proba, cf. historique du fichier).
    X = pd.DataFrame([agg]).reindex(columns=features).astype(float)
    probs = [m.predict_proba(X)[0, 1] for m in models.values()]
    player_proba = float(sum(probs) / len(probs))

    calibration = _load_calibration()
    closest = min(calibration, key=lambda c: abs(c["mean_proba"] - player_proba))
    return {
        "predicted_rank": closest["rank"],
        "proba": round(player_proba, 4),
        "n_games_used": len(adc_games),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/web/test_ml_rank.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full web test suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/web/ -v`
Expected: PASS — en particulier `tests/web/test_api.py` (qui mocke déjà
`routers.predicted_rank.ml_rank.predict_rank`, donc insensible au changement interne).

- [ ] **Step 6: Commit**

```bash
git add web/backend/ml_rank.py tests/web/test_ml_rank.py
git commit -m "feat(web): switch predicted-rank to the per-player ensemble (min 5 games)"
```

---

## Task 7: Vérification end-to-end

**Files:** aucun changement de fichier — vérification manuelle.

- [ ] **Step 1: Suite de tests complète**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS, aucune régression sur les tests existants.

- [ ] **Step 2: Démarrer le serveur web et vérifier l'endpoint réel**

```bash
cd web/backend && ../../.venv/bin/python -m uvicorn main:app --reload --port 8000 &
sleep 2
curl -s http://localhost:8000/api/c/<slug-existant>/predicted-rank | python3 -m json.tool
```
(remplacer `<slug-existant>` par un compte déjà collecté avec >= 5 games ADC connues,
ex. `spadzze-euw` si présent — vérifier via `readers.py`/la base de comptes existante).

Expected : réponse JSON `{"predicted_rank": "...", "proba": ..., "n_games_used": ...}`
avec `n_games_used >= 5`, ou `{"predicted_rank": null, "proba": null, "n_games_used": 0}`
si le compte a moins de 5 games ADC.

Arrêter le serveur après vérification (`kill %1` ou `Ctrl+C` selon le mode de lancement).

- [ ] **Step 3: Comparer au comportement précédent (sanity check qualitatif)**

Noter le `predicted_rank`/`proba` obtenu et le comparer mentalement au rang réel du
compte testé (`entries_by_puuid` / rang affiché sur `/c/{slug}`) — pas un test
automatisé, juste une vérification de non-absurdité avant de considérer la tâche
terminée.

---

## Task 8: Documentation

**Files:**
- Modify: `poc/README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Ajouter un pointeur prod dans `poc/README.md`**

Ajouter en tête de fichier, juste après le titre `# POC — hypothèse « challenger = consistance, pas pic »` :

```markdown
> **Implémenté en prod** (2026-07) : cf.
> `docs/superpowers/specs/2026-07-03-per-player-consistency-design.md` et
> `docs/superpowers/plans/2026-07-03-per-player-consistency.md`. Pipeline :
> `src/core/ml_features.py` (agrégation partagée) → `src/01_data_engineering/build_player_dataset.py`
> → `src/02_data_science/train_player_ensemble.py` → `calibrate_player_rank.py` →
> `web/backend/ml_rank.py` (seuil relevé à 5 games ADC, cf. `player_metrics.json`
> pour les métriques mesurées en prod).
```

- [ ] **Step 2: Mettre à jour `CLAUDE.md`**

Dans la section « Pipeline ML (en cours de structuration) », ajouter après le
paragraphe décrivant `calibrate_rank.py` (sous `src/02_data_science/`) :

```markdown
    **Pipeline per-player (features de constance)** : `src/core/ml_features.py`
    (FEATURES canonique + `aggregate_player_features` mean/std/p10/p50/p90, partagé
    train/serve) → `src/01_data_engineering/build_player_dataset.py` (1 ligne = 1
    joueur ≥5 games ADC référentiel) → `train_player_ensemble.py` (ensemble xgb/rf/ebm,
    StratifiedKFold, conserve le test d'hypothèse dispersion vs tendance centrale du
    POC dans `player_metrics.json`) → `calibrate_player_rank.py`. Implémente en prod
    `poc/per_player_hypothesis.py` : `web/backend/ml_rank.py` utilise désormais ce
    modèle (seuil `MIN_ADC_GAMES=5`, pas de fallback en dessous) au lieu de la moyenne
    per-game. Cf. `docs/superpowers/specs/2026-07-03-per-player-consistency-design.md`.
```

Dans la section « État d'avancement », ajouter une puce après le paragraphe
« Rang ML estimé (web) ✅ » :

```markdown
- **Rang ML per-player (constance)** ✅ — `web/backend/ml_rank.py` bascule sur le
  modèle per-player (features mean/std/p10/p50/p90, seuil 5 games ADC), qui reprend
  en prod l'hypothèse validée par `poc/per_player_hypothesis.py` (dispersion/plancher
  > tendance centrale). AUC per-player vs AUC per-game précédent (0.589) : voir
  `data/05_model/player_metrics.json` pour la valeur mesurée lors de l'entraînement.
```

- [ ] **Step 3: Commit**

```bash
git add poc/README.md CLAUDE.md
git commit -m "docs: point to the per-player pipeline now running in prod"
```
