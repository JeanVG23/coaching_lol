# POC Régression LP (Master→GM→Challenger) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Déterminer si le LP réel (échelle continue partagée par Master/GM/Challenger) porte un signal prédictif au-delà du modèle binaire high/low actuel (AUC purgée 0.6354), via un POC confiné à `poc/`.

**Architecture:** Deux scripts indépendants — `poc/script/fetch_apex_lp.py` (3 appels API, écrit le LP courant de tous les joueurs apex dans `poc/output/apex_lp.json`) et `poc/script/train_lp_regression.py` (0 appel API, purged CV + XGBRegressor, écrit `poc/output/lp_regression_metrics.json` avec Spearman pooled et par tier).

**Tech Stack:** Python (poetry), pandas, xgboost (`XGBRegressor`), scikit-learn (`StratifiedKFold`), scipy (`spearmanr`), pytest.

## Global Constraints

- 0 nouveau scraping de games : tout part de `data/04_dataset/adc_dataset.parquet` (référentiel déjà collecté) + 3 appels API légers (`apex_league`).
- Diamond exclu entièrement (LP non comparable, divisions I-IV avec reset).
- Décalage temporel LP (games jusqu'à 13j vs LP actuel) **ignoré** pour ce POC — documenté comme limite connue dans les sorties.
- Purged CV obligatoire (réutilise `purged_train_features` de `src/02_data_science/train_player_ensemble.py`, importée telle quelle, jamais dupliquée) — sans elle le Spearman serait gonflé par la fuite des ~37% de games partagées entre les 2 ADC d'une game.
- Aucun artefact prod (`web/backend/ml_rank.py`, `*_player_highelo.pkl`, `player_metrics.json`) n'est lu en écriture ni modifié.
- **Déviation actée vs le spec** : le spec dit de partir de `adc_player_dataset.parquet`, mais ce fichier contient déjà le balance-cap prod (`n_min = min(pos, neg)` dans `build_player_dataset.py`) — il n'a que 378 joueurs master (sur 787 qualifiés réels) et 113 diamond (sur 237), car l'undersampling a déjà jeté le surplus low-elo avant l'écriture du parquet. Utiliser ce fichier tel quel donnerait un pool tronqué, contredisant l'objectif même du POC (récupérer les 1278 joueurs qualifiés Master/GM/Chall). Le plan reconstruit donc le pool qualifié directement depuis `adc_dataset.parquet` (per-game, non balancé) via une nouvelle fonction `qualified_apex_players` (Task 3) — `build_player_dataset.py` n'est pas modifié.

---

### Task 1: `fetch_apex_lp.py` — fetch en masse du LP apex

**Files:**
- Create: `poc/script/fetch_apex_lp.py`
- Modify: `.gitignore` (ajoute `poc/output/` — les sorties contiennent des puuids réels, même convention que `data/` qui est gitignoré)
- Test: `tests/test_poc_lp_regression.py` (nouveau fichier, partagé avec les tasks 2 et 3)

**Interfaces:**
- Produces: `fetch_apex_lp.build_lp_lookup(entries_by_tier: dict[str, list[dict]]) -> dict[str, dict]` — clé `puuid`, valeur `{"tier": str, "leaguePoints": int}`. Consommé par `train_lp_regression.py` (Task 3/4) via le fichier `poc/output/apex_lp.json` qu'il écrit.

- [ ] **Step 1: Ajouter `poc/output/` au `.gitignore`**

Ouvrir `.gitignore` et ajouter une ligne à la fin :

```
poc/output/
```

- [ ] **Step 2: Créer le fichier de test avec le premier test (échoue)**

Créer `tests/test_poc_lp_regression.py` :

```python
"""Tests des fonctions pures du POC régression LP (poc/script/). Le reste (appels
API, CV complète) est vérifié par exécution réelle, cf.
docs/superpowers/specs/2026-07-07-lp-regression-poc-design.md."""
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src" / "core"))
sys.path.insert(0, str(_ROOT / "poc" / "script"))

import fetch_apex_lp


# --- build_lp_lookup ---------------------------------------------------------

def test_build_lp_lookup_merges_tiers_and_skips_missing_puuid():
    entries_by_tier = {
        "challenger": [{"puuid": "a", "leaguePoints": 800}],
        "master": [{"puuid": "b", "leaguePoints": 50},
                   {"puuid": None, "leaguePoints": 10}],
    }
    lookup = fetch_apex_lp.build_lp_lookup(entries_by_tier)
    assert lookup == {
        "a": {"tier": "challenger", "leaguePoints": 800},
        "b": {"tier": "master", "leaguePoints": 50},
    }
```

- [ ] **Step 3: Lancer le test, vérifier qu'il échoue**

Run: `poetry run pytest tests/test_poc_lp_regression.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'fetch_apex_lp'` (le fichier n'existe pas encore).

- [ ] **Step 4: Créer `poc/script/fetch_apex_lp.py`**

```python
#!/usr/bin/env python3
"""
poc — fetch en masse du LP courant des joueurs apex (Master/GM/Challenger).

`apex_league(tier)` (riotlib) retourne la liste COMPLETE d'un tier apex (puuid +
leaguePoints pour chaque joueur classé) en 1 seul appel — donc récupérer le LP de
tous les joueurs apex ne coûte que 3 appels API, pas 1 par joueur.

Étape 1/2 du POC régression LP, cf.
docs/superpowers/specs/2026-07-07-lp-regression-poc-design.md.

Sortie : poc/output/apex_lp.json = {puuid: {"tier": str, "leaguePoints": int}}
Usage : poetry run python3 poc/script/fetch_apex_lp.py --region euw1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src" / "core"))
import riotlib as rl

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "output" / "apex_lp.json"
TIERS = ("challenger", "grandmaster", "master")


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def build_lp_lookup(entries_by_tier: dict[str, list[dict]]) -> dict[str, dict]:
    """entries_by_tier : {"challenger": [...], "grandmaster": [...], "master": [...]}
    (format brut apex_league, une entrée par joueur classé). Retourne
    {puuid: {"tier": str, "leaguePoints": int}}. Entrées sans puuid ignorées."""
    lookup: dict[str, dict] = {}
    for tier, entries in entries_by_tier.items():
        for e in entries:
            puuid = e.get("puuid")
            if not puuid:
                continue
            lookup[puuid] = {"tier": tier, "leaguePoints": int(e.get("leaguePoints", 0))}
    return lookup


def main() -> int:
    env = rl.load_env()
    api_key = env.get("RIOT_API_ID")
    platform = (arg("--region") or env.get("RIOT_REGION", "")).lower()
    if not api_key or not platform:
        print("✗ RIOT_API_ID + RIOT_REGION requis (--region euw1).", file=sys.stderr)
        return 1
    regional = rl.PLATFORM_TO_REGIONAL.get(platform)
    if not regional:
        print(f"✗ Région inconnue: {platform!r}", file=sys.stderr)
        return 1

    client = rl.RiotClient(api_key, regional, platform)

    entries_by_tier = {}
    for tier in TIERS:
        entries = client.apex_league(tier)
        entries_by_tier[tier] = entries
        print(f"  {tier}: {len(entries)} entrées", file=sys.stderr)

    lookup = build_lp_lookup(entries_by_tier)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(lookup, indent=2))
    print(f"\n✓ {len(lookup)} joueurs (LP courant) écrits dans {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Lancer le test, vérifier qu'il passe**

Run: `poetry run pytest tests/test_poc_lp_regression.py -v`
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add .gitignore poc/script/fetch_apex_lp.py tests/test_poc_lp_regression.py
git commit -m "poc(lp-regression): fetch en masse du LP apex (3 appels API)"
```

---

### Task 2: `lp_metrics.py` — Spearman pooled + par tier

**Files:**
- Create: `poc/script/lp_metrics.py`
- Test: `tests/test_poc_lp_regression.py` (ajout de tests)

**Interfaces:**
- Consumes: rien (module pur, `pandas` + `scipy.stats.spearmanr` seulement).
- Produces: `lp_metrics.spearman_report(df: pd.DataFrame) -> dict`. `df` doit avoir les colonnes `rank` (str), `y_true` (float), `y_pred` (float). Retourne
  `{"spearman_pooled": float, "spearman_by_tier": {tier: {"spearman": float|None, "n": int}}, "rmse_pooled": float, "n_players_total": int}`.
  Consommé par `train_lp_regression.py` (Task 3/4).

- [ ] **Step 1: Ajouter les tests (échouent)**

Ajouter à la fin de `tests/test_poc_lp_regression.py` :

```python
import lp_metrics


# --- spearman_report ----------------------------------------------------------

def test_spearman_report_pooled_and_by_tier():
    df = pd.DataFrame({
        "rank": ["master"] * 4 + ["challenger"] * 4,
        "y_true": [10, 20, 30, 40, 500, 600, 700, 800],
        "y_pred": [12, 18, 33, 38, 510, 590, 710, 790],
    })
    report = lp_metrics.spearman_report(df)
    assert report["spearman_pooled"] > 0.9
    assert report["spearman_by_tier"]["master"]["spearman"] == 1.0
    assert report["spearman_by_tier"]["challenger"]["spearman"] == 1.0
    assert report["spearman_by_tier"]["master"]["n"] == 4
    assert report["n_players_total"] == 8


def test_spearman_report_handles_small_tier_gracefully():
    df = pd.DataFrame({
        "rank": ["master", "master", "challenger"],
        "y_true": [10, 20, 500],
        "y_pred": [12, 18, 510],
    })
    report = lp_metrics.spearman_report(df)
    assert report["spearman_by_tier"]["challenger"]["spearman"] is None
    assert report["spearman_by_tier"]["challenger"]["n"] == 1


def test_spearman_report_rmse():
    df = pd.DataFrame({"rank": ["master", "master"], "y_true": [0, 10], "y_pred": [0, 0]})
    report = lp_metrics.spearman_report(df)
    assert report["rmse_pooled"] == pytest.approx(7.07, abs=0.01)
```

- [ ] **Step 2: Lancer les tests, vérifier qu'ils échouent**

Run: `poetry run pytest tests/test_poc_lp_regression.py -v -k spearman_report`
Expected: FAIL avec `ModuleNotFoundError: No module named 'lp_metrics'`.

- [ ] **Step 3: Créer `poc/script/lp_metrics.py`**

```python
#!/usr/bin/env python3
"""
poc — métriques de la régression LP : Spearman pooled + par tier, RMSE pooled.

Le Spearman pooled seul peut juste redécouvrir la frontière de tier connue (un
Challenger a par définition un LP plus haut qu'un Master) : le Spearman PAR TIER
(calculé séparément à l'intérieur de master, GM, challenger) est le vrai test de
l'hypothèse — il isole si le modèle discrimine une granularité de skill au-delà du
tier. Cf. docs/superpowers/specs/2026-07-07-lp-regression-poc-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_TIER_N = 3  # sous ce seuil, spearman non significatif -> None plutôt qu'une valeur trompeuse


def spearman_report(df: pd.DataFrame) -> dict:
    """df : colonnes rank (str), y_true (float), y_pred (float). Une ligne par
    joueur. Retourne spearman pooled + par tier (None si <MIN_TIER_N lignes ou
    y_true constant sur le tier) + rmse pooled."""
    pooled_rho = float(spearmanr(df["y_true"], df["y_pred"])[0])

    by_tier: dict[str, dict] = {}
    for tier, g in df.groupby("rank"):
        if len(g) < MIN_TIER_N or g["y_true"].nunique() < 2:
            by_tier[str(tier)] = {"spearman": None, "n": int(len(g))}
            continue
        rho = float(spearmanr(g["y_true"], g["y_pred"])[0])
        by_tier[str(tier)] = {"spearman": round(rho, 4), "n": int(len(g))}

    rmse = float(np.sqrt(np.mean((df["y_true"] - df["y_pred"]) ** 2)))

    return {
        "spearman_pooled": round(pooled_rho, 4),
        "spearman_by_tier": by_tier,
        "rmse_pooled": round(rmse, 2),
        "n_players_total": int(len(df)),
    }
```

- [ ] **Step 4: Lancer les tests, vérifier qu'ils passent**

Run: `poetry run pytest tests/test_poc_lp_regression.py -v -k spearman_report`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add poc/script/lp_metrics.py tests/test_poc_lp_regression.py
git commit -m "poc(lp-regression): métrique Spearman pooled + par tier"
```

---

### Task 3: `train_lp_regression.py` — pool qualifié Master/GM/Chall (non balancé)

**Files:**
- Create: `poc/script/train_lp_regression.py` (fonction pure de cette task ; le `main()` est ajouté en Task 4)
- Test: `tests/test_poc_lp_regression.py` (ajout de tests)

**Interfaces:**
- Consumes: `ml_features.FEATURES`, `ml_features.resolve_rank`, `ml_features.aggregate_player_features` (`src/core/ml_features.py`, inchangé).
- Produces: `train_lp_regression.qualified_apex_players(ref: pd.DataFrame, min_games: int = 15, features: list[str] | None = None) -> pd.DataFrame`. Colonnes en sortie : `puuid`, `rank`, puis `{feature}__{stat}` pour chaque feature × 5 stats, `win_rate`, `n_games`. Consommé par le `main()` de la Task 4.

- [ ] **Step 1: Ajouter les tests (échouent)**

Ajouter à la fin de `tests/test_poc_lp_regression.py` :

```python
import train_lp_regression


# --- qualified_apex_players ---------------------------------------------------

def _games(puuid, rank, n, csm10):
    return pd.DataFrame({
        "puuid": [puuid] * n,
        "rank": [rank] * n,
        "win": [1] * n,
        "csm10": csm10,
    })


def test_qualified_apex_players_filters_min_games_and_excludes_diamond():
    ref = pd.concat([
        _games("p1", "master", 3, [4.0, 6.0, 8.0]),   # qualifie : master, 3>=2 games
        _games("p2", "master", 1, [5.0]),              # exclu : trop peu de games
        _games("p3", "diamond", 3, [4.0, 6.0, 8.0]),   # exclu : diamond hors scope LP
    ], ignore_index=True)
    out = train_lp_regression.qualified_apex_players(ref, min_games=2, features=["csm10"])
    assert set(out["puuid"]) == {"p1"}
    assert out.iloc[0]["csm10__mean"] == pytest.approx(6.0)


def test_qualified_apex_players_resolves_rank_by_mode_across_all_games():
    ref = pd.concat([
        _games("p4", "master", 2, [4.0, 6.0]),
        _games("p4", "diamond", 1, [5.0]),
    ], ignore_index=True)
    out = train_lp_regression.qualified_apex_players(ref, min_games=2, features=["csm10"])
    assert list(out["rank"]) == ["master"]     # mode sur tout l'historique : 2 master > 1 diamond
```

- [ ] **Step 2: Lancer les tests, vérifier qu'ils échouent**

Run: `poetry run pytest tests/test_poc_lp_regression.py -v -k qualified_apex_players`
Expected: FAIL avec `ModuleNotFoundError: No module named 'train_lp_regression'`.

- [ ] **Step 3: Créer `poc/script/train_lp_regression.py` (partie pure seulement)**

```python
#!/usr/bin/env python3
"""
poc — régression LP (Master/GM/Challenger) : le pool qualifié ADC (>=15 games) est
reconstruit DIRECTEMENT depuis adc_dataset.parquet (per-game, non balancé) plutôt que
depuis adc_player_dataset.parquet, qui contient déjà le balance-cap prod
(n_min = min(pos, neg) dans build_player_dataset.py — ne garde que 378 des 787
masters qualifiés réels). Reconstruire le pool ici récupère les 1278 joueurs
Master/GM/Chall qualifiés sans toucher au pipeline prod. Diamond exclu (LP non
comparable, divisions avec reset). Cf.
docs/superpowers/specs/2026-07-07-lp-regression-poc-design.md.

Étape 2/2 du POC (0 appel API, lit poc/output/apex_lp.json produit par
fetch_apex_lp.py). Usage : poetry run python3 poc/script/train_lp_regression.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src" / "core"))              # riotlib, ml_features
sys.path.insert(0, str(_ROOT / "src" / "02_data_science"))   # purged_train_features
import ml_features as mf

APEX_TIERS = {"master", "grandmaster", "challenger"}


def qualified_apex_players(ref: pd.DataFrame, min_games: int = 15,
                            features: list[str] | None = None) -> pd.DataFrame:
    """Réplique build_player_rows (build_player_dataset.py) SANS l'étape de balance
    (undersampling) et restreint à Master/GM/Challenger (Diamond exclu). Le rang est
    résolu par mode sur TOUT l'historique du joueur (même logique que
    ml_features.resolve_rank), pas seulement ses games apex, pour rester cohérent
    avec le pipeline prod."""
    features = mf.FEATURES if features is None else features
    rows = []
    for puuid, g in ref.groupby("puuid"):
        if len(g) < min_games:
            continue
        rank = mf.resolve_rank(g)
        if rank not in APEX_TIERS:
            continue
        rec = {"puuid": puuid, "rank": rank}
        rec.update(mf.aggregate_player_features(g, features))
        rows.append(rec)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Lancer les tests, vérifier qu'ils passent**

Run: `poetry run pytest tests/test_poc_lp_regression.py -v -k qualified_apex_players`
Expected: PASS (2 tests).

- [ ] **Step 5: Lancer toute la suite de tests du fichier**

Run: `poetry run pytest tests/test_poc_lp_regression.py -v`
Expected: PASS (6 tests au total : 1 build_lp_lookup + 3 spearman_report + 2 qualified_apex_players).

- [ ] **Step 6: Commit**

```bash
git add poc/script/train_lp_regression.py tests/test_poc_lp_regression.py
git commit -m "poc(lp-regression): pool qualifié Master/GM/Chall non balancé"
```

---

### Task 4: `train_lp_regression.py` — assemblage complet (purged CV + XGBRegressor)

**Files:**
- Modify: `poc/script/train_lp_regression.py` (ajoute `main()`, importe `purged_train_features` et `spearman_report`)

**Interfaces:**
- Consumes: `train_player_ensemble.purged_train_features(ref, train_puuids, val_puuids, features=None) -> tuple[pd.DataFrame, list[str]]` (import direct depuis `src/02_data_science/train_player_ensemble.py`, non dupliqué — signature confirmée en Task 3 de l'exploration) ; `lp_metrics.spearman_report(df) -> dict` (Task 2) ; `qualified_apex_players` (Task 3).
- Produces: `poc/output/lp_regression_metrics.json`.

Ce `main()` est une intégration de pièces déjà testées unitairement (Task 2, Task 3) plus des appels sklearn/xgboost — pas de nouveau test unitaire ici, vérifié par exécution réelle en Task 5 (cohérent avec la convention du repo : `train_player_ensemble.py` ne teste pas non plus sa boucle CV/fit unitairement, cf. commentaire en tête de `tests/test_train_player_ensemble.py`).

- [ ] **Step 1: Ajouter les imports et `main()` à la fin de `poc/script/train_lp_regression.py`**

Ajouter après les imports existants (`import ml_features as mf`) :

```python
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from train_player_ensemble import purged_train_features
from lp_metrics import spearman_report  # même dossier (poc/script), déjà sur sys.path
import riotlib as rl

DATASET_PER_GAME = rl.DATA / "04_dataset" / "adc_dataset.parquet"
LP_LOOKUP_PATH = Path(__file__).resolve().parent.parent / "output" / "apex_lp.json"
OUTPUT = Path(__file__).resolve().parent.parent / "output" / "lp_regression_metrics.json"
MIN_GAMES = 15
```

Puis, à la fin du fichier (après `qualified_apex_players`) :

```python
def make_regressor() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, tree_method="hist", random_state=42,
    )


def main() -> int:
    import json

    full = pd.read_parquet(DATASET_PER_GAME)
    ref = full[full["source"] == "referentiel"].copy()
    df = qualified_apex_players(ref, min_games=MIN_GAMES)
    print(f"  {len(df)} joueurs qualifiés Master/GM/Chall (>={MIN_GAMES} games)")
    print(f"  par tier : {df['rank'].value_counts().to_dict()}")

    if not LP_LOOKUP_PATH.exists():
        print(f"✗ {LP_LOOKUP_PATH} introuvable — lancer fetch_apex_lp.py d'abord.",
              file=sys.stderr)
        return 1
    lp_lookup = json.loads(LP_LOOKUP_PATH.read_text())

    df["leaguePoints"] = df["puuid"].map(
        lambda p: lp_lookup.get(p, {}).get("leaguePoints"))
    n_dropped = int(df["leaguePoints"].isna().sum())
    df = df.dropna(subset=["leaguePoints"]).reset_index(drop=True)
    df["leaguePoints"] = df["leaguePoints"].astype(float)
    print(f"  {n_dropped} joueurs sans LP actuel (tier changé depuis la collecte)")
    print(f"  {len(df)} joueurs retenus pour l'entraînement")
    if len(df) < 20:
        print("✗ Pool trop petit après filtrage LP, abandon.", file=sys.stderr)
        return 1

    features = mf.player_feature_names(mf.FEATURES)
    X = df.reindex(columns=features)
    y = df["leaguePoints"]
    y_of = dict(zip(df["puuid"], y))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(df))
    for train_idx, val_idx in cv.split(X, df["rank"]):
        X_val = X.iloc[val_idx]
        train_puuids = df["puuid"].iloc[train_idx].tolist()
        val_puuids = set(df["puuid"].iloc[val_idx])
        Xtr, dropped = purged_train_features(ref, train_puuids, val_puuids,
                                             features=mf.FEATURES)
        Xtr = Xtr[Xtr["puuid"].isin(y_of)]  # sécurité : ne garder que des puuids connus
        y_train = Xtr["puuid"].map(y_of).astype(float)
        model = make_regressor()
        model.fit(Xtr.reindex(columns=features), y_train)
        oof[val_idx] = model.predict(X_val)

    report_df = pd.DataFrame({"rank": df["rank"].values, "y_true": y.values, "y_pred": oof})
    report = spearman_report(report_df)
    report["n_players_by_tier"] = {k: int(v) for k, v in df["rank"].value_counts().items()}
    report["n_dropped_no_lp"] = n_dropped

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2))

    print(f"\n  Spearman pooled = {report['spearman_pooled']}  "
          f"(rmse={report['rmse_pooled']})")
    for tier, r in report["spearman_by_tier"].items():
        print(f"    {tier:<12} spearman={r['spearman']}  n={r['n']}")
    print(f"\n✓ Métriques écrites dans {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Relancer la suite de tests (non-régression des fonctions pures)**

Run: `poetry run pytest tests/test_poc_lp_regression.py -v`
Expected: PASS (6 tests, inchangé — `main()` n'est pas testé unitairement, cf. rationale ci-dessus).

- [ ] **Step 3: Vérifier que le script s'importe sans erreur (sans l'exécuter)**

Run: `poetry run python3 -c "import sys; sys.path.insert(0, 'poc/script'); sys.path.insert(0, 'src/core'); sys.path.insert(0, 'src/02_data_science'); import train_lp_regression"`
Expected: pas d'erreur (vérifie que tous les imports résolvent, y compris `purged_train_features` depuis `train_player_ensemble`).

- [ ] **Step 4: Commit**

```bash
git add poc/script/train_lp_regression.py
git commit -m "poc(lp-regression): purged CV + XGBRegressor, écrit lp_regression_metrics.json"
```

---

### Task 5: Exécution réelle et lecture des résultats

**Files:** aucun fichier créé (les sorties vont dans `poc/output/`, gitignoré depuis Task 1).

- [ ] **Step 1: Lancer `fetch_apex_lp.py` contre l'API réelle**

Run: `poetry run python3 poc/script/fetch_apex_lp.py --region euw1`
Expected: 3 lignes `<tier>: N entrées` (ordre de grandeur attendu : ~300 challenger, quelques centaines à ~1000 grandmaster, plusieurs milliers master), puis `✓ N joueurs (LP courant) écrits dans .../poc/output/apex_lp.json`.

- [ ] **Step 2: Lancer `train_lp_regression.py`**

Run: `poetry run python3 poc/script/train_lp_regression.py`
Expected : logs de pool (attendu proche de 787 master / 385 challenger / 106 grandmaster avant filtrage LP, cf. `dataset_report.py` — un sous-ensemble sera droppé faute de LP actuel), puis le bloc Spearman pooled + par tier, puis `✓ Métriques écrites dans .../poc/output/lp_regression_metrics.json`.

- [ ] **Step 3: Lire `poc/output/lp_regression_metrics.json` et trancher**

Appliquer la règle de décision du spec (§Séquencement) :
- Si `spearman_by_tier` est proche de 0 dans au moins 2 des 3 tiers → signal LP négatif, ne pas poursuivre vers une implémentation prod.
- Si `spearman_by_tier` est notablement > 0 dans au moins `master` (le tier le plus peuplé) → signal prometteur, discuter d'une vraie implémentation avec l'utilisateur (hors scope de ce plan).

- [ ] **Step 4: Rapporter la conclusion à l'utilisateur**

Résumer dans le chat : `n_players_total`, `n_players_by_tier`, `n_dropped_no_lp`, `spearman_pooled`, `spearman_by_tier` (les 3 tiers), `rmse_pooled`, et la conclusion (signal réel vs redécouverte du tier vs négatif). Ne pas committer de fichier à cette étape (sorties gitignorées) — la trace du POC reste dans les 3 scripts + tests committés aux tasks précédentes.

## Critères de succès du plan

- Les 6 tests de `tests/test_poc_lp_regression.py` passent.
- `poc/output/apex_lp.json` et `poc/output/lp_regression_metrics.json` existent après exécution réelle et ont la forme attendue.
- La conclusion signal/pas-signal est tranchée sans ambiguïté à partir de `spearman_by_tier` (pas seulement `spearman_pooled`).
