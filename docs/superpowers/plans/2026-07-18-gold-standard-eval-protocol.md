# Protocole d'éval gold standard per-player + arrêt per-game — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faire passer les deux modèles per-player servis (rang high/low, régression LP) au protocole held-out test + CV-in-train via un split canonique partagé, et documenter l'arrêt du pipeline per-game.

**Architecture:** Un split canonique unique (`data/04_dataset/split.json`, par `puuid`, stratifié, graine fixe, 70/15/15) est généré une fois et consommé par tous les modèles per-player. La sélection des hyperparamètres se fait en k-fold **sur le train seul** ; le *headline* est mesuré sur le **test held-out** ; les sets **calibration** et **test** ne rentrent jamais dans le modèle servi (calibration réservée à une future couche AOS4, hors périmètre). La purge existante (`purged_train_features`) est réutilisée telle quelle en lui passant l'ensemble à exclure élargi (fold-val ∪ holdout).

**Tech Stack:** Python 3, Poetry, pandas, numpy, scikit-learn (StratifiedKFold, RandomForest), xgboost, interpret (EBM), shap, pytest. Convention flat-import (`sys.path.insert` de `src/` puis `src/core/` avant `import riotlib`/`ml_features`).

## Global Constraints

- **0 appel API** dans tous les scripts touchés (relecture de datasets/artefacts déjà construits).
- **Aucune logique conformal / intervalle de confiance** dans ce plan. Le set calibration est créé et *réservé*, rien de plus.
- **Aucune suppression** de code ni d'artefact existant (per-game déprécié = documenté, pas supprimé).
- `data/04_dataset/adc_player_dataset.parquet` et `adc_player_lp_dataset.parquet` **inchangés** (pas de refonte dataset/features).
- **Graine fixe** partout : `SEED = 42`. **Proportions** `train=0.70, calibration=0.15, test=0.15`.
- **Holdout par modèle** = `(calibration ∪ test) ∩ {puuid du dataset de CE modèle}` — chaque modèle purge son train des matchs partagés avec les joueurs holdout de sa propre population.
- **Interface de `web/backend/ml_rank.py` inchangée** (mêmes noms d'artefacts `.pkl` / `.json`, mêmes clés servies).
- Convention flat-import respectée : `sys.path.insert(0, .../src)` (riotlib) puis `.../src/core` (ml_features, dataset_split).

---

## Structure de fichiers

- **Create** `src/core/dataset_split.py` — chargement/lookup du split partagé (loader, `puuids_in`, `partition`).
- **Create** `src/01_data_engineering/build_split.py` — générateur du `split.json` (stratifié, déterministe).
- **Create** `tests/test_dataset_split.py` — tests du split (déterminisme, disjonction, couverture, stratification, garde d'absence).
- **Modify** `src/02_data_science/train_player_ensemble.py` — protocole held-out (CV sur train, headline sur test, refit sur train, export OOF train). Retrait des variantes naïve/contrôle (superseded par le test held-out).
- **Modify** `src/02_data_science/train_player_lp.py` — protocole held-out (recherche sur train, headline sur test, refit sur train).
- **Modify** `src/02_data_science/calibrate_player_rank.py` — calibration proba→rang sur les **OOF du train** (via `player_train_oof.json`), plus de proba in-sample.
- **Modify** `src/02_data_science/train_ensemble.py` — en-tête « DÉPRÉCIÉ ».
- **Modify** `src/02_data_science/calibrate_rank.py` — en-tête « DÉPRÉCIÉ ».
- **Modify** `CLAUDE.md` — note protocole gold standard + arrêt per-game + attente de baisse des métriques.

---

## Task 1: Split canonique — module, générateur, tests

**Files:**
- Create: `src/core/dataset_split.py`
- Create: `src/01_data_engineering/build_split.py`
- Test: `tests/test_dataset_split.py`

**Interfaces:**
- Produces:
  - `dataset_split.SPLIT_PATH: Path`, `dataset_split.BUCKETS: tuple[str,str,str]`
  - `dataset_split.load_split(path: Path = SPLIT_PATH) -> dict` (lève `FileNotFoundError` si absent)
  - `dataset_split.puuids_in(split: dict, bucket: str) -> set[str]`
  - `dataset_split.partition(df: pd.DataFrame, split: dict, bucket: str, col: str = "puuid") -> pd.DataFrame`
  - `build_split.assign(ranks: dict[str,str], proportions=..., seed=42) -> dict[str,str]`
  - Artefact `data/04_dataset/split.json` (clés : `seed`, `proportions`, `created_from`, `n_by_bucket_by_rank`, `assignment`)

- [ ] **Step 1: Écrire le module `src/core/dataset_split.py`**

```python
#!/usr/bin/env python3
"""core — split canonique train/calibration/test au niveau joueur, partagé par tout
le pipeline per-player. Généré une fois par build_split.py, consommé par lookup puuid.
Voir docs/superpowers/specs/2026-07-18-gold-standard-eval-protocol-design.md."""
from __future__ import annotations

import json
from pathlib import Path

import riotlib as rl

SPLIT_PATH = rl.DATA / "04_dataset" / "split.json"
BUCKETS = ("train", "calibration", "test")


def load_split(path: Path = SPLIT_PATH) -> dict:
    """Charge le split. Lève FileNotFoundError explicite si absent (pas de repli)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"split introuvable ({path}). Lance d'abord "
            "`poetry run python3 src/01_data_engineering/build_split.py`."
        )
    return json.loads(Path(path).read_text())


def puuids_in(split: dict, bucket: str) -> set[str]:
    if bucket not in BUCKETS:
        raise ValueError(f"bucket inconnu: {bucket!r} (attendus: {BUCKETS})")
    return {p for p, b in split["assignment"].items() if b == bucket}


def partition(df, split: dict, bucket: str, col: str = "puuid"):
    """Sous-DataFrame des lignes dont le `col` est dans `bucket`."""
    return df[df[col].isin(puuids_in(split, bucket))].copy()
```

- [ ] **Step 2: Écrire le générateur `src/01_data_engineering/build_split.py`**

```python
#!/usr/bin/env python3
"""01_data_engineering — génère le split canonique train/calibration/test.

Au niveau joueur (puuid), stratifié par rang résolu, graine fixe. Population = UNION
des puuid des deux datasets per-player (rang + LP) : aucun joueur consommé par un
modèle n'échappe au split. Écrit data/04_dataset/split.json.

0 appel API. Voir docs/superpowers/specs/2026-07-18-gold-standard-eval-protocol-design.md.
Usage : poetry run python3 src/01_data_engineering/build_split.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # riotlib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core")) # dataset_split
import numpy as np
import pandas as pd
import riotlib as rl
from dataset_split import SPLIT_PATH, BUCKETS

DATASET_DIR = rl.DATA / "04_dataset"
RANK_DATASET = DATASET_DIR / "adc_player_dataset.parquet"
LP_DATASET = DATASET_DIR / "adc_player_lp_dataset.parquet"
SEED = 42
PROPORTIONS = {"train": 0.70, "calibration": 0.15, "test": 0.15}


def rank_by_puuid() -> dict[str, str]:
    """puuid -> rang, union des deux datasets per-player (rang prioritaire, LP en repli)."""
    mapping: dict[str, str] = {}
    if LP_DATASET.exists():
        lp = pd.read_parquet(LP_DATASET, columns=["puuid", "rank"])
        mapping.update(dict(zip(lp["puuid"], lp["rank"])))
    rk = pd.read_parquet(RANK_DATASET, columns=["puuid", "rank"])
    mapping.update(dict(zip(rk["puuid"], rk["rank"])))  # priorité au dataset rang
    return mapping


def assign(ranks: dict[str, str], proportions: dict = PROPORTIONS,
           seed: int = SEED) -> dict[str, str]:
    """Assigne chaque puuid à un bucket, stratifié par rang, déterministe (tri des
    puuid avant shuffle -> indépendant de l'ordre d'insertion)."""
    rng = np.random.RandomState(seed)
    by_rank: dict[str, list[str]] = defaultdict(list)
    for puuid, rank in sorted(ranks.items()):
        by_rank[rank].append(puuid)
    assignment: dict[str, str] = {}
    for rank in sorted(by_rank):
        puuids = by_rank[rank]
        rng.shuffle(puuids)
        n = len(puuids)
        n_train = int(round(n * proportions["train"]))
        n_calib = int(round(n * proportions["calibration"]))
        for i, puuid in enumerate(puuids):
            if i < n_train:
                assignment[puuid] = "train"
            elif i < n_train + n_calib:
                assignment[puuid] = "calibration"
            else:
                assignment[puuid] = "test"
    return assignment


def main() -> int:
    ranks = rank_by_puuid()
    assignment = assign(ranks)
    n_by = {b: defaultdict(int) for b in BUCKETS}
    for puuid, bucket in assignment.items():
        n_by[bucket][ranks[puuid]] += 1
    out = {
        "seed": SEED,
        "proportions": PROPORTIONS,
        "created_from": [RANK_DATASET.name]
                        + ([LP_DATASET.name] if LP_DATASET.exists() else []),
        "n_by_bucket_by_rank": {b: dict(sorted(n_by[b].items())) for b in BUCKETS},
        "assignment": assignment,
    }
    SPLIT_PATH.write_text(json.dumps(out, indent=2))
    print("  " + " / ".join(f"{b}:{sum(n_by[b].values())}" for b in BUCKETS)
          + f"  (total {len(assignment)} joueurs)")
    for b in BUCKETS:
        print(f"    {b:<12} {dict(sorted(n_by[b].items()))}")
    print(f"\n✓ Split écrit dans {SPLIT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Écrire les tests `tests/test_dataset_split.py`**

```python
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))                    # riotlib
sys.path.insert(0, str(ROOT / "src" / "core"))           # dataset_split
sys.path.insert(0, str(ROOT / "src" / "01_data_engineering"))  # build_split
import build_split as bs
import dataset_split as ds


def _ranks(n_per_rank: dict) -> dict:
    return {f"{rank}-{i}": rank for rank, n in n_per_rank.items() for i in range(n)}


def test_assign_deterministic():
    ranks = _ranks({"master": 100, "challenger": 40, "grandmaster": 12})
    assert bs.assign(ranks) == bs.assign(ranks)


def test_assign_disjoint_and_covers_all():
    ranks = _ranks({"master": 100, "challenger": 40, "grandmaster": 12})
    a = bs.assign(ranks)
    assert set(a) == set(ranks)                  # tout le monde est assigné
    assert set(a.values()) <= set(ds.BUCKETS)    # buckets valides seulement


def test_assign_stratified_proportions():
    a = bs.assign(_ranks({"master": 100}))
    counts = {b: sum(1 for v in a.values() if v == b) for b in ds.BUCKETS}
    assert counts == {"train": 70, "calibration": 15, "test": 15}


def test_load_split_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ds.load_split(tmp_path / "nope.json")


def test_puuids_in_rejects_unknown_bucket():
    with pytest.raises(ValueError):
        ds.puuids_in({"assignment": {}}, "validation")


def test_partition_filters_by_bucket():
    split = {"assignment": {"a": "train", "b": "test", "c": "train"}}
    df = pd.DataFrame({"puuid": ["a", "b", "c", "d"], "v": [1, 2, 3, 4]})
    assert set(ds.partition(df, split, "train")["puuid"]) == {"a", "c"}
    assert set(ds.partition(df, split, "test")["puuid"]) == {"b"}
```

- [ ] **Step 4: Lancer les tests, vérifier qu'ils passent**

Run: `poetry run pytest tests/test_dataset_split.py -v`
Expected: 6 tests PASS.

- [ ] **Step 5: Générer le split réel**

Run: `poetry run python3 src/01_data_engineering/build_split.py`
Expected: crée `data/04_dataset/split.json` ; affiche `train:… / calibration:… / test:…` avec le détail par rang. Vérifier à l'œil que les 3 buckets sont non vides et que GM apparaît dans chacun (petit, flaw assumé).

- [ ] **Step 6: Commit**

```bash
git add src/core/dataset_split.py src/01_data_engineering/build_split.py tests/test_dataset_split.py data/04_dataset/split.json
git commit -m "feat(split): split canonique train/calibration/test par joueur"
```

> Note : `data/` est gitignoré. Si `git add data/04_dataset/split.json` est refusé, faire `git add -f data/04_dataset/split.json` (le split est un artefact de config reproductible, comme `champion_traits.json` est force-add). Si le projet préfère ne pas versionner le split, l'omettre du commit — il est régénérable par `build_split.py`.

---

## Task 2: `train_player_ensemble.py` — protocole held-out

**Files:**
- Modify: `src/02_data_science/train_player_ensemble.py` (ajout import + réécriture de `main()`)

**Interfaces:**
- Consumes: `dataset_split.load_split/puuids_in/partition`, `purged_train_features` (inchangé), `make_models`, `shap_dispersion_analysis` (inchangés).
- Produces: `data/05_model/{xgb,rf}_player_highelo.pkl` (refités sur train), `player_features.json`, `player_metrics.json` (clés `cv_train`, `test`, `split`), et **nouveau** `player_train_oof.json` (`{puuid: proba_ensemble_oof}` sur le train) consommé par Task 4.

- [ ] **Step 1: Ajouter l'import du split**

Dans le bloc d'imports (après `import ml_features as mf`, ligne ~48), ajouter :

```python
import dataset_split as ds
```

- [ ] **Step 2: Remplacer entièrement `main()` (lignes 180-275) par la version held-out**

Retire les variantes `naive`/`control` (le test held-out remplace le diagnostic de fuite ; `purged_train_features` et `control_train_features` restent dans le fichier, `control_train_features` devient inutilisée mais **n'est pas supprimée** — cf. contrainte « aucune suppression »).

```python
def main() -> int:
    split = ds.load_split()
    df = pd.read_parquet(DATASET)
    ref = pd.read_parquet(DATASET_PER_GAME)
    ref = ref[(ref["source"] == "referentiel")
              & ref["puuid"].isin(set(df["puuid"]))].copy()
    features = mf.player_feature_names(mf.FEATURES)
    y_of = dict(zip(df["puuid"], df["rank"].isin(HIGH_ELO).astype(int)))

    pop = set(df["puuid"])
    train_p = ds.puuids_in(split, "train") & pop
    holdout = (ds.puuids_in(split, "calibration") | ds.puuids_in(split, "test")) & pop
    df_train = df[df["puuid"].isin(train_p)].copy()
    df_test = ds.partition(df, split, "test")
    print(f"  split: train={len(df_train)} test={len(df_test)} "
          f"(calibration réservée, non utilisée) | {len(ref)} games pour la purge")

    # --- Diagnostic/observabilité : CV purgée SUR LE TRAIN (purge externe = holdout) ---
    X_train_natural = df_train.reindex(columns=features)
    y_train = df_train["puuid"].map(y_of).astype(int).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = {name: np.zeros(len(df_train)) for name in make_models()}
    for tr_idx, va_idx in cv.split(X_train_natural, y_train):
        inner_train = df_train["puuid"].iloc[tr_idx].tolist()
        inner_val = set(df_train["puuid"].iloc[va_idx])
        Xtr, _ = purged_train_features(ref, inner_train, inner_val | holdout)
        y_inner = Xtr["puuid"].map(y_of).astype(int)
        Xva = X_train_natural.iloc[va_idx]
        for name, model in make_models().items():
            model.fit(Xtr.reindex(columns=features), y_inner)
            oof[name][va_idx] = model.predict_proba(Xva)[:, 1]
    ens_oof = np.mean(list(oof.values()), axis=0)
    auc_cv = roc_auc_score(y_train, ens_oof)
    acc_cv = accuracy_score(y_train, (ens_oof >= 0.5).astype(int))
    per_model = {name: {"auc": round(roc_auc_score(y_train, p), 4)}
                 for name, p in oof.items()}
    print(f"  CV train (purgée) : AUC={auc_cv:.3f}  acc={acc_cv:.3f}  n={len(df_train)}")

    # --- Modèle servi : refit sur le TRAIN, features purgées de holdout ---
    Xtr_final, _ = purged_train_features(ref, df_train["puuid"].tolist(), holdout)
    y_final = Xtr_final["puuid"].map(y_of).astype(int)
    Xtr_final_feat = Xtr_final.reindex(columns=features)
    final_models = make_models()
    for model in final_models.values():
        model.fit(Xtr_final_feat, y_final)

    # --- Headline : TEST held-out (features naturelles, comme au serving) ---
    X_test = df_test.reindex(columns=features)
    y_test = df_test["puuid"].map(y_of).astype(int).values
    ens_test = np.mean([m.predict_proba(X_test)[:, 1]
                        for m in final_models.values()], axis=0)
    auc_test = roc_auc_score(y_test, ens_test)
    acc_test = accuracy_score(y_test, (ens_test >= 0.5).astype(int))
    print(f"  TEST held-out (HEADLINE) : AUC={auc_test:.3f}  acc={acc_test:.3f}  "
          f"n={len(df_test)}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in final_models.items():
        with open(MODEL_DIR / f"{name}_player_highelo.pkl", "wb") as f:
            pickle.dump(model, f)
    # OOF du train exporté pour calibrate_player_rank.py (calibration hors in-sample)
    train_oof = {p: float(v) for p, v in zip(df_train["puuid"], ens_oof)}
    (MODEL_DIR / "player_train_oof.json").write_text(json.dumps(train_oof, indent=2))

    dispersion = shap_dispersion_analysis(
        Xtr_final_feat.fillna(Xtr_final_feat.median()), y_final, final_models)
    print(f"  dispersion (std/p10/p90) = {dispersion['shap_dispersion_share']:.1%} "
          f"du signal | EBM cross-check = {dispersion['ebm_dispersion_share']:.1%}")

    (MODEL_DIR / "player_features.json").write_text(json.dumps(features, indent=2))
    (MODEL_DIR / "player_metrics.json").write_text(json.dumps({
        "cv_train": {"auc": round(auc_cv, 4), "acc": round(acc_cv, 4),
                     "per_model": per_model, "n": len(df_train),
                     "n_pos": int(y_train.sum()), "n_neg": int((1 - y_train).sum())},
        "test": {"auc": round(auc_test, 4), "acc": round(acc_test, 4),
                 "n": len(df_test), "n_pos": int(y_test.sum()),
                 "n_neg": int((1 - y_test).sum())},
        "split": {"proportions": split["proportions"],
                  "n_by_bucket_by_rank": split["n_by_bucket_by_rank"]},
        "features": features,
        "dispersion_analysis": dispersion,
    }, indent=2))
    print("\n✓ Modèles per-player écrits (HEADLINE = TEST held-out ; "
          "calibration réservée AOS4)")
    return 0
```

- [ ] **Step 3: Vérifier que les tests de purge existants passent toujours**

`purged_train_features` est inchangée (on l'appelle juste avec un ensemble à exclure élargi).
Run: `poetry run pytest tests/test_train_player_ensemble.py -v`
Expected: PASS (aucune régression sur la logique de purge).

- [ ] **Step 4: Lancer l'entraînement, vérifier les artefacts et le sens des chiffres**

Run: `poetry run python3 src/02_data_science/train_player_ensemble.py`
Expected : affiche `CV train (purgée)` puis `TEST held-out (HEADLINE)` ; crée/écrase `xgb_player_highelo.pkl`, `rf_player_highelo.pkl`, `ebm` non requis ici (make_models en produit un mais seuls xgb/rf sont servis — laisser tel quel), `player_features.json`, `player_metrics.json` (avec clés `cv_train`/`test`/`split`), `player_train_oof.json`. **Attendu et normal** : `test.auc` plus bas que l'ancien `auc_cv` 0.635 (dé-optimisme + moins de données de train).

- [ ] **Step 5: Commit**

```bash
git add src/02_data_science/train_player_ensemble.py data/05_model/player_metrics.json data/05_model/player_features.json
git commit -m "feat(rank): held-out test + CV-in-train per-player, export OOF train"
```

---

## Task 3: `train_player_lp.py` — protocole held-out

**Files:**
- Modify: `src/02_data_science/train_player_lp.py` (import + `prepare_folds` + `main()`)

**Interfaces:**
- Consumes: `dataset_split.*`, `purged_train_features` (import existant depuis `train_player_ensemble`), `search_best`/`sample_configs`/`make_model`/`shap_top20` (inchangés), `spearman_report`/`_safe_spearman`.
- Produces: `data/05_model/{xgb,rf,ebm}_player_lp.pkl` (refités sur train), `player_lp_features.json`, `player_lp_metrics.json` (clés `cv_train`, `test`, `per_model_cv`, `split`, `shap`).

- [ ] **Step 1: Ajouter l'import du split**

Après `from train_player_ensemble import purged_train_features, dispersion_share_analysis` (ligne ~50) :

```python
import dataset_split as ds
```

- [ ] **Step 2: Modifier `prepare_folds` pour purger aussi du holdout externe**

Remplacer la signature et le corps (lignes ~115-133) :

```python
def prepare_folds(df_train: pd.DataFrame, ref: pd.DataFrame, features: list[str],
                  holdout_puuids: set) -> list[tuple]:
    """Précalcule par fold (sur le TRAIN uniquement) : (X_train purgé, y_train, X_val,
    val_idx). Chaque fold purge son train-fold des matchs partagés avec (fold-val ∪
    holdout externe) — les agrégats purgés ne dépendent pas des hyperparamètres, donc
    calculés UNE fois et réutilisés pour toutes les configs du random search."""
    X = df_train.reindex(columns=features)
    y_of = dict(zip(df_train["puuid"], df_train["lp"].astype(float)))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for train_idx, val_idx in cv.split(X, df_train["rank"]):
        train_puuids = df_train["puuid"].iloc[train_idx].tolist()
        val_puuids = set(df_train["puuid"].iloc[val_idx])
        Xtr, dropped = purged_train_features(ref, train_puuids,
                                             val_puuids | holdout_puuids,
                                             features=mf.FEATURES)
        if dropped:
            print(f"    (purge : {len(dropped)} joueurs de train droppés sur ce fold)")
        y_train = Xtr["puuid"].map(y_of).astype(float)
        folds.append((Xtr.reindex(columns=features), y_train, X.iloc[val_idx], val_idx))
    return folds
```

- [ ] **Step 3: Réécrire `main()` (lignes ~176-238) pour le held-out**

```python
def main() -> int:
    split = ds.load_split()
    df = pd.read_parquet(DATASET)
    meta = json.loads(DATASET_META.read_text()) if DATASET_META.exists() else {}
    ref = pd.read_parquet(DATASET_PER_GAME)
    ref = ref[(ref["source"] == "referentiel")
              & ref["puuid"].isin(set(df["puuid"]))].copy()
    features = mf.player_feature_names(mf.FEATURES)

    pop = set(df["puuid"])
    train_p = ds.puuids_in(split, "train") & pop
    holdout = (ds.puuids_in(split, "calibration") | ds.puuids_in(split, "test")) & pop
    df_train = df[df["puuid"].isin(train_p)].reset_index(drop=True)
    df_test = ds.partition(df, split, "test").reset_index(drop=True)
    y_train = df_train["lp"].astype(float).values
    print(f"  {len(df)} joueurs | train={len(df_train)} test={len(df_test)} "
          f"(calibration réservée) | tiers train : "
          f"{df_train['rank'].value_counts().to_dict()} | "
          f"label fetched_at={meta.get('fetched_at', '?')}")

    print("\n  Précalcul des folds purgés (train seul, 1 fois, réutilisés)…")
    folds = prepare_folds(df_train, ref, features, holdout)

    best, per_model = {}, {}
    for name, spec in GRIDS.items():
        print(f"\n  Random search {name} ({spec['n_configs']} configs max)…")
        best[name] = search_best(name, spec, folds, y_train, len(df_train))
        per_model[name] = {
            "spearman_pooled": round(best[name]["spearman"], 4),
            "best_config": {k: (v if v is None or isinstance(v, (int, float, str))
                                else str(v)) for k, v in best[name]["config"].items()},
        }
        print(f"    -> spearman={best[name]['spearman']:.4f}  "
              f"config={best[name]['config']}")

    ens_oof = np.mean([best[n]["oof"] for n in GRIDS], axis=0)
    cv_report = spearman_report(pd.DataFrame({
        "rank": df_train["rank"].values, "y_true": y_train, "y_pred": ens_oof}))
    print(f"\n  CV train (purgé) : spearman pooled = {cv_report['spearman_pooled']}  "
          f"(baseline POC {POC_BASELINE_SPEARMAN_POOLED})  rmse={cv_report['rmse_pooled']}")

    print("\n  Refit final sur le TRAIN (features purgées de holdout)…")
    Xtr_final, _ = purged_train_features(ref, df_train["puuid"].tolist(), holdout,
                                         features=mf.FEATURES)
    Xtr_final_feat = Xtr_final.reindex(columns=features)
    y_of = dict(zip(df_train["puuid"], df_train["lp"].astype(float)))
    y_final = Xtr_final["puuid"].map(y_of).astype(float)
    final_models = {name: make_model(name, best[name]["config"]) for name in GRIDS}
    for model in final_models.values():
        model.fit(Xtr_final_feat, y_final)

    # --- Headline : TEST held-out (features naturelles) ---
    X_test = df_test.reindex(columns=features)
    ens_test = np.mean([m.predict(X_test) for m in final_models.values()], axis=0)
    test_report = spearman_report(pd.DataFrame({
        "rank": df_test["rank"].values, "y_true": df_test["lp"].astype(float).values,
        "y_pred": ens_test}))
    print(f"\n  TEST held-out (HEADLINE) : spearman pooled = "
          f"{test_report['spearman_pooled']}  rmse={test_report['rmse_pooled']}  "
          f"n={len(df_test)}")
    for tier, r in test_report["spearman_by_tier"].items():
        print(f"    {tier:<12} spearman={r['spearman']}  n={r['n']}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in final_models.items():
        with open(MODEL_DIR / f"{name}_player_lp.pkl", "wb") as f:
            pickle.dump(model, f)
    (MODEL_DIR / "player_lp_features.json").write_text(json.dumps(features, indent=2))

    shap_block = shap_top20(final_models["xgb"], Xtr_final_feat.fillna(Xtr_final_feat.median()))
    print(f"  dispersion (std/p10/p90) = {shap_block['shap_dispersion_share']:.1%} "
          "du signal SHAP (xgb)")

    (MODEL_DIR / "player_lp_metrics.json").write_text(json.dumps({
        "cv_train": cv_report,
        "test": test_report,
        "per_model_cv": per_model,
        "split": {"proportions": split["proportions"],
                  "n_by_bucket_by_rank": split["n_by_bucket_by_rank"]},
        "n_players_train": len(df_train),
        "n_players_test": len(df_test),
        "n_dropped_no_lp": meta.get("n_dropped_no_lp"),
        "lp_fetched_at": meta.get("fetched_at"),
        "poc_baseline_spearman_pooled": POC_BASELINE_SPEARMAN_POOLED,
        "features": features,
        "shap": shap_block,
    }, indent=2))

    print(f"\n✓ Modèles LP écrits dans {MODEL_DIR}/ (HEADLINE = TEST held-out)")
    return 0
```

- [ ] **Step 4: Lancer l'entraînement LP (peut prendre ~10-30 min à cause de l'EBM)**

Run: `poetry run python3 src/02_data_science/train_player_lp.py`
Expected : affiche `CV train (purgé)` puis `TEST held-out (HEADLINE)` avec Spearman par tier ; écrit `{xgb,rf,ebm}_player_lp.pkl`, `player_lp_features.json`, `player_lp_metrics.json` (clés `cv_train`/`test`/`split`). **Attendu** : `test.spearman_pooled` proche mais probablement sous l'ancien 0.5186, GM très bruité (n≈12, flaw assumé).

- [ ] **Step 5: Commit**

```bash
git add src/02_data_science/train_player_lp.py data/05_model/player_lp_metrics.json data/05_model/player_lp_features.json
git commit -m "feat(lp): held-out test + recherche sur train pour la régression LP"
```

---

## Task 4: `calibrate_player_rank.py` — calibration sur OOF du train

**Files:**
- Modify: `src/02_data_science/calibrate_player_rank.py`

**Interfaces:**
- Consumes: `data/05_model/player_train_oof.json` (produit par Task 2), `adc_player_dataset.parquet` (colonnes `puuid`, `rank`).
- Produces: `data/05_model/player_rank_calibration.json` (inchangé de forme : liste de `{rank, mean_proba, median_proba, n}`).

- [ ] **Step 1: Réécrire `main()` (lignes 31-57) pour consommer l'OOF du train**

Supprime le chargement des `.pkl` et le calcul de proba in-sample (remplacés par l'OOF). Les imports `pickle` et `numpy` deviennent inutilisés — **les laisser** (aucune suppression requise, et flake n'est pas bloquant ici) ou les retirer si le lint du projet l'exige.

```python
def main() -> int:
    oof_path = MODEL_DIR / "player_train_oof.json"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"{oof_path} absent. Lance d'abord "
            "`train_player_ensemble.py` (il exporte les OOF du train)."
        )
    train_oof = json.loads(oof_path.read_text())
    df = pd.read_parquet(DATASET)
    df = df[df["puuid"].isin(train_oof)].copy()
    df["ensemble_proba"] = df["puuid"].map(train_oof)

    calibration = []
    print("  Calibration proba -> rang (per-player, OOF du TRAIN, hors in-sample) :")
    for rank in RANKS:
        sub = df[df["rank"] == rank]["ensemble_proba"]
        if not len(sub):
            continue
        row = {"rank": rank, "mean_proba": round(float(sub.mean()), 4),
               "median_proba": round(float(sub.median()), 4), "n": int(len(sub))}
        calibration.append(row)
        print(f"    {rank:<12} mean={row['mean_proba']:.3f} "
              f"median={row['median_proba']:.3f}  n={row['n']}")

    (MODEL_DIR / "player_rank_calibration.json").write_text(
        json.dumps(calibration, indent=2))
    print(f"\n✓ Calibration écrite dans {MODEL_DIR}/player_rank_calibration.json")
    return 0
```

- [ ] **Step 2: Lancer la calibration**

Run: `poetry run python3 src/02_data_science/calibrate_player_rank.py`
Expected : lit `player_train_oof.json`, écrit `player_rank_calibration.json` (mean_proba croissant de diamond→challenger attendu). Si l'ordre n'est pas monotone → le noter, ne pas bloquer (signal faible connu sur cette frontière).

- [ ] **Step 3: Commit**

```bash
git add src/02_data_science/calibrate_player_rank.py data/05_model/player_rank_calibration.json
git commit -m "feat(calib): calibration proba->rang sur les OOF du train (hors in-sample)"
```

---

## Task 5: Arrêt documenté du per-game

**Files:**
- Modify: `src/02_data_science/train_ensemble.py` (docstring)
- Modify: `src/02_data_science/calibrate_rank.py` (docstring)

**Interfaces:** aucune (documentation seulement, 0 changement de comportement).

- [ ] **Step 1: Ajouter l'en-tête de dépréciation en tête de docstring de `train_ensemble.py`**

Insérer ces lignes juste après le `"""` ouvrant (avant la ligne existante `02_data_science — ...`) :

```python
"""
DÉPRÉCIÉ — arrêté le 2026-07-18. Modèle per-game (1 ligne = 1 ADC d'une game) NON servi
en prod : le web (web/backend/ml_rank.py) tourne sur le per-player « constance/plancher ».
AUC trop basse (~0.63 dia_chall / ~0.59 high_elo) — 1 game porte un signal quasi aléatoire
(RNG matchmaking/stomps), sans valeur prédictive utile. Conservé pour l'historique et la
reproductibilité (aucune suppression). Ne pas migrer au protocole gold standard.
Voir docs/superpowers/specs/2026-07-18-gold-standard-eval-protocol-design.md.

02_data_science — ...  (docstring d'origine conservée ci-dessous)
```

(Concrètement : garder tout le texte existant, ajouter le paragraphe DÉPRÉCIÉ au-dessus de la première ligne descriptive.)

- [ ] **Step 2: Même en-tête de dépréciation en tête de docstring de `calibrate_rank.py`**

```python
"""
DÉPRÉCIÉ — arrêté le 2026-07-18. Calibration du modèle per-game (déprécié) : non servie
en prod (le web utilise player_rank_calibration.json via calibrate_player_rank.py).
Conservé pour l'historique. Voir
docs/superpowers/specs/2026-07-18-gold-standard-eval-protocol-design.md.

02_data_science — ...  (docstring d'origine conservée ci-dessous)
```

- [ ] **Step 3: Vérifier que rien n'importe ces modules (garde anti-casse)**

Run: `grep -rn --include="*.py" -e "import train_ensemble" -e "import calibrate_rank" -e "from train_ensemble" -e "from calibrate_rank" src/ web/ tests/`
Expected : aucune ligne (ces scripts ne sont importés nulle part ; seul `ml_features.py` les cite en commentaire). Confirme que la dépréciation ne casse aucun import.

- [ ] **Step 4: Commit**

```bash
git add src/02_data_science/train_ensemble.py src/02_data_science/calibrate_rank.py
git commit -m "docs(per-game): marque train_ensemble/calibrate_rank dépréciés (arrêt 2026-07-18)"
```

---

## Task 6: Vérif bout-en-bout du serving + mise à jour CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:** vérifie que `web/backend/ml_rank.py` sert toujours sans changement d'interface.

- [ ] **Step 1: Vérifier que le serving charge les nouveaux artefacts sans erreur**

Run:
```bash
poetry run python3 -c "
import sys; from pathlib import Path
r = Path('.').resolve()
sys.path.insert(0, str(r/'src'/'core')); sys.path.insert(0, str(r/'web'/'backend'))
import ml_rank
print('models OK:', list(ml_rank._load_models().keys()))
print('calibration ranks:', [c['rank'] for c in ml_rank._load_calibration()])
print('lp bundle present:', ml_rank._load_lp_bundle() is not None)
"
```
Expected : `models OK: ['xgb', 'rf']`, la liste des rangs calibrés, `lp bundle present: True`. Aucune exception. (Confirme que les `.pkl` refités sur train et les JSON régénérés se chargent — interface serving intacte.)

- [ ] **Step 2: Lancer toute la suite de tests**

Run: `poetry run pytest tests/ -q`
Expected : vert (les tests de split ajoutés + les tests de purge existants ; aucune régression).

- [ ] **Step 3: Mettre à jour CLAUDE.md — protocole + métriques + arrêt per-game**

Dans la section « État d'avancement » / pipeline ML, ajouter une entrée (adapter les chiffres réels lus dans `player_metrics.json` / `player_lp_metrics.json` après les runs des Tasks 2-3) :

```markdown
- **Protocole d'éval gold standard (per-player)** ✅ — 2026-07-18. Split canonique unique
  `data/04_dataset/split.json` (par joueur, stratifié, graine fixe, 70/15/15, cf.
  `src/core/dataset_split.py` + `src/01_data_engineering/build_split.py`). Sélection des
  hyperparamètres en k-fold SUR LE TRAIN, headline sur le TEST held-out ; calibration + test
  hors du modèle servi (calibration RÉSERVÉE à une future couche AOS4, non encore implémentée).
  Purge étendue via `purged_train_features` (fold-val ∪ holdout). ⚠ Le headline test est
  volontairement plus bas que les anciens OOF-à-plat (fin de l'optimisme de sélection + modèle
  sur ~70 % des joueurs) : c'est la mesure honnête. **FLAW ASSUMÉ (GM)** : ~78 GM au total →
  calib/test GM petits (~12 chacun), métriques GM bruitées ; remédiation renvoyée à un script
  ultérieur. Spec : `docs/superpowers/specs/2026-07-18-gold-standard-eval-protocol-design.md`.
  Métriques : `player_metrics.json` (rang : cv_train.auc=<X> / test.auc=<Y>) et
  `player_lp_metrics.json` (LP : cv_train / test spearman + by_tier).
- **Per-game DÉPRÉCIÉ** — 2026-07-18. `train_ensemble.py` / `calibrate_rank.py` arrêtés (non
  servis, AUC ~0.63/0.59 trop aléatoire) ; code et artefacts conservés pour l'historique.
```

Repérer aussi la ligne existante « ⚠️ `xgb/rf/ebm_highelo.pkl` à ré-entraîner avant de servir »
(section Phase 1.8) et y ajouter : « — per-game déprécié 2026-07-18, non servi ; le serving
utilise les `*_player_highelo.pkl`. »

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: acte le protocole gold standard per-player + arrêt per-game dans CLAUDE.md"
```

---

## Self-review (fait à l'écriture du plan)

- **Couverture de la spec** : §4 split → Task 1 ; §5 purge étendue → Tasks 2-3 (`| holdout`) ; §6 refactor scripts → Tasks 2-3-4 ; §7 format métriques `cv_train`/`test` → Tasks 2-3 ; §8 arrêt per-game → Task 5 + Task 6 (CLAUDE.md) ; §10 tests → Task 1 (+ Task 6 suite complète) ; §11 critère de succès → Task 6 (serving + tests verts). ✅
- **Placeholders** : aucun `<X>`/`<Y>` sauf les métriques réelles à recopier depuis les JSON après exécution (Task 6 Step 3), explicitement marqués « adapter les chiffres réels ». Pas de « TODO/TBD » de code.
- **Cohérence des types** : `holdout` = `set[str]` partout ; `purged_train_features(ref, train_puuids, val∪holdout, features=…)` signature conforme au fichier source ; `player_train_oof.json` produit en Task 2, consommé en Task 4 sous le même nom ; `ds.load_split/puuids_in/partition` cohérents entre Tasks 1-2-3.
- **Décision assumée** : retrait des variantes CV naïve/contrôle du headline (superseded par le test held-out) — `control_train_features` reste dans le fichier (non supprimée) conformément à la contrainte « aucune suppression ».
