#!/usr/bin/env python3
"""
02_data_science — régression LP per-player (Master/GM/Challenger), ensemble tuné.

Prédit le LP courant d'un joueur apex depuis ses features agrégées
(mean/std/p10/p50/p90 + win_rate, cf. ml_features). Suite prod du POC
poc/script/train_lp_regression.py (spearman pooled 0.5028 avec un XGBRegressor
unique non tuné — baseline rappelée dans les métriques pour mesurer le gain).

Optimisation : random search à graine fixe par modèle (xgb ~40 configs, rf ~20,
ebm ~8) en PURGED CV 5 folds (StratifiedKFold sur le TIER — y est continu, on
stratifie la catégorie pour équilibrer master/GM/chall par fold). La purge
(agrégats de train recalculés sans les matchs partagés avec la val, cf.
train_player_ensemble.purged_train_features : ~37 % des games opposent 2 ADC du
dataset, features en miroir) ne dépend QUE du découpage en folds, pas des
hyperparamètres → précalculée UNE FOIS par fold et réutilisée pour toutes les
configs (sinon chaque config repayerait le recalcul des agrégats, ~×70 le coût).
Critère de sélection : Spearman pooled OOF (le within-tier reste la métrique de
REPORTING décisive mais trop bruitée par tier pour piloter une recherche — GM
n'a que ~78 joueurs).

N'écrase AUCUN artefact du pipeline binaire : marqueur "player_lp" partout.
Durée attendue : quelques minutes (xgb/rf) + ~10-30 min pour les 8 configs EBM.

Sorties : data/05_model/{xgb,rf,ebm}_player_lp.pkl, player_lp_features.json,
player_lp_metrics.json
Usage : poetry run python3 src/02_data_science/train_player_lp.py
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # lp_metrics, train_player_ensemble
import numpy as np
import pandas as pd
import riotlib as rl
import ml_features as mf
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from interpret.glassbox import ExplainableBoostingRegressor

from lp_metrics import _safe_spearman, spearman_report
from train_player_ensemble import purged_train_features, dispersion_share_analysis

DATASET = rl.DATA / "04_dataset" / "adc_player_lp_dataset.parquet"
DATASET_META = rl.DATA / "04_dataset" / "adc_player_lp_dataset.meta.json"
DATASET_PER_GAME = rl.DATA / "04_dataset" / "adc_dataset.parquet"  # pour la purge
MODEL_DIR = rl.DATA / "05_model"
SEED = 42
POC_BASELINE_SPEARMAN_POOLED = 0.5028

GRIDS = {
    "xgb": {
        "n_configs": 40,
        "grid": {
            "max_depth": [2, 3, 4],
            "n_estimators": [200, 300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "min_child_weight": [3, 5, 10],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_lambda": [0.5, 1.0, 3.0],
        },
    },
    "rf": {
        "n_configs": 20,
        "grid": {
            "n_estimators": [300, 500],
            "max_depth": [None, 8, 12],
            "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt", 0.3, 0.5],
        },
    },
    "ebm": {
        # EBM est lent à fitter (~minutes/config) : budget volontairement réduit
        "n_configs": 8,
        "grid": {
            "max_bins": [128, 256],
            "interactions": [0, 10, 20],
            "learning_rate": [0.01, 0.02],
        },
    },
}


def sample_configs(grid: dict, n: int, seed: int = SEED) -> list[dict]:
    """n combinaisons distinctes tirées uniformément du produit cartésien de grid
    (toutes si le produit est <= n). Déterministe à graine fixe (clés triées)."""
    keys = sorted(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    if len(combos) > n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(combos), size=n, replace=False)
        combos = [combos[i] for i in idx]
    return [dict(zip(keys, c)) for c in combos]


def make_model(name: str, config: dict):
    if name == "xgb":
        return xgb.XGBRegressor(tree_method="hist", random_state=SEED, **config)
    if name == "rf":
        return RandomForestRegressor(n_jobs=-1, random_state=SEED, **config)
    if name == "ebm":
        return ExplainableBoostingRegressor(random_state=SEED, **config)
    raise ValueError(f"modèle inconnu: {name!r}")


def prepare_folds(df: pd.DataFrame, ref: pd.DataFrame,
                  features: list[str]) -> list[tuple]:
    """Précalcule par fold : (X_train purgé, y_train, X_val, val_idx). Les agrégats
    purgés ne dépendent pas des hyperparamètres — calculés UNE fois, réutilisés
    pour toutes les configs du random search."""
    X = df.reindex(columns=features)
    y_of = dict(zip(df["puuid"], df["lp"].astype(float)))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for train_idx, val_idx in cv.split(X, df["rank"]):
        train_puuids = df["puuid"].iloc[train_idx].tolist()
        val_puuids = set(df["puuid"].iloc[val_idx])
        Xtr, dropped = purged_train_features(ref, train_puuids, val_puuids,
                                             features=mf.FEATURES)
        if dropped:
            print(f"    (purge : {len(dropped)} joueurs de train droppés sur ce fold)")
        y_train = Xtr["puuid"].map(y_of).astype(float)
        folds.append((Xtr.reindex(columns=features), y_train, X.iloc[val_idx], val_idx))
    return folds


def oof_predictions(name: str, config: dict, folds: list[tuple],
                    n_rows: int) -> np.ndarray:
    oof = np.zeros(n_rows)
    for X_train, y_train, X_val, val_idx in folds:
        model = make_model(name, config)
        model.fit(X_train, y_train)
        oof[val_idx] = model.predict(X_val)
    return oof


def search_best(name: str, spec: dict, folds: list[tuple],
                y_true: np.ndarray, n_rows: int) -> dict:
    """Random search : retourne {"config", "spearman", "oof"} de la meilleure config
    au Spearman pooled OOF. Une config au spearman indéfini (prédiction constante)
    est traitée comme -1 (ne gagne jamais face à un vrai score)."""
    best = None
    for config in sample_configs(spec["grid"], spec["n_configs"]):
        oof = oof_predictions(name, config, folds, n_rows)
        rho = _safe_spearman(y_true, oof)
        rho = -1.0 if rho is None else rho
        if best is None or rho > best["spearman"]:
            best = {"config": config, "spearman": rho, "oof": oof}
    return best


def shap_top20(model, X: pd.DataFrame) -> dict:
    """SHAP TreeExplainer sur le XGB final : top-20 features + part de dispersion
    (std/p10/p90 vs mean/p50) — vérifie si l'hypothèse constance tient sur la
    cible LP comme elle tenait sur le binaire (~58-62 % du signal)."""
    vals = np.abs(shap.TreeExplainer(model).shap_values(X))
    per_feat = dict(zip(X.columns, vals.mean(axis=0).tolist()))
    disp = dispersion_share_analysis(per_feat)
    top20 = sorted(per_feat.items(), key=lambda kv: kv[1], reverse=True)[:20]
    return {
        "shap_share_by_stat": disp["share_by_stat"],
        "shap_dispersion_share": disp["dispersion_share_of_signal"],
        "top20_shap": [{"feature": k, "mean_abs_shap": round(v, 5)} for k, v in top20],
    }


def main() -> int:
    df = pd.read_parquet(DATASET)
    meta = json.loads(DATASET_META.read_text()) if DATASET_META.exists() else {}
    ref = pd.read_parquet(DATASET_PER_GAME)
    ref = ref[(ref["source"] == "referentiel")
              & ref["puuid"].isin(set(df["puuid"]))].copy()
    features = mf.player_feature_names(mf.FEATURES)
    X = df.reindex(columns=features)
    y = df["lp"].astype(float).values
    print(f"  {len(df)} joueurs | tiers : {df['rank'].value_counts().to_dict()} | "
          f"{len(ref)} games per-game pour la purge | "
          f"label fetched_at={meta.get('fetched_at', '?')}")

    print("\n  Précalcul des folds purgés (1 fois, réutilisés par toutes les configs)…")
    folds = prepare_folds(df, ref, features)

    best, per_model = {}, {}
    for name, spec in GRIDS.items():
        print(f"\n  Random search {name} ({spec['n_configs']} configs max)…")
        best[name] = search_best(name, spec, folds, y, len(df))
        per_model[name] = {"spearman_pooled": round(best[name]["spearman"], 4),
                           "best_config": {k: (v if v is None or isinstance(v, (int, float, str))
                                               else str(v))
                                           for k, v in best[name]["config"].items()}}
        print(f"    -> spearman={best[name]['spearman']:.4f}  "
              f"config={best[name]['config']}")

    ens_oof = np.mean([best[n]["oof"] for n in GRIDS], axis=0)
    report = spearman_report(pd.DataFrame({
        "rank": df["rank"].values, "y_true": y, "y_pred": ens_oof}))
    print(f"\n  Ensemble OOF (purgé) : spearman pooled = {report['spearman_pooled']}  "
          f"(baseline POC {POC_BASELINE_SPEARMAN_POOLED})  rmse={report['rmse_pooled']}")
    for tier, r in report["spearman_by_tier"].items():
        print(f"    {tier:<12} spearman={r['spearman']}  n={r['n']}")

    print("\n  Refit final sur 100 % du dataset…")
    final_models = {name: make_model(name, best[name]["config"]) for name in GRIDS}
    for model in final_models.values():
        model.fit(X, df["lp"].astype(float))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in final_models.items():
        with open(MODEL_DIR / f"{name}_player_lp.pkl", "wb") as f:
            pickle.dump(model, f)
    (MODEL_DIR / "player_lp_features.json").write_text(json.dumps(features, indent=2))

    shap_block = shap_top20(final_models["xgb"], X.fillna(X.median()))
    print(f"  dispersion (std/p10/p90) = {shap_block['shap_dispersion_share']:.1%} "
          "du signal SHAP (xgb)")

    (MODEL_DIR / "player_lp_metrics.json").write_text(json.dumps({
        **report,
        "per_model_cv": per_model,
        "n_players_by_tier": {k: int(v) for k, v in df["rank"].value_counts().items()},
        "n_dropped_no_lp": meta.get("n_dropped_no_lp"),
        "lp_fetched_at": meta.get("fetched_at"),
        "poc_baseline_spearman_pooled": POC_BASELINE_SPEARMAN_POOLED,
        "features": features,
        "shap": shap_block,
    }, indent=2))

    print(f"\n✓ Modèles LP écrits dans {MODEL_DIR}/ (marqueur 'player_lp')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
