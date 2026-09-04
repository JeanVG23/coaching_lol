#!/usr/bin/env python3
"""
02_data_science — entraîne l'ensemble per-player (XGBoost, Random Forest, EBM) pour
séparer high-elo (GM+Chall) de low (master+diamond) à partir des features agrégées
par joueur (mean/std/p10/p50/p90), cf. poc/per_player_hypothesis.py.

Reprend l'architecture de train_ensemble.py (mêmes 3 biais inductifs : GBDT / bagging
/ GA²M glass-box), appliquée au dataset per-player
(data/04_dataset/adc_player_dataset.parquet, 1 ligne = 1 joueur >= MIN_PLAYER_GAMES
games ADC, cf. build_player_dataset.py).
EBM interactions=0 (vs 10 en per-game) : pas assez de rows pour des paires fiables sur
un espace de features ~5x plus large.

N'écrase JAMAIS les artefacts du pipeline per-game (xgb_highelo.pkl, features.json,
etc., cf. docs/superpowers/specs/2026-07-03-per-player-consistency-design.md) : tous
les fichiers de sortie portent le marqueur "player".

Protocole held-out (gold-standard-eval-protocol) : le dataset per-player est d'abord
scindé en buckets train/calibration/test (joueurs disjoints, cf. dataset_split.py).
Les hyperparamètres et le choix de modèle sont sélectionnés en CV purgée
(StratifiedKFold sur les joueurs + purge des games partagées, cf. historique ci-dessus)
UNIQUEMENT sur le bucket TRAIN ; le headline honnête (`test.auc`) est l'AUC mesurée
une seule fois sur le bucket TEST held-out, jamais vu pendant la sélection. Le bucket
calibration est réservé à un usage ultérieur (calibration proba->rang/LP, cf. AOS4) et
n'intervient pas dans ce script. player_metrics.json expose `cv_train` (diagnostic CV
sur le train) et `test` (headline held-out) ; `player_train_oof.json` (prédictions
out-of-fold sur le train) est exporté pour la calibration en aval.

Conserve le test d'hypothèse "constance" du POC (masse |SHAP| groupée par type
d'agrégat, dispersion vs tendance centrale) dans player_metrics.json, pour garder
l'observabilité en prod.

Sorties : data/05_model/{xgb,rf,ebm}_player_highelo.pkl, player_features.json,
player_metrics.json
Usage : poetry run python3 src/02_data_science/train_player_ensemble.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core")) # ml_features
sys.path.insert(0, str(Path(__file__).resolve().parent))  # cv_common
import numpy as np
import pandas as pd
import riotlib as rl
from ranks import HIGH_ELO
import ml_features as mf
import dataset_split as ds
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import shap

DATASET = rl.DATA / "04_dataset" / "adc_player_dataset.parquet"
DATASET_PER_GAME = rl.DATA / "04_dataset" / "adc_dataset.parquet"  # pour la purge
MODEL_DIR = rl.DATA / "05_model"


# Briques de CV partagées (cf. cv_common.py) — réexportées ici : le module reste le
# point d'entrée historique de `purged_train_features` pour les tests et train_player_lp.
from cv_common import (SEED, control_train_features, make_models,  # noqa: E402,F401
                       purged_train_features)


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


if __name__ == "__main__":
    sys.exit(main())
