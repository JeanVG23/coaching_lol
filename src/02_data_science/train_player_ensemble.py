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

CV : StratifiedKFold (pas de group CV — 1 ligne = 1 joueur, aucune fuite joueur->fold
possible par construction, contrairement au per-game qui groupe par puuid).

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
    print(f"  {len(df)} joueurs | pos={int(y.sum())} / neg={int((1-y).sum())}")

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
