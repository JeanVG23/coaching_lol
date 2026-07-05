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

CV : StratifiedKFold sur les joueurs + PURGE des games partagées. 1 ligne = 1 joueur
(pas de fuite joueur->fold), MAIS ~37 % des games des qualifiés opposent deux joueurs
du dataset (2 ADC extraits par game, features en miroir) et le graphe des games
partagées est une composante géante (98.7 % des joueurs) -> group-CV par composantes
impossible. À la place, à chaque fold les agrégats des joueurs de TRAIN sont
recalculés en excluant les matchs joués par un joueur de VAL (purged CV, exact) ;
une passe contrôle retire le même nombre de games au hasard pour distinguer l'effet
"fuite retirée" de l'effet "moins de games". auc_cv (headline) = AUC purgée ;
auc_cv_naive / auc_cv_control exposés dans player_metrics.json pour comparaison.

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
DATASET_PER_GAME = rl.DATA / "04_dataset" / "adc_dataset.parquet"  # pour la purge
MODEL_DIR = rl.DATA / "05_model"
HIGH_ELO = {"grandmaster", "challenger"}


def purged_train_features(ref: pd.DataFrame, train_puuids, val_puuids,
                          features: list[str] = None) -> tuple[pd.DataFrame, list[str]]:
    """Agrégats des joueurs de train recalculés SANS les matchs joués par un joueur
    de val (purge exacte de la fuite par games partagées : les deux ADC d'une game
    portent des features en miroir, cf. tests/test_train_player_ensemble.py).
    Retourne (DataFrame 1 ligne/joueur survivant, ordre de train_puuids ; liste des
    joueurs droppés faute de game restante après purge)."""
    features = mf.FEATURES if features is None else features
    val_matches = set(ref.loc[ref["puuid"].isin(set(val_puuids)), "match_id"])
    sub = ref[ref["puuid"].isin(set(train_puuids))
              & ~ref["match_id"].isin(val_matches)]
    by_puuid = dict(tuple(sub.groupby("puuid")))
    rows, dropped = [], []
    for puuid in train_puuids:
        g = by_puuid.get(puuid)
        if g is None or g.empty:
            dropped.append(puuid)
            continue
        rec = {"puuid": puuid}
        rec.update(mf.aggregate_player_features(g, features))
        rows.append(rec)
    return pd.DataFrame(rows), dropped


def control_train_features(ref: pd.DataFrame, train_puuids, n_drop: dict,
                           features: list[str] = None,
                           seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    """Contrôle du purged CV : retire à chaque joueur de train le MÊME NOMBRE de
    games que la purge (n_drop = {puuid: n}), mais tirées au hasard au lieu des
    games partagées — la différence d'AUC contrôle-purgé isole la fuite pure de
    l'effet 'agrégats sur moins de games'."""
    features = mf.FEATURES if features is None else features
    rng = np.random.RandomState(seed)
    by_puuid = dict(tuple(ref[ref["puuid"].isin(set(train_puuids))].groupby("puuid")))
    rows, dropped = [], []
    for puuid in train_puuids:
        g = by_puuid.get(puuid)
        if g is None:
            dropped.append(puuid)
            continue
        n = n_drop.get(puuid, 0)
        if n >= len(g):
            dropped.append(puuid)
            continue
        if n:
            g = g.drop(index=rng.choice(g.index, size=n, replace=False))
        rec = {"puuid": puuid}
        rec.update(mf.aggregate_player_features(g, features))
        rows.append(rec)
    return pd.DataFrame(rows), dropped


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
    ref = pd.read_parquet(DATASET_PER_GAME)
    ref = ref[(ref["source"] == "referentiel")
              & ref["puuid"].isin(set(df["puuid"]))].copy()
    features = mf.player_feature_names(mf.FEATURES)
    X = df.reindex(columns=features)
    y = df["rank"].isin(HIGH_ELO).astype(int)
    y_of = dict(zip(df["puuid"], y))
    orig_counts = ref.groupby("puuid").size()
    print(f"  {len(df)} joueurs | pos={int(y.sum())} / neg={int((1-y).sum())} | "
          f"{len(ref)} games per-game pour la purge")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    variants = ("purged", "naive", "control")
    oof = {v: {name: np.zeros(len(X)) for name in make_models()} for v in variants}
    games_purged, games_train, dropped_players = 0, 0, 0
    for train_idx, val_idx in cv.split(X, y):
        X_val = X.iloc[val_idx]
        train_puuids = df["puuid"].iloc[train_idx].tolist()
        val_puuids = set(df["puuid"].iloc[val_idx])

        # purged : agrégats train sans les matchs des joueurs de val (fuite = 0)
        Xtr_p, drop_p = purged_train_features(ref, train_puuids, val_puuids)
        n_drop = {p: int(orig_counts[p]) for p in drop_p}
        n_drop.update({row["puuid"]: int(orig_counts[row["puuid"]] - row["n_games"])
                       for _, row in Xtr_p.iterrows()})
        # control : même nombre de games retirées, au hasard
        Xtr_c, drop_c = control_train_features(ref, train_puuids, n_drop)
        games_purged += sum(n_drop.values())
        games_train += int(orig_counts.loc[train_puuids].sum())
        dropped_players += len(drop_p)

        fits = {
            "naive": (X.iloc[train_idx], y.iloc[train_idx]),
            "purged": (Xtr_p.reindex(columns=features),
                       Xtr_p["puuid"].map(y_of).astype(int)),
            "control": (Xtr_c.reindex(columns=features),
                        Xtr_c["puuid"].map(y_of).astype(int)),
        }
        for variant, (X_train, y_train) in fits.items():
            for name, model in make_models().items():
                model.fit(X_train, y_train)
                oof[variant][name][val_idx] = model.predict_proba(X_val)[:, 1]

    per_model = {}
    print("\n  Perf par modèle (CV out-of-fold, PURGÉE) :")
    for name, preds in oof["purged"].items():
        m_auc = roc_auc_score(y, preds)
        m_acc = accuracy_score(y, (preds >= 0.5).astype(int))
        per_model[name] = {"auc": round(m_auc, 4), "acc": round(m_acc, 4)}
        print(f"    {name:<4} AUC={m_auc:.3f}  accuracy={m_acc:.3f}")

    ens = {v: np.mean(list(oof[v].values()), axis=0) for v in variants}
    auc = roc_auc_score(y, ens["purged"])
    acc = accuracy_score(y, (ens["purged"] >= 0.5).astype(int))
    auc_naive = roc_auc_score(y, ens["naive"])
    auc_control = roc_auc_score(y, ens["control"])
    frac_purged = games_purged / max(1, games_train)
    print(f"\n  Ensemble CV out-of-fold : AUC purgée={auc:.3f} (headline)  "
          f"naïve={auc_naive:.3f}  contrôle={auc_control:.3f}")
    print(f"  purge : {frac_purged:.1%} des games de train retirées, "
          f"{dropped_players} joueurs droppés (cumul 5 folds)")
    print(f"  fuite pure ≈ contrôle - purgée = {auc_control - auc:+.4f} ; "
          f"effet 'moins de games' ≈ naïve - contrôle = {auc_naive - auc_control:+.4f}")
    print(classification_report(y, (ens["purged"] >= 0.5).astype(int),
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
        "auc_cv": round(auc, 4), "acc_cv": round(acc, 4),   # PURGÉE (honnête)
        "auc_cv_naive": round(auc_naive, 4),
        "auc_cv_control": round(auc_control, 4),
        "purge": {"train_games_removed_frac": round(frac_purged, 4),
                  "dropped_players_5folds": dropped_players},
        "per_model_cv": per_model,
        "n_players": len(df), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "features": features,
        "dispersion_analysis": dispersion,
    }, indent=2))

    print(f"\n✓ Modèles per-player écrits dans {MODEL_DIR}/ (marqueur 'player_highelo')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
