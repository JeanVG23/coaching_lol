#!/usr/bin/env python3
"""02_data_science — briques de CV partagées par les entraînements per-player.

`make_models()` était recopié caractère pour caractère entre `train_player_ensemble`
et `analyze_auc_vs_ngames`, et `purged_train_features` réimplémenté dans le second
alors qu'il était importable. Ces scripts sont censés CARACTÉRISER le modèle servi :
un tweak d'hyperparamètre appliqué d'un seul côté invalide silencieusement la
comparaison d'AUC entre runs, et la purge (le garde-fou de fuite) pouvait être
corrigée d'un seul côté.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import numpy as np
import pandas as pd
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

import ml_features as mf

SEED = 42


def make_models(seed: int = SEED) -> dict:
    """Ensemble à 3 biais inductifs : GBDT (xgb) / bagging (rf) / GA²M glass-box (ebm)."""
    return {
        "xgb": xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.0, eval_metric="logloss", tree_method="hist",
            random_state=seed,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=5,
            max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=seed,
        ),
        "ebm": ExplainableBoostingClassifier(interactions=0, random_state=seed),
    }


def purged_train_features(ref: pd.DataFrame, train_puuids, val_puuids,
                          features: list[str] | None = None
                          ) -> tuple[pd.DataFrame, list[str]]:
    """Agrégats des joueurs de train recalculés SANS les matchs joués par un joueur
    de val (purge exacte de la fuite par games partagées : les deux ADC d'une game
    portent des features en miroir, cf. tests/test_train_player_ensemble.py).
    Retourne (DataFrame 1 ligne/joueur survivant, ordre de train_puuids ; liste des
    joueurs droppés faute de game restante après purge).

    `features` = noms de features BRUTES (csm10, ...), PAS les noms agrégés
    (csm10__mean) — `aggregate_player_features` produit lui-même les suffixes.
    """
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
                           features: list[str] | None = None,
                           seed: int = SEED) -> tuple[pd.DataFrame, list[str]]:
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
