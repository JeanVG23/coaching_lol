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
