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
