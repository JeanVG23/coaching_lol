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

from ranks import RANK_ORD  # noqa: F401  (réexport historique, cf. src/core/ranks.py)

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

# Une seule table {nom: calcul} : la liste AGG_STATS, l'ordre des colonnes et le
# calcul en dérivent tous. Avant, les 5 stats étaient écrites trois fois (liste +
# branche NaN + branche valeur) dans le module même censé empêcher le drift train/serve.
AGG_FUNCS = {
    "mean": lambda v: v.mean(),
    "std": lambda v: v.std(ddof=1) if len(v) > 1 else 0.0,
    "p10": lambda v: np.percentile(v, 10),
    "p50": lambda v: np.percentile(v, 50),
    "p90": lambda v: np.percentile(v, 90),
}
AGG_STATS = list(AGG_FUNCS)
DISPERSION_STATS = {"std", "p10", "p90"}
CENTRAL_STATS = {"mean", "p50"}


def resolve_rank(group: pd.DataFrame) -> str:
    """Rang du joueur = mode de ses games ; tie-break sur le rang le plus bas
    (ne pas gonfler high_elo aux frontières — cf. CLAUDE.md)."""
    counts = group["rank"].value_counts()
    top = counts[counts == counts.max()]
    return sorted(top.index, key=lambda r: RANK_ORD[r])[0]


def player_feature_names(features: list[str] = FEATURES) -> list[str]:
    """Ordre canonique des colonnes agrégées : {feature}__{stat}, puis win_rate, n_games."""
    return [f"{f}__{s}" for f in features for s in AGG_STATS] + ["win_rate", "n_games"]


def aggregate_player_features(df: pd.DataFrame, features: list[str] = FEATURES) -> dict:
    """Games d'UN joueur (1 ligne par game) -> dict plat {feature}__{stat} + win_rate +
    n_games. std ddof=1 (0.0 si une seule game). NaN propagée si la feature est absente
    ou vide sur tout le groupe (XGBoost/EBM gèrent le NaN nativement, pas d'imputation).

    win_rate : les features de perf (CS/gold/deaths/positioning) sont mécaniquement
    pires en défaite, indépendamment du skill — sans ce signal, un joueur avec une
    série de défaites dans l'échantillon voit ses stats agrégées tirées vers le profil
    "low elo" à tort. Exposé en scalaire brut (pas de mean/std/percentiles : c'est déjà
    un taux) pour que le modèle apprenne à corriger l'effet plutôt que de le confondre
    avec un vrai signal de rang."""
    rec: dict = {}
    for f in features:
        vals = df[f].dropna() if f in df.columns else pd.Series([], dtype=float)
        for stat, fn in AGG_FUNCS.items():
            rec[f"{f}__{stat}"] = np.nan if vals.empty else fn(vals)
    rec["win_rate"] = df["win"].mean() if "win" in df.columns and len(df) else np.nan
    rec["n_games"] = len(df)
    return rec
