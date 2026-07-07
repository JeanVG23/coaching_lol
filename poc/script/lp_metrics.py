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
