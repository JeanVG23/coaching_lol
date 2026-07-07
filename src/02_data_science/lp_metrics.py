#!/usr/bin/env python3
"""
02_data_science — métriques de la régression LP : Spearman pooled + par tier, RMSE.

Le Spearman pooled seul peut juste redécouvrir la frontière de tier connue (un
Challenger a par définition un LP plus haut qu'un Master) : le Spearman PAR TIER
(calculé séparément à l'intérieur de master, GM, challenger) est le vrai test —
il isole si le modèle discrimine une granularité de skill au-delà du tier.
Copie prod du module POC poc/script/lp_metrics.py (le code prod n'importe pas
depuis poc/). Cf. docs/superpowers/specs/2026-07-07-lp-production-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_TIER_N = 3  # sous ce seuil, spearman non significatif -> None plutôt qu'une valeur trompeuse


def _safe_spearman(a, b) -> float | None:
    """Spearman, ou None s'il est indéfini : moins de MIN_TIER_N points, ou a/b
    constant (nunique < 2). Jamais de NaN silencieux."""
    a, b = pd.Series(a), pd.Series(b)
    if len(a) < MIN_TIER_N or a.nunique() < 2 or b.nunique() < 2:
        return None
    return round(float(spearmanr(a, b)[0]), 4)


def spearman_report(df: pd.DataFrame) -> dict:
    """df : colonnes rank (str), y_true (float), y_pred (float). Une ligne par
    joueur. Retourne spearman pooled + par tier (None si <MIN_TIER_N lignes ou
    y_true/y_pred constant sur le tier) + rmse pooled."""
    pooled_rho = _safe_spearman(df["y_true"], df["y_pred"])

    by_tier: dict[str, dict] = {}
    for tier, g in df.groupby("rank"):
        rho = _safe_spearman(g["y_true"], g["y_pred"])
        by_tier[str(tier)] = {"spearman": rho, "n": int(len(g))}

    rmse = float(np.sqrt(np.mean((df["y_true"] - df["y_pred"]) ** 2)))

    return {
        "spearman_pooled": pooled_rho,
        "spearman_by_tier": by_tier,
        "rmse_pooled": round(rmse, 2),
        "n_players_total": int(len(df)),
    }
