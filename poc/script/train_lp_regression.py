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
