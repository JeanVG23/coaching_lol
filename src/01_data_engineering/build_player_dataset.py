#!/usr/bin/env python3
"""
01_data_engineering — dataset per-player (1 ligne = 1 joueur >= MIN_PLAYER_GAMES
games ADC référentiel).

Agrège adc_dataset.parquet (référentiel, 1 ligne = 1 ADC d'une game, déjà construit
par build_dataset.py) par puuid : pour chaque joueur à >= MIN_PLAYER_GAMES games,
mean/std/p10/p50/p90 par feature sur la TOTALITÉ de son historique disponible
(cf. poc/per_player_hypothesis.py — hypothèse "constance/plancher" validée sur données
densifiées, +0.12 AUC vs per-game). Rang résolu au mode (tie-break rang le plus bas,
cf. ml_features.resolve_rank).

MIN_PLAYER_GAMES=15 (relevé depuis 5) : la dispersion (std/p10/p90) est le cœur du
signal du modèle, or l'écart-type sur 5 games est dominé par le bruit de matchmaking
(RNG, AFK, stomps) plutôt que par la vraie régularité du joueur. Sous 5 games, l'AUC
mesurée tombait à 0.531 (à peine mieux que pile ou face) — la variance intra-joueur
noyait le signal inter-joueur. Un joueur ne calcule plus ses stats sur ses 5 dernières
games mais sur tout son historique référentiel disponible (>= 15) : la régression vers
la moyenne y réduit aussi l'impact d'une série de défaites malchanceuse isolée, en
complément du signal win_rate (cf. ml_features.aggregate_player_features).

0 appel API (relit un dataset déjà construit).

Sortie : data/04_dataset/adc_player_dataset.parquet (+ .csv pour inspection).
Usage : poetry run python3 src/01_data_engineering/build_player_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core")) # ml_features
import pandas as pd
import riotlib as rl
from ranks import HIGH_ELO
import ml_features as mf

DATASET_DIR = rl.DATA / "04_dataset"
MIN_PLAYER_GAMES = 15


def build_player_rows(df: pd.DataFrame, min_games: int = MIN_PLAYER_GAMES) -> pd.DataFrame:
    """df : rows per-game référentiel (colonnes puuid, rank, win, + mf.FEATURES).
    Retourne 1 ligne par joueur ayant >= min_games games, colonnes agrégées
    {feature}__{stat} + win_rate + n_games, rank, high_elo. Vide si aucun joueur ne
    qualifie. Agrégation sur TOUT l'historique disponible du joueur (pas de troncature
    à un nombre fixe de games) : plus un joueur a de games, plus sa mesure de
    régularité est fiable."""
    rows = []
    for puuid, g in df.groupby("puuid"):
        if len(g) < min_games:
            continue
        rec = {"puuid": puuid, "rank": mf.resolve_rank(g)}
        rec.update(mf.aggregate_player_features(g, mf.FEATURES))
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["high_elo"] = out["rank"].isin(HIGH_ELO).astype(int)
        
        # Balance classes (undersampling)
        pos = out[out["high_elo"] == 1]
        neg = out[out["high_elo"] == 0]
        n_min = min(len(pos), len(neg))
        if n_min > 0:
            pos = pos.sample(n=n_min, random_state=42)
            neg = neg.sample(n=n_min, random_state=42)
            out = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
            
    return out


def main() -> int:
    df = pd.read_parquet(DATASET_DIR / "adc_dataset.parquet")
    ref = df[df["source"] == "referentiel"].copy()
    print(f"  {len(ref)} games référentiel | {ref['puuid'].nunique()} joueurs uniques")

    out = build_player_rows(ref)
    print(f"  >= {MIN_PLAYER_GAMES} games : {len(out)} joueurs")
    if out.empty:
        print("  ⚠ aucun joueur ne qualifie -> rien à écrire")
        return 1
    print(f"  répartition rangs : {dict(out['rank'].value_counts())}")
    print(f"  high_elo (GM+Chall=1) : {dict(out['high_elo'].value_counts())}")

    out.to_parquet(DATASET_DIR / "adc_player_dataset.parquet", index=False)
    out.to_csv(DATASET_DIR / "adc_player_dataset.csv", index=False)
    print(f"\n✓ Dataset per-player écrit dans {DATASET_DIR}/adc_player_dataset.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
