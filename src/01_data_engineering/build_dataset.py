#!/usr/bin/env python3
"""
01_data_engineering — silver -> dataset ML consolidé (1 ligne = 1 game ADC).

Lit la couche silver (par-game), filtre les ADC (lane features cohérentes), aplatit
en table de features + label de rang, et écrit un dataset unique pour data_science.

Les trous (gd20/csd14 None sur games courtes) sont LAISSÉS en NaN : XGBoost gère les
valeurs manquantes nativement (pas d'imputation arbitraire).

Sortie : data/04_dataset/adc_dataset.parquet (+ .csv pour inspection).
Usage : .venv/bin/python src/01_data_engineering/build_dataset.py
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # accès à riotlib
import numpy as np
import pandas as pd
import riotlib as rl

DATASET_DIR = rl.DATA / "04_dataset"
RANKS = ["diamond", "master", "grandmaster", "challenger"]
RANK_ORD = {r: i for i, r in enumerate(RANKS)}
HIGH_ELO = {"grandmaster", "challenger"}  # cible binaire


def game_to_row(g: dict, rank: str | None, source: str) -> dict:
    lane = g.get("lane", {})
    deaths = g.get("deaths", [])
    n = len(deaths)
    ph = collections.Counter(d["phase"] for d in deaths)
    gs = collections.Counter(d.get("gold_state") for d in deaths if d.get("gold_state"))
    gs_tot = sum(gs.values())
    return {
        # méta (non-features)
        "match_id": g["match_id"], "puuid": g.get("puuid"), "source": source,
        "rank": rank, "champion": g["champion"], "win": int(g["win"]),
        # features de lane (diffs vs adversaire)
        "gd10": lane.get("gd10"), "gd14": lane.get("gd14"), "gd20": lane.get("gd20"),
        "csd10": lane.get("csd10"), "csd14": lane.get("csd14"), "xpd10": lane.get("xpd10"),
        # features de morts
        "n_deaths": n,
        "deaths_early": ph.get("early", 0),
        "deaths_mid": ph.get("mid", 0),
        "deaths_late": ph.get("late", 0),
        "frac_behind": gs.get("behind", 0) / gs_tot if gs_tot else np.nan,
        "frac_ahead": gs.get("ahead", 0) / gs_tot if gs_tot else np.nan,
    }


def main() -> int:
    rows = []
    # référentiels (labellisés par rang) — ADC uniquement
    for rank in RANKS:
        games = rl.read_jsonl(rl.SILVER_DIR / "referentiel" / rank / "games.jsonl")
        adc = [g for g in games if g.get("role") == "BOTTOM"]
        rows += [game_to_row(g, rank, "referentiel") for g in adc]
        print(f"  referentiel/{rank:<12}: {len(adc)} games ADC")
    # perso (non labellisé : pour inférence ultérieure)
    perso_root = rl.SILVER_DIR / "personal"
    if perso_root.exists():
        for d in sorted(perso_root.iterdir()):
            games = rl.read_jsonl(d / "games.jsonl")
            adc = [g for g in games if g.get("role") == "BOTTOM"]
            rows += [game_to_row(g, None, f"personal:{d.name}") for g in adc]
            print(f"  personal/{d.name:<14}: {len(adc)} games ADC")

    df = pd.DataFrame(rows)
    # dédup par (match_id, puuid) : garde les perspectives distinctes d'une même game
    df = df.drop_duplicates(subset=["match_id", "puuid"]).reset_index(drop=True)
    df["rank_ord"] = df["rank"].map(RANK_ORD)
    df["high_elo"] = df["rank"].isin(HIGH_ELO).astype("Int64")
    df.loc[df["rank"].isna(), "high_elo"] = pd.NA  # perso : label inconnu

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_DIR / "adc_dataset.parquet", index=False)
    df.to_csv(DATASET_DIR / "adc_dataset.csv", index=False)

    ref = df[df["source"] == "referentiel"]
    print(f"\n✓ Dataset : {len(df)} games ({len(ref)} référentiel labellisés)")
    print(f"  Répartition rangs : {dict(ref['rank'].value_counts())}")
    print(f"  high_elo (GM+Chall=1) : {dict(ref['high_elo'].value_counts())}")
    print(f"  Écrit dans {DATASET_DIR}/adc_dataset.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
