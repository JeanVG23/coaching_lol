#!/usr/bin/env python3
"""01_data_engineering — génère le split canonique train/calibration/test.

Au niveau joueur (puuid), stratifié par rang résolu, graine fixe. Population = UNION
des puuid des deux datasets per-player (rang + LP) : aucun joueur consommé par un
modèle n'échappe au split. Écrit data/04_dataset/split.json.

0 appel API. Voir docs/superpowers/specs/2026-07-18-gold-standard-eval-protocol-design.md.
Usage : poetry run python3 src/01_data_engineering/build_split.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # riotlib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core")) # dataset_split
import numpy as np
import pandas as pd
import riotlib as rl
from dataset_split import SPLIT_PATH, BUCKETS

DATASET_DIR = rl.DATA / "04_dataset"
RANK_DATASET = DATASET_DIR / "adc_player_dataset.parquet"
LP_DATASET = DATASET_DIR / "adc_player_lp_dataset.parquet"
SEED = 42
PROPORTIONS = {"train": 0.70, "calibration": 0.15, "test": 0.15}


def rank_by_puuid() -> dict[str, str]:
    """puuid -> rang, union des deux datasets per-player (rang prioritaire, LP en repli)."""
    mapping: dict[str, str] = {}
    if LP_DATASET.exists():
        lp = pd.read_parquet(LP_DATASET, columns=["puuid", "rank"])
        mapping.update(dict(zip(lp["puuid"], lp["rank"])))
    rk = pd.read_parquet(RANK_DATASET, columns=["puuid", "rank"])
    mapping.update(dict(zip(rk["puuid"], rk["rank"])))  # priorité au dataset rang
    return mapping


def assign(ranks: dict[str, str], proportions: dict = PROPORTIONS,
           seed: int = SEED) -> dict[str, str]:
    """Assigne chaque puuid à un bucket, stratifié par rang, déterministe (tri des
    puuid avant shuffle -> indépendant de l'ordre d'insertion)."""
    rng = np.random.RandomState(seed)
    by_rank: dict[str, list[str]] = defaultdict(list)
    for puuid, rank in sorted(ranks.items()):
        by_rank[rank].append(puuid)
    assignment: dict[str, str] = {}
    for rank in sorted(by_rank):
        puuids = by_rank[rank]
        rng.shuffle(puuids)
        n = len(puuids)
        n_train = int(round(n * proportions["train"]))
        n_calib = int(round(n * proportions["calibration"]))
        for i, puuid in enumerate(puuids):
            if i < n_train:
                assignment[puuid] = "train"
            elif i < n_train + n_calib:
                assignment[puuid] = "calibration"
            else:
                assignment[puuid] = "test"
    return assignment


def main() -> int:
    ranks = rank_by_puuid()
    assignment = assign(ranks)
    n_by = {b: defaultdict(int) for b in BUCKETS}
    for puuid, bucket in assignment.items():
        n_by[bucket][ranks[puuid]] += 1
    out = {
        "seed": SEED,
        "proportions": PROPORTIONS,
        "created_from": [RANK_DATASET.name]
                        + ([LP_DATASET.name] if LP_DATASET.exists() else []),
        "n_by_bucket_by_rank": {b: dict(sorted(n_by[b].items())) for b in BUCKETS},
        "assignment": assignment,
    }
    SPLIT_PATH.write_text(json.dumps(out, indent=2))
    print("  " + " / ".join(f"{b}:{sum(n_by[b].values())}" for b in BUCKETS)
          + f"  (total {len(assignment)} joueurs)")
    for b in BUCKETS:
        print(f"    {b:<12} {dict(sorted(n_by[b].items()))}")
    print(f"\n✓ Split écrit dans {SPLIT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
