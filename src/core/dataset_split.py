#!/usr/bin/env python3
"""core — split canonique train/calibration/test au niveau joueur, partagé par tout
le pipeline per-player. Généré une fois par build_split.py, consommé par lookup puuid.
Voir docs/superpowers/specs/2026-07-18-gold-standard-eval-protocol-design.md."""
from __future__ import annotations

import json
from pathlib import Path

import riotlib as rl

SPLIT_PATH = rl.DATA / "04_dataset" / "split.json"
BUCKETS = ("train", "calibration", "test")


def load_split(path: Path = SPLIT_PATH) -> dict:
    """Charge le split. Lève FileNotFoundError explicite si absent (pas de repli)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"split introuvable ({path}). Lance d'abord "
            "`poetry run python3 src/01_data_engineering/build_split.py`."
        )
    return json.loads(Path(path).read_text())


def puuids_in(split: dict, bucket: str) -> set[str]:
    if bucket not in BUCKETS:
        raise ValueError(f"bucket inconnu: {bucket!r} (attendus: {BUCKETS})")
    return {p for p, b in split["assignment"].items() if b == bucket}


def partition(df, split: dict, bucket: str, col: str = "puuid"):
    """Sous-DataFrame des lignes dont le `col` est dans `bucket`."""
    return df[df[col].isin(puuids_in(split, bucket))].copy()
