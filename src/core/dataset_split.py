#!/usr/bin/env python3
"""core — split canonique train/calibration/test au niveau joueur, partagé par tout
le pipeline per-player. Généré une fois par build_split.py, consommé par lookup puuid.
Voir docs/superpowers/specs/2026-07-18-gold-standard-eval-protocol-design.md."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
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


def compute_dataset_identity(
    puuids: Iterable[str],
    features: Iterable[str] | None = None,
    split_assignment: dict[str, str] | None = None,
    min_player_games: int | None = None,
) -> dict:
    """Empreinte canonique (déterministe) d'un jeu de données et configuration ML."""
    sorted_puuids = sorted(set(puuids))
    puuid_digest = hashlib.sha256(",".join(sorted_puuids).encode()).hexdigest()[:16]

    feat_digest = None
    if features is not None:
        feat_digest = hashlib.sha256(",".join(sorted(features)).encode()).hexdigest()[:16]

    split_digest = None
    if split_assignment is not None:
        split_entries = [f"{p}:{split_assignment[p]}" for p in sorted_puuids if p in split_assignment]
        split_digest = hashlib.sha256(",".join(split_entries).encode()).hexdigest()[:16]

    components = [
        f"puuids:{puuid_digest}",
        f"features:{feat_digest or 'none'}",
        f"split:{split_digest or 'none'}",
        f"min_games:{min_player_games or 0}",
    ]
    fingerprint = hashlib.sha256(";".join(components).encode()).hexdigest()[:16]

    return {
        "fingerprint": fingerprint,
        "n_players": len(sorted_puuids),
        "puuid_digest": puuid_digest,
        "features_digest": feat_digest,
        "split_digest": split_digest,
        "min_player_games": min_player_games,
    }
