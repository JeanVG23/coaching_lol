#!/usr/bin/env python3
"""
02_data_science — couche donnée/CV partagée par train_sequence_model et
pretrain_sequence_model (DRY). Chargement npz, labels par tâche, folds joueur-groupés,
purge miroir, standardisation per-feature (z-score sur le train du fold).

⚠ Standardisation non négociable (spec 2026-07-18 décision 3) : totalGold ~15000 dans la
même projection que position [0,1] rend l'entraînement instable ; sans standardisation un
null est un artefact d'optimisation, pas une réponse à la question de recherche.
"""
from __future__ import annotations

import sys
from pathlib import Path

_CORE = Path(__file__).resolve().parent.parent / "core"
sys.path.insert(0, str(_CORE))
import numpy as np
import riotlib as rl
import ranks as rank_defs

DATASET = rl.DATA / "04_dataset" / "adc_sequence_dataset.npz"
# Réexports depuis la source unique src/core/ranks.py (cf. rank_defs.TARGETS).
HIGH_ELO = rank_defs.HIGH_ELO
DIA_CHALL = set(rank_defs.TARGETS["dia_chall"]["ranks"])
TASKS = tuple(rank_defs.TARGETS)


def load_dataset(path: Path = DATASET) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def task_subset(data: dict, task: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (idx rows, y binaire). Les rangs retenus et la classe positive viennent de
    `ranks.TARGETS` : high_elo garde les 4 rangs et labellise GM/C ; dia_chall filtre
    diamond+challenger et labellise challenger. Le perso (rank None) est toujours exclu."""
    spec = rank_defs.TARGETS.get(task)
    if spec is None:
        raise ValueError(f"task inconnue: {task}")
    allowed, positive = set(spec["ranks"]), spec["pos"]
    row_ranks = data["rank"]
    idx = np.array([i for i, r in enumerate(row_ranks) if r in allowed])
    y = np.array([1 if row_ranks[i] in positive else 0 for i in idx], dtype=np.int64)
    return idx, y


def player_folds(puuids: np.ndarray, y: np.ndarray,
                 n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold sur les joueurs (chaque joueur -> 1 fold), expandu en indices de rows.
    Stratification sur le label du joueur (mode de ses rows). Aucun joueur à cheval train/val."""
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    df = pd.DataFrame({"puuid": puuids, "y": y})
    player_label = df.groupby("puuid")["y"].agg(lambda s: s.mode().iloc[0]).reset_index()
    player_label.columns = ["puuid", "plabel"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    player_arr = player_label["puuid"].to_numpy()
    plabel_arr = player_label["plabel"].to_numpy()
    for p_tr, p_va in cv.split(player_arr, plabel_arr):
        tr_p = set(player_arr[p_tr]); va_p = set(player_arr[p_va])
        tr_idx = np.array([i for i, p in enumerate(puuids) if p in tr_p])
        va_idx = np.array([i for i, p in enumerate(puuids) if p in va_p])
        folds.append((tr_idx, va_idx))
    return folds


def mirror_purge(train_idx: np.ndarray, val_puuids: set,
                 match_ids: np.ndarray, puuids: np.ndarray) -> np.ndarray:
    """Drop les rows de train dont l'ADC adverse (autre puuid du même match_id) est un
    joueur de val (fuite par games miroir : les 2 ADC d'une game sont des lignes en miroir).
    O(N) : pré-indexe match_id -> puuids une fois, puis lookup par row de train (vs O(n²)
    naïf qui coûterait ~500M itérations Python sur 5 folds × 2 tâches)."""
    val_puuids = set(val_puuids)
    match_to_puuids: dict[object, list] = {}
    for mid, p in zip(match_ids, puuids):
        match_to_puuids.setdefault(mid, []).append(p)
    keep = []
    for i in train_idx:
        mid = match_ids[i]; me = puuids[i]
        opps = match_to_puuids.get(mid, [])
        if any(o != me and o in val_puuids for o in opps):
            continue
        keep.append(i)
    return np.array(keep, dtype=train_idx.dtype)


def standardize_fit(sequences: np.ndarray, mask: np.ndarray,
                    train_idx: np.ndarray, bin_cols=None) -> tuple[np.ndarray, np.ndarray]:
    """mean/std par feature sur les frames valides des rows de train. bin_cols = indices de
    colonnes à laisser brutes (canaux events binaires v2) : on force mean=0/std=1 -> apply
    est l'identité sur ces cols. Défaut None = toutes cols standardisées (v1, backward-compat)."""
    X = sequences[train_idx][mask[train_idx]]      # [n_valid, F]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0                          # garde-fou feature constante
    if bin_cols is not None:
        bc = list(bin_cols)
        mean[bc] = 0.0
        std[bc] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(sequences: np.ndarray, mean: np.ndarray,
                      std: np.ndarray) -> np.ndarray:
    return ((sequences - mean) / std).astype(np.float32)