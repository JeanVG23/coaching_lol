#!/usr/bin/env python3
"""
01_data_engineering — raw -> dataset séquentiel (1 ADC d'une game = 1 séquence [40,20]).

Construit les séquences d'états par-minute DEPUIS LE RAW (0 API), pour le transformer
séquentiel. Réutilise build_dataset (adc_puuids, build_rank_map, _load_raw, RANKS/HIGH_ELO)
via importlib — zéro duplication de la logique métier.

State vector 20-dim/frame = ADC ciblé (8) + ADC adverse (8) + 4 diffs relatives
(gold, cs, xp, level). Aucun event en v1 (les events discrets = étape 2 d'enrichissement,
cf. spec 2026-07-18 §Pièges : un null v1 ne réfute pas la thèse séquence).

Sortie : data/04_dataset/adc_sequence_dataset.npz
Usage : poetry run python3 src/01_data_engineering/build_sequence_dataset.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_CORE = Path(__file__).resolve().parent.parent / "core"
sys.path.insert(0, str(_CORE))                       # import riotlib
import numpy as np
import riotlib as rl

# reutilise build_dataset (vit dans un dossier non-importable) via importlib
_BD = Path(__file__).resolve().parent / "build_dataset.py"
_spec = importlib.util.spec_from_file_location("build_dataset", _BD)
build_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_dataset)

MAP_SIZE = 14800.0          # Summoner's Rift (approx, normalisation position -> [0,1])
MAX_LEN = 40                 # cap minutes ; pad au-delà de la durée réelle
STATE_FIELDS = ["pos_x", "pos_y", "totalGold", "currentGold", "xp",
                "level", "minionsKilled", "jungleMinionsKilled"]


def frame_state(pf: dict) -> list[float]:
    """Vecteur d'état 8-dim depuis un participantFrame. Position normalisée à [0,1]."""
    pos = pf.get("position") or {}
    return [
        float(pos.get("x") or 0) / MAP_SIZE,
        float(pos.get("y") or 0) / MAP_SIZE,
        float(pf.get("totalGold") or 0),
        float(pf.get("currentGold") or 0),
        float(pf.get("xp") or 0),
        float(pf.get("level") or 0),
        float(pf.get("minionsKilled") or 0),
        float(pf.get("jungleMinionsKilled") or 0),
    ]


def participant_pid(match: dict, puuid: str) -> int:
    """puuid -> participantId (1-indexé)."""
    return match["metadata"]["participants"].index(puuid) + 1


def opponent_pid(match: dict, target_puuid: str) -> int | None:
    """participantId de l'adversaire de même rôle (BOTTOM), équipe opposée. None si introuvable."""
    meta = match["metadata"]
    parts = match["info"]["participants"]
    pidx = meta["participants"].index(target_puuid)
    me = parts[pidx]
    my_team = me["teamId"]
    my_role = me.get("teamPosition") or ""
    for i, p in enumerate(parts):
        if p["teamId"] != my_team and (p.get("teamPosition") or "") == my_role and my_role:
            return i + 1
    return None


def _diffs(self_state: list[float], opp_state: list[float]) -> list[float]:
    """4 diffs relatives : gold, cs, xp, level (signaux de lane)."""
    return [
        self_state[2] - opp_state[2],                              # totalGold
        (self_state[6] + self_state[7]) - (opp_state[6] + opp_state[7]),  # cs
        self_state[4] - opp_state[4],                              # xp
        self_state[5] - opp_state[5],                              # level
    ]


def build_sequence(match: dict, timeline: dict,
                   target_puuid: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Une game -> (seq[40,20] float32, mask[40] bool). None si pas d'opponent ou 0 frame."""
    pid = participant_pid(match, target_puuid)
    opp = opponent_pid(match, target_puuid)
    if opp is None:
        return None
    my_fr = rl._frames_by_minute(timeline, pid)      # {minute_int: participantFrame}
    opp_fr = rl._frames_by_minute(timeline, opp)
    seq = np.zeros((MAX_LEN, 20), dtype=np.float32)
    mask = np.zeros(MAX_LEN, dtype=bool)
    for minute, pf in my_fr.items():
        if minute >= MAX_LEN:
            continue
        self_s = frame_state(pf)
        opp_s = frame_state(opp_fr.get(minute, {}))  # frame adverse manquante -> zeros
        seq[minute] = self_s + opp_s + _diffs(self_s, opp_s)
        mask[minute] = True
    if mask.sum() == 0:
        return None
    return seq, mask