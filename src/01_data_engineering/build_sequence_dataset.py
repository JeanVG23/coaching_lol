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
import game_journal as gj      # réutilise OBJECTIVES, _objective_kills, _events, _recalls (DRY)

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
    """puuid -> participantId (1-indexé). Délègue à riotlib.participant_id."""
    return rl.participant_id(match, puuid)


def opponent_pid(match: dict, target_puuid: str) -> int | None:
    """participantId de l'adversaire de même rôle, équipe opposée. None si introuvable.

    Délègue aux primitives riotlib (`find_pid`/`enemy_team_of`) : cette résolution y
    était recopiée à l'identique, à l'ordre des conditions près.
    """
    pidx = match["metadata"]["participants"].index(target_puuid)
    me = match["info"]["participants"][pidx]
    my_role = me.get("teamPosition") or ""
    if not my_role:
        return None
    return rl.find_pid(match, team=rl.enemy_team_of(me["teamId"]), role=my_role)


def _diffs(self_state: list[float], opp_state: list[float]) -> list[float]:
    """4 diffs relatives : gold, cs, xp, level (signaux de lane)."""
    return [
        self_state[2] - opp_state[2],                              # totalGold
        (self_state[6] + self_state[7]) - (opp_state[6] + opp_state[7]),  # cs
        self_state[4] - opp_state[4],                              # xp
        self_state[5] - opp_state[5],                              # level
    ]


def _obj_up(obj_kills: dict, name: str, t_ms: int) -> bool:
    """Objectif `name` respawné (up) à t_ms ? Réutilise game_journal.OBJECTIVES (timers)."""
    cfg = gj.OBJECTIVES[name]
    past = [k for k in obj_kills[name] if k <= t_ms]
    next_spawn = past[-1] + cfg["respawn"] if past else cfg["first"]
    return t_ms >= next_spawn


def _event_channels(timeline: dict, pid: int, opp_pid: int,
                    enemy_jungle_pid: int | None,
                    obj_kills: dict | None = None) -> np.ndarray:
    """-> [40, 7] float32. Canaux events binaires par minute, COACHING_SAFE (info que le
    joueur avait : sa mort, l'annonce de la mort adverse, timers objectifs = HUD public).
    Ordre : self_death_m, opp_death_m, self_recall_m, drake_up, baron_up, is_ganked,
    is_solo_death. Réutilise game_journal (OBJECTIVES, _objective_kills, _events, _recalls)
    — version allégée : on ne calcule que le bucket minute + gank/solo, PAS gold_state/
    consequences (coût inutile sur 43-95k games × 2 ADC)."""
    ch = np.zeros((MAX_LEN, 7), dtype=np.float32)
    # obj_kills injectable : identique pour les 2 ADC d'une même game.
    if obj_kills is None:
        obj_kills = gj._objective_kills(timeline)
    for m in range(MAX_LEN):
        t_end = (m + 1) * 60000 - 1                       # fin de la minute m
        if _obj_up(obj_kills, "DRAGON", t_end):
            ch[m, 3] = 1.0                                # drake_up
        if _obj_up(obj_kills, "BARON_NASHOR", t_end):
            ch[m, 4] = 1.0                                # baron_up
    for ev in gj._events(timeline):
        et = ev.get("timestamp", 0)
        m = et // 60000
        if m >= MAX_LEN:
            continue
        if ev.get("type") == "CHAMPION_KILL":
            vid = ev.get("victimId")
            if vid == pid:
                ch[m, 0] = 1.0                            # self_death_m
                assisters = ev.get("assistingParticipantIds") or []
                involved = ({ev.get("killerId")} | set(assisters)) - {None}
                if enemy_jungle_pid is not None and enemy_jungle_pid in involved:
                    ch[m, 5] = 1.0                        # is_ganked
                if len(assisters) == 0:
                    ch[m, 6] = 1.0                        # is_solo_death
            elif vid == opp_pid:
                ch[m, 1] = 1.0                            # opp_death_m
    for rec in gj.recalls_for(timeline, pid, obj_kills):  # visites de shop (clusters d'achats)
        m = rec["t_ms"] // 60000
        if m < MAX_LEN:
            ch[m, 2] = 1.0                               # self_recall_m
    return ch


def build_sequence(match: dict, timeline: dict, target_puuid: str,
                   obj_kills: dict | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Une game -> (seq[40,27] float32, mask[40] bool). None si pas d'opponent ou 0 frame.
    Frame = self(8) + opp(8) + diffs(4) + event_channels(7) = 27-d.

    `obj_kills` est injectable : les timers d'objectifs sont propres à la game, pas au
    joueur, et étaient recalculés à l'identique pour chacun des 2 ADC.
    """
    pid = participant_pid(match, target_puuid)
    opp = opponent_pid(match, target_puuid)
    if opp is None:
        return None
    pidx = match["metadata"]["participants"].index(target_puuid)
    my_team = match["info"]["participants"][pidx]["teamId"]
    enemy_jungle_pid = rl.find_pid(match, team=rl.enemy_team_of(my_team), role="JUNGLE")
    my_fr = rl.frames_by_minute(timeline, pid)
    opp_fr = rl.frames_by_minute(timeline, opp)
    ev = _event_channels(timeline, pid, opp, enemy_jungle_pid, obj_kills)
    seq = np.zeros((MAX_LEN, 27), dtype=np.float32)
    mask = np.zeros(MAX_LEN, dtype=bool)
    for minute, pf in my_fr.items():
        if minute >= MAX_LEN:
            continue
        self_s = frame_state(pf)
        opp_s = frame_state(opp_fr.get(minute, {}))   # frame adverse manquante -> zeros
        seq[minute] = self_s + opp_s + _diffs(self_s, opp_s) + list(ev[minute])
        mask[minute] = True
    if mask.sum() == 0:
        return None
    return seq, mask


DATASET_DIR = rl.DATA / "04_dataset"


def champion_of(match: dict, puuid: str) -> str:
    pidx = match["metadata"]["participants"].index(puuid)
    return match["info"]["participants"][pidx].get("championName") or "unknown"


def main() -> int:
    rank_of, multi = build_dataset.build_rank_map()
    print(f"  {len(rank_of)} games référentiel distinctes ({multi} multi-rang)")
    seqs, masks, labels, ranks, puuids, mids, champs = [], [], [], [], [], [], []
    raw_miss = 0
    for mid, rank in rank_of.items():
        raw = build_dataset._load_raw(mid)
        if not raw:
            raw_miss += 1
            continue
        match, timeline = raw
        obj_kills = gj._objective_kills(timeline)   # une fois par game, pas par ADC
        for puuid in build_dataset.adc_puuids(match):
            out = build_sequence(match, timeline, puuid, obj_kills)
            if out is None:
                continue
            seq, mask = out
            seqs.append(seq)
            masks.append(mask)
            ranks.append(rank)
            puuids.append(puuid)
            mids.append(mid)
            champs.append(champion_of(match, puuid))
            labels.append(1 if rank in build_dataset.HIGH_ELO else 0)
    if raw_miss:
        print(f"  ⚠ {raw_miss} games sans raw lisible -> ignorées")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATASET_DIR / "adc_sequence_dataset.npz",
        sequences=np.stack(seqs).astype(np.float32),
        mask=np.stack(masks).astype(bool),
        label_highelo=np.array(labels, dtype=np.int64),
        rank=np.array(ranks, dtype=object),
        puuid=np.array(puuids, dtype=object),
        match_id=np.array(mids, dtype=object),
        champion=np.array(champs, dtype=object),
    )
    print(f"\n✓ {len(seqs)} séquences ADC -> {DATASET_DIR}/adc_sequence_dataset.npz")
    print(f"  high_elo (GM+Chall=1) : {int(np.sum(labels))} / {len(labels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())