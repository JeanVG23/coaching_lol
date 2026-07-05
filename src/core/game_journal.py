"""Journal structuré d'UNE game depuis match + timeline (0 CV, module pur).

Événements ancrés (timestamp exact) du joueur ciblé : morts et recalls, chacun
avec son contexte — gold-state, gold non dépensé, objectif up/imminent. C'est la
granularité qui manque aux médianes agrégées : le coach peut citer « mort à 17:42
en BOT, drake dans 30 s, 1 450 g non dépensés » au lieu de « tu meurs trop ».

ASYMÉTRIE : tout champ du journal est une information que le joueur AVAIT —
sa propre mort (killer affiché dans le death recap), son propre gold, les timers
d'objectifs (HUD public). Aucun proxy de vision (fog, unaccounted) : ces features
restent ML_ONLY dans positioning.py et ne doivent jamais entrer ici.
"""
from __future__ import annotations

from riotlib import (SR_MAP_ID, approx_zone, patch_of, phase_of,
                     _frames_by_minute, _gold_state)

# Timers d'objectifs (ms) — v1 approximative, À AJUSTER PAR PATCH.
# Elder (après l'âme) et Atakhan ignorés en v1 : le drake standard porte
# l'essentiel du signal coaching ADC (botside, récurrent).
OBJECTIVES = {
    "DRAGON": {"first": 5 * 60000, "respawn": 5 * 60000},
    "BARON_NASHOR": {"first": 25 * 60000, "respawn": 6 * 60000},
}
IMMINENT_WINDOW_S = 90        # objectif "imminent" si spawn dans <= 90 s
OPENING_BUY_MS = 90000        # achats avant 1:30 = shopping de départ, pas un recall
RECALL_CLUSTER_GAP_MS = 30000 # achats espacés de <= 30 s = même visite de shop


def _clock(t_ms: int) -> str:
    return f"{t_ms // 60000}:{(t_ms % 60000) // 1000:02d}"


def _events(timeline: dict):
    for fr in timeline["info"]["frames"]:
        yield from fr.get("events", [])


def _objective_kills(timeline: dict) -> dict[str, list[int]]:
    kills = {name: [] for name in OBJECTIVES}
    for ev in _events(timeline):
        if ev.get("type") == "ELITE_MONSTER_KILL" and ev.get("monsterType") in kills:
            kills[ev["monsterType"]].append(ev["timestamp"])
    return {name: sorted(ts) for name, ts in kills.items()}


def _objective_at(kills: dict[str, list[int]], t_ms: int) -> dict | None:
    """Objectif up ou imminent à t_ms. Up prioritaire, sinon le plus proche."""
    up, imminent = [], []
    for name, cfg in OBJECTIVES.items():
        past = [k for k in kills[name] if k <= t_ms]
        next_spawn = past[-1] + cfg["respawn"] if past else cfg["first"]
        if t_ms >= next_spawn:
            up.append((t_ms - next_spawn, name))
        elif next_spawn - t_ms <= IMMINENT_WINDOW_S * 1000:
            imminent.append((next_spawn - t_ms, name))
    if up:
        delta, name = min(up)
        return {"type": name, "status": "up", "delta_s": round(delta / 1000)}
    if imminent:
        delta, name = min(imminent)
        return {"type": name, "status": "imminent", "delta_s": round(delta / 1000)}
    return None


def _frame_before(timeline: dict, pid: int, t_ms: int) -> dict | None:
    """Dernière participantFrame du joueur pid avec timestamp <= t_ms."""
    best = None
    for fr in timeline["info"]["frames"]:
        if fr["timestamp"] > t_ms:
            break
        pf = fr["participantFrames"].get(str(pid))
        if pf:
            best = pf
    return best


def _deaths(timeline: dict, pid: int, pid_champ: dict, pid_role: dict,
            enemy_jungle_pid: int | None, gold_state_at, obj_kills) -> list[dict]:
    out = []
    for ev in _events(timeline):
        if ev.get("type") != "CHAMPION_KILL" or ev.get("victimId") != pid:
            continue
        t = ev["timestamp"]
        kpid = ev.get("killerId")
        assisters = ev.get("assistingParticipantIds", [])
        involved = ({kpid} | set(assisters)) - {None}
        pos = ev.get("position", {})
        pf = _frame_before(timeline, pid, t)
        out.append({
            "t_ms": t, "clock": _clock(t),
            "minute": t // 60000, "phase": phase_of(t // 60000),
            "zone": approx_zone(pos.get("x", 0), pos.get("y", 0)),
            "killer_champ": pid_champ.get(kpid, "?"),
            "killer_role": pid_role.get(kpid, "?"),
            "is_solo": len(assisters) == 0,
            "is_ganked_by_jungle": (enemy_jungle_pid is not None
                                    and enemy_jungle_pid in involved),
            "gold_state": gold_state_at(t // 60000),
            "unspent_gold": pf.get("currentGold") if pf else None,
            "level": pf.get("level") if pf else None,
            "objective": _objective_at(obj_kills, t),
        })
    out.sort(key=lambda d: d["t_ms"])
    return out


def _recalls(timeline: dict, pid: int, obj_kills) -> list[dict]:
    """Visites de shop (clusters d'achats), hors shopping de départ.

    Approximation v1 : un achat implique la présence au shop (recall ou reset
    après mort — les deux sont des « resets » à coacher). gold_before = currentGold
    de la dernière frame avant la visite (léger plancher, frames espacées de 60 s).
    """
    buys = sorted(ev["timestamp"] for ev in _events(timeline)
                  if ev.get("type") == "ITEM_PURCHASED"
                  and ev.get("participantId") == pid
                  and ev["timestamp"] >= OPENING_BUY_MS)
    visits: list[list[int]] = []
    for t in buys:
        if visits and t - visits[-1][-1] <= RECALL_CLUSTER_GAP_MS:
            visits[-1].append(t)
        else:
            visits.append([t])
    out = []
    for visit in visits:
        t0 = visit[0]
        pf = _frame_before(timeline, pid, t0)
        out.append({
            "t_ms": t0, "clock": _clock(t0),
            "minute": t0 // 60000, "phase": phase_of(t0 // 60000),
            "items_bought": len(visit),
            "gold_before": pf.get("currentGold") if pf else None,
            "objective": _objective_at(obj_kills, t0),
        })
    return out


def game_journal(match: dict, timeline: dict, puuid: str) -> dict | None:
    """Une game -> journal d'événements ancrés du joueur. None si hors Faille."""
    info = match["info"]
    if info.get("mapId") != SR_MAP_ID:
        return None
    meta = match["metadata"]
    if puuid not in meta["participants"]:
        return None
    pid = meta["participants"].index(puuid) + 1
    parts = info["participants"]
    me = parts[pid - 1]
    my_team = me["teamId"]

    pid_champ = {i + 1: p["championName"] for i, p in enumerate(parts)}
    pid_role = {i + 1: p.get("teamPosition") or "?" for i, p in enumerate(parts)}
    my_role = me.get("teamPosition") or ""
    opp_pid = next((i + 1 for i, p in enumerate(parts)
                    if p["teamId"] != my_team and my_role
                    and (p.get("teamPosition") or "") == my_role), None)
    enemy_jungle_pid = next((i + 1 for i, p in enumerate(parts)
                             if p["teamId"] != my_team
                             and p.get("teamPosition") == "JUNGLE"), None)

    my_fr = _frames_by_minute(timeline, pid)
    opp_fr = _frames_by_minute(timeline, opp_pid) if opp_pid else {}

    def gold_state_at(minute: int) -> str | None:
        for m in range(minute, -1, -1):   # frame la plus récente <= minute
            if m in my_fr and m in opp_fr:
                return _gold_state(my_fr[m].get("totalGold", 0)
                                   - opp_fr[m].get("totalGold", 0))
        return None

    obj_kills = _objective_kills(timeline)
    return {
        "match_id": meta["matchId"],
        "patch": patch_of(info.get("gameVersion", "")),
        "champion": me["championName"],
        "role": my_role,
        "win": me.get("win"),
        "duration_min": round(info.get("gameDuration", 0) / 60, 1),
        "kda": {"kills": me.get("kills", 0), "deaths": me.get("deaths", 0),
                "assists": me.get("assists", 0)},
        "opponent": pid_champ.get(opp_pid),
        "deaths": _deaths(timeline, pid, pid_champ, pid_role,
                          enemy_jungle_pid, gold_state_at, obj_kills),
        "recalls": _recalls(timeline, pid, obj_kills),
    }
