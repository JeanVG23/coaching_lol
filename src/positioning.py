"""Features macro-positionnement depuis la timeline Riot (0 CV, module pur).

extract_game (riotlib) appelle positioning_features en import PARESSEUX et niche
le retour sous record["position"]. Voir docs/superpowers/specs/2026-06-30-macro-positioning-design.md.
"""
from __future__ import annotations

from riotlib import approx_zone, phase_of, MAP_W, MAP_H

SIGHT = 1350.0                 # portée de vue d'un champion (proxy vision)
OVEREXT_THRESHOLD = 2000.0     # profondeur en terrain ennemi = "over-extended"
_MAP_MID = (MAP_W + MAP_H) / 2.0
_SQRT2 = 2 ** 0.5

# Base respawn wait (s) par niveau 1..18 (patch 16.x). v1 : facteur temps late-game
# ignoré (négligeable <30 min, sous-estimation conservatrice) -> raffinement v2.
_BRW = {1: 10, 2: 10, 3: 12, 4: 12, 5: 14, 6: 16, 7: 20, 8: 25, 9: 28, 10: 32.5,
        11: 35, 12: 37.5, 13: 40, 14: 42.5, 15: 45, 16: 47.5, 17: 50, 18: 52.5}

# Zone "lane" attendue par rôle (pour frac_own_lane / roam).
_ROLE_ZONE = {"TOP": "TOP", "MIDDLE": "MID", "BOTTOM": "BOT",
              "UTILITY": "BOT", "JUNGLE": "JUNGLE/RIVER"}

COACHING_SAFE = {
    "frac_own_lane_early", "frac_river_early", "frac_roam_mid", "frac_enemy_half",
    "frac_base", "avg_map_depth", "max_map_depth", "frac_overextended",
    "avg_dist_to_ally", "gold_dead_time",
    "wards_placed", "wards_placed_early", "control_wards_placed", "wards_killed",
}
ML_ONLY = {"frac_deaths_in_fog", "avg_unaccounted_enemies", "overext_x_unaccounted"}
ALL_FEATURES = COACHING_SAFE | ML_ONLY


def _build_snaps(timeline: dict) -> list[tuple[int, int, dict, dict]]:
    """Une passe : [(t_ms, minute, {pid:(x,y)}, {pid:level})] pour les 10 joueurs."""
    snaps = []
    for fr in timeline["info"]["frames"]:
        t = fr["timestamp"]
        pos, lvl = {}, {}
        for pid_s, pf in fr["participantFrames"].items():
            pid = int(pid_s)
            p = pf.get("position")
            if p and p.get("x") is not None and p.get("y") is not None:
                pos[pid] = (p["x"], p["y"])
            lvl[pid] = pf.get("level", 1)
        snaps.append((t, round(t / 60000), pos, lvl))
    return snaps


def _depth(x: float, y: float, my_team: int) -> float:
    """Profondeur signée dans le terrain ennemi (>0 = chez l'ennemi)."""
    raw = (x + y - _MAP_MID) / _SQRT2
    return raw if my_team == 100 else -raw


def _territory(snaps: list, pid: int, my_team: int) -> dict:
    """Calcule la fraction de temps en terrain ennemi, profondeur moyenne/max, over-extension."""
    depths, n, enemy_half, overext = [], 0, 0, 0
    for _t, _m, pos, _lvl in snaps:
        if pid not in pos:
            continue
        n += 1
        d = _depth(pos[pid][0], pos[pid][1], my_team)
        depths.append(max(0.0, d))
        if d > 0:
            enemy_half += 1
        if d > OVEREXT_THRESHOLD:
            overext += 1
    if not n:
        return {k: None for k in ("frac_enemy_half", "avg_map_depth",
                                  "max_map_depth", "frac_overextended")}
    return {
        "frac_enemy_half": enemy_half / n,
        "avg_map_depth": sum(depths) / n,
        "max_map_depth": max(depths),
        "frac_overextended": overext / n,
    }


def _zone_presence(snaps: list, pid: int, my_role: str) -> dict:
    """Calcule la fraction de temps passé dans chaque zone par phase."""
    own = _ROLE_ZONE.get(my_role, "BOT")
    e_own = e_river = e_tot = m_roam = m_tot = 0
    for _t, minute, pos, _lvl in snaps:
        if pid not in pos:
            continue
        z = approx_zone(*pos[pid])
        ph = phase_of(minute)
        if ph == "early":
            e_tot += 1
            if z == own:
                e_own += 1
            if z == "JUNGLE/RIVER":
                e_river += 1
        elif ph == "mid":
            m_tot += 1
            # roam = autre lane que la sienne (pas river/jungle, pas sa lane)
            if z in ("TOP", "MID", "BOT") and z != own:
                m_roam += 1
    return {
        "frac_own_lane_early": e_own / e_tot if e_tot else None,
        "frac_river_early": e_river / e_tot if e_tot else None,
        "frac_roam_mid": m_roam / m_tot if m_tot else None,
    }


def _in_base(x: float, y: float, my_team: int) -> bool:
    """Vérifie si la position est dans la base (boîte d'équipe)."""
    if my_team == 100:
        return x < 3500 and y < 3500
    return x > 11300 and y > 11300


def _base_and_isolation(snaps: list, pid: int, my_team: int, allies: list) -> dict:
    """Calcule la fraction de temps en base et la distance moyenne aux alliés."""
    n = base = 0
    dists = []
    others = [a for a in allies if a != pid]
    for _t, _m, pos, _lvl in snaps:
        if pid not in pos:
            continue
        n += 1
        x, y = pos[pid]
        if _in_base(x, y, my_team):
            base += 1
        near = [((pos[a][0] - x) ** 2 + (pos[a][1] - y) ** 2) ** 0.5
                for a in others if a in pos]
        if near:
            dists.append(min(near))
    if not n:
        return {"frac_base": None, "avg_dist_to_ally": None}
    return {
        "frac_base": base / n,
        "avg_dist_to_ally": sum(dists) / len(dists) if dists else None,
    }


def _ward_counts(timeline: dict, pid: int) -> dict:
    """Compte les wards placés et tués par le joueur pid (Famille C exact)."""
    placed = early = control = killed = 0
    for fr in timeline["info"]["frames"]:
        for ev in fr.get("events", []):
            t = ev.get("type")
            if t == "WARD_PLACED" and ev.get("creatorId") == pid:
                placed += 1
                if round(ev["timestamp"] / 60000) < 14:
                    early += 1
                if ev.get("wardType") == "CONTROL_WARD":
                    control += 1
            elif t == "WARD_KILL" and ev.get("killerId") == pid:
                killed += 1
    return {"wards_placed": placed, "wards_placed_early": early,
            "control_wards_placed": control, "wards_killed": killed}


def _vision_frames(snaps: list, pid: int, allies: list, enemies: list,
                   my_team: int) -> dict:
    """Calcule le nombre moyen d'ennemis non-accountés et l'overextension × unaccounted."""
    unacc_per_frame, overext_unacc = [], []
    for _t, _m, pos, _lvl in snaps:
        if pid not in pos:
            continue
        seen = [pos[a] for a in allies if a in pos]
        unacc = 0
        for e in enemies:
            if e not in pos:
                continue
            ex, ey = pos[e]
            if not any(((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5 <= SIGHT
                       for sx, sy in seen):
                unacc += 1
        unacc_per_frame.append(unacc)
        depth = max(0.0, _depth(pos[pid][0], pos[pid][1], my_team))
        overext_unacc.append(depth * unacc)
    if not unacc_per_frame:
        return {"avg_unaccounted_enemies": None, "overext_x_unaccounted": None}
    return {
        "avg_unaccounted_enemies": sum(unacc_per_frame) / len(unacc_per_frame),
        "overext_x_unaccounted": sum(overext_unacc) / len(overext_unacc),
    }
