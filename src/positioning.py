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
