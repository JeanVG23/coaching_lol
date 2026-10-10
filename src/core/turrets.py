"""Tourelles de la Faille : table de positions et tourelle amie debout la plus proche.

Module pur, 0 reseau, 0 API. Riot ne publie pas les coordonnees des tourelles :
elles sont reconstituees depuis les `BUILDING_KILL` du raw par
`src/pipeline_ops/build_turret_map.py`, qui ecrit `data/00_static/sr_turrets.json`
(configuration source force-add, au meme titre que `champion_traits.json`).

Point de correction, et raison d'etre du parametre `destroyed` : une tourelle
DETRUITE ne protege plus personne. Dire « tu etais a 2 850 unites de ta
tourelle » alors qu'elle est tombee a la minute 12 serait faux.
"""
from __future__ import annotations

from pathlib import Path

import champion_profiles as cprof
import riotlib as rl

OUTER = "OUTER_TURRET"

# Role de la timeline -> lane de la table. Le jungler n'a pas de lane propre :
# `None` fait omettre `beyond_own_outer_turret`, plutot que d'inventer une
# frontiere qui n'existe pas pour lui.
LANE_OF_ROLE: dict[str, str | None] = {
    "TOP": "TOP_LANE", "MIDDLE": "MID_LANE",
    "BOTTOM": "BOT_LANE", "UTILITY": "BOT_LANE", "JUNGLE": None,
}

# Cache manuel indexe par chemin, et non `lru_cache` : la pile medaillon est
# redirigee en test par substitution d'attribut de module (`cp.STATIC_DIR`),
# et un cache sans cle de chemin servirait la table reelle a la demo.
_CACHE: dict[Path, list[dict]] = {}


def clear_cache() -> None:
    """Vide le cache mémoire des tables de tourelles."""
    _CACHE.clear()


def _path(static_dir=None) -> Path:
    base = Path(static_dir) if static_dir is not None else cprof.STATIC_DIR
    return base / "sr_turrets.json"


def load(static_dir=None) -> list[dict]:
    """[{"team", "lane", "tier", "x", "y"}]. Table absente -> liste vide.

    Table absente n'est pas une erreur : le journal omet alors simplement les
    champs de tourelle, comme il omet les blocs de spike sans catalogue d'items.
    """
    path = _path(static_dir)
    if path not in _CACHE:
        _CACHE[path] = rl.read_json(path) or []
    return _CACHE[path]


def destroyed_at(events, t_ms: int) -> set[tuple[int, str, str]]:
    """Cles `(team, lane, tier)` des tourelles tombees a `t_ms`.

    Semantique timeline : `BUILDING_KILL.teamId` = equipe qui PERD le
    batiment. La cle porte donc l'equipe proprietaire, ce qui est exactement ce
    dont `nearest_standing` a besoin.
    """
    out = set()
    for ev in events:
        if ev.get("type") != "BUILDING_KILL" or ev.get("timestamp", 0) > t_ms:
            continue
        tower = ev.get("towerType")
        if not tower:                     # inhibiteur ou nexus : hors table
            continue
        out.add((ev.get("teamId"), ev.get("laneType"), tower))
    return out


def nearest_standing(x: float, y: float, my_team: int,
                     destroyed: set, table=None, static_dir=None) -> dict | None:
    """Tourelle de MON equipe encore debout la plus proche de (x, y).

    Jamais une tourelle ennemie : « ta tourelle » est un abri, pas une cible.
    """
    rows = load(static_dir) if table is None else table
    best = None
    for row in rows:
        if row["team"] != my_team:
            continue
        if (row["team"], row["lane"], row["tier"]) in destroyed:
            continue
        distance = ((row["x"] - x) ** 2 + (row["y"] - y) ** 2) ** 0.5
        if best is None or distance < best[0]:
            best = (distance, row)
    if best is None:
        return None
    distance, row = best
    return {"lane": row["lane"], "tier": row["tier"], "distance": round(distance)}


def own_outer(my_team: int, role: str, destroyed: set,
              table=None, static_dir=None) -> dict | None:
    """Tourelle exterieure ENCORE DEBOUT de ma propre lane, sinon None.

    Tombee, elle ne definit plus de frontiere : l'appelant omet alors
    `beyond_own_outer_turret` au lieu de trancher sur un repere disparu.
    """
    lane = LANE_OF_ROLE.get(role or "")
    if lane is None:
        return None
    for row in (load(static_dir) if table is None else table):
        if (row["team"] == my_team and row["lane"] == lane
                and row["tier"] == OUTER
                and (row["team"], lane, OUTER) not in destroyed):
            return row
    return None
