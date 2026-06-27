"""Identité des champions : Data Dragon (statique) + table de traits curée.

Module isolé, sans appel réseau au runtime (cache disque). Donne un vecteur
d'identité par champion et dérive les axes de contexte de botlane.
"""
from __future__ import annotations

AXES_CURATED = ("power_curve", "lane_pattern", "playstyle", "gank_threat", "roam")
RANGED_MIN = 500  # attackrange >= 500 => ranged


def champion_vector(name: str, traits: dict | None = None,
                    ddragon: dict | None = None) -> dict:
    if traits is None:
        traits = load_traits()
    if ddragon is None:
        ddragon = load_ddragon()
    dd = ddragon.get(name, {})
    tr = traits.get(name, {})
    rng = dd.get("attackrange")
    range_class = "unknown" if rng is None else ("ranged" if rng >= RANGED_MIN else "melee")
    v = {"name": name, "range_class": range_class, "tags": dd.get("tags", [])}
    for axis in AXES_CURATED:
        v[axis] = tr.get(axis, "unknown")
    return v


def load_traits() -> dict:  # remplacé Task 3
    return {}


def load_ddragon() -> dict:  # remplacé Task 3
    return {}
