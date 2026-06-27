"""Identité des champions : Data Dragon (statique) + table de traits curée.

Module isolé, sans appel réseau au runtime (cache disque). Donne un vecteur
d'identité par champion et dérive les axes de contexte de botlane.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "data" / "00_static"
DDRAGON_VERSION = "16.13.1"  # figée ; refresh = action manuelle (fetch_ddragon)
TRAITS_PATH = STATIC_DIR / "champion_traits.json"

AXES_CURATED = ("power_curve", "lane_pattern", "playstyle", "gank_threat", "roam")
RANGED_MIN = 500  # attackrange >= 500 => ranged


def fetch_ddragon(version: str | None = None) -> Path:
    version = version or DDRAGON_VERSION
    url = (f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/championFull.json")
    dest = STATIC_DIR / "ddragon" / version / "championFull.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_text(resp.text)
    return dest


@lru_cache(maxsize=1)
def load_ddragon() -> dict:
    path = STATIC_DIR / "ddragon" / DDRAGON_VERSION / "championFull.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())["data"]
    out = {}
    for champ in raw.values():
        out[champ["id"]] = {
            "attackrange": champ.get("stats", {}).get("attackrange"),
            "tags": champ.get("tags", []),
        }
    return out


@lru_cache(maxsize=1)
def load_traits() -> dict:
    if not TRAITS_PATH.exists():
        return {}
    data = json.loads(TRAITS_PATH.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


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
    v = {"name": name, "range_class": range_class, "tags": list(dd.get("tags", []))}
    for axis in AXES_CURATED:
        v[axis] = tr.get(axis, "unknown")
    return v


def _vec(name, traits, ddragon):
    """Récupère le vecteur d'un champion, ou {} si None."""
    return champion_vector(name, traits, ddragon) if name else {}


def _lane_pattern(enemy_adc_v, enemy_supp_v) -> str:
    """Dérive le pattern de lane du duo ennemi.

    Règles:
    - tous deux unknown => "unknown"
    - un pattern == "all_in" => "all_in"
    - sinon "poke" présent => "poke"
    - sinon tous ∈ {"scaling","sustain"} => "scaling"
    - sinon => "mixed"
    """
    pats = [v.get("lane_pattern", "unknown") for v in (enemy_adc_v, enemy_supp_v)]
    pats = [p for p in pats if p and p != "unknown"]
    if not pats:
        return "unknown"
    if "all_in" in pats:
        return "all_in"
    if "poke" in pats:
        return "poke"
    if all(p in ("scaling", "sustain") for p in pats):
        return "scaling"
    return "mixed"


_THREAT = {"high": 2, "med": 1, "low": 0, "unknown": 0}
_ROAM = {"high": 2, "med": 1, "low": 0, "unknown": 0}
_MITIG = {"ganking": -1, "skirmish": 0, "farming": 1, "unknown": 0}


def _gank_exposure(enemy_jgl_v, enemy_mid_v, self_jgl_v) -> str:
    """Dérive le niveau de gank exposure.

    Score:
    - enemy_jungle.gank_threat: high=+2, med=+1, low=0, unknown=0
    - enemy_mid.roam: high=+2, med=+1, low=0, unknown=0
    - atténuation self_jungle.playstyle: ganking=−1, skirmish=0, farming=+1, unknown=0

    Si tous 3 unknown => "unknown"
    Sinon: score ≤1 => "low", 2–3 => "med", ≥4 => "high"
    """
    jt = enemy_jgl_v.get("gank_threat", "unknown")
    mr = enemy_mid_v.get("roam", "unknown")
    sp = self_jgl_v.get("playstyle", "unknown")
    if jt == "unknown" and mr == "unknown" and sp == "unknown":
        return "unknown"
    score = max(0, _THREAT.get(jt, 0) + _ROAM.get(mr, 0) + _MITIG.get(sp, 0))
    if score <= 1:
        return "low"
    if score <= 3:
        return "med"
    return "high"


def derive_context(comp: dict, traits=None, ddragon=None) -> dict:
    """Dérive les deux axes coarse de contexte botlane.

    Args:
        comp: dict avec clés {self_adc, self_support, enemy_adc, enemy_support,
              self_jungle, enemy_jungle, enemy_mid}, valeurs = noms de champions ou None.
        traits: dict de traits curés (par défaut load_traits()).
        ddragon: dict Data Dragon (par défaut load_ddragon()).

    Returns:
        {"lane_pattern": <bucket>, "gank_exposure": <bucket>}
    """
    if traits is None:
        traits = load_traits()
    if ddragon is None:
        ddragon = load_ddragon()

    def vec(k):
        return _vec(comp.get(k), traits, ddragon)

    return {
        "lane_pattern": _lane_pattern(vec("enemy_adc"), vec("enemy_support")),
        "gank_exposure": _gank_exposure(vec("enemy_jungle"), vec("enemy_mid"), vec("self_jungle")),
    }
