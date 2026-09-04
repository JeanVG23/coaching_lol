"""Identité des champions : Data Dragon (statique) + table de traits curée.

Module isolé, sans appel réseau au runtime (cache disque). Donne un vecteur
d'identité par champion et dérive les axes de contexte de botlane.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
# Même racine surchargeable que riotlib.DATA : sans elle, une pile démo déportée
# irait quand même lire les catalogues Data Dragon des données réelles.
_DATA = Path(os.environ.get("COACHING_DATA_DIR") or (ROOT / "data")).resolve()
STATIC_DIR = _DATA / "00_static"
DDRAGON_VERSION = "16.13.1"  # figée ; refresh = action manuelle (fetch_ddragon)
TRAITS_PATH = STATIC_DIR / "champion_traits.json"

AXES_CURATED = ("power_curve", "lane_pattern", "playstyle", "gank_threat", "roam")
RANGED_MIN = 500  # attackrange >= 500 => ranged


def fetch_ddragon(version: str | None = None) -> Path:
    version = version or DDRAGON_VERSION
    dest = STATIC_DIR / "ddragon" / version / "championFull.json"
    if dest.exists():
        return dest  # idempotent : cache déjà chaud (refresh = supprimer le fichier)
    url = (f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/championFull.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_text(resp.text)
    _invalidate_catalogs()  # invalide un éventuel {} mis en cache avant le fetch
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


def fetch_ddragon_items(version: str | None = None) -> Path:
    version = version or DDRAGON_VERSION
    dest = STATIC_DIR / "ddragon" / version / "item.json"
    if dest.exists():
        return dest  # idempotent : cache déjà chaud (refresh = supprimer le fichier)
    url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/item.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_text(resp.text)
    load_items.cache_clear()
    return dest


def _parse_items(raw: dict) -> dict:
    return {int(iid): {"name": it.get("name", f"item_{iid}"),
                       "cost": it.get("gold", {}).get("total")}
            for iid, it in raw.items()}


@lru_cache(maxsize=1)
def load_items() -> dict:
    path = STATIC_DIR / "ddragon" / DDRAGON_VERSION / "item.json"
    if not path.exists():
        return {}
    return _parse_items(json.loads(path.read_text())["data"])


@lru_cache(maxsize=1)
def load_traits() -> dict:
    if not TRAITS_PATH.exists():
        return {}
    data = json.loads(TRAITS_PATH.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _ci_get(name: str, d: dict, index: dict | None = None) -> dict:
    """d[name] (exact d'abord), repli insensible à la casse, sinon {}.

    `index` = table {nom minuscule: entrée} pré-construite : évite de rebalayer les
    ~170 entrées du catalogue à chaque miss (le scan linéaire était sur le chemin
    chaud de `_by_lane_context`, ~7 vecteurs par game).
    """
    if name in d:
        return d[name]
    lower = name.lower()
    if index is not None:
        return index.get(lower, {})
    for k, v in d.items():
        if k.lower() == lower:
            return v
    return {}


@lru_cache(maxsize=2)
def _default_index(which: str) -> dict:
    """Index insensible à la casse des catalogues par défaut, construit une fois."""
    catalog = load_traits() if which == "traits" else load_ddragon()
    return {k.lower(): v for k, v in catalog.items()}


def _build_vector(name: str, traits: dict, ddragon: dict,
                  tr_index=None, dd_index=None) -> dict:
    dd = _ci_get(name, ddragon, dd_index)
    tr = _ci_get(name, traits, tr_index)
    rng = dd.get("attackrange")
    range_class = "unknown" if rng is None else ("ranged" if rng >= RANGED_MIN else "melee")
    v = {"name": name, "range_class": range_class, "tags": list(dd.get("tags", []))}
    for axis in AXES_CURATED:
        v[axis] = tr.get(axis, "unknown")
    return v


@lru_cache(maxsize=None)
def _cached_vector(name: str) -> dict:
    return _build_vector(name, load_traits(), load_ddragon(),
                         _default_index("traits"), _default_index("ddragon"))


def _invalidate_catalogs() -> None:
    """Vide d'un coup tous les caches dérivés des catalogues (traits + Data Dragon).

    Un seul point d'invalidation : oublier l'un des caches dérivés après un fetch
    laissait servir un catalogue vide mis en cache avant le téléchargement.
    """
    load_ddragon.cache_clear()
    load_traits.cache_clear()
    _default_index.cache_clear()
    _cached_vector.cache_clear()


def champion_vector(name: str, traits: dict | None = None,
                    ddragon: dict | None = None) -> dict:
    """Vecteur d'identité d'un champion (Data Dragon + table curée).

    Mémoïsé quand les catalogues sont ceux par défaut (le cas de tout le pipeline) :
    ~170 champions distincts étaient reconstruits des dizaines de milliers de fois sur
    un `rebuild_gold`. Copie retournée pour que le cache reste immuable côté appelant.
    """
    on_defaults = ((traits is None or traits is load_traits())
                   and (ddragon is None or ddragon is load_ddragon()))
    if on_defaults:
        return dict(_cached_vector(name))
    return _build_vector(name, traits if traits is not None else load_traits(),
                         ddragon if ddragon is not None else load_ddragon())


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
