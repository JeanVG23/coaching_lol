# src/pipeline_ops/audit_polyvalence.py
"""Audit polyvalence des champions : croise champion_traits.json avec
champion-rune-recommendations.json (Community Dragon), et signale les
incohérences potentielles avec les axes curés.

0 appel API Riot. 1 fetch Community Dragon (idempotent, caché localement).

Sources de données :
- champion_traits.json (data/00_static/) : axes curés (lane_pattern, power_curve,
  playstyle, gank_threat, roam).
- champion-rune-recommendations.json (Community Dragon) : positions officielles
  par champion (par mapId + isDefaultPosition).

Cas d'incohérence détectés :
- "polyvalence" : champion avec builds SR dans 2+ rôles distincts (UTILITY+BOTTOM,
  MIDDLE+TOP, ...) — le JSON curé peut l'avoir classifié dans un seul rôle.
- "axe_orphelin" : champion avec axe curé (gank_threat, roam) qui n'a PAS de
  build officiel dans le rôle attendu (jgl curé "ganking" sans build JUNGLE).
- "alias_introuvable" : champion du traits absent de CDragon (release trop
  récente ou alias qui a changé).

Usage :
    python3 src/pipeline_ops/audit_polyvalence.py
    python3 src/pipeline_ops/audit_polyvalence.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import champion_profiles as cp
import riotlib as rl

# Même convention que champion_profiles : STATIC_DIR = data/00_static
STATIC_DIR = rl.ROOT / "data" / "00_static"

CDRAGON_RECS_URL = (
    "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/"
    "global/default/v1/champion-rune-recommendations.json"
)
CDRAGON_SUMMARY_URL = (
    "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/"
    "global/default/v1/champion-summary.json"
)
CDRAGON_DIR = STATIC_DIR / "cdragon"
CDRAGON_RECS_CACHE = CDRAGON_DIR / "champion-rune-recommendations.json"
CDRAGON_SUMMARY_CACHE = CDRAGON_DIR / "champion-summary.json"

# Positions de jeu de LoL (CDragon mappe sur ces libellés pour mapId=11).
# On exclut NONE (réservé ARAM, pas une vraie lane).
_SR_POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")

# Axes et leurs positions "porteuses". Pour chaque axe, on liste les positions
# où l'axe a du sens. Si un champion a l'axe curé mais aucune de ses positions
# n'est porteuse de l'axe, c'est suspect.
#
# - playstyle / gank_threat : n'a de sens qu'en JUNGLE. Un champion marqué
#   "ganking" qui n'a pas de build JUNGLE officiel = label faux.
# - roam : MIDDLE ou UTILITY. Un support qui roam (Bard, Pyke, Rakan, Karma)
#   a légitimement roam=high.
# - lane_pattern : BOTTOM ou UTILITY. ADC et supports de botlane.
# - power_curve : BOTTOM, MIDDLE ou UTILITY. Carry scalings (Kassadin, Veigar,
#   Senna) sont légitimes dans plusieurs rôles.
_AXE_PORTEUSES = {
    "playstyle": ("JUNGLE",),
    "gank_threat": ("JUNGLE",),
    "roam": ("MIDDLE", "UTILITY"),
    "lane_pattern": ("BOTTOM", "UTILITY"),
    "power_curve": ("BOTTOM", "MIDDLE", "UTILITY"),
}


# ---------- CDragon fetch + parse ----------

def _fetch(url: str, cache: Path) -> dict | list:
    """Fetch (et cache) un JSON CDragon. Idempotent."""
    if cache.exists():
        return json.loads(cache.read_text())
    CDRAGON_DIR.mkdir(parents=True, exist_ok=True)
    import requests
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    cache.write_text(resp.text)
    return json.loads(resp.text)


def fetch_cdragon_recs() -> list:
    return _fetch(CDRAGON_RECS_URL, CDRAGON_RECS_CACHE)


def fetch_cdragon_summary() -> list:
    return _fetch(CDRAGON_SUMMARY_URL, CDRAGON_SUMMARY_CACHE)


def _id_to_alias(summary: list) -> dict[int, str]:
    """Construit {championId: alias} depuis champion-summary.json. Skip les -1."""
    out: dict[int, str] = {}
    for entry in summary:
        cid = entry.get("id")
        alias = entry.get("alias")
        if cid is None or cid == -1 or not alias or alias == "None":
            continue
        out[cid] = alias
    return out


def parse_positions(recs: list, id_to_alias: dict[int, str],
                    map_id: int = 11) -> dict[str, set[str]]:
    """Renvoie {alias_champion: {positions_officielles_sur_la_map}}.

    Le fichier est une liste d'entrées indexées par `championId`. On filtre sur
    mapId, on déduplique par champion. `isDefaultPosition` n'est pas utilisé
    pour l'audit : on veut la liste brute des positions officiellement
    recommandées (la polyvalence est un signal même si elle n'est pas "default").

    Les clés du dict retourné sont en **lowercase** (insensible à la casse :
    CDragon utilise `FiddleSticks`, ton JSON peut avoir `Fiddlesticks`).
    """
    out: dict[str, set[str]] = defaultdict(set)
    for entry in recs:
        if not isinstance(entry, dict):
            continue
        cid = entry.get("championId")
        alias = id_to_alias.get(cid or -1)
        if not alias:
            continue
        for r in entry.get("runeRecommendations", []):
            if r.get("mapId") != map_id:
                continue
            pos = r.get("position")
            if pos in _SR_POSITIONS:
                out[alias.lower()].add(pos)
    return out


# ---------- Audit ----------

def _coherence_check(alias: str, positions: set[str],
                     traits: dict) -> list[tuple[str, str]]:
    """Alertes de CE champion, en (type, détail).

    Le type est explicite : `audit` les classait en testant
    `msg.startswith("polyvalence")` sur un texte destiné à l'utilisateur —
    reformuler un message reclassait silencieusement toutes les alertes.
    """
    alerts: list[tuple[str, str]] = []
    trait = traits.get(alias, {})

    # 1) Polyvalence : 2+ positions distinctes (hors axes vides, sans filtre
    # "default"). On l'affiche, c'est au curateur de juger si c'est OK.
    if len(positions) >= 2:
        alerts.append(("polyvalence", f"polyvalence: positions SR = {sorted(positions)}"))

    # 2) Cohérence axe/rôle : pour chaque axe curé non-unknown, on vérifie qu'il
    # existe au moins UNE position SR où cet axe est censé s'appliquer.
    axes_curés = {k: v for k, v in trait.items()
                  if k in _AXE_PORTEUSES and v not in (None, "unknown", "")}

    for axe, val in axes_curés.items():
        porteuses = _AXE_PORTEUSES.get(axe, ())
        positions_qui_porte_l_axe = [p for p in positions if p in porteuses]
        # Si l'axe curé n'a aucune position où il a du sens, on signale.
        # Ex. lane_pattern sans BOTTOM/UTILITY, roam sans MIDDLE,
        # gank_threat/playstyle sans JUNGLE. Le curateur tranchera.
        if not positions_qui_porte_l_axe:
            alerts.append(("axe_orphelin",
                           f"axe '{axe}={val}' sans position SR cohérente "
                           f"(positions SR = {sorted(positions) or '∅'})"))

    return alerts


def audit(traits: dict, positions_by_alias: dict[str, set[str]]) -> dict:
    """Croise traits et positions, renvoie un rapport structuré."""
    polyvalence: list[dict] = []
    axe_orphelin: list[dict] = []
    alias_introuvable: list[str] = []

    for alias in sorted(traits.keys()):
        alias_lc = alias.lower()
        if alias_lc not in positions_by_alias:
            alias_introuvable.append(alias)
            continue
        positions = positions_by_alias[alias_lc]
        buckets = {"polyvalence": polyvalence, "axe_orphelin": axe_orphelin}
        for kind, msg in _coherence_check(alias, positions, traits):
            buckets[kind].append({
                "champion": alias,
                "positions": sorted(positions),
                "detail": msg,
            })

    return {
        "summary": {
            "n_traits": len(traits),
            "n_in_cdragon": len(traits) - len(alias_introuvable),
            "n_introuvables": len(alias_introuvable),
            "n_polyvalents": len(polyvalence),
            "n_axe_orphelin": len(axe_orphelin),
        },
        "polyvalence": polyvalence,
        "axe_orphelin": axe_orphelin,
        "alias_introuvable": alias_introuvable,
    }


# ---------- Reporting ----------

def render_text(report: dict) -> str:
    s = report["summary"]
    lines = [
        f"=== Audit polyvalence ({s['n_traits']} champions curés) ===",
        f"présents dans CDragon : {s['n_in_cdragon']}",
        f"introuvables          : {s['n_introuvables']}",
        f"polyvalents (≥2 pos.) : {s['n_polyvalents']}",
        f"axes orphelins        : {s['n_axe_orphelin']}",
        "",
    ]

    if report["alias_introuvable"]:
        lines.append("--- Champions du traits absents de CDragon ---")
        for n in report["alias_introuvable"]:
            lines.append(f"  • {n}")
        lines.append("")

    if report["polyvalence"]:
        lines.append("--- Polyvalence (≥2 positions SR officielles) ---")
        for r in report["polyvalence"]:
            lines.append(f"  • {r['champion']:<18} {r['detail']}")
        lines.append("")

    if report["axe_orphelin"]:
        lines.append("--- Axes curés sans position SR cohérente ---")
        for r in report["axe_orphelin"]:
            lines.append(f"  • {r['champion']:<18} {r['detail']}")
        lines.append("")

    if not (report["polyvalence"] or report["axe_orphelin"] or report["alias_introuvable"]):
        lines.append("✓ Aucun signal. champion_traits.json cohérent avec CDragon.")

    return "\n".join(lines)


# ---------- Main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description="Audit polyvalence champions (CDragon).")
    ap.add_argument("--json", metavar="PATH", help="écrit le rapport JSON à PATH")
    args = ap.parse_args()

    print("Fetch champion-rune-recommendations.json (Community Dragon)…")
    recs = fetch_cdragon_recs()
    summary = fetch_cdragon_summary()
    id_to_alias = _id_to_alias(summary)
    print(f"  → {len(id_to_alias)} champions connus de CDragon")
    positions_by_alias = parse_positions(recs, id_to_alias)
    print(f"  → {len(positions_by_alias)} champions avec recos SR parsées")

    traits = cp.load_traits()
    print(f"  → {len(traits)} champions dans champion_traits.json")

    report = audit(traits, positions_by_alias)
    print()
    print(render_text(report))

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nRapport JSON écrit : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
