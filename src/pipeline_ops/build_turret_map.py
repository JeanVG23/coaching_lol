#!/usr/bin/env python3
"""One-shot : reconstitue `data/00_static/sr_turrets.json` depuis le raw. 0 API.

Chaque `BUILDING_KILL` de la timeline porte la `position` du batiment detruit :
un balayage de quelques dizaines de timelines suffit a retrouver la table exacte
des tourelles, sans ecrire une seule coordonnee de memoire.

Le script ECHOUE plutot que d'ecrire une table douteuse, dans deux cas :

- une cle porte plus d'une position (deux pour `NEXUS_TURRET`, qui va par paire) :
  cela signalerait un remaniement de la carte, pas un fichier a ecraser en silence ;
- l'echantillon ne couvre pas les 20 cles / 22 positions attendues : une cle
  manquante ferait passer une tourelle jamais observee pour debout en permanence,
  et `turrets.nearest_standing` mentirait sans jamais lever.

Le compte 20/22 ne vaut que pour les TOURELLES : `BUILDING_KILL` couvre aussi
les inhibiteurs (`buildingType: INHIBITOR_BUILDING`, sans `towerType`). Sans le
filtre sur `towerType`, le compte est 26/28 et l'assertion echouerait sur une
table pourtant correcte.

Usage :
  poetry run python3 src/pipeline_ops/build_turret_map.py [--limit 200] [--dry-run]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import champion_profiles as cprof
import riotlib as rl
from cli import flag, int_arg

# 3 tiers x 3 lanes x 2 equipes + 1 cle de nexus par equipe = 20 cles ;
# les deux tourelles de nexus partagent leur cle, d'ou 22 positions.
EXPECTED_KEYS = 20
EXPECTED_POSITIONS = 22
NEXUS = "NEXUS_TURRET"
MAX_POSITIONS = {NEXUS: 2}


def scan(read_raw, names) -> dict[tuple, set[tuple[int, int]]]:
    """{(team, lane, tier): {(x, y)}} sur les timelines nommees. Tourelles seules."""
    out: dict[tuple, set[tuple[int, int]]] = defaultdict(set)
    for name in names:
        timeline = read_raw(name)
        if not timeline:
            continue
        for ev in rl.iter_events(timeline):
            if ev.get("type") != "BUILDING_KILL":
                continue
            tower = ev.get("towerType")
            if not tower:                 # inhibiteur : pas une tourelle
                continue
            pos = ev.get("position") or {}
            if pos.get("x") is None or pos.get("y") is None:
                continue
            out[(ev.get("teamId"), ev.get("laneType"), tower)].add(
                (int(pos["x"]), int(pos["y"])))
    return dict(out)


def validate(table: dict[tuple, set]) -> dict[tuple, set]:
    """Rend la table telle quelle, ou sort en erreur. Jamais de correction muette."""
    for key, positions in sorted(table.items(), key=lambda kv: str(kv[0])):
        limit = MAX_POSITIONS.get(key[2], 1)
        if len(positions) > limit:
            sys.exit(f"✗ {key} : {len(positions)} positions pour un maximum "
                     f"de {limit}. Remaniement de carte probable, table non "
                     f"ecrite ({sorted(positions)})")
    keys, total = len(table), sum(len(v) for v in table.values())
    if keys != EXPECTED_KEYS or total != EXPECTED_POSITIONS:
        sys.exit(f"✗ echantillon incomplet : {keys} cles / {total} positions, "
                 f"attendu {EXPECTED_KEYS}/{EXPECTED_POSITIONS}. Augmente "
                 f"--limit (80 timelines suffisent en pratique).")
    return table


def rows(table: dict[tuple, set]) -> list[dict]:
    """Table -> lignes triees, serialisables et stables d'un run a l'autre."""
    out = []
    for (team, lane, tier), positions in table.items():
        for x, y in sorted(positions):
            out.append({"team": team, "lane": lane, "tier": tier, "x": x, "y": y})
    return sorted(out, key=lambda r: (r["team"], r["lane"], r["tier"], r["x"]))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    limit = int_arg("--limit", 200, argv)
    dry = flag("--dry-run", argv)
    raw_dir = rl.raw_dir()
    names = sorted(p.name.split(".json")[0]
                   for p in raw_dir.glob("*_timeline.json*"))[:limit]
    if not names:
        sys.exit(f"✗ aucune timeline dans {raw_dir}")
    table = validate(scan(rl._read_raw, names))
    out = rows(table)
    dest = cprof.STATIC_DIR / "sr_turrets.json"
    if dry:
        print(f"[dry-run] {len(names)} timelines -> {len(out)} positions "
              f"({len(table)} cles) vers {dest}")
        return 0
    rl.dump_json_atomic(dest, out, indent=2)
    print(f"✔ {len(out)} positions ({len(table)} cles) ecrites dans {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
