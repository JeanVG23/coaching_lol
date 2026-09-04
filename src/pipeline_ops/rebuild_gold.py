#!/usr/bin/env python3
"""
rebuild_gold — régénère la couche gold depuis le silver, sans aucun appel API.

Le gold est entièrement dérivé du silver : à chaque évolution de la logique
d'agrégation (ex. ajout des facettes win/loss), on régénère tout en quelques ms.

Usage :
    python3 rebuild_gold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import riotlib as rl

SCOPES = ["all", "adc", "zeri", "smolder", "jinx", "caitlyn", "ezreal", "aphelios", "kaisa"]


def patch_of_pool(games: list[dict]) -> str:
    return games[0].get("patch", "?") if games else "?"


def main() -> int:
    n = 0
    # Les deux couches (referentiel/<rank>, personal/<player>) ne diffèrent que par le
    # nom du label passé au gold — une seule boucle sur rl.silver_roots().
    label_key = {rl.KIND_REF: "rank", rl.KIND_PERSONAL: "player"}
    for kind, root in rl.silver_roots():
        for d in sorted(root.iterdir()):
            games = rl.read_jsonl(d / "games.jsonl")
            if not games:
                continue
            rl.write_gold(rl.gold_base(kind, d.name), games, SCOPES,
                          patch=patch_of_pool(games), **{label_key[kind]: d.name})
            print(f"  gold ◄ {kind}/{d.name}  ({len(games)} games)")
            n += 1

    if not n:
        print("✗ Aucun silver trouvé. Lance d'abord aggregate_games / build_referential.",
              file=sys.stderr)
        return 1
    print(f"\n✓ {n} jeux gold régénérés.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
