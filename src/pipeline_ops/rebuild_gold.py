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
    # référentiels : silver/referentiel/<rank>/games.jsonl
    ref_root = rl.SILVER_DIR / "referentiel"
    if ref_root.exists():
        for d in sorted(ref_root.iterdir()):
            games = rl.read_jsonl(d / "games.jsonl")
            if not games:
                continue
            rl.write_gold(rl.GOLD_DIR / "referentiel" / d.name, games, SCOPES,
                          rank=d.name, patch=patch_of_pool(games))
            print(f"  gold ◄ referentiel/{d.name}  ({len(games)} games)")
            n += 1
    # perso : silver/personal/<player>/games.jsonl
    perso_root = rl.SILVER_DIR / "personal"
    if perso_root.exists():
        for d in sorted(perso_root.iterdir()):
            games = rl.read_jsonl(d / "games.jsonl")
            if not games:
                continue
            rl.write_gold(rl.GOLD_DIR / "personal" / d.name, games, SCOPES,
                          player=d.name, patch=patch_of_pool(games))
            print(f"  gold ◄ personal/{d.name}  ({len(games)} games)")
            n += 1

    if not n:
        print("✗ Aucun silver trouvé. Lance d'abord aggregate_games / build_referential.",
              file=sys.stderr)
        return 1
    print(f"\n✓ {n} jeux gold régénérés.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
