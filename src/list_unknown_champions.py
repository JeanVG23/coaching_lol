# src/list_unknown_champions.py
"""Liste les champions présents dans le silver mais absents de champion_traits.json
(pour compléter la table au fil de l'eau). 0 appel API."""
from __future__ import annotations

import collections

import champion_profiles as cp
import riotlib as rl

ROLES = ("self_adc", "self_support", "enemy_adc", "enemy_support",
         "self_jungle", "enemy_jungle", "enemy_mid")


def main() -> int:
    traits = cp.load_traits()
    seen = collections.Counter()
    for root in (rl.SILVER_DIR / "referentiel", rl.SILVER_DIR / "personal"):
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            for g in rl.read_jsonl(d / "games.jsonl"):
                comp = g.get("comp") or {}
                for role in ROLES:
                    name = comp.get(role)
                    if name and name not in traits:
                        seen[name] += 1
    if not seen:
        print("✓ Tous les champions du silver sont dans la table (ou pas de comp).")
        return 0
    print(f"Champions manquants ({len(seen)}), triés par fréquence :")
    for name, n in seen.most_common():
        print(f"  {name:<18} {n}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
