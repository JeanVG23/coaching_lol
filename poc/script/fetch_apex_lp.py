#!/usr/bin/env python3
"""
poc — fetch en masse du LP courant des joueurs apex (Master/GM/Challenger).

`apex_league(tier)` (riotlib) retourne la liste COMPLETE d'un tier apex (puuid +
leaguePoints pour chaque joueur classé) en 1 seul appel — donc récupérer le LP de
tous les joueurs apex ne coûte que 3 appels API, pas 1 par joueur.

Étape 1/2 du POC régression LP, cf.
docs/superpowers/specs/2026-07-07-lp-regression-poc-design.md.

Sortie : poc/output/apex_lp.json = {puuid: {"tier": str, "leaguePoints": int}}
Usage : poetry run python3 poc/script/fetch_apex_lp.py --region euw1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src" / "core"))
import riotlib as rl

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "output" / "apex_lp.json"
TIERS = ("challenger", "grandmaster", "master")


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def build_lp_lookup(entries_by_tier: dict[str, list[dict]]) -> dict[str, dict]:
    """entries_by_tier : {"challenger": [...], "grandmaster": [...], "master": [...]}
    (format brut apex_league, une entrée par joueur classé). Retourne
    {puuid: {"tier": str, "leaguePoints": int}}. Entrées sans puuid ignorées."""
    lookup: dict[str, dict] = {}
    for tier, entries in entries_by_tier.items():
        for e in entries:
            puuid = e.get("puuid")
            if not puuid:
                continue
            lookup[puuid] = {"tier": tier, "leaguePoints": int(e.get("leaguePoints", 0))}
    return lookup


def main() -> int:
    env = rl.load_env()
    api_key = env.get("RIOT_API_ID")
    platform = (arg("--region") or env.get("RIOT_REGION", "")).lower()
    if not api_key or not platform:
        print("✗ RIOT_API_ID + RIOT_REGION requis (--region euw1).", file=sys.stderr)
        return 1
    regional = rl.PLATFORM_TO_REGIONAL.get(platform)
    if not regional:
        print(f"✗ Région inconnue: {platform!r}", file=sys.stderr)
        return 1

    client = rl.RiotClient(api_key, regional, platform)

    entries_by_tier = {}
    for tier in TIERS:
        entries = client.apex_league(tier)
        entries_by_tier[tier] = entries
        print(f"  {tier}: {len(entries)} entrées", file=sys.stderr)

    lookup = build_lp_lookup(entries_by_tier)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(lookup, indent=2))
    print(f"\n✓ {len(lookup)} joueurs (LP courant) écrits dans {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
