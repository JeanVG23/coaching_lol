#!/usr/bin/env python3
"""
collection — fetch en masse du LP courant des joueurs apex (Master/GM/Challenger).

`apex_league(tier)` (riotlib) retourne la liste COMPLÈTE d'un tier apex (puuid +
leaguePoints pour chaque joueur classé) en 1 seul appel — récupérer le LP de tous
les joueurs apex ne coûte que 3 appels API. À relancer juste avant chaque
entraînement du modèle LP (le label LP dérive avec le temps : drift borné par la
fraîcheur du dataset, limite connue actée en spec —
docs/superpowers/specs/2026-07-07-lp-production-design.md). Promu du POC
poc/script/fetch_apex_lp.py (qui reste intact, référence historique).

Sortie : data/04_dataset/apex_lp.json =
  {"fetched_at": ISO-8601 UTC, "players": {puuid: {"tier": str, "leaguePoints": int}}}
Usage : poetry run python3 src/collection/fetch_apex_lp.py --region euw1
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import riotlib as rl

OUTPUT = rl.DATA / "04_dataset" / "apex_lp.json"
TIERS = ("challenger", "grandmaster", "master")
# Au POC, apex_league("master") a renvoyé pile 10 000 entrées — possible cap de
# l'API (non confirmé). On loggue un warning si ça se reproduit, sans bloquer :
# ~89 % des qualifiés master matchaient quand même (churn normal de tier).
SUSPECT_MASTER_COUNT = 10_000


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


def build_payload(entries_by_tier: dict[str, list[dict]], fetched_at: str) -> dict:
    """Enveloppe du fichier de sortie : fetched_at trace la fraîcheur du label LP
    (reporté jusque dans player_lp_metrics.json)."""
    return {"fetched_at": fetched_at, "players": build_lp_lookup(entries_by_tier)}


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
    if len(entries_by_tier.get("master", [])) == SUSPECT_MASTER_COUNT:
        print(f"  ⚠ master = {SUSPECT_MASTER_COUNT} pile — possible cap de l'API, "
              "une partie des masters peut manquer du lookup.", file=sys.stderr)

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = build_payload(entries_by_tier, fetched_at)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(f"\n✓ {len(payload['players'])} joueurs (LP courant, fetched_at={fetched_at}) "
          f"écrits dans {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
