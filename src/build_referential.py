#!/usr/bin/env python3
"""
build_referential — collecte les jeux de données de référence (benchmarks) par rang.

Pour chaque rang : échantillonne des joueurs (league-v4 / league-exp-v4), tire leurs
games ranked solo récentes (filtrées Faille + patch courant), écrit la couche silver
(games.jsonl + sources.json) puis la couche gold (agrégats par scope all/adc/zeri).

Usage :
    python3 build_referential.py                              # 4 rangs, 25j × 20g
    python3 build_referential.py --rank challenger --players 3 --games 10   # test rapide
    python3 build_referential.py --rank master,diamond --patch 16.13
"""
from __future__ import annotations

import json
import sys
import time

import riotlib as rl

APEX = {"challenger", "grandmaster", "master"}
ALL_RANKS = ["challenger", "grandmaster", "master", "diamond"]
SCOPES = ["all", "adc", "zeri"]


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def get_player_puuids(client: rl.RiotClient, rank: str, n: int) -> list[str]:
    if rank in APEX:
        entries = client.apex_league(rank)
        entries.sort(key=lambda e: e.get("leaguePoints", 0), reverse=True)
        puuids = [e["puuid"] for e in entries if e.get("puuid")]
    elif rank == "diamond":
        puuids = []
        for div in ("I", "II", "III", "IV"):
            page = 1
            while len(puuids) < n and page <= 5:
                entries = client.league_exp_entries("DIAMOND", div, page=page)
                if not entries:
                    break
                puuids += [e["puuid"] for e in entries if e.get("puuid")]
                page += 1
            if len(puuids) >= n:
                break
    else:
        raise ValueError(f"rang inconnu: {rank}")
    return puuids[:n]


def detect_patch(client: rl.RiotClient, puuids: list[str]) -> str | None:
    """Patch courant = patch MAJORITAIRE sur un échantillon de games récentes.

    On ne prend pas la 1re game trouvée (un joueur inactif imposerait un vieux patch) :
    on agrège plusieurs games et on retourne le mode.
    """
    import collections
    seen = collections.Counter()
    for puuid in puuids[:5]:
        for mid in client.match_ids(puuid, count=5, queue=rl.QUEUE_SOLO):
            got = rl.get_match_timeline(client, mid)
            if got:
                seen[rl.patch_of(got[0]["info"].get("gameVersion", ""))] += 1
    return seen.most_common(1)[0][0] if seen else None


def collect_rank(client: rl.RiotClient, rank: str, n_players: int,
                 n_games: int, patch: str) -> list[dict]:
    puuids = get_player_puuids(client, rank, n_players)
    print(f"\n=== {rank.upper()} : {len(puuids)} joueurs samplés ===", file=sys.stderr)
    pool: list[dict] = []
    seen: set[str] = set()
    for i, puuid in enumerate(puuids, 1):
        kept = 0
        for mid in client.match_ids(puuid, count=30, queue=rl.QUEUE_SOLO):
            if mid in seen:
                continue
            got = rl.get_match_timeline(client, mid)
            if not got:
                continue
            g = rl.extract_game(got[0], got[1], puuid, rank=rank)
            if not g or g["patch"] != patch:
                continue
            seen.add(mid)
            pool.append(g)
            kept += 1
            if kept >= n_games:
                break
        print(f"  [{i}/{len(puuids)}] +{kept} games (pool={len(pool)})", file=sys.stderr)
    return pool


def main() -> int:
    env = rl.load_env()
    api_key = env.get("RIOT_API_ID")
    platform = (arg("--region") or env.get("RIOT_REGION", "")).lower()
    if not api_key or not platform:
        print("✗ RIOT_API_ID + RIOT_REGION requis.", file=sys.stderr)
        return 1
    regional = rl.PLATFORM_TO_REGIONAL.get(platform)
    if not regional:
        print(f"✗ Région inconnue: {platform!r}", file=sys.stderr)
        return 1

    n_players = int(arg("--players", 25))
    n_games = int(arg("--games", 20))
    ranks = (arg("--rank") or ",".join(ALL_RANKS)).split(",")
    patch_override = arg("--patch")

    client = rl.RiotClient(api_key, regional, platform)

    patch = patch_override
    if not patch:
        print("→ Détection du patch courant…", file=sys.stderr)
        probe = get_player_puuids(client, ranks[0], 5)
        patch = detect_patch(client, probe)
        if not patch:
            print("✗ Impossible de détecter le patch.", file=sys.stderr)
            return 1
    print(f"→ Patch cible : {patch}", file=sys.stderr)

    for rank in ranks:
        pool = collect_rank(client, rank, n_players, n_games, patch)
        if not pool:
            print(f"  ⚠ {rank}: aucun game collecté.", file=sys.stderr)
            continue
        # silver
        silver_base = rl.SILVER_DIR / "referentiel" / rank
        rl.write_jsonl(silver_base / "games.jsonl", pool)
        (silver_base / "sources.json").write_text(json.dumps({
            "rank": rank, "patch": patch,
            "collected_at": time.strftime("%Y-%m-%d %H:%M"),
            "n_players": n_players, "n_games": len(pool),
        }, indent=2))
        # gold
        gold_base = rl.GOLD_DIR / "referentiel" / rank
        rl.write_gold(gold_base, pool, SCOPES, rank=rank, patch=patch)

        # récap console
        for scope in SCOPES:
            agg = rl.aggregate(pool, scope, rank=rank, patch=patch)
            print(f"  {rank}/{scope:<4} : {agg['n_games']:>3} games, "
                  f"{agg['overall']['deaths_per_game']} morts/game, WR {agg['winrate']:.0%}")

    print("\n✓ Référentiels écrits (silver/ + gold/).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
