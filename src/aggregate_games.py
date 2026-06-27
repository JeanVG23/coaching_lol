#!/usr/bin/env python3
"""
aggregate_games — pipeline perso (médaillon raw -> silver -> gold).

Tire les N dernières games ranked d'un joueur, écrit la couche silver
(silver/personal/<player>/games.jsonl) et la couche gold (agrégats par scope
all/adc/zeri sous gold/personal/<player>/<scope>/), puis affiche un récap.

Usage :
    python3 aggregate_games.py "Spadzze#euw" euw1 -n 20
    python3 aggregate_games.py "Spadzze#euw" euw1 -n 20 --role BOTTOM
"""
from __future__ import annotations

import collections
import json
import sys

import riotlib as rl

SCOPES = ["all", "adc", "zeri"]


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> int:
    env = rl.load_env()
    api_key = env.get("RIOT_API_ID")
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    riot_id = pos[0] if pos else env.get("RIOT_ID")
    platform = (pos[1] if len(pos) > 1 else env.get("RIOT_REGION", "")).lower()
    n = int(arg("-n", 20))
    role_filter = (arg("--role") or "").upper() or None

    if not (api_key and riot_id and "#" in riot_id):
        print("✗ RIOT_API_ID + Riot ID requis.", file=sys.stderr)
        return 1
    regional = rl.PLATFORM_TO_REGIONAL.get(platform)
    if not regional:
        print(f"✗ Région inconnue: {platform!r}", file=sys.stderr)
        return 1

    game_name, tag_line = riot_id.split("#", 1)
    player = game_name.lower()
    client = rl.RiotClient(api_key, regional, platform)

    print(f"→ Résolution de {riot_id}…")
    puuid = client.puuid_from_riot_id(game_name, tag_line)
    if not puuid:
        print("✗ Riot ID introuvable.", file=sys.stderr)
        return 1

    print(f"→ {n} dernières games ranked…")
    games = []
    for mid in client.match_ids(puuid, count=n, queue=rl.QUEUE_SOLO):
        got = rl.get_match_timeline(client, mid)
        if not got:
            continue
        g = rl.extract_game(got[0], got[1], puuid)
        if g and (not role_filter or g["role"] == role_filter):
            games.append(g)

    if not games:
        print("✗ Aucune game Faille exploitable.", file=sys.stderr)
        return 1

    # silver
    rl.write_jsonl(rl.SILVER_DIR / "personal" / player / "games.jsonl", games)
    # gold
    gold_base = rl.GOLD_DIR / "personal" / player
    rl.write_gold(gold_base, games, SCOPES, player=player)

    # récap console (scope dominant = le rôle principal joué)
    roles = collections.Counter(g["role"] for g in games)
    champs = collections.Counter(g["champion"] for g in games)
    agg_all = rl.aggregate(games, "all", player=player)
    print("\n" + "=" * 60)
    print(f"  {player.upper()} — {agg_all['n_games']} games  |  "
          f"WR {agg_all['winrate']:.0%}  |  {agg_all['overall']['deaths_per_game']} morts/game")
    print(f"  Rôles : {dict(roles)}")
    print(f"  Champions : {dict(champs.most_common(5))}")
    print("=" * 60)
    for scope in SCOPES:
        a = rl.aggregate(games, scope, player=player)
        f = a["overall"]
        if a["n_games"]:
            top = sorted(f["by_zone_phase"].items(), key=lambda x: -x[1])[:3]
            top_s = ", ".join(f"{k} {v:.0%}" for k, v in top)
            print(f"  {scope:<4}: {a['n_games']:>2} games, {f['deaths_per_game']} m/g "
                  f"→ top morts: {top_s}")

    print(f"\n✓ silver + gold écrits sous data/.../personal/{player}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
