#!/usr/bin/env python3
"""
phase1_pull — spike de faisabilité (détail d'UNE game).

Affiche le déplacement minute par minute + le contexte des morts d'une game,
pour inspection visuelle. Le pipeline de production est aggregate_games.py.

Usage :
    python3 phase1_pull.py "Spadzze#euw" euw1
    python3 phase1_pull.py "Spadzze#euw" euw1 -n 5   # parmi les 5 dernières
"""
from __future__ import annotations

import json
import sys

import riotlib as rl


def main() -> int:
    env = rl.load_env()
    api_key = env.get("RIOT_API_ID")
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    riot_id = pos[0] if pos else env.get("RIOT_ID")
    platform = (pos[1] if len(pos) > 1 else env.get("RIOT_REGION", "")).lower()
    n = int(sys.argv[sys.argv.index("-n") + 1]) if "-n" in sys.argv else 1

    if not (api_key and riot_id and "#" in riot_id):
        print("✗ RIOT_API_ID + Riot ID requis (.env ou args).", file=sys.stderr)
        return 1
    regional = rl.PLATFORM_TO_REGIONAL.get(platform)
    if not regional:
        print(f"✗ Région inconnue: {platform!r}", file=sys.stderr)
        return 1

    game_name, tag_line = riot_id.split("#", 1)
    client = rl.RiotClient(api_key, regional, platform)

    print(f"→ Résolution de {riot_id} ({platform} / {regional})…")
    puuid = client.puuid_from_riot_id(game_name, tag_line)
    ids = client.match_ids(puuid, count=n, queue=rl.QUEUE_SOLO)
    if not ids:
        print("✗ Aucune game.", file=sys.stderr)
        return 1
    match_id = ids[0]
    print(f"→ Game {match_id} (détail + timeline)…")
    got = rl.get_match_timeline(client, match_id)
    if not got:
        print("✗ Timeline indispo.", file=sys.stderr)
        return 1
    match, timeline = got

    meta, info = match["metadata"], match["info"]
    pidx = meta["participants"].index(puuid)
    pid = pidx + 1
    me = info["participants"][pidx]

    print("\n" + "=" * 56)
    print(f"  {me['championName']} ({me.get('teamPosition')}) — "
          f"{'VICTOIRE' if me['win'] else 'DÉFAITE'} en {round(info['gameDuration']/60,1)} min")
    print(f"  KDA {me['kills']}/{me['deaths']}/{me['assists']}  |  "
          f"patch {rl.patch_of(info.get('gameVersion',''))}")
    print("=" * 56)

    print("\n  Déplacements (zone approx. par minute) :")
    for frame in timeline["info"]["frames"]:
        pf = frame["participantFrames"].get(str(pid))
        if pf and "position" in pf:
            x, y = pf["position"]["x"], pf["position"]["y"]
            minute = round(frame["timestamp"] / 60000)
            print(f"    min {minute:>2}  {rl.approx_zone(x, y):<12}  "
                  f"gold={pf.get('totalGold')}  lvl={pf.get('level')}")

    g = rl.extract_game(match, timeline, puuid)
    print(f"\n  Morts ({len(g['deaths'])}) :")
    for d in g["deaths"]:
        print(f"    min {d['minute']:>2}  zone={d['zone']:<12} "
              f"tué par {d['killer_champ']} ({d['killer_role']})")
    print("\n✓ Spike : positionnement reconstruit sans aucune vision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
