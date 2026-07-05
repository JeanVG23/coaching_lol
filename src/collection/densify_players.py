#!/usr/bin/env python3
"""
densify_players.py — reprend les joueurs existants du dataset silver
et collecte des parties supplémentaires (historique profond) pour
atteindre un nombre plus élevé de parties par joueur (ex: 20+).

Avec `--target-list` pointant vers une sortie de `densify_targets.py`
({puuid: {"rank": ..., "gap": ...}}), s'arrête PAR JOUEUR dès que `gap` games
ADC (BOTTOM, sur le patch courant) neuves ont été trouvées, au lieu d'épuiser
tout `--history` pour chacun. Rétro-compatible avec l'ancien format plat
{puuid: "rank"} (pas de gap -> pas d'arrêt anticipé, comportement historique).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import riotlib as rl

ALL_RANKS = ["challenger", "grandmaster", "master", "diamond"]
SCOPES = ["all", "adc", "zeri", "smolder", "jinx", "caitlyn", "ezreal", "aphelios", "kaisa"]


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def has_flag(flag: str) -> bool:
    return flag in sys.argv


def parse_target_list(targets: dict, rank: str) -> tuple[list[str], dict[str, int]]:
    """(puuids du rang, gap_by_puuid) depuis un target-list JSON. Supporte le format
    enrichi {"rank": ..., "gap": ...} (densify_targets.py) et l'ancien format plat
    {puuid: "rank"} (rétro-compat, pas de gap -> pas d'arrêt anticipé)."""
    puuids: list[str] = []
    gap_by_puuid: dict[str, int] = {}
    for p, v in targets.items():
        r = v["rank"] if isinstance(v, dict) else v
        if r != rank:
            continue
        puuids.append(p)
        if isinstance(v, dict) and "gap" in v:
            gap_by_puuid[p] = v["gap"]
    return puuids, gap_by_puuid


def closes_gap(games: list[dict], puuid: str) -> bool:
    """True si CE joueur est ADC (BOTTOM) dans les games extraites d'un match — seul
    rôle qui alimente adc_dataset.parquet / le seuil ML per-player."""
    return any(g["puuid"] == puuid and g["role"] == "BOTTOM" for g in games)


def main() -> int:
    env = rl.load_env()
    api_key = env.get("RIOT_API_ID")
    platform = (arg("--region") or env.get("RIOT_REGION", "euw1")).lower()
    if not api_key:
        print("✗ RIOT_API_ID requis.", file=sys.stderr)
        return 1
        
    regional = rl.PLATFORM_TO_REGIONAL.get(platform)
    if not regional:
        print(f"✗ Région inconnue: {platform!r}", file=sys.stderr)
        return 1

    client = rl.RiotClient(api_key, regional, platform)

    ranks = (arg("--rank") or ",".join(ALL_RANKS)).split(",")
    max_history = int(arg("--history", 50))
    days_back = int(arg("--days", 28))
    checkpoint_every = int(arg("--checkpoint", 10))

    start_time = int(time.time()) - days_back * 86400

    # 1. Charger tous les match_ids déjà connus pour éviter les appels d'API inutiles
    global_seen_matches = set()
    for r in ALL_RANKS:
        path = rl.SILVER_DIR / "referentiel" / r / "games.jsonl"
        if path.exists():
            for row in rl.read_jsonl(path):
                if row.get("match_id"):
                    global_seen_matches.add(row["match_id"])
                    
    print(f"→ {len(global_seen_matches)} match_ids uniques déjà en base.", file=sys.stderr)

    for rank in ranks:
        silver_base = rl.SILVER_DIR / "referentiel" / rank
        path = silver_base / "games.jsonl"
        if not path.exists():
            print(f"  ⚠ {rank}: aucun games.jsonl trouvé, skip.", file=sys.stderr)
            continue
            
        gap_by_puuid: dict[str, int] = {}
        if has_flag("--target-list"):
            target_file = arg("--target-list")
            targets = json.loads(Path(target_file).read_text())
            puuids, gap_by_puuid = parse_target_list(targets, rank)
            print(f"\n=== {rank.upper()} : {len(puuids)} joueurs CIBLÉS à densifier "
                  f"({len(gap_by_puuid)} avec objectif chiffré = arrêt anticipé) ===", file=sys.stderr)
            if not puuids:
                continue
        else:
            existing = rl.read_jsonl(path)
            # Extraire les puuids uniques associés à ce rang
            puuids = list(set(row["puuid"] for row in existing if row.get("puuid")))
            print(f"\n=== {rank.upper()} : {len(puuids)} joueurs existants à densifier ===", file=sys.stderr)
        
        # Trouver le patch courant de ce rang dans sources.json
        patch = None
        sources_file = silver_base / "sources.json"
        if sources_file.exists():
            try:
                patch = json.loads(sources_file.read_text()).get("patch")
            except Exception:
                pass
        
        if not patch:
            print("  ⚠ Patch introuvable, skip.", file=sys.stderr)
            continue

        pool: list[dict] = []
        for i, puuid in enumerate(puuids, 1):
            perf_kept = 0
            adc_games_added = 0
            gap = gap_by_puuid.get(puuid)  # None = pas d'objectif chiffré -> historique complet (comportement historique)
            # Récupérer l'historique complet
            history = client.match_ids(puuid, count=max_history, queue=rl.QUEUE_SOLO, start_time=start_time)

            for mid in history:
                if mid in global_seen_matches:
                    continue  # Déjà analysé par le passé

                global_seen_matches.add(mid)

                got = rl.get_match_timeline(client, mid)
                if not got:
                    continue

                # Vérifier que c'est bien sur le même patch
                if rl.patch_of(got[0]["info"].get("gameVersion", "")) != patch:
                    continue

                games = rl.extract_all_games(got[0], got[1], rank=rank)
                if games:
                    pool.extend(games)
                    perf_kept += len(games)

                    if gap is not None and closes_gap(games, puuid):
                        adc_games_added += 1
                        if adc_games_added >= gap:
                            break  # objectif atteint, joueur suivant plutôt que d'épuiser --history

            if perf_kept > 0:
                goal = f" (objectif ADC {adc_games_added}/{gap} atteint)" if gap and adc_games_added >= gap else ""
                print(f"  [{i}/{len(puuids)}] +{perf_kept} performances (pool_buffer={len(pool)}){goal}", file=sys.stderr)

            # Checkpoint
            if pool and i % checkpoint_every == 0:
                merged = rl.merge_jsonl(path, pool)
                
                existing_players = 0
                if sources_file.exists():
                    try:
                        existing_players = json.loads(sources_file.read_text()).get("n_players", 0)
                    except json.JSONDecodeError:
                        pass
                        
                sources_file.write_text(json.dumps({
                    "rank": rank, "patch": patch,
                    "collected_at": time.strftime("%Y-%m-%d %H:%M"),
                    "n_players": existing_players, "n_games": len(merged),
                }, indent=2))
                pool.clear()
                print(f"  · checkpoint @ {i}/{len(puuids)} → silver={len(merged)} games", file=sys.stderr)

        # Flush final
        if pool:
            merged = rl.merge_jsonl(path, pool)
            existing_players = 0
            if sources_file.exists():
                try:
                    existing_players = json.loads(sources_file.read_text()).get("n_players", 0)
                except json.JSONDecodeError:
                    pass
            sources_file.write_text(json.dumps({
                "rank": rank, "patch": patch,
                "collected_at": time.strftime("%Y-%m-%d %H:%M"),
                "n_players": existing_players, "n_games": len(merged),
            }, indent=2))
            
            # Mettre à jour la couche Gold
            gold_base = rl.GOLD_DIR / "referentiel" / rank
            rl.write_gold(gold_base, merged, SCOPES, rank=rank, patch=patch)

    print("\n✓ Densification terminée.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
