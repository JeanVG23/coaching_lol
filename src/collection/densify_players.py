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
from ranks import COLLECT_ORDER as ALL_RANKS
from cli import arg

SCOPES = ["all", "adc", "zeri", "smolder", "jinx", "caitlyn", "ezreal", "aphelios", "kaisa"]


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


def collect_player_games(client, puuid: str, *, rank: str, patch: str,
                         max_history: int, start_time: int, gap: int | None,
                         seen_matches: set[str]) -> tuple[list[dict], int]:
    """Collecte les nouvelles performances d'un joueur, avec arrêt au gap ADC."""
    history = client.match_ids(
        puuid, count=max_history, queue=rl.QUEUE_SOLO, start_time=start_time)
    collected: list[dict] = []
    adc_games_added = 0

    for match_id in history:
        if match_id in seen_matches:
            continue
        seen_matches.add(match_id)

        got = rl.get_match_timeline(client, match_id)
        if not got or rl.patch_of(got[0]["info"].get("gameVersion", "")) != patch:
            continue

        games = rl.extract_all_games(got[0], got[1], rank=rank)
        if not games:
            continue
        collected.extend(games)
        if gap is not None and closes_gap(games, puuid):
            adc_games_added += 1
            if adc_games_added >= gap:
                break

    return collected, adc_games_added


def _known_match_ids() -> set[str]:
    seen: set[str] = set()
    for rank in ALL_RANKS:
        path = rl.silver_games(rl.KIND_REF, rank)
        if not path.exists():
            continue
        seen.update(row["match_id"] for row in rl.read_jsonl(path) if row.get("match_id"))
    return seen


def _patch_from_sources(sources_file: Path) -> str | None:
    if not sources_file.exists():
        return None
    try:
        return json.loads(sources_file.read_text()).get("patch")
    except (json.JSONDecodeError, OSError):
        return None


def _existing_player_count(sources_file: Path) -> int:
    if not sources_file.exists():
        return 0
    try:
        return json.loads(sources_file.read_text()).get("n_players", 0)
    except (json.JSONDecodeError, OSError):
        return 0


def _persist_pool(path: Path, sources_file: Path, pool: list[dict], *,
                  rank: str, patch: str) -> list[dict]:
    merged = rl.merge_jsonl(path, pool)
    sources_file.write_text(json.dumps({
        "rank": rank,
        "patch": patch,
        "collected_at": time.strftime("%Y-%m-%d %H:%M"),
        "n_players": _existing_player_count(sources_file),
        "n_games": len(merged),
    }, indent=2))
    pool.clear()
    return merged


def _players_for_rank(path: Path, rank: str, target_file: str | None) \
        -> tuple[list[str], dict[str, int]]:
    if target_file:
        targets = json.loads(Path(target_file).read_text())
        puuids, gaps = parse_target_list(targets, rank)
        print(f"\n=== {rank.upper()} : {len(puuids)} joueurs CIBLÉS à densifier "
              f"({len(gaps)} avec objectif chiffré = arrêt anticipé) ===", file=sys.stderr)
        return puuids, gaps

    existing = rl.read_jsonl(path)
    puuids = list({row["puuid"] for row in existing if row.get("puuid")})
    print(f"\n=== {rank.upper()} : {len(puuids)} joueurs existants à densifier ===",
          file=sys.stderr)
    return puuids, {}


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

    global_seen_matches = _known_match_ids()
    print(f"→ {len(global_seen_matches)} match_ids uniques déjà en base.", file=sys.stderr)
    target_file = arg("--target-list") if has_flag("--target-list") else None

    for rank in ranks:
        silver_base = rl.SILVER_DIR / rl.KIND_REF / rank
        path = silver_base / "games.jsonl"
        if not path.exists():
            print(f"  ⚠ {rank}: aucun games.jsonl trouvé, skip.", file=sys.stderr)
            continue
            
        puuids, gap_by_puuid = _players_for_rank(path, rank, target_file)
        if not puuids:
            continue
        
        sources_file = silver_base / "sources.json"
        patch = _patch_from_sources(sources_file)
        if not patch:
            print("  ⚠ Patch introuvable, skip.", file=sys.stderr)
            continue

        pool: list[dict] = []
        for i, puuid in enumerate(puuids, 1):
            gap = gap_by_puuid.get(puuid)
            games, adc_games_added = collect_player_games(
                client, puuid, rank=rank, patch=patch, max_history=max_history,
                start_time=start_time, gap=gap, seen_matches=global_seen_matches,
            )
            pool.extend(games)
            perf_kept = len(games)

            if perf_kept > 0:
                goal = f" (objectif ADC {adc_games_added}/{gap} atteint)" if gap and adc_games_added >= gap else ""
                print(f"  [{i}/{len(puuids)}] +{perf_kept} performances (pool_buffer={len(pool)}){goal}", file=sys.stderr)

            # Checkpoint
            if pool and i % checkpoint_every == 0:
                merged = _persist_pool(path, sources_file, pool, rank=rank, patch=patch)
                print(f"  · checkpoint @ {i}/{len(puuids)} → silver={len(merged)} games", file=sys.stderr)

        # Flush final
        if pool:
            merged = _persist_pool(path, sources_file, pool, rank=rank, patch=patch)
            # Mettre à jour la couche Gold
            gold_base = rl.gold_base(rl.KIND_REF, rank)
            rl.write_gold(gold_base, merged, SCOPES, rank=rank, patch=patch)

    print("\n✓ Densification terminée.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
