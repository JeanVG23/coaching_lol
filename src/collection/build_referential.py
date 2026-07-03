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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import riotlib as rl

APEX = {"challenger", "grandmaster", "master"}
ALL_RANKS = ["challenger", "grandmaster", "master", "diamond"]
SCOPES = ["all", "adc", "zeri", "smolder", "jinx", "caitlyn", "ezreal", "aphelios", "kaisa"]


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def has_flag(flag: str) -> bool:
    return flag in sys.argv


def known_puuids(rank: str) -> set[str]:
    """Puuids déjà présents dans le silver du rang (évite le re-échantillonnage
    des mêmes top joueurs à chaque run — densification réelle par deeper paging)."""
    path = rl.SILVER_DIR / "referentiel" / rank / "games.jsonl"
    seen_path = rl.SILVER_DIR / "referentiel" / rank / "seen_puuids.txt"
    known = set()
    
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    known.add(json.loads(line).get("puuid"))
                except json.JSONDecodeError:
                    continue
                    
    if seen_path.exists():
        with open(seen_path) as f:
            for line in f:
                known.add(line.strip())
                
    known.discard(None)
    known.discard("")
    return known


def get_player_puuids(client: rl.RiotClient, rank: str, n: int,
                      skip: set[str] | None = None, max_pages: int = 5,
                      start_page: int = 1) -> list[str]:
    skip = skip or set()
    if rank in APEX:
        entries = client.apex_league(rank)
        entries.sort(key=lambda e: e.get("leaguePoints", 0), reverse=True)
        puuids = [e["puuid"] for e in entries
                  if e.get("puuid") and e["puuid"] not in skip]
        puuids = puuids[start_page * 100:]   # offset sur la liste triée par LP
    elif rank == "diamond":
        puuids = []
        for div in ("I", "II", "III", "IV"):
            page = start_page
            while len(puuids) < n and page <= max_pages:
                entries = client.league_exp_entries("DIAMOND", div, page=page)
                if not entries:
                    break
                puuids += [e["puuid"] for e in entries
                           if e.get("puuid") and e["puuid"] not in skip]
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


def _write_sources(silver_base, rank: str, patch: str, n_games: int) -> None:
    """Met à jour sources.json en préservant n_players (cumul des runs précédents).
    Utilisé par les checkpoints : on ne touche pas à n_players ici (incrément final
    fait par main), on rafraîchit juste n_games + collected_at pour refléter le
    checkpoint courant."""
    sf = silver_base / "sources.json"
    existing: dict = {}
    if sf.exists():
        try:
            existing = json.loads(sf.read_text())
        except json.JSONDecodeError:
            pass
    existing.update({"rank": rank, "patch": patch,
                     "collected_at": time.strftime("%Y-%m-%d %H:%M"),
                     "n_games": n_games})
    sf.write_text(json.dumps(existing, indent=2))


def collect_rank(client: rl.RiotClient, rank: str, n_players: int,
                 n_games: int, patch: str,
                 skip: set[str] | None = None, max_pages: int = 5,
                 days_back: int = 28, start_page: int = 1,
                 silver_base=None, checkpoint_every: int = 25,
                 target_role: str | None = None,
                 max_match_history: int = 50) -> list[dict]:
    puuids = get_player_puuids(client, rank, n_players, skip=skip,
                               max_pages=max_pages, start_page=start_page)
                               
    if silver_base is not None and puuids:
        seen_path = silver_base / "seen_puuids.txt"
        seen_path.parent.mkdir(parents=True, exist_ok=True)
        with open(seen_path, "a") as f:
            for p in puuids:
                f.write(p + "\n")
                
    print(f"\n=== {rank.upper()} : {len(puuids)} joueurs samplés "
          f"(skip-known={'on' if skip else 'off'}, max_pages={max_pages}, "
          f"start_page={start_page}) ===", file=sys.stderr)
    # Filtre startTime : ne demander que les matches des `days_back` derniers jours.
    # Le patch courant (~2 semaines) est inclus ; on évite de fetcher des timelines
    # de patches précédents juste pour les filtrer (joueur inactif = 1 appel, pas 30).
    start_time = int(time.time()) - days_back * 86400
    print(f"  filtre match_ids startTime = {days_back}j avant now", file=sys.stderr)
    pool: list[dict] = []
    seen: set[str] = set()
    for i, puuid in enumerate(puuids, 1):
        matches_kept = 0
        perf_kept = 0
        for mid in client.match_ids(puuid, count=max_match_history, queue=rl.QUEUE_SOLO,
                                    start_time=start_time):
            if mid in seen:
                continue
            seen.add(mid)
            got = rl.get_match_timeline(client, mid, target_puuid=puuid if target_role else None, target_role=target_role)
            if not got:
                continue

            if rl.patch_of(got[0]["info"].get("gameVersion", "")) != patch:
                continue
                
            games = rl.extract_all_games(got[0], got[1], rank=rank)
            if not games:
                continue
                
            pool.extend(games)
            perf_kept += len(games)
            matches_kept += 1
            if matches_kept >= n_games:
                break
        print(f"  [{i}/{len(puuids)}] +{perf_kept} performances (pool={len(pool)})", file=sys.stderr)

        # Checkpoint incrémental : on écrit le silver tous les `checkpoint_every`
        # joueurs. Si le process est tué en cours de route, on perd au plus
        # checkpoint_every-1 joueurs de travail au lieu de tout le batch.
        if silver_base is not None and i % checkpoint_every == 0:
            merged = rl.merge_jsonl(silver_base / "games.jsonl", pool)
            _write_sources(silver_base, rank, patch, len(merged))
            print(f"  · checkpoint @ {i}/{len(puuids)} → silver={len(merged)} games",
                  file=sys.stderr)
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
    skip_known = has_flag("--skip-known")
    max_pages = int(arg("--max-pages", 5))
    start_page = int(arg("--start-page", 1))
    target_role = arg("--target-role")
    max_match_history = int(arg("--max-match-history", 50))

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

    # Check for patch mismatch before collecting
    for rank in ranks:
        sources_file = rl.SILVER_DIR / "referentiel" / rank / "sources.json"
        if sources_file.exists():
            try:
                existing = json.loads(sources_file.read_text())
                existing_patch = existing.get("patch")
                if existing_patch and existing_patch != patch:
                    print(f"✗ Erreur : Le patch détecté ({patch}) est différent du patch actuel en base ({existing_patch}).", file=sys.stderr)
                    print("Veuillez d'abord archiver les données avec 'python src/archive_patch.py'.", file=sys.stderr)
                    return 1
            except json.JSONDecodeError:
                pass

    for rank in ranks:
        skip = known_puuids(rank) if skip_known else None
        if skip is not None:
            print(f"  [{rank}] skip-known: {len(skip)} puuids déjà en silver → "
                  f"échantillonnage plus profond", file=sys.stderr)
        silver_base = rl.SILVER_DIR / "referentiel" / rank
        pool = collect_rank(client, rank, n_players, n_games, patch,
                            skip=skip, max_pages=max_pages, start_page=start_page,
                            silver_base=silver_base, target_role=target_role,
                            max_match_history=max_match_history)
        if not pool:
            print(f"  ⚠ {rank}: aucun game collecté.", file=sys.stderr)
            continue
        # silver (final merge — les checkpoints ont déjà écrit une partie ; idempotent)
        merged_pool = rl.merge_jsonl(silver_base / "games.jsonl", pool)
        
        existing_players = 0
        if (silver_base / "sources.json").exists():
            try:
                existing_players = json.loads((silver_base / "sources.json").read_text()).get("n_players", 0)
            except json.JSONDecodeError:
                pass
                
        (silver_base / "sources.json").write_text(json.dumps({
            "rank": rank, "patch": patch,
            "collected_at": time.strftime("%Y-%m-%d %H:%M"),
            "n_players": existing_players + n_players, "n_games": len(merged_pool),
        }, indent=2))
        
        # gold
        gold_base = rl.GOLD_DIR / "referentiel" / rank
        rl.write_gold(gold_base, merged_pool, SCOPES, rank=rank, patch=patch)

        # récap console
        for scope in SCOPES:
            agg = rl.aggregate(merged_pool, scope, rank=rank, patch=patch)
            print(f"  {rank}/{scope:<4} : {agg['n_games']:>3} games, "
                  f"{agg['overall']['deaths_per_game']} morts/game, WR {agg['winrate']:.0%}")

    print("\n✓ Référentiels écrits (silver/ + gold/).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
