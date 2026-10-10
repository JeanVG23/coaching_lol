# web/backend/pipeline.py
"""Wrappers autour du pipeline existant pour le web, avec callback de progression.

Réutilise riotlib (pull/silver/gold) et coach (payload/generate/persist). Ne réécrit
rien : juste un point d'entrée callable avec on_progress.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable

import riotlib as rl
from aggregate_games import SCOPES

import coach
import payload
import settings


def _write_rank(player: str, entries: list[dict]) -> None:
    """Cache le rang solo/duo courant (rafraîchi à chaque fetch, pas à la volée) :
    sert de repère de fraîcheur des données pour l'utilisateur côté web."""
    solo = next((e for e in entries if e.get("queueType") == "RANKED_SOLO_5x5"), None)
    data = {
        "tier": solo.get("tier") if solo else None,
        "division": solo.get("rank") if solo else None,
        "league_points": solo.get("leaguePoints") if solo else None,
        "wins": solo.get("wins") if solo else None,
        "losses": solo.get("losses") if solo else None,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = rl.SILVER_DIR / "personal" / player / "rank.json"
    rl.dump_json_atomic(path, data, indent=2)


def fetch_games(account: dict, n: int = 20,
                on_progress: Callable[[str], None] | None = None, *,
                start_time: int | None = None,
                end_time: int | None = None,
                only_new: bool = False,
                allow_empty: bool = False) -> dict:
    """Pull Riot -> silver -> gold pour un compte. Bloquant : lancer via threadpool.

    ``start_time``/``end_time`` bornent la liste côté Match-V5. ``only_new``
    évite même la lecture du raw pour les parties déjà présentes en silver ;
    ``allow_empty`` fait d'une journée sans partie un succès normal.
    """
    key = settings.riot_api_key()
    if not key:
        raise RuntimeError("RIOT_API_ID manquant")
    platform = account["region"]
    regional = rl.PLATFORM_TO_REGIONAL[platform]
    client = rl.RiotClient(key, regional, platform)
    game_name, tag_line = account["riot_id"].split("#", 1)
    # Le slug de configuration est l'identité de stockage et d'URL. Le dériver
    # du pseudo cassait les Riot ID contenant des espaces (`Bobby Lupo` écrivait
    # dans `personal/bobby lupo`, puis le sync cherchait `personal/bobby-lupo`).
    # Il garantit aussi que deux comptes au nom proche restent indépendants.
    player = account["slug"]
    puuid = settings.resolve_puuid(account)
    if not puuid:
        raise RuntimeError("Riot ID introuvable")

    _write_rank(player, client.entries_by_puuid(puuid))

    match_kwargs: dict[str, int] = {}
    if start_time is not None:
        match_kwargs["start_time"] = start_time
    if end_time is not None:
        match_kwargs["end_time"] = end_time
    match_ids = client.match_ids(puuid, count=n, queue=rl.QUEUE_SOLO, **match_kwargs)

    silver_path = rl.SILVER_DIR / "personal" / player / "games.jsonl"
    if only_new:
        known_ids = {
            row.get("match_id") for row in rl.iter_jsonl(silver_path)
            if row.get("match_id")
        }
        match_ids = [mid for mid in match_ids if mid not in known_ids]

    games: list[dict] = []
    total = len(match_ids)
    for i, mid in enumerate(match_ids, 1):
        got = rl.get_match_timeline(client, mid)
        if not got:
            continue
        g = rl.extract_game(got[0], got[1], puuid)
        if g:
            games.append(g)
        if on_progress:
            on_progress(f"{i}/{total}")

    if not games:
        if allow_empty:
            return {
                "n_games": len(rl.read_jsonl(silver_path)),
                "n_new_games": 0,
                "player": player,
            }
        raise RuntimeError("Aucune game exploitable")
    merged = rl.merge_jsonl(silver_path, games)
    rl.write_gold(rl.GOLD_DIR / "personal" / player, merged, SCOPES, player=player)
    return {"n_games": len(merged), "n_new_games": len(games), "player": player}


def run_coach(player: str, scope: str = "adc", outcome: str = "loss",
              target: str = "challenger", model: str | None = None,
              on_progress: Callable[[str], None] | None = None) -> dict:
    """Payload -> LLM -> persist. Bloquant : lancer via threadpool."""
    if model is None:
        model = settings.ollama_model()
    if on_progress:
        on_progress("payload")
    pl = payload.build(player, scope, target, outcome)
    if on_progress:
        on_progress("llm")
    review = coach.generate_review(pl, model)
    ts = datetime.now().isoformat(timespec="seconds")
    coach.persist(player, model, pl, review, ts)
    return {"ts": ts}
