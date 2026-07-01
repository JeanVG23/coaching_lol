# web/backend/pipeline.py
"""Wrappers autour du pipeline existant pour le web, avec callback de progression.

Réutilise riotlib (pull/silver/gold) et coach (payload/generate/persist). Ne réécrit
rien : juste un point d'entrée callable avec on_progress.
"""
from __future__ import annotations

import json
import os
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def fetch_games(account: dict, n: int = 20,
                on_progress: Callable[[str], None] | None = None) -> dict:
    """Pull Riot -> silver -> gold pour un compte. Bloquant : lancer via threadpool."""
    key = settings.riot_api_key()
    if not key:
        raise RuntimeError("RIOT_API_ID manquant")
    platform = account["region"]
    regional = rl.PLATFORM_TO_REGIONAL[platform]
    client = rl.RiotClient(key, regional, platform)
    game_name, tag_line = account["riot_id"].split("#", 1)
    player = game_name.lower()
    puuid = settings.resolve_puuid(account)
    if not puuid:
        raise RuntimeError("Riot ID introuvable")

    _write_rank(player, client.entries_by_puuid(puuid))

    games: list[dict] = []
    for i, mid in enumerate(client.match_ids(puuid, count=n, queue=rl.QUEUE_SOLO), 1):
        got = rl.get_match_timeline(client, mid)
        if not got:
            continue
        g = rl.extract_game(got[0], got[1], puuid)
        if g:
            games.append(g)
        if on_progress:
            on_progress(f"{i}/{n}")

    if not games:
        raise RuntimeError("Aucune game exploitable")
    merged = rl.merge_jsonl(rl.SILVER_DIR / "personal" / player / "games.jsonl", games)
    rl.write_gold(rl.GOLD_DIR / "personal" / player, merged, SCOPES, player=player)
    return {"n_games": len(merged), "player": player}


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