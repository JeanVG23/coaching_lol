# web/backend/pipeline.py
"""Wrappers autour du pipeline existant pour le web, avec callback de progression.

Réutilise riotlib (pull/silver/gold) et coach (payload/generate/persist). Ne réécrit
rien : juste un point d'entrée callable avec on_progress.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Callable

import riotlib as rl
from aggregate_games import SCOPES

import coach
import payload
import settings


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