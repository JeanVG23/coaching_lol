# web/backend/settings.py
"""Config web : comptes préconfigurés, chemins, secrets lus via env puis .env."""
from __future__ import annotations

import json
import os
from pathlib import Path

import riotlib as rl

HERE = Path(__file__).resolve().parent
ACCOUNTS_FILE = HERE / "accounts.json"
JOBS_FILE = rl.DATA / "08_jobs" / "jobs.jsonl"

_PUUID_CACHE: dict[str, str] = {}


def _env() -> dict[str, str]:
    return rl.load_env()


def riot_api_key() -> str | None:
    return os.environ.get("RIOT_API_ID") or _env().get("RIOT_API_ID")


def ollama_key() -> str | None:
    return os.environ.get("OLLAMA_API_KEY") or _env().get("OLLAMA_API_KEY")


def ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL") or _env().get("OLLAMA_MODEL", "kimi-k2.6")


def load_accounts() -> list[dict]:
    return json.loads(ACCOUNTS_FILE.read_text())


def account_for(slug: str) -> dict | None:
    for a in load_accounts():
        if a["slug"] == slug:
            return a
    return None


def resolve_puuid(account: dict) -> str:
    """Résout le puuid via account-v1 (cache en mémoire). Lève si introuvable."""
    slug = account["slug"]
    if slug in _PUUID_CACHE:
        return _PUUID_CACHE[slug]
    regional = rl.PLATFORM_TO_REGIONAL[account["region"]]
    client = rl.RiotClient(riot_api_key(), regional, account["region"])
    game_name, tag_line = account["riot_id"].split("#", 1)
    puuid = client.puuid_from_riot_id(game_name, tag_line)
    if not puuid:
        raise RuntimeError(f"Riot ID introuvable : {account['riot_id']}")
    _PUUID_CACHE[slug] = puuid
    return puuid