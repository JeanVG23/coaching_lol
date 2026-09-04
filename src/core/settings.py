"""Config du pipeline web : comptes suivis, chemins, secrets lus via env puis .env.

Vivait dans `web/backend/` du temps de l'app FastAPI. Le backend est parti, pas ce
module : `pipeline.py` et `sync_cloudflare.py` en dependent pour la collecte locale
qui alimente Cloudflare KV."""
from __future__ import annotations

import json
import os
from pathlib import Path

import riotlib as rl

# Les comptes suivis sont des donnees personnelles : accounts.json est ignore par git,
# accounts.example.json sert de gabarit et de repli.
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
ACCOUNTS_FILE = CONFIG_DIR / "accounts.json"
ACCOUNTS_EXAMPLE_FILE = CONFIG_DIR / "accounts.example.json"
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
    # accounts.json n'est pas versionné (comptes suivis = données perso) :
    # on retombe sur l'exemple pour qu'un clone frais démarre quand même.
    path = ACCOUNTS_FILE if ACCOUNTS_FILE.exists() else ACCOUNTS_EXAMPLE_FILE
    return json.loads(path.read_text())


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