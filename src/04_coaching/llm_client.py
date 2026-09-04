"""Client Ollama Cloud (structured output). Aucune logique métier.

POST https://ollama.com/api/chat avec Authorization: Bearer <OLLAMA_API_KEY>
et `format` = JSON-schema -> renvoie le dict parsé du message. Retries/backoff
sur 429/5xx/timeout, message clair sur clé absente / 401.

`generate` renvoie en plus la télémétrie du run (latence, tokens, tentatives,
coût estimé) : sans elle une review persistée n'est pas reproductible ni
comparable d'un modèle à l'autre. `generate_json` reste la façade sans
télémétrie (appelants qui ne veulent que la sortie).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))  # accès src/core/
import requests
import riotlib as rl

OLLAMA_URL = "https://ollama.com/api/chat"
_MAX_ATTEMPTS = 4

# Tarif $/million de tokens, par modèle. Ollama Cloud est facturé à l'abonnement
# (forfait mensuel), pas à l'usage : la table reste donc vide par défaut et
# `cost_usd` vaut None. La remplir (ex. {"kimi-k2.6": (0.6, 2.5)}) suffit à
# obtenir un coût estimé par review, sans toucher au reste du code.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {}


class LLMError(RuntimeError):
    pass


class Generation(NamedTuple):
    """Sortie du modèle + télémétrie du run (persistée avec la review)."""
    data: dict
    usage: dict


def estimate_cost(model: str, prompt_tokens: int | None,
                  completion_tokens: int | None) -> float | None:
    price = PRICE_PER_MTOK.get(model)
    if price is None or prompt_tokens is None or completion_tokens is None:
        return None
    price_in, price_out = price
    return round((prompt_tokens * price_in + completion_tokens * price_out) / 1e6, 6)


def _usage(model: str, body: dict, latency_ms: int, attempts: int) -> dict:
    """Télémétrie normalisée. Ollama renvoie `prompt_eval_count`/`eval_count`
    et des durées en nanosecondes ; l'absence d'un champ donne None, jamais 0
    (0 tokens serait un chiffre faux dans les agrégats)."""
    def _int(name):
        value = body.get(name)
        return int(value) if isinstance(value, (int, float)) else None

    prompt_tokens, completion_tokens = _int("prompt_eval_count"), _int("eval_count")
    total_ns = _int("total_duration")
    total = None if (prompt_tokens is None or completion_tokens is None) \
        else prompt_tokens + completion_tokens
    return {
        "latency_ms": latency_ms,
        "attempts": attempts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
        "server_duration_ms": None if total_ns is None else round(total_ns / 1e6),
        "cost_usd": estimate_cost(model, prompt_tokens, completion_tokens),
    }


def generate(model: str, system: str, user: str, schema: dict,
             temperature: float = 0.2, timeout: int = 180) -> Generation:
    key = rl.load_env().get("OLLAMA_API_KEY")
    if not key:
        raise LLMError("OLLAMA_API_KEY absente du .env")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "format": schema,
        "stream": False,
        "options": {"temperature": temperature},
    }
    headers = {"Authorization": f"Bearer {key}"}
    last = None
    started = time.monotonic()
    for attempt in range(_MAX_ATTEMPTS):
        try:
            r = requests.post(OLLAMA_URL, json=body, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last = e
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1))
            continue
        code = r.status_code
        if code == 401:
            raise LLMError("401 — vérifie OLLAMA_API_KEY")
        if code == 429 or code >= 500:
            last = LLMError(f"HTTP {code}")
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1))
            continue
        if code >= 400:
            raise LLMError(f"HTTP {code} — requête rejetée par Ollama")
        latency_ms = round((time.monotonic() - started) * 1000)
        try:
            payload = r.json()
            content = payload["message"]["content"]
            return Generation(json.loads(content),
                              _usage(model, payload, latency_ms, attempt + 1))
        except (ValueError, KeyError, TypeError) as e:
            raise LLMError(f"réponse Ollama inexploitable : {e}")
    raise LLMError(f"échec après {_MAX_ATTEMPTS} tentatives : {last}")


def generate_json(model: str, system: str, user: str, schema: dict,
                  temperature: float = 0.2, timeout: int = 180) -> dict:
    return generate(model, system, user, schema, temperature, timeout).data
