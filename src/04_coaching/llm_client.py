"""Client Ollama Cloud (structured output). Aucune logique métier.

POST https://ollama.com/api/chat avec Authorization: Bearer <OLLAMA_API_KEY>
et `format` = JSON-schema -> renvoie le dict parsé du message. Retries/backoff
sur 429/5xx/timeout, message clair sur clé absente / 401.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # accès src/
import requests
import riotlib as rl

OLLAMA_URL = "https://ollama.com/api/chat"
_MAX_ATTEMPTS = 4


class LLMError(RuntimeError):
    pass


def generate_json(model: str, system: str, user: str, schema: dict,
                  temperature: float = 0.2, timeout: int = 180) -> dict:
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
    for attempt in range(_MAX_ATTEMPTS):
        try:
            r = requests.post(OLLAMA_URL, json=body, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last = e
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 401:
            raise LLMError("401 — vérifie OLLAMA_API_KEY")
        if r.status_code == 429 or r.status_code >= 500:
            last = LLMError(f"HTTP {r.status_code}")
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return json.loads(content)
    raise LLMError(f"échec après {_MAX_ATTEMPTS} tentatives : {last}")