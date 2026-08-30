#!/usr/bin/env python3
"""Synchronise les données locales et prédictions précalculées vers Workers KV.

Le Worker ne parle jamais à Riot et ne charge aucun modèle ML. Ce script relit les
couches silver/gold/SHAP locales, calcule le rang via ``web/backend/ml_rank.py`` et
pousse une valeur KV par fichier logique. Les clés ``coaching:*`` restent la
propriété du Worker ; ``--seed-reviews`` ne les amorce que si elles sont absentes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[2]
for module_path in (ROOT / "src" / "core", ROOT / "web" / "backend"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import riotlib as rl  # noqa: E402

ACCOUNTS_FILE = ROOT / "web" / "backend" / "accounts.json"
KV_URL = (
    "https://api.cloudflare.com/client/v4/accounts/{account}"
    "/storage/kv/namespaces/{namespace}/values/{key}"
)


def load_accounts() -> list[dict[str, str]]:
    """Charge les comptes préconfigurés qui constituent la source Python."""
    return json.loads(ACCOUNTS_FILE.read_text())


def _match_seq(match_id: str) -> int:
    try:
        return int(match_id.rsplit("_", 1)[-1])
    except (ValueError, AttributeError):
        return 0


def read_games(slug: str) -> list[dict[str, Any]]:
    """Lit les games silver et les trie par séquence décroissante."""
    path = rl.DATA / "02_silver" / "personal" / slug / "games.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda row: _match_seq(row.get("match_id", "")), reverse=True)


class KV:
    """Client REST minimal Workers KV (PUT/GET) et journal des clés poussées."""

    def __init__(self, account: str, namespace: str, token: str):
        self.account = account
        self.namespace = namespace
        self.token = token
        self.puts: list[str] = []

    def _url(self, key: str) -> str:
        return KV_URL.format(
            account=self.account,
            namespace=self.namespace,
            key=quote(key, safe=""),
        )

    def put(self, key: str, value: str) -> None:
        response = requests.put(
            self._url(key),
            data=value.encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "text/plain; charset=utf-8",
            },
            timeout=30,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"KV PUT {key} -> HTTP {response.status_code} : {response.text[:200]}"
            )
        self.puts.append(key)

    def get(self, key: str) -> str | None:
        response = requests.get(
            self._url(key),
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30,
        )
        if response.status_code == 200:
            return response.text
        if response.status_code == 404:
            return None
        raise RuntimeError(
            f"KV GET {key} -> HTTP {response.status_code} : {response.text[:200]}"
        )


class DryKV(KV):
    """Journalise les écritures sans accès réseau."""

    def __init__(self):
        super().__init__("dry-run", "dry-run", "dry-run")

    def put(self, key: str, value: str) -> None:
        del value
        self.puts.append(key)

    def get(self, key: str) -> str | None:
        del key
        return None


def put_json(kv: KV, key: str, value: Any) -> None:
    kv.put(key, json.dumps(value, ensure_ascii=False))


def sync_account(kv: KV, slug: str, *, seed_reviews: bool = False) -> None:
    games = read_games(slug)
    silver = rl.DATA / "02_silver" / "personal" / slug
    games_file = silver / "games.jsonl"
    rank_file = silver / "rank.json"
    if games_file.exists():
        kv.put(f"silver:{slug}:games", games_file.read_text())
    if rank_file.exists():
        kv.put(f"silver:{slug}:rank", rank_file.read_text())

    gold = rl.DATA / "03_gold" / "personal" / slug
    if gold.is_dir():
        for scope_dir in sorted(path for path in gold.iterdir() if path.is_dir()):
            aggregate = scope_dir / "aggregate.json"
            if aggregate.exists():
                put_json(
                    kv,
                    f"gold:{slug}:{scope_dir.name}",
                    json.loads(aggregate.read_text()),
                )

    import ml_rank  # noqa: E402  (artefacts ML chargés uniquement pour le sync)

    prediction = ml_rank.predict_rank(games[:20])
    if prediction is not None:
        put_json(kv, f"pred:{slug}", prediction)

    shap = rl.DATA / "06_shap" / f"{slug}_drivers.json"
    if shap.exists():
        kv.put(f"shap:{slug}:drivers", shap.read_text())

    if seed_reviews:
        reviews = rl.DATA / "07_coaching" / slug / "reviews.jsonl"
        review_key = f"coaching:{slug}:reviews"
        if reviews.exists() and kv.get(review_key) is None:
            kv.put(review_key, reviews.read_text())


def sync_referential(kv: KV) -> None:
    referential = rl.DATA / "03_gold" / "referentiel"
    if not referential.is_dir():
        return
    for rank_dir in sorted(path for path in referential.iterdir() if path.is_dir()):
        for scope_dir in sorted(path for path in rank_dir.iterdir() if path.is_dir()):
            aggregate = scope_dir / "aggregate.json"
            if aggregate.exists():
                put_json(
                    kv,
                    f"ref:{rank_dir.name}:{scope_dir.name}",
                    json.loads(aggregate.read_text()),
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", help="limite le sync à un compte")
    parser.add_argument("--skip-ref", action="store_true", help="ne pousse pas le référentiel")
    parser.add_argument(
        "--seed-reviews",
        action="store_true",
        help="amorce les reviews locales uniquement si la clé KV est absente",
    )
    parser.add_argument("--dry-run", action="store_true", help="journalise sans écrire dans KV")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    env = rl.load_env()
    for key in ("CF_API_TOKEN", "CF_ACCOUNT_ID", "CF_NAMESPACE_ID"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    if args.dry_run:
        kv: KV = DryKV()
    else:
        token = env.get("CF_API_TOKEN", "")
        account = env.get("CF_ACCOUNT_ID", "")
        namespace = env.get("CF_NAMESPACE_ID", "")
        if not (token and account and namespace):
            raise SystemExit(
                "CF_API_TOKEN / CF_ACCOUNT_ID / CF_NAMESPACE_ID manquants dans .env"
            )
        kv = KV(account, namespace, token)

    accounts = load_accounts()
    if args.slug:
        accounts = [account for account in accounts if account["slug"] == args.slug]
        if not accounts:
            raise SystemExit(f"compte inconnu : {args.slug}")
    for account in accounts:
        sync_account(kv, account["slug"], seed_reviews=args.seed_reviews)
    if not args.skip_ref:
        sync_referential(kv)

    print(f"{len(kv.puts)} clés {'à pousser' if args.dry_run else 'poussées'} :")
    for key in kv.puts:
        print(f"  {key}")


if __name__ == "__main__":
    main()
