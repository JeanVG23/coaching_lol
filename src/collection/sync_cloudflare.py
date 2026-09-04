#!/usr/bin/env python3
"""Synchronise les données locales et prédictions précalculées vers Workers KV.

Le Worker ne parle jamais à Riot et ne charge aucun modèle ML. Ce script relit les
couches silver/gold/SHAP locales, calcule le rang via ``src/core/ml_rank.py`` et
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
for module_path in (ROOT / "src" / "core",):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import riotlib as rl  # noqa: E402
from kv_keys import key as kv_key  # noqa: E402

ACCOUNTS_FILE = ROOT / "config" / "accounts.json"
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


def parse_jsonl(raw: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def parse_games(raw: str) -> list[dict[str, Any]]:
    """Parse un JSONL silver déjà lu et trie par séquence de match décroissante."""
    return sorted(parse_jsonl(raw), key=lambda row: _match_seq(row.get("match_id", "")), reverse=True)


def read_games(slug: str) -> list[dict[str, Any]]:
    """Lit les games silver d'un joueur et les trie par séquence décroissante."""
    path = rl.silver_games(rl.KIND_PERSONAL, slug)
    return parse_games(path.read_text()) if path.exists() else []


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


def merge_jsonl(remote: str | None, local: list[dict], id_key: str = "ts") -> str:
    """Fusionne des lignes locales dans un JSONL distant, clé `id_key`.

    Les reviews et feedbacks arrivent des DEUX côtés : le site écrit dans KV
    (bouton coaching, annotations web) et la CLI locale écrit dans
    ``data/07_coaching/``. Écraser la clé perdrait le web ; ne rien pousser
    laisserait les annotations CLI invisibles sur le site. On garde l'ordre
    distant, on remplace ligne à ligne sur `id_key` (le local, plus récent au
    moment du sync, gagne) et on ajoute le reste à la fin.
    """
    rows = parse_jsonl(remote) if remote else []
    by_id = {row.get(id_key): row for row in local if row.get(id_key) is not None}
    merged, used = [], set()
    for row in rows:
        rid = row.get(id_key)
        if rid in by_id:
            merged.append(by_id[rid])
            used.add(rid)
        else:
            merged.append(row)
    merged.extend(row for rid, row in by_id.items() if rid not in used)
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in merged) + "\n"


def push_coaching(kv: KV, slug: str) -> None:
    """Pousse reviews + feedbacks locaux dans KV en fusionnant l'existant.

    Sans cette étape, les reviews générées en CLI et surtout les annotations de
    la boucle d'éval restent locales : le taux d'utilité publié sur le site
    (`/api/c/<slug>/eval`) ignorerait la moitié des données."""
    base = rl.DATA / "07_coaching" / slug
    for name, filename in (("reviews", "reviews.jsonl"), ("feedback", "feedback.jsonl")):
        path = base / filename
        if not path.exists():
            continue
        key = kv_key(name, slug=slug)
        kv.put(key, merge_jsonl(kv.get(key), parse_jsonl(path.read_text())))


def sync_account(kv: KV, slug: str, *, seed_reviews: bool = False,
                 coaching: bool = False) -> None:
    # Une seule lecture du JSONL : le texte brut part tel quel dans KV et sert aussi
    # de source au parse local (il était lu deux fois : read_games + read_text).
    games_file = rl.silver_games(rl.KIND_PERSONAL, slug)
    games_raw = games_file.read_text() if games_file.exists() else None
    games = parse_games(games_raw) if games_raw is not None else []
    if games_raw is not None:
        kv.put(kv_key("games", slug=slug), games_raw)
    rank_file = games_file.parent / "rank.json"
    if rank_file.exists():
        kv.put(kv_key("rank", slug=slug), rank_file.read_text())

    gold = rl.gold_base(rl.KIND_PERSONAL, slug)
    if gold.is_dir():
        for scope_dir in sorted(path for path in gold.iterdir() if path.is_dir()):
            aggregate = scope_dir / "aggregate.json"
            if aggregate.exists():
                put_json(
                    kv,
                    kv_key("gold", slug=slug, scope=scope_dir.name),
                    json.loads(aggregate.read_text()),
                )

    import ml_rank  # noqa: E402  (artefacts ML chargés uniquement pour le sync)

    prediction = ml_rank.predict_rank(games[:20])
    if prediction is not None:
        put_json(kv, kv_key("pred", slug=slug), prediction)

    shap = rl.DATA / "06_shap" / f"{slug}_drivers.json"
    if shap.exists():
        kv.put(kv_key("shap", slug=slug), shap.read_text())

    if coaching:
        push_coaching(kv, slug)
    elif seed_reviews:
        reviews = rl.DATA / "07_coaching" / slug / "reviews.jsonl"
        review_key = kv_key("reviews", slug=slug)
        if reviews.exists() and kv.get(review_key) is None:
            kv.put(review_key, reviews.read_text())


def sync_referential(kv: KV) -> None:
    referential = rl.gold_dir() / rl.KIND_REF
    if not referential.is_dir():
        return
    for rank_dir in sorted(path for path in referential.iterdir() if path.is_dir()):
        for scope_dir in sorted(path for path in rank_dir.iterdir() if path.is_dir()):
            aggregate = scope_dir / "aggregate.json"
            if aggregate.exists():
                put_json(
                    kv,
                    kv_key("ref", rank=rank_dir.name, scope=scope_dir.name),
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
    parser.add_argument(
        "--push-coaching",
        action="store_true",
        help="fusionne reviews + annotations locales dans KV (le site les publie)",
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
        sync_account(kv, account["slug"], seed_reviews=args.seed_reviews,
                     coaching=args.push_coaching)
    if not args.skip_ref:
        sync_referential(kv)

    print(f"{len(kv.puts)} clés {'à pousser' if args.dry_run else 'poussées'} :")
    for key in kv.puts:
        print(f"  {key}")


if __name__ == "__main__":
    main()
