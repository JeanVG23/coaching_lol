#!/usr/bin/env python3
"""Rafraichit les donnees Riot locales puis les publie sur Cloudflare KV.

Cette commande est le point d'entree operationnel du site Cloudflare : elle
recupere les parties et le rang via le pipeline Python existant, reconstruit les
agregats, puis appelle ``sync_cloudflare``. Les referentiels, beaucoup plus
statiques, ne sont republies que sur demande avec ``--with-ref``.

Usage :
    poetry run python src/collection/refresh_cloudflare.py
    poetry run python src/collection/refresh_cloudflare.py --slug spadzze -n 20
    poetry run python src/collection/refresh_cloudflare.py --with-ref
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
for module_path in (
    ROOT / "src" / "core",
    ROOT / "src" / "04_coaching",
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import pipeline  # noqa: E402
import riotlib as rl  # noqa: E402
import sync_cloudflare  # noqa: E402

CF_CONFIG_KEYS = ("CF_API_TOKEN", "CF_ACCOUNT_ID", "CF_NAMESPACE_ID")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", help="limite le rafraichissement a un compte")
    parser.add_argument(
        "-n",
        "--games",
        type=int,
        default=20,
        help="nombre de parties recentes demandees a Riot (defaut : 20)",
    )
    parser.add_argument(
        "--with-ref",
        action="store_true",
        help="republie aussi les referentiels de rang, normalement inchanges",
    )
    return parser


def select_accounts(slug: str | None) -> list[dict[str, Any]]:
    """Retourne les comptes a rafraichir ou termine sur un slug inconnu."""
    accounts = sync_cloudflare.load_accounts()
    if slug is None:
        return accounts
    selected = [account for account in accounts if account["slug"] == slug]
    if not selected:
        raise SystemExit(f"compte inconnu : {slug}")
    return selected


def ensure_cloudflare_configured() -> None:
    """Refuse la collecte si sa publication KV ne peut pas aboutir."""
    env = rl.load_env()
    missing = [key for key in CF_CONFIG_KEYS if not os.environ.get(key) and not env.get(key)]
    if missing:
        raise SystemExit(
            "configuration Cloudflare manquante dans .env : " + ", ".join(missing)
        )


def refresh_accounts(
    accounts: list[dict[str, Any]],
    games: int,
    *,
    fetch: Callable[..., dict[str, Any]] | None = None,
) -> None:
    """Collecte tous les comptes avant de commencer la publication distante."""
    fetch = fetch or pipeline.fetch_games
    for account in accounts:
        slug = account["slug"]
        print(f"\n== Rafraichissement Riot : {slug} ==")

        def progress(message: str, *, _slug: str = slug) -> None:
            print(f"  [{_slug}] {message}")

        result = fetch(account, n=games, on_progress=progress)
        print(f"  {result['n_games']} parties disponibles localement")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.games < 1 or args.games > 100:
        raise SystemExit("--games doit etre compris entre 1 et 100")

    ensure_cloudflare_configured()
    accounts = select_accounts(args.slug)
    try:
        refresh_accounts(accounts, args.games)
    except Exception as exc:
        raise SystemExit(f"rafraichissement Riot interrompu : {exc}") from exc

    print("\n== Publication Cloudflare KV ==")
    sync_args: list[str] = []
    if args.slug:
        sync_args.extend(["--slug", args.slug])
    if not args.with_ref:
        sync_args.append("--skip-ref")
    sync_cloudflare.main(sync_args)
    print("\nOK : le site Cloudflare utilise maintenant les nouvelles donnees.")


if __name__ == "__main__":
    main()
