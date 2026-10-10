#!/usr/bin/env python3
"""Synchronise les données locales et prédictions précalculées vers Workers KV.

Le Worker ne parle jamais à Riot et ne charge aucun modèle ML. Ce script relit les
couches silver/gold locales, calcule le rang via ``src/core/ml_rank.py`` puis les
drivers EBM via ``src/core/ebm_explain.py`` (sur la même ligne de features que la
prédiction publiée), et pousse une valeur KV par fichier logique. Les clés
``riftsense:*`` restent la
propriété du Worker ; ``--seed-reviews`` ne les amorce que si elles sont absentes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for module_path in (ROOT / "src" / "core", ROOT / "src" / "04_coaching"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import riotlib as rl  # noqa: E402
import positioning as pos  # noqa: E402  (manifeste d'asymétrie : COACHING_SAFE vs ML_ONLY)
import payload as coaching_payload  # noqa: E402
from kv_keys import key as kv_key  # noqa: E402
from kv_client import KV, DryKV, put_json  # noqa: E402

ACCOUNTS_FILE = ROOT / "config" / "accounts.json"


def load_accounts() -> list[dict[str, str]]:
    """Charge les comptes préconfigurés qui constituent la source Python."""
    return rl.read_json(ACCOUNTS_FILE) or []


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
        kv.put(key, merge_jsonl(kv.get(key), rl.read_jsonl(path)))


# Garde-fou asymétrie côté PUBLICATION. Les features per-player sont nommées
# `pos_<base>__<stat>` (ml_features.aggregate_player_features), donc un proxy
# ML_ONLY se reconnaît à son préfixe. Ces 3 proxys de vision alimentent le MODÈLE
# (ils sont dans FEATURES, c'est voulu) mais l'onglet est lu par le joueur : lui
# montrer une barre « morts en fog » revient à lui opposer une info que le modèle
# reconstruit a posteriori et qu'il n'avait pas. compare.py garde le même manifeste
# par un assert au chargement ; ici la liste des features vient d'un artefact
# (player_features.json), donc le contrôle est à l'exécution, sur le payload sortant.
_ML_ONLY_PREFIXES = tuple(f"pos_{name}__" for name in sorted(pos.ML_ONLY))


def _is_ml_only(feature: str) -> bool:
    return feature.startswith(_ML_ONLY_PREFIXES)


def _ebm_drivers(bundle: tuple[dict, int] | None, n: int = 20) -> list[dict] | None:
    """Top-n drivers EBM d'un joueur depuis son agrégat de features, ou None si
    l'agrégat est indisponible (sous MIN_ADC_GAMES : sémantique de la clé `pred`,
    pas de mise à jour de la clé) ou si les artefacts d'analyse manquent.

    Le rang servi vient de xgb+rf ; les drivers sont la décomposition exacte de
    l'EBM per-player (l'additif rend la somme des contributions identique au score,
    par construction) sur la même ligne de features que la prédiction publiée :
    l'explication ne peut pas diverger de ce qui est servi. La paire est légitimée
    au niveau population par le cross-check de l'analyse
    (06_shap/player/high_elo/crosscheck_tree_vs_ebm.json). Payload = array JSON nu :
    le Worker (readers.ts readShap) exige Array.isArray.

    Ce qui est publié est un EXTRAIT, pas la décomposition entière : top-n sur 125
    features, proxys ML_ONLY retirés (cf. _is_ml_only). La somme des barres
    affichées ne vaut donc pas le score, et c'est assumé : l'onglet répond « quels
    indicateurs pèsent », pas « refais l'addition »."""
    if bundle is None:
        return None
    try:
        import ebm_explain  # noqa: E402  (artefacts ML chargés uniquement pour le sync)
        loaded = ebm_explain.load_level("player")
    except (FileNotFoundError, OSError) as exc:
        # Dégradation propre : avant cette clé, un artefact ML absent ne pouvait pas
        # abattre le sync d'un compte. load_level charge 3 pkl ET un parquet ; un seul
        # manquant ne doit pas emporter games/gold/reviews avec lui.
        print(f"  ⚠ drivers EBM ignorés (artefact d'analyse absent : {exc})")
        return None
    contribs = ebm_explain.explain_player_row(
        loaded["models"]["ebm"], bundle[0], loaded["features"])
    drivers = ebm_explain.top_drivers(
        [c for c in contribs if not _is_ml_only(c["feature"])], n)
    assert not any(_is_ml_only(d["feature"]) for d in drivers), \
        "un proxy ML_ONLY a atteint le payload publié — violation d'asymétrie"
    return drivers


def sync_account(kv: KV, slug: str, *, seed_reviews: bool = False,
                 coaching: bool = False, game_payloads: bool = True) -> None:
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
        for scope_dir in sorted(path for path in gold.iterdir()
                                if path.is_dir() and path.name in rl.ROLE_SCOPES):
            aggregate = scope_dir / "aggregate.json"
            if aggregate.exists():
                kv.put(
                    kv_key("gold", slug=slug, scope=scope_dir.name),
                    aggregate.read_text(encoding="utf-8"),
                )

    if games and game_payloads:
        bundle = coaching_payload.build_game_bundle(slug, records=games)
        encoded = coaching_payload.encode_game_bundle(bundle)
        kv.put(kv_key("game_payloads", slug=slug), encoded)

    import ml_rank  # noqa: E402  (artefacts ML chargés uniquement pour le sync)

    prediction = ml_rank.predict_rank(games[:20])
    if prediction is not None:
        put_json(kv, kv_key("pred", slug=slug), prediction)
        # drivers EBM per-player : décomposition exacte de la prédiction publiée.
        # La double agrégation (predict_rank l'a déjà calculée en interne) est
        # acceptée : 20 games, coût négligeable, et predict_rank garde sa shape.
        drivers = _ebm_drivers(ml_rank.player_aggregate(games[:20]))
        if drivers is not None:
            put_json(kv, kv_key("shap", slug=slug), drivers)

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
        for scope_dir in sorted(path for path in rank_dir.iterdir()
                                if path.is_dir() and path.name in rl.ROLE_SCOPES):
            aggregate = scope_dir / "aggregate.json"
            if aggregate.exists():
                kv.put(
                    kv_key("ref", rank=rank_dir.name, scope=scope_dir.name),
                    aggregate.read_text(encoding="utf-8"),
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
    parser.add_argument(
        "--coaching-only",
        action="store_true",
        help="ne synchronise que reviews + annotations (aucun silver/gold/ML/référentiel)",
    )
    parser.add_argument("--dry-run", action="store_true", help="journalise sans écrire dans KV")
    parser.add_argument(
        "--skip-game-payloads", action="store_true",
        help="ne construit pas les payloads unitaires depuis le cache raw local",
    )
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
        if args.coaching_only:
            push_coaching(kv, account["slug"])
        else:
            sync_account(kv, account["slug"], seed_reviews=args.seed_reviews,
                         coaching=args.push_coaching,
                         game_payloads=not args.skip_game_payloads)
    if not args.skip_ref and not args.coaching_only:
        sync_referential(kv)

    print(f"{len(kv.puts)} clés {'à pousser' if args.dry_run else 'poussées'} :")
    for key in kv.puts:
        print(f"  {key}")


if __name__ == "__main__":
    main()
