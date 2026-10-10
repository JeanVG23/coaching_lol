#!/usr/bin/env python3
"""01_data_engineering — raw/silver -> dataset ML per-game par rôle.

Pour chaque game du référentiel, les deux joueurs du rôle demandé sont ré-extraits
depuis le raw, sans appel API. Le rang approximatif du dossier silver n'est plus
transféré à ces lignes : le label officiel sera joint au niveau joueur à partir
d'un snapshot daté du ladder par ``build_role_player_dataset.py``.

Les valeurs manquantes restent en NaN pour être traitées par les modèles. Sans
``--role``, le comportement ADC historique reste inchangé pour le graphe Make.
Avec ``--role``, la sortie est ``data/04_dataset/{role}_dataset.parquet``.

Usage : poetry run python3 src/01_data_engineering/build_dataset.py --role JUNGLE
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))  # accès à riotlib
import pandas as pd
import cli
from game_rows import game_to_row  # noqa: F401  (réexport : appelé par les tests existants)
import riotlib as rl
import role_features as rf
from ranks import HIGH_ELO, RANKS, RANK_ORD  # réexports du pipeline ADC historique

DATASET_DIR = rl.DATA / "04_dataset"


def role_puuids(match: dict, role: str) -> list[str]:
    """Puuids des joueurs occupant ``role`` — un par équipe, donc environ 2."""
    riot_role = rf.riot_role(role)
    puuids = match["metadata"]["participants"]
    parts = match["info"]["participants"]
    return [puuids[i] for i, p in enumerate(parts)
            if (p.get("teamPosition") or "") == riot_role]


def adc_puuids(match: dict) -> list[str]:
    """Compatibilité du pipeline ADC historique."""
    return role_puuids(match, "BOTTOM")


def dataset_path(role: str) -> Path:
    return DATASET_DIR / f"{rf.normalize_role(role).lower()}_dataset.parquet"


def build_rank_map() -> tuple[dict[str, str], int]:
    """match_id -> rang de collecte (référentiel). Une game peut être collectée sous
    plusieurs rangs (lobby à cheval master/GM) : on résout au mode, tie-break sur le
    rang le plus bas (RANK_ORD min) pour ne pas gonfler high_elo aux frontières."""
    seen: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for rank in RANKS:
        for g in rl.iter_jsonl(rl.silver_games(rl.KIND_REF, rank)):
            seen[g["match_id"]][rank] += 1
    rank_of, multi = {}, 0
    for mid, cnt in seen.items():
        if len(cnt) > 1:
            multi += 1
        rank_of[mid] = min(cnt.items(), key=lambda kv: (-kv[1], RANK_ORD[kv[0]]))[0]
    return rank_of, multi


def _load_raw(match_id: str):
    try:
        match = rl._read_raw(f"{match_id}_match")
        timeline = rl._read_raw(f"{match_id}_timeline")
    except Exception as e:  # raw corrompu / illisible -> traité comme manquant
        print(f"  ⚠ raw illisible {match_id}: {e}", file=sys.stderr)
        return None
    return (match, timeline) if match and timeline else None


def _legacy_adc_main() -> int:
    """Reconstruit les artefacts ADC historiques attendus par ``make dataset``."""
    rows = []
    rank_of, multi = build_rank_map()
    print(f"  {len(rank_of)} games référentiel distinctes "
          f"({multi} multi-rang -> mode)")
    per_rank, raw_miss = collections.Counter(), 0
    for mid, rank in rank_of.items():
        raw = _load_raw(mid)
        if not raw:
            raw_miss += 1
            continue
        match, timeline = raw
        for puuid in adc_puuids(match):
            rec = rl.extract_game(match, timeline, puuid, rank=rank)
            if rec and rec.get("role") == "BOTTOM":
                rows.append(game_to_row(rec, rank, rl.KIND_REF))
                per_rank[rank] += 1
    for rank in RANKS:
        print(f"  referentiel/{rank:<12}: {per_rank[rank]} rows ADC")
    if raw_miss:
        print(f"  ⚠ {raw_miss} games sans raw lisible -> ignorées")

    perso_root = rl.silver_dir() / rl.KIND_PERSONAL
    if perso_root.exists():
        for directory in sorted(perso_root.iterdir()):
            games = rl.read_jsonl(directory / "games.jsonl")
            adc = [game for game in games if game.get("role") == "BOTTOM"]
            rows += [game_to_row(game, None, f"personal:{directory.name}")
                     for game in adc]
            print(f"  personal/{directory.name:<14}: {len(adc)} games ADC")

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["match_id", "puuid"]).reset_index(drop=True)
    df["rank_ord"] = df["rank"].map(RANK_ORD)
    df["high_elo"] = df["rank"].isin(HIGH_ELO).astype("Int64")
    df.loc[df["rank"].isna(), "high_elo"] = pd.NA

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_DIR / "adc_dataset.parquet", index=False)
    df.to_csv(DATASET_DIR / "adc_dataset.csv", index=False)
    ref = df[df["source"] == "referentiel"]
    print(f"\n✓ Dataset : {len(df)} games ({len(ref)} référentiel labellisés)")
    print(f"  Répartition rangs : {dict(ref['rank'].value_counts())}")
    print(f"  high_elo (GM+Chall=1) : {dict(ref['high_elo'].value_counts())}")
    print(f"  Écrit dans {DATASET_DIR}/adc_dataset.parquet")
    return 0


def _role_dataset_main(role: str) -> int:
    """Construit le dataset per-game non labellisé du rôle demandé."""
    role = rf.normalize_role(role)
    if role not in rf.ROLES:
        print(f"rôle inconnu : {role} (attendus : {', '.join(rf.ROLES)})",
              file=sys.stderr)
        return 2

    rows = []
    # Le rang du dossier silver ne labellise plus chaque game. Le snapshot du
    # ladder sera joint après agrégation, dans build_role_player_dataset.py.
    match_ids = {g["match_id"] for rank in RANKS
                 for g in rl.iter_jsonl(rl.silver_games(rl.KIND_REF, rank))}
    print(f"  {len(match_ids)} games référentiel distinctes")
    raw_miss = 0
    for mid in sorted(match_ids):
        raw = _load_raw(mid)
        if not raw:
            raw_miss += 1
            continue
        match, timeline = raw
        for puuid in role_puuids(match, role):
            rec = rl.extract_game(match, timeline, puuid, rank=None)
            if rec and rec.get("role") == rf.riot_role(role):
                row = game_to_row(rec, None, rl.KIND_REF)
                row.pop("rank", None)
                row["role"] = role
                rows.append(row)
    if raw_miss:
        print(f"  ⚠ {raw_miss} games sans raw lisible -> ignorées")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["match_id", "puuid"]).reset_index(drop=True)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dataset_path(role), index=False)
    print(f"\n✓ {len(df)} lignes {role} écrites dans {dataset_path(role)}")
    return 0


def main() -> int:
    if not cli.flag("--role"):
        return _legacy_adc_main()
    return _role_dataset_main(rf.normalize_role(cli.arg("--role", "")))


if __name__ == "__main__":
    sys.exit(main())
