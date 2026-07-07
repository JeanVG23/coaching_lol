#!/usr/bin/env python3
"""
01_data_engineering — dataset per-player pour la régression LP (Master/GM/Chall).

Comme build_player_dataset.py (binaire) mais SANS balance-cap : une régression n'a
pas de classes à équilibrer, donc on garde TOUS les joueurs qualifiés (le cap du
binaire jette ~40 % des masters qualifiés — c'est précisément le pool que le modèle
LP récupère, cf. spec). Restreint aux tiers apex (diamond exclu : divisions I-IV
avec reset, LP non comparable à l'échelle continue master→challenger).

Label : LP courant depuis data/04_dataset/apex_lp.json (fetch_apex_lp.py, à relancer
avant ce script pour un label frais). Joueurs qualifiés absents du lookup = tier
changé depuis la collecte → droppés et comptés (n_dropped_no_lp).

0 appel API. Sorties : data/04_dataset/adc_player_lp_dataset.parquet (+ .csv)
et adc_player_lp_dataset.meta.json (fetched_at du label, n_dropped_no_lp, n_by_tier).
Usage : poetry run python3 src/01_data_engineering/build_player_lp_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import pandas as pd
import riotlib as rl
import ml_features as mf

DATASET_DIR = rl.DATA / "04_dataset"
LP_PATH = DATASET_DIR / "apex_lp.json"
MIN_PLAYER_GAMES = 15
APEX_TIERS = {"master", "grandmaster", "challenger"}


def build_lp_player_rows(ref: pd.DataFrame, lp_players: dict,
                         min_games: int = MIN_PLAYER_GAMES,
                         features: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """ref : rows per-game référentiel (colonnes puuid, rank, win, + features).
    lp_players : {puuid: {"tier", "leaguePoints"}} (clé "players" d'apex_lp.json).
    Retourne (1 ligne par joueur qualifié apex AVEC LP courant — colonnes puuid,
    rank, lp, agrégats + win_rate + n_games ; nombre de qualifiés droppés faute de
    LP courant). PAS de balance-cap. Rang résolu au mode sur tout l'historique
    (même sémantique que le binaire, cf. ml_features.resolve_rank)."""
    features = mf.FEATURES if features is None else features
    rows, n_dropped = [], 0
    for puuid, g in ref.groupby("puuid"):
        if len(g) < min_games:
            continue
        rank = mf.resolve_rank(g)
        if rank not in APEX_TIERS:
            continue
        entry = lp_players.get(puuid)
        if entry is None:
            n_dropped += 1
            continue
        rec = {"puuid": puuid, "rank": rank, "lp": int(entry["leaguePoints"])}
        rec.update(mf.aggregate_player_features(g, features))
        rows.append(rec)
    return pd.DataFrame(rows), n_dropped


def main() -> int:
    if not LP_PATH.exists():
        print(f"✗ {LP_PATH} introuvable — lancer "
              "src/collection/fetch_apex_lp.py d'abord.", file=sys.stderr)
        return 1
    lp_file = json.loads(LP_PATH.read_text())

    df = pd.read_parquet(DATASET_DIR / "adc_dataset.parquet")
    ref = df[df["source"] == "referentiel"].copy()
    print(f"  {len(ref)} games référentiel | {ref['puuid'].nunique()} joueurs uniques")
    print(f"  label LP : {len(lp_file['players'])} joueurs apex, "
          f"fetched_at={lp_file['fetched_at']}")

    out, n_dropped = build_lp_player_rows(ref, lp_file["players"])
    print(f"  >= {MIN_PLAYER_GAMES} games apex avec LP courant : {len(out)} joueurs "
          f"({n_dropped} qualifiés droppés, tier changé depuis la collecte)")
    if out.empty:
        print("  ⚠ aucun joueur ne qualifie -> rien à écrire")
        return 1
    n_by_tier = {k: int(v) for k, v in out["rank"].value_counts().items()}
    print(f"  répartition tiers : {n_by_tier}")

    out.to_parquet(DATASET_DIR / "adc_player_lp_dataset.parquet", index=False)
    out.to_csv(DATASET_DIR / "adc_player_lp_dataset.csv", index=False)
    (DATASET_DIR / "adc_player_lp_dataset.meta.json").write_text(json.dumps({
        "fetched_at": lp_file["fetched_at"],
        "n_dropped_no_lp": n_dropped,
        "n_by_tier": n_by_tier,
    }, indent=2))
    print(f"\n✓ Dataset LP per-player écrit dans {DATASET_DIR}/adc_player_lp_dataset.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
