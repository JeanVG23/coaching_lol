#!/usr/bin/env python3
"""
01_data_engineering — raw/silver -> dataset ML consolidé (1 ligne = 1 ADC d'une game).

Référentiel : on extrait LES DEUX ADC (botlane des deux équipes) de chaque game
DEPUIS LE RAW (0 appel API), pas seulement la perspective collectée. Le silver ne
stocke qu'un joueur ciblé par game (cf. extract_game(puuid) unique) ; s'y limiter
jetait ~9/10 des perspectives et ne récupérait l'ADC que des games où le joueur
ciblé ÉTAIT ADC. En relisant le raw (qui contient les 10 joueurs), on densifie le
dataset ADC d'environ ×8.7 (≈ games × 2).

⚠️ FLAW ASSUMÉ — transfert de rang : le rang d'une game = le rang de collecte du
joueur ciblé (dossier silver). On le transfère AUX DEUX ADC en supposant un MMR
égal dans le lobby (vrai en solo queue high-elo, matchmaking serré). L'ADC ennemi
n'a donc pas son rang réel mesuré mais celui, approché, de la game. Acceptable pour
un classif high/low ; à revoir si on descend en elo où l'écart de MMR intra-lobby
s'élargit. Games collectées sous plusieurs rangs (lobbies master∩GM) : rang résolu
au mode, tie-break sur le rang le plus bas (ne pas gonfler high_elo aux frontières).

Perso (Spadzze) : on garde SA perspective ADC (inférence), pas l'ADC ennemi.

Les trous (gd20/csd14 None sur games courtes) sont LAISSÉS en NaN : XGBoost gère les
valeurs manquantes nativement (pas d'imputation arbitraire).

Sortie : data/04_dataset/adc_dataset.parquet (+ .csv pour inspection).
Usage : poetry run python3 src/01_data_engineering/build_dataset.py
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))  # accès à riotlib
import numpy as np
import pandas as pd
import riotlib as rl
from ranks import RANKS, RANK_ORD, HIGH_ELO  # cible binaire + tie-break

DATASET_DIR = rl.DATA / "04_dataset"


def game_to_row(g: dict, rank: str | None, source: str) -> dict:
    lane = g.get("lane", {})
    deaths = g.get("deaths", [])
    kills = g.get("kills", [])
    assists = g.get("assists", [])
    n = len(deaths)
    ph = collections.Counter(d["phase"] for d in deaths)
    gs = collections.Counter(d.get("gold_state") for d in deaths if d.get("gold_state"))
    gs_tot = sum(gs.values())
    return {
        # méta (non-features)
        "match_id": g["match_id"], "puuid": g.get("puuid"), "source": source,
        "rank": rank, "champion": g["champion"], "win": int(g["win"]),
        "patch": g.get("patch"), "game_ts": g.get("game_ts"),
        # features de lane (diffs vs adversaire)
        "gd10": lane.get("gd10"), "gd14": lane.get("gd14"), "gd20": lane.get("gd20"),
        "csd10": lane.get("csd10"), "csd14": lane.get("csd14"), "xpd10": lane.get("xpd10"),
        "csm10": lane.get("csm10"), "csm14": lane.get("csm14"),
        "gpm10": lane.get("gpm10"), "gpm14": lane.get("gpm14"),
        "xppm10": lane.get("xppm10"),
        # features de morts
        "n_deaths": n,
        "deaths_early": ph.get("early", 0),
        "deaths_mid": ph.get("mid", 0),
        "deaths_late": ph.get("late", 0),
        "deaths_solo": sum(1 for d in deaths if d.get("is_solo")),
        "deaths_teamfight": sum(1 for d in deaths if not d.get("is_solo")),
        "deaths_early_jungle": sum(1 for d in deaths if d.get("phase") == "early" and d.get("is_ganked_by_jungle")),
        "deaths_early_2v2": sum(1 for d in deaths if d.get("phase") == "early" and d.get("is_2v2")),
        "kills_solo": sum(1 for k in kills if k.get("is_solo")),
        "kills_2v2": sum(1 for k in kills if k.get("is_2v2")),
        "assists_2v2": sum(1 for a in assists if a.get("is_2v2")),
        "kda_1v1": sum(1 for k in kills if k.get("is_solo")) / max(1, sum(1 for d in deaths if d.get("is_solo"))),
        "kda_2v2": (sum(1 for k in kills if k.get("is_2v2")) + sum(1 for a in assists if a.get("is_2v2"))) / max(1, sum(1 for d in deaths if d.get("is_2v2"))),
        "frac_behind": gs.get("behind", 0) / gs_tot if gs_tot else np.nan,
        "frac_ahead": gs.get("ahead", 0) / gs_tot if gs_tot else np.nan,
        # macro
        "avg_dragon_prox": g.get("avg_dragon_prox") if g.get("avg_dragon_prox") is not None else np.nan,
        "support_deaths_early": g.get("support_deaths_early", 0),
        "plates_diff_early": g.get("plates_diff_early", 0),
        "frames_in_base_early": g.get("frames_in_base_early", 0),
        **{f"pos_{k}": v for k, v in (g.get("position") or {}).items()},
    }


def adc_puuids(match: dict) -> list[str]:
    """puuids des ADC (teamPosition BOTTOM) d'une game — un par équipe, donc ~2."""
    puuids = match["metadata"]["participants"]
    parts = match["info"]["participants"]
    return [puuids[i] for i, p in enumerate(parts)
            if (p.get("teamPosition") or "") == "BOTTOM"]


def build_rank_map() -> tuple[dict[str, str], int]:
    """match_id -> rang de collecte (référentiel). Une game peut être collectée sous
    plusieurs rangs (lobby à cheval master/GM) : on résout au mode, tie-break sur le
    rang le plus bas (RANK_ORD min) pour ne pas gonfler high_elo aux frontières."""
    seen: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for rank in RANKS:
        for g in rl.read_jsonl(rl.silver_games(rl.KIND_REF, rank)):
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


def main() -> int:
    rows = []
    # référentiel : LES DEUX ADC de chaque game, ré-extraits depuis le raw (0 API).
    rank_of, multi = build_rank_map()
    print(f"  {len(rank_of)} games référentiel distinctes ({multi} multi-rang -> mode)")
    per_rank, raw_miss = collections.Counter(), 0
    for mid, rank in rank_of.items():
        raw = _load_raw(mid)
        if not raw:
            raw_miss += 1
            continue
        match, timeline = raw
        for puuid in adc_puuids(match):
            rec = rl.extract_game(match, timeline, puuid, rank=rank)
            if rec and rec.get("role") == "BOTTOM":  # extract_game filtre déjà non-SR
                rows.append(game_to_row(rec, rank, rl.KIND_REF))
                per_rank[rank] += 1
    for rank in RANKS:
        print(f"  referentiel/{rank:<12}: {per_rank[rank]} rows ADC")
    if raw_miss:
        print(f"  ⚠ {raw_miss} games sans raw lisible -> ignorées")

    # perso (non labellisé : pour inférence ultérieure) — perspective de Spadzze, pas
    # l'ADC ennemi. Reste basé sur le silver collecté.
    perso_root = rl.SILVER_DIR / rl.KIND_PERSONAL
    if perso_root.exists():
        for d in sorted(perso_root.iterdir()):
            games = rl.read_jsonl(d / "games.jsonl")
            adc = [g for g in games if g.get("role") == "BOTTOM"]
            rows += [game_to_row(g, None, f"personal:{d.name}") for g in adc]
            print(f"  personal/{d.name:<14}: {len(adc)} games ADC")

    df = pd.DataFrame(rows)
    # dédup par (match_id, puuid) : garde les perspectives distinctes d'une même game
    df = df.drop_duplicates(subset=["match_id", "puuid"]).reset_index(drop=True)
    df["rank_ord"] = df["rank"].map(RANK_ORD)
    df["high_elo"] = df["rank"].isin(HIGH_ELO).astype("Int64")
    df.loc[df["rank"].isna(), "high_elo"] = pd.NA  # perso : label inconnu

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_DIR / "adc_dataset.parquet", index=False)
    df.to_csv(DATASET_DIR / "adc_dataset.csv", index=False)

    ref = df[df["source"] == "referentiel"]
    print(f"\n✓ Dataset : {len(df)} games ({len(ref)} référentiel labellisés)")
    print(f"  Répartition rangs : {dict(ref['rank'].value_counts())}")
    print(f"  high_elo (GM+Chall=1) : {dict(ref['high_elo'].value_counts())}")
    print(f"  Écrit dans {DATASET_DIR}/adc_dataset.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
