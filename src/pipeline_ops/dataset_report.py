#!/usr/bin/env python3
"""État des lieux des datasets ML (per-game + per-player). 0 appel API.

Répond en une commande aux questions de visibilité : combien de games, de joueurs
ADC, répartition par rang / classe high-low / issue, profondeur games-par-joueur
(seuils du pipeline per-player), et cohérence avec le dernier modèle entraîné
(dérive dataset <-> player_metrics.json).

⚠ Le 50/50 win/loss du per-game est MÉCANIQUE (les 2 ADC de chaque game sont
extraits -> 1 win + 1 loss par game) : c'est une propriété de construction,
pas une mesure de qualité de collecte.

Usage : poetry run python3 src/pipeline_ops/dataset_report.py [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import pandas as pd
import riotlib as rl
import ml_features as mf

DATASET_DIR = rl.DATA / "04_dataset"
MODEL_DIR = rl.DATA / "05_model"
HIGH_ELO = {"grandmaster", "challenger"}          # même définition que le train
THRESHOLDS = (5, 10, 15, 20, 30)
MIN_PLAYER_GAMES = 15                             # aligné build_player_dataset


def per_game_stats(df: pd.DataFrame) -> dict:
    """Dataset per-game complet -> volumes et équilibres (stats rang/win/classe
    calculées sur le référentiel seul : le perso n'a pas de rang mesuré)."""
    ref = df[df["source"] == "referentiel"]
    wins = int(ref["win"].sum())
    high = int(ref["rank"].isin(HIGH_ELO).sum())
    return {
        "rows_total": int(len(df)),
        "by_source": {k: int(v) for k, v in df["source"].value_counts().items()},
        "n_games": int(ref["match_id"].nunique()),
        "n_players": int(ref["puuid"].nunique()),
        "by_rank": {k: int(v) for k, v in ref["rank"].value_counts().items()},
        "win_balance": {"win": wins, "loss": int(len(ref) - wins)},
        "class_balance": {"high": high, "low": int(len(ref) - high)},
    }


def depth_stats(ref: pd.DataFrame, thresholds=THRESHOLDS,
                min_games: int = MIN_PLAYER_GAMES) -> dict:
    """Profondeur games-par-joueur du référentiel + joueurs qualifiés per-player
    (>= min_games), rang résolu au mode comme le train (ml_features.resolve_rank)."""
    counts = ref.groupby("puuid").size()
    qualified = ref[ref["puuid"].isin(counts[counts >= min_games].index)]
    by_rank = (qualified.groupby("puuid").apply(mf.resolve_rank, include_groups=False)
               if len(qualified) else pd.Series(dtype=object))
    return {
        "games_per_player": {
            "mean": round(float(counts.mean()), 1),
            "median": float(counts.median()),
            "p25": float(counts.quantile(0.25)),
            "p75": float(counts.quantile(0.75)),
            "max": int(counts.max()),
        },
        "at_least": {int(t): int((counts >= t).sum()) for t in thresholds},
        "min_games": int(min_games),
        "qualified_by_rank": {k: int(v) for k, v in by_rank.value_counts().items()},
    }


_DAY_MS = 86_400_000


def temporal_stats(ref: pd.DataFrame, now_ms: int | None = None) -> dict:
    """Fenêtre temporelle du référentiel : répartition par patch et âge des games
    (le rang d'un joueur est celui mesuré à la collecte — plus les games sont
    vieilles, plus le label rang est approximatif). Dégradation propre si le
    parquet date d'avant l'ajout des colonnes patch/game_ts."""
    if "game_ts" not in ref.columns or ref["game_ts"].notna().sum() == 0:
        return {"available": False}
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    ts = ref["game_ts"].dropna()
    age = (now_ms - ts) / _DAY_MS
    by_patch = (ref["patch"].value_counts() if "patch" in ref.columns
                else pd.Series(dtype=int))
    return {
        "available": True,
        "by_patch": {str(k): int(v) for k, v in by_patch.items()},
        "age_days": {"newest": round(float(age.min()), 1),
                     "median": round(float(age.median()), 1),
                     "oldest": round(float(age.max()), 1)},
        "missing_ts": int(ref["game_ts"].isna().sum()),
    }


def per_player_stats(pdf: pd.DataFrame) -> dict:
    """Dataset per-player (déjà équilibré) -> volumes, classes, et composition
    intra-classe (le rang dominant de chaque classe et sa part : dit quelle
    frontière de rang le modèle apprend réellement)."""
    intra = {}
    for label, is_high in (("high", 1), ("low", 0)):
        cls = pdf[pdf["high_elo"] == is_high]["rank"].value_counts()
        if len(cls):
            intra[label] = {"dominant": str(cls.index[0]),
                            "share": round(float(cls.iloc[0] / cls.sum()), 4)}
    return {
        "n_players": int(len(pdf)),
        "by_rank": {k: int(v) for k, v in pdf["rank"].value_counts().items()},
        "class_balance": {"high": int((pdf["high_elo"] == 1).sum()),
                          "low": int((pdf["high_elo"] == 0).sum())},
        "intra_class": intra,
        "n_games_median_by_rank": {k: float(v) for k, v in
                                   pdf.groupby("rank")["n_games"].median().items()},
    }


def model_crosscheck(metrics: dict | None, n_players_dataset: int) -> dict:
    """Compare le dataset per-player actuel au dernier modèle entraîné : une
    dérive (n différents) signifie que player_metrics.json / les .pkl servis
    ne reflètent plus le dataset -> ré-entraîner avant d'en citer les chiffres."""
    if metrics is None:
        return {"available": False}
    return {
        "available": True,
        "n_players_model": int(metrics["n_players"]),
        "auc_cv": float(metrics["auc_cv"]),
        "n_players_dataset": int(n_players_dataset),
        "drift": int(metrics["n_players"]) != int(n_players_dataset),
    }


def _pct(part: int, total: int) -> str:
    return f"{100 * part / total:.1f}%" if total else "n/a"


def render(report: dict) -> str:
    pg, dp, tp, pp, md = (report["per_game"], report["depth"], report["temporal"],
                          report["per_player"], report["model"])
    lines = ["=== Dataset per-game (1 ligne = 1 ADC d'une game) ==="]
    lines.append(f"  lignes : {pg['rows_total']}  "
                 f"({', '.join(f'{k} {v}' for k, v in pg['by_source'].items())})")
    lines.append(f"  référentiel : {pg['n_games']} games uniques, "
                 f"{pg['n_players']} joueurs ADC uniques")
    lines.append("  par rang : " + ", ".join(
        f"{k} {v} ({_pct(v, sum(pg['by_rank'].values()))})"
        for k, v in pg["by_rank"].items()))
    wb, cb = pg["win_balance"], pg["class_balance"]
    lines.append(f"  win/loss : {wb['win']}/{wb['loss']} "
                 "(50/50 mécanique : 2 ADC extraits par game)")
    lines.append(f"  high/low elo : {cb['high']}/{cb['low']} "
                 f"({_pct(cb['high'], cb['high'] + cb['low'])} high)")

    g = dp["games_per_player"]
    lines.append("\n=== Profondeur games/joueur (référentiel) ===")
    lines.append(f"  médiane {g['median']:.0f}, p25 {g['p25']:.0f}, p75 {g['p75']:.0f}, "
                 f"moyenne {g['mean']}, max {g['max']}")
    lines.append("  seuils : " + ", ".join(
        f">={t} : {n}" for t, n in dp["at_least"].items()))
    lines.append(f"  qualifiés per-player (>= {dp['min_games']} games, rang au mode) : "
                 + ", ".join(f"{k} {v}" for k, v in dp["qualified_by_rank"].items()))

    lines.append("\n=== Fenêtre temporelle (référentiel) ===")
    if not tp["available"]:
        lines.append("  colonnes temporelles absentes — relancer build_dataset.py "
                     "(parquet antérieur à l'ajout patch/game_ts)")
    else:
        lines.append("  par patch : " + ", ".join(
            f"{k} {v}" for k, v in tp["by_patch"].items()))
        a = tp["age_days"]
        lines.append(f"  âge des games : plus récente {a['newest']:.0f} j, "
                     f"médiane {a['median']:.0f} j, plus vieille {a['oldest']:.0f} j")
        if tp["missing_ts"]:
            lines.append(f"  ⚠ {tp['missing_ts']} games sans game_ts")

    lines.append("\n=== Dataset per-player (équilibré, 1 ligne = 1 joueur) ===")
    lines.append(f"  joueurs : {pp['n_players']}  "
                 f"(high {pp['class_balance']['high']} / low {pp['class_balance']['low']})")
    lines.append("  par rang : " + ", ".join(
        f"{k} {v}" for k, v in pp["by_rank"].items()))
    for label, d in pp["intra_class"].items():
        lines.append(f"  classe {label} : dominée par {d['dominant']} ({d['share']:.0%})")
    lines.append("  n_games médian par rang : " + ", ".join(
        f"{k} {v:.0f}" for k, v in pp["n_games_median_by_rank"].items()))

    lines.append("\n=== Modèle per-player entraîné ===")
    if not md["available"]:
        lines.append("  player_metrics.json absent — modèle jamais entraîné ?")
    else:
        lines.append(f"  AUC_cv {md['auc_cv']} sur {md['n_players_model']} joueurs")
        if md["drift"]:
            lines.append(f"  ⚠ DÉRIVE : dataset à {md['n_players_dataset']} joueurs "
                         f"vs modèle entraîné sur {md['n_players_model']} — "
                         "ré-entraîner avant de citer/servir ces chiffres")
        else:
            lines.append("  ✓ modèle aligné sur le dataset courant")
    return "\n".join(lines)


def build_report() -> dict:
    df = pd.read_parquet(DATASET_DIR / "adc_dataset.parquet")
    pdf = pd.read_parquet(DATASET_DIR / "adc_player_dataset.parquet")
    metrics_path = MODEL_DIR / "player_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    ref = df[df["source"] == "referentiel"]
    return {
        "per_game": per_game_stats(df),
        "depth": depth_stats(ref),
        "temporal": temporal_stats(ref),
        "per_player": per_player_stats(pdf),
        "model": model_crosscheck(metrics, len(pdf)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                    help="sortie JSON (pour scripter/comparer entre densifications)")
    args = ap.parse_args()
    report = build_report()
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json
          else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
