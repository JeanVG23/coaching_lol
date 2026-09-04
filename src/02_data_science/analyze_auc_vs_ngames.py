#!/usr/bin/env python3
"""
02_data_science — analyse AUC du modèle per-player en fonction du nombre de games
agrégées par ligne (cap par joueur).

Question : comment se comporte l'AUC quand on fait varier N (le nombre de games
agrégées pour construire les features mean/std/p10/p50/p90 d'un joueur) ?

Deux courbes pour démêler les deux effets confondus :

1. « qualify=N, cap=N » — seuil de qualification = cap. Réaliste (tradeoff production) :
   plus N grandit, moins de joueurs qualifient (pool rétrécit), mais chaque ligne porte
   plus de signal. Montre le compromis AUC vs taille de dataset.

2. « pool fixe ≥50, cap=N » — isole l'effet PUR de la profondeur d'agrégation en gardant
   le MÊME pool de joueurs (ceux avec ≥50 games disponibles). Cap chaque joueur à N games
   (N ≤ 50). Si l'AUC monte avec N ici, c'est bien la profondeur qui parle, pas la
   composition du pool.

Méthode : reprend le purged CV de train_player_ensemble.py (mêmes 3 modèles, purge des
games partagées entre folds). Capping par échantillonnage aléatoire (seed fixe par
joueur) — pas de biais temporel, estimateur propre de « N games de signal ».

0 appel API (relit adc_dataset.parquet).
Sortie : data/05_model/auc_vs_ngames.json + .png
Usage : poetry run python3 src/02_data_science/analyze_auc_vs_ngames.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core")) # ml_features
sys.path.insert(0, str(Path(__file__).resolve().parent))  # cv_common
import numpy as np
import pandas as pd
import riotlib as rl
from ranks import HIGH_ELO
import ml_features as mf
from cv_common import make_models, purged_train_features
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATASET_PER_GAME = rl.DATA / "04_dataset" / "adc_dataset.parquet"
MODEL_DIR = rl.DATA / "05_model"
FEATURES = mf.player_feature_names(mf.FEATURES)
SEED = 42


def cap_player_games(ref: pd.DataFrame, n_cap: int | None,
                     pool_puuids: set[str] | None = None,
                     seed: int = SEED) -> pd.DataFrame:
    """Cappe chaque joueur à n_cap games (échantillonnage aléatoire, seed fixe).
    n_cap=None -> tout l'historique disponible. Optionnellement restreint au pool."""
    sub = ref if pool_puuids is None else ref[ref["puuid"].isin(pool_puuids)]
    if n_cap is None:
        return sub
    # Index accumulés puis un seul .loc : le pd.concat par joueur construisait
    # ~1 000 petits DataFrames par appel, et l'appelant boucle sur les caps.
    rng = np.random.RandomState(seed)
    keep = []
    for _puuid, idx in sub.groupby("puuid").indices.items():
        positions = sub.index[idx]
        if len(positions) > n_cap:
            positions = rng.choice(positions, size=n_cap, replace=False)
        keep.extend(positions)
    return sub.loc[keep]


def build_player_rows(ref: pd.DataFrame, min_games: int,
                      n_cap: int | None = None,
                      seed: int = SEED,
                      counts: pd.Series | None = None) -> pd.DataFrame:
    """Agrège ref (per-game) par joueur. Qualifie >= min_games games sur l'historique
    COMPLET. Rang résolu sur l'historique COMPLET (label fixe = vrai rang du joueur,
    sémantique production) — SEULES les features sont cappées à n_cap games. Isole
    proprement 'qualité des features à profondeur N' sans confondre avec la stabilité
    du label. Balance les classes (undersampling seed)."""
    if counts is None:
        counts = ref.groupby("puuid").size()
    qualified = set(counts[counts >= min_games].index)
    capped = cap_player_games(ref, n_cap, pool_puuids=qualified, seed=seed)
    by_puuid_cap = dict(tuple(capped.groupby("puuid")))
    rows = []
    for puuid, g_full in ref[ref["puuid"].isin(qualified)].groupby("puuid"):
        g_cap = by_puuid_cap.get(puuid, g_full)
        rec = {"puuid": puuid, "rank": mf.resolve_rank(g_full)}  # label fixe
        rec.update(mf.aggregate_player_features(g_cap, mf.FEATURES))  # features cappées
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["high_elo"] = out["rank"].isin(HIGH_ELO).astype(int)
    pos = out[out["high_elo"] == 1]
    neg = out[out["high_elo"] == 0]
    n_min = min(len(pos), len(neg))
    if n_min == 0:
        return out
    pos = pos.sample(n=n_min, random_state=seed)
    neg = neg.sample(n=n_min, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def run_cv(df: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Purged CV (5 folds) sur df (1 ligne/joueur, déjà balancé). ref = per-game CAPPÉ
    correspondant (pour la purge). Retourne AUC ensemble purgée + naïve + par modèle."""
    X = df.reindex(columns=FEATURES)
    y = df["rank"].isin(HIGH_ELO).astype(int)
    y_of = dict(zip(df["puuid"], y))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_p, oof_n = {n: np.zeros(len(X)) for n in make_models()}, {n: np.zeros(len(X)) for n in make_models()}
    for train_idx, val_idx in cv.split(X, y):
        X_val = X.iloc[val_idx]
        train_puuids = df["puuid"].iloc[train_idx].tolist()
        val_puuids = set(df["puuid"].iloc[val_idx])
        Xtr_p, _ = purged_train_features(ref, train_puuids, val_puuids)
        for name, model in make_models().items():
            model.fit(Xtr_p.reindex(columns=FEATURES),
                      Xtr_p["puuid"].map(y_of).astype(int))
            oof_p[name][val_idx] = model.predict_proba(X_val)[:, 1]
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_n[name][val_idx] = model.predict_proba(X_val)[:, 1]
    per_model = {n: round(roc_auc_score(y, p), 4) for n, p in oof_p.items()}
    auc_p = round(roc_auc_score(y, np.mean(list(oof_p.values()), axis=0)), 4)
    auc_n = round(roc_auc_score(y, np.mean(list(oof_n.values()), axis=0)), 4)
    return {"auc_purged": auc_p, "auc_naive": auc_n, "per_model_purged": per_model,
            "n_players": len(df), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}


def run_curve(ref_all: pd.DataFrame, ns: list[int],
              min_games) -> list[dict]:
    """Balaie les caps N pour une règle de qualification donnée.

    `min_games` : int fixe (courbe pool fixe) ou callable(n) (courbe qualify=N).
    Les deux courbes étaient deux boucles identiques au `min_games` près.
    """
    out = []
    print(f"  {'N':>4} {'joueurs':>8} {'pos/neg':>10} {'AUC purgée':>11} {'AUC naïve':>11}")
    for n in ns:
        qualify = min_games(n) if callable(min_games) else min_games
        df_n = build_player_rows(ref_all, min_games=qualify, n_cap=n)
        if df_n.empty or df_n["high_elo"].nunique() < 2:
            out.append({"n_games": n, "skipped": "no balanced classes"})
            print(f"  {n:>4} {'-':>8} (classes insuffisantes)")
            continue
        # ref cappé au même N pour la purge (pool = qualifiés)
        ref_cap = cap_player_games(ref_all, n, pool_puuids=set(df_n["puuid"]))
        r = run_cv(df_n, ref_cap)
        out.append(r | {"n_games": n})
        cls = f"{r['n_pos']}/{r['n_neg']}"
        print(f"  {n:>4} {r['n_players']:>8} {cls:>10} "
              f"{r['auc_purged']:>11.4f} {r['auc_naive']:>11.4f}")
    return out


def main() -> int:
    ref_all = pd.read_parquet(DATASET_PER_GAME)
    ref_all = ref_all[ref_all["source"] == "referentiel"].copy()
    counts = ref_all.groupby("puuid").size()
    print(f"  {len(ref_all)} games référentiel | {ref_all['puuid'].nunique()} joueurs uniques")
    print(f"  qualification: ≥15={int((counts>=15).sum())} ≥20={int((counts>=20).sum())} "
          f"≥25={int((counts>=25).sum())} ≥30={int((counts>=30).sum())} ≥40={int((counts>=40).sum())} "
          f"≥50={int((counts>=50).sum())} ≥75={int((counts>=75).sum())}")

    # --- Courbe 1 : qualify=N, cap=N (tradeoff réaliste) ---
    print("\n  Courbe 1 : qualify=N, cap=N (pool rétrécit quand N grandit)")
    curve1 = run_curve(ref_all, [15, 20, 25, 30, 40, 50, 75], min_games=lambda n: n)

    # --- Courbe 2 : pool fixe ≥50, cap=N (effet pur de la profondeur) ---
    pool50 = set(counts[counts >= 50].index)
    print(f"\n  Courbe 2 : pool fixe ≥50 ({len(pool50)} joueurs), cap=N "
          f"(effet pur de la profondeur d'agrégation)")
    curve2 = run_curve(ref_all, [15, 20, 25, 30, 40, 50], min_games=50)

    # --- Référence : production actuelle (qualify=15, cap=all) ---
    df_ref = build_player_rows(ref_all, min_games=15, n_cap=None)
    ref_ref = cap_player_games(ref_all, None, pool_puuids=set(df_ref["puuid"]))
    r_ref = run_cv(df_ref, ref_ref)
    print(f"\n  Référence production (qualify=15, cap=tout l'historique) : "
          f"AUC purgée={r_ref['auc_purged']:.4f} | {r_ref['n_players']} joueurs "
          f"({r_ref['n_pos']}/{r_ref['n_neg']})")

    out = {
        "curve1_qualify_eq_cap": curve1,
        "curve2_fixed_pool_ge50_cap": curve2,
        "reference_production_qualify15_cap_all": r_ref | {"n_games": "all"},
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "auc_vs_ngames.json").write_text(json.dumps(out, indent=2))
    print(f"\n✓ JSON écrit : {MODEL_DIR}/auc_vs_ngames.json")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        c1 = [(d["n_games"], d["auc_purged"], d["n_players"]) for d in curve1
              if "auc_purged" in d]
        ax1.plot([x[0] for x in c1], [x[1] for x in c1], "o-", color="#2E86AB", label="AUC purgée")
        ax1.axhline(r_ref["auc_purged"], color="#A23B72", ls="--", alpha=.7,
                    label=f"prod (cap=all)={r_ref['auc_purged']:.3f}")
        for x, auc, np_ in c1:
            ax1.annotate(f"{np_}j", (x, auc), textcoords="offset points",
                         xytext=(0, -14), ha="center", fontsize=8, color="#555")
        ax1.set_xlabel("N games agrégées par ligne (cap = seuil qualif.)")
        ax1.set_ylabel("AUC (ensemble, purged CV)")
        ax1.set_title("Courbe 1 : qualify=N, cap=N\n(pool rétrécit quand N grandit)")
        ax1.legend(); ax1.grid(alpha=.3); ax1.set_ylim(0.45, max(0.85, max(x[1] for x in c1)+.03))

        c2 = [(d["n_games"], d["auc_purged"]) for d in curve2 if "auc_purged" in d]
        ax2.plot([x[0] for x in c2], [x[1] for x in c2], "o-", color="#2E86AB", label="AUC purgée")
        ax2.axhline(r_ref["auc_purged"], color="#A23B72", ls="--", alpha=.7,
                    label=f"prod (cap=all)={r_ref['auc_purged']:.3f}")
        ax2.set_xlabel("N games agrégées par ligne (cap)")
        ax2.set_ylabel("AUC (ensemble, purged CV)")
        ax2.set_title(f"Courbe 2 : pool fixe ≥50 ({len(pool50)} joueurs), cap=N\n(effet pur de la profondeur)")
        ax2.legend(); ax2.grid(alpha=.3); ax2.set_ylim(0.45, max(0.85, max(x[1] for x in c2)+.03))
        fig.tight_layout()
        fig.savefig(MODEL_DIR / "auc_vs_ngames.png", dpi=130)
        print(f"✓ Plot écrit : {MODEL_DIR}/auc_vs_ngames.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())