#!/usr/bin/env python3
"""
audit_leakage — vérifie l'intégrité OOF du modèle high-elo et sonde l'AUC.

3 angles :
  1. Fuite OOF : per-model / per-fold AUC + vérif que le groupage (composantes
     connexes puuid ∪ match_id) isole vraiment joueurs et games.
  2. Null test : shuffle du label → AUC doit retomber à ~0.5 (sanité pipeline).
  3. Confound de durée : gameDuration (lu depuis le raw) prédit-il high_elo seul ?
     + décomposition counts (scalent avec durée) vs rates (normalisés/min).
Usage : .venv/bin/python src/audit_leakage.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import numpy as np
import pandas as pd
import riotlib as rl
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

DATASET = rl.DATA / "04_dataset" / "adc_dataset.parquet"
FEATURES = json.loads((rl.DATA / "05_model" / "features.json").read_text())

RATES = ["csm10", "csm14", "gpm10", "gpm14", "xppm10",
         "frac_behind", "frac_ahead", "avg_dragon_prox",
         "plates_diff_early", "frames_in_base_early",
         "kda_1v1", "kda_2v2"]  # normalisés ou bornés early
COUNTS = ["n_deaths", "deaths_early", "deaths_mid", "deaths_late",
          "deaths_solo", "deaths_teamfight", "deaths_early_jungle",
          "deaths_early_2v2", "kills_solo", "kills_2v2", "assists_2v2",
          "support_deaths_early"]


def leak_groups(df):
    n = len(df); parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[a] = b
    for idx in df.groupby("puuid").groups.values():
        idx = list(idx)
        for j in idx[1:]: union(idx[0], j)
    for idx in df.groupby("match_id").groups.values():
        idx = list(idx)
        for j in idx[1:]: union(idx[0], j)
    return np.array([find(i) for i in range(n)])


def cv_auc(X, y, groups, feats, seed=42):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X)); fold_aucs = []
    for tr, va in cv.split(X, y, groups):
        m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                              reg_lambda=1.0, eval_metric="logloss", tree_method="hist",
                              random_state=42)
        m.fit(X.iloc[tr][feats], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va][feats])[:, 1]
        fold_aucs.append(roc_auc_score(y.iloc[va], oof[va]))
    return roc_auc_score(y, oof), fold_aucs


def main():
    df = pd.read_parquet(DATASET)
    ref = df[df["source"] == "referentiel"].copy().reset_index(drop=True)
    X = ref[FEATURES]; y = ref["high_elo"].astype(int)
    groups = leak_groups(ref[["puuid", "match_id"]])

    print("=" * 64)
    print("1. INTÉGRITÉ OOF & GROUPAGE")
    print("=" * 64)
    print(f"  n={len(ref)}  puuids={ref['puuid'].nunique()}  match_ids={ref['match_id'].nunique()}")
    print(f"  composantes connexes={len(set(groups))}  taille max={pd.Series(groups).value_counts().max()}")
    print(f"  high={int(y.sum())}  low={int((1-y).sum())}  (base rate={y.mean():.2f})")

    # Vérif groupage : aucun puuid à cheval sur 2 folds
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    leaks = 0
    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        puuids_tr = set(ref.iloc[tr]["puuid"]); puuids_va = set(ref.iloc[va]["puuid"])
        overlap = puuids_tr & puuids_va
        matches_tr = set(ref.iloc[tr]["match_id"]); matches_va = set(ref.iloc[va]["match_id"])
        moverlap = matches_tr & matches_va
        leaks += len(overlap) + len(moverlap)
        print(f"  fold {fold}: train={len(tr)} val={len(va)} "
              f"high_val={int(y.iloc[va].sum())}  puuid_overlap={len(overlap)}  match_overlap={len(moverlap)}")
    print(f"  → fuites joueur/game détectées : {leaks} (doit être 0)")

    print("\n" + "=" * 64)
    print("2. AUC PAR MODÈLE (OOF) + spread par fold")
    print("=" * 64)
    auc_all, folds_all = cv_auc(X, y, groups, FEATURES)
    print(f"  xgb OOF AUC (toutes features) = {auc_all:.4f}")
    print(f"  AUC par fold : {[round(a,3) for a in folds_all]}  (spread={max(folds_all)-min(folds_all):.3f})")

    print("\n" + "=" * 64)
    print("3. NULL TEST — label mélangé (AUC doit → ~0.5)")
    print("=" * 64)
    rng = np.random.RandomState(42)
    y_shuf = pd.Series(rng.permutation(y.values))
    auc_null, _ = cv_auc(X, y_shuf, groups, FEATURES)
    print(f"  AUC label shuffled = {auc_null:.4f}  (attendu ~0.5)")

    print("\n" + "=" * 64)
    print("4. CONFOND DE DURÉE — gameDuration prédit-il high_elo seul ?")
    print("=" * 64)
    # gameDuration depuis le raw, join par match_id
    dur = {}
    for mid in ref["match_id"].unique():
        data = rl._read_raw(f"{mid}_match")
        if data is None:
            continue
        dur[mid] = data["info"]["gameDuration"]
    ref["dur"] = ref["match_id"].map(dur)
    n_dur = ref["dur"].notna().sum()
    print(f"  durations récupérées : {n_dur}/{len(ref)}")
    if n_dur > 0:
        d_hi = ref.loc[ref["high_elo"]==1, "dur"].mean()
        d_lo = ref.loc[ref["high_elo"]==0, "dur"].mean()
        print(f"  durée moyenne (s) : high={d_hi:.0f}  low={d_lo:.0f}  (diff={d_hi-d_lo:+.0f})")
        sub = ref.dropna(subset=["dur"]).reset_index(drop=True)
        g2 = leak_groups(sub[["puuid", "match_id"]])
        auc_dur, _ = cv_auc(sub, sub["high_elo"].astype(int), g2, ["dur"])
        print(f"  AUC modèle durée-seule = {auc_dur:.4f}")

    print("\n" + "=" * 64)
    print("5. DÉCOMPOSITION COUNTS vs RATES")
    print("=" * 64)
    auc_counts, _ = cv_auc(X, y, groups, COUNTS)
    auc_rates, _ = cv_auc(X, y, groups, RATES)
    print(f"  AUC counts-only ({len(COUNTS)} feats) = {auc_counts:.4f}")
    print(f"  AUC rates-only  ({len(RATES)} feats) = {auc_rates:.4f}")
    print(f"  AUC all         ({len(FEATURES)} feats) = {auc_all:.4f}")

    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    dur_str = f"{auc_dur:.3f}" if n_dur else "NA"
    print(f"  fuites={leaks}  null={auc_null:.3f}  durée-only={dur_str}  "
          f"counts={auc_counts:.3f}  rates={auc_rates:.3f}  all={auc_all:.3f}")


if __name__ == "__main__":
    sys.exit(main())