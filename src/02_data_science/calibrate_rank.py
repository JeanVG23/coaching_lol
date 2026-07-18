#!/usr/bin/env python3
"""
DÉPRÉCIÉ — arrêté le 2026-07-18. Calibration du modèle per-game (déprécié) : non servie
en prod (le web utilise player_rank_calibration.json via calibrate_player_rank.py).
Conservé pour l'historique. Voir
docs/superpowers/specs/2026-07-18-gold-standard-eval-protocol-design.md.

02_data_science — calibration proba -> rang pour le placement ML web.

Le modèle high_elo (xgb+rf) est binaire (low M/D vs high GM/C) : il ne "connaît"
que 2 classes. Pour placer un joueur sur les 4 rangs (diamond/master/grandmaster/
challenger) dans l'UI web, on calibre : moyenne de la proba ensemble (xgb+rf,
mêmes membres que le SHAP-arbres) par rang réel sur le référentiel labellisé. Le
web (`web/backend/ml_rank.py`) compare ensuite la proba moyenne d'un joueur au
rang calibré le plus proche.

Sortie : data/05_model/rank_calibration.json
Usage : poetry run python3 src/02_data_science/calibrate_rank.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import numpy as np
import pandas as pd
import riotlib as rl

DATASET = rl.DATA / "04_dataset" / "adc_dataset.parquet"
MODEL_DIR = rl.DATA / "05_model"
RANKS = ["diamond", "master", "grandmaster", "challenger"]


def main() -> int:
    features = json.loads((MODEL_DIR / "features.json").read_text())
    models = {}
    for name in ("xgb", "rf"):
        with open(MODEL_DIR / f"{name}_highelo.pkl", "rb") as f:
            models[name] = pickle.load(f)

    df = pd.read_parquet(DATASET)
    ref = df[df["source"] == "referentiel"].copy()
    X = ref.reindex(columns=features)
    proba = np.mean([m.predict_proba(X)[:, 1] for m in models.values()], axis=0)
    ref["ensemble_proba"] = proba

    calibration = []
    print("  Calibration proba -> rang (référentiel) :")
    for rank in RANKS:
        sub = ref[ref["rank"] == rank]["ensemble_proba"]
        if not len(sub):
            continue
        row = {"rank": rank, "mean_proba": round(float(sub.mean()), 4),
               "median_proba": round(float(sub.median()), 4), "n": int(len(sub))}
        calibration.append(row)
        print(f"    {rank:<12} mean={row['mean_proba']:.3f} "
              f"median={row['median_proba']:.3f}  n={row['n']}")

    (MODEL_DIR / "rank_calibration.json").write_text(json.dumps(calibration, indent=2))
    print(f"\n✓ Calibration écrite dans {MODEL_DIR}/rank_calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
