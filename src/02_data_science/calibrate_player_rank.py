#!/usr/bin/env python3
"""
02_data_science — calibration proba -> rang pour le modèle per-player.

Miroir de calibrate_rank.py, appliqué aux modèles/dataset per-player. Le modèle
player_highelo est binaire (low M/D vs high GM/C) : on calibre la proba moyenne
ensemble (xgb+rf) par rang réel sur le dataset per-player, pour placer un joueur sur
les 4 rangs dans web/backend/ml_rank.py.

Sortie : data/05_model/player_rank_calibration.json
Usage : .venv/bin/python src/02_data_science/calibrate_player_rank.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # riotlib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core")) # (cohérent avec les autres scripts du pipeline, cf. Task 3/4)
import numpy as np
import pandas as pd
import riotlib as rl

DATASET = rl.DATA / "04_dataset" / "adc_player_dataset.parquet"
MODEL_DIR = rl.DATA / "05_model"
RANKS = ["diamond", "master", "grandmaster", "challenger"]


def main() -> int:
    features = json.loads((MODEL_DIR / "player_features.json").read_text())
    models = {}
    for name in ("xgb", "rf"):
        with open(MODEL_DIR / f"{name}_player_highelo.pkl", "rb") as f:
            models[name] = pickle.load(f)

    df = pd.read_parquet(DATASET)
    X = df.reindex(columns=features)
    proba = np.mean([m.predict_proba(X)[:, 1] for m in models.values()], axis=0)
    df["ensemble_proba"] = proba

    calibration = []
    print("  Calibration proba -> rang (per-player) :")
    for rank in RANKS:
        sub = df[df["rank"] == rank]["ensemble_proba"]
        if not len(sub):
            continue
        row = {"rank": rank, "mean_proba": round(float(sub.mean()), 4),
               "median_proba": round(float(sub.median()), 4), "n": int(len(sub))}
        calibration.append(row)
        print(f"    {rank:<12} mean={row['mean_proba']:.3f} "
              f"median={row['median_proba']:.3f}  n={row['n']}")

    (MODEL_DIR / "player_rank_calibration.json").write_text(json.dumps(calibration, indent=2))
    print(f"\n✓ Calibration écrite dans {MODEL_DIR}/player_rank_calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
