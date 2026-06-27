#!/usr/bin/env python3
"""
03_data_analyse — SHAP sur le modèle high-elo.

1. SHAP global : quelles features séparent le plus high-elo de low-elo (ranking + plots).
2. SHAP appliqué aux games de Spadzze : quelles features le tirent vers le LOW-elo
   (= ses axes de progression prioritaires, data-driven).

Sorties : data/06_shap/{shap_bar.png, shap_beeswarm.png, ranking.json, spadzze_drivers.json}
Usage : .venv/bin/python src/03_data_analyse/shap_analysis.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import riotlib as rl
import shap
import xgboost as xgb

DATASET = rl.DATA / "04_dataset" / "adc_dataset.parquet"
MODEL_DIR = rl.DATA / "05_model"
OUT = rl.DATA / "06_shap"


def main() -> int:
    FEATURES = json.loads((MODEL_DIR / "features.json").read_text())
    with open(MODEL_DIR / "xgb_highelo.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_parquet(DATASET)
    ref = df[df["source"] == "referentiel"].copy()
    Xref = ref[FEATURES]

    explainer = shap.TreeExplainer(model)
    sv = explainer(Xref)  # shap values sur le référentiel

    OUT.mkdir(parents=True, exist_ok=True)

    # --- ranking global (mean |shap|) ---
    mean_abs = np.abs(sv.values).mean(axis=0)
    ranking = sorted(zip(FEATURES, mean_abs), key=lambda t: -t[1])
    print("  SHAP global — features qui séparent le plus high/low elo :")
    for f, v in ranking:
        print(f"    {f:<14} {v:.3f}")
    (OUT / "ranking.json").write_text(json.dumps(
        [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in ranking], indent=2))

    # --- plots ---
    plt.figure()
    shap.plots.bar(sv, max_display=len(FEATURES), show=False)
    plt.tight_layout(); plt.savefig(OUT / "shap_bar.png", dpi=130); plt.close()

    plt.figure()
    shap.plots.beeswarm(sv, max_display=len(FEATURES), show=False)
    plt.tight_layout(); plt.savefig(OUT / "shap_beeswarm.png", dpi=130); plt.close()

    # --- SHAP sur Spadzze : qu'est-ce qui le tire vers le low-elo ? ---
    spad = df[df["source"].str.startswith("personal:spadzze", na=False)].copy()
    if len(spad):
        sv_spad = explainer(spad[FEATURES])
        # contribution moyenne signée (négatif = pousse vers low-elo)
        mean_signed = sv_spad.values.mean(axis=0)
        drivers = sorted(zip(FEATURES, mean_signed), key=lambda t: t[1])  # plus négatif d'abord
        print(f"\n  SHAP Spadzze ({len(spad)} games) — ce qui te tire vers le LOW-elo (négatif) :")
        for f, v in drivers:
            arrow = "↓ low" if v < 0 else "↑ high"
            print(f"    {f:<14} {v:+.3f}  {arrow}")
        (OUT / "spadzze_drivers.json").write_text(json.dumps(
            [{"feature": f, "mean_shap": round(float(v), 4)} for f, v in drivers], indent=2))

    print(f"\n✓ SHAP écrit dans {OUT}/ (bar + beeswarm + ranking + spadzze_drivers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
