#!/usr/bin/env python3
"""
02_data_science — entraîne un XGBoost à séparer high-elo (GM+Chall) de low (master+diam).

But : PAS la précision (prédire le rang depuis UNE game est intrinsèquement bruité,
AUC ~0.65-0.75 attendu), mais obtenir un modèle dont SHAP révélera quelles features
de game séparent le plus les paliers. C'est la version data-driven et vérifiable du
coaching « qu'est-ce qui me sépare du rang au-dessus ».

Anti-fuite : on EXCLUT `win` et toute colonne dérivée du rang. Seules des features
comportementales/lane servent.

Sorties : data/05_model/xgb_highelo.json, metrics.json, features.json
Usage : .venv/bin/python src/02_data_science/train_xgb.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import riotlib as rl
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb

DATASET = rl.DATA / "04_dataset" / "adc_dataset.parquet"
MODEL_DIR = rl.DATA / "05_model"

FEATURES = [
    "gd10", "gd14", "gd20", "csd10", "csd14", "xpd10",
    "n_deaths", "deaths_early", "deaths_mid", "deaths_late",
    "frac_behind", "frac_ahead",
]


def make_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, eval_metric="logloss", tree_method="hist",
    )


def main() -> int:
    df = pd.read_parquet(DATASET)
    train = df[df["source"] == "referentiel"].copy()
    X = train[FEATURES]
    y = train["high_elo"].astype(int)
    print(f"  {len(train)} games référentiel | high_elo={int(y.sum())} / low={int((1-y).sum())}")

    # --- évaluation honnête en CV stratifiée (out-of-fold) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    acc = accuracy_score(y, (proba >= 0.5).astype(int))
    print(f"\n  CV out-of-fold : AUC={auc:.3f}  accuracy={acc:.3f}")
    print("  (AUC ~0.65-0.75 = normal : 1 game prédit mal le rang ; on vise SHAP, pas l'accuracy)")
    print(classification_report(y, (proba >= 0.5).astype(int),
                                target_names=["low(M/D)", "high(GM/C)"], digits=3))

    # --- modèle final sur toutes les données (pour SHAP) ---
    model = make_model().fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # pickle plutôt que save_model (bug _estimator_type du wrapper sklearn xgboost 2.1.x)
    with open(MODEL_DIR / "xgb_highelo.pkl", "wb") as f:
        pickle.dump(model, f)
    (MODEL_DIR / "features.json").write_text(json.dumps(FEATURES, indent=2))
    (MODEL_DIR / "metrics.json").write_text(json.dumps({
        "auc_cv": round(auc, 4), "acc_cv": round(acc, 4),
        "n_train": len(train), "n_high": int(y.sum()), "n_low": int((1 - y).sum()),
        "features": FEATURES,
    }, indent=2))

    # aperçu importance native (gain) — SHAP affinera dans 03
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])
    print("  Importance native (gain) — top 6 :")
    for f, v in imp[:6]:
        print(f"    {f:<14} {v:.3f}")
    print(f"\n✓ Modèle écrit dans {MODEL_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
