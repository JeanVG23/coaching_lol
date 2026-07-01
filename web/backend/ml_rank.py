"""Estime le rang du joueur via l'ensemble ML (xgb+rf) entraîné sur le référentiel
high-elo, appliqué aux dernières games ADC (BOTTOM) du joueur.

Le modèle est binaire (low M/D vs high GM/C) : on place le joueur sur les 4 rangs
en comparant sa probabilité moyenne à une calibration par rang (mean_proba par rang
sur le référentiel), précalculée hors-ligne par
`src/02_data_science/calibrate_rank.py` et écrite dans
`data/05_model/rank_calibration.json`.

On réutilise xgb+rf (pas l'EBM) : mêmes membres que le SHAP-arbres déjà exposé côté
web (cf. shap_analysis.py `sv_ensemble`), et ça évite d'embarquer `interpret`
(lourd) dans l'image de prod pour ce seul usage.
"""
from __future__ import annotations

import functools
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import riotlib as rl

DATA_ENG = Path(__file__).resolve().parents[2] / "src" / "01_data_engineering"
if str(DATA_ENG) not in sys.path:
    sys.path.insert(0, str(DATA_ENG))
import build_dataset  # noqa: E402

MODEL_DIR = rl.DATA / "05_model"
MIN_ADC_GAMES = 3


@functools.lru_cache(maxsize=1)
def _load_models() -> dict:
    models = {}
    for name in ("xgb", "rf"):
        with open(MODEL_DIR / f"{name}_highelo.pkl", "rb") as f:
            models[name] = pickle.load(f)
    return models


@functools.lru_cache(maxsize=1)
def _load_features() -> list[str]:
    return json.loads((MODEL_DIR / "features.json").read_text())


@functools.lru_cache(maxsize=1)
def _load_calibration() -> list[dict]:
    return json.loads((MODEL_DIR / "rank_calibration.json").read_text())


def _game_proba(game: dict, models: dict, features: list[str]) -> float:
    row = build_dataset.game_to_row(game, rank=None, source="inference")
    # astype(float) : une seule ligne avec un None isolé (ex. pos_* absente sur
    # game courte) reste en dtype object sans valeur de comparaison pour upcaster
    # en float -> XGBoost refuse. Sur le dataset d'entraînement (des milliers de
    # lignes) pandas upcast déjà tout seul, donc invisible côté train_ensemble.py.
    X = pd.DataFrame([row]).reindex(columns=features).astype(float)
    probs = [m.predict_proba(X)[0, 1] for m in models.values()]
    return sum(probs) / len(probs)


def predict_rank(games: list[dict]) -> dict | None:
    """None si moins de MIN_ADC_GAMES games ADC (BOTTOM) dans l'historique fourni."""
    adc_games = [g for g in games if g.get("role") == "BOTTOM"]
    if len(adc_games) < MIN_ADC_GAMES:
        return None
    models = _load_models()
    features = _load_features()
    probas = [_game_proba(g, models, features) for g in adc_games]
    player_proba = float(sum(probas) / len(probas))
    calibration = _load_calibration()
    closest = min(calibration, key=lambda c: abs(c["mean_proba"] - player_proba))
    return {
        "predicted_rank": closest["rank"],
        "proba": round(player_proba, 4),
        "n_games_used": len(adc_games),
    }
