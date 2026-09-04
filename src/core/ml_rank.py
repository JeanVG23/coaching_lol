"""Estime le rang du joueur via l'ensemble ML per-player (xgb+rf) entraîné sur le
référentiel high-elo, appliqué aux features agrégées (mean/std/p10/p50/p90) des
dernières games ADC (BOTTOM) du joueur — cf.
docs/superpowers/specs/2026-07-03-per-player-consistency-design.md et
poc/per_player_hypothesis.py (hypothèse "constance/plancher" validée, +0.12 AUC vs
un modèle per-game moyenné).

Le modèle est binaire (low M/D vs high GM/C) : on place le joueur sur les 4 rangs en
comparant sa probabilité à une calibration par rang (mean_proba par rang sur le
dataset per-player), précalculée hors-ligne par
`src/02_data_science/calibrate_player_rank.py` et écrite dans
`data/05_model/player_rank_calibration.json`.

MIN_ADC_GAMES = 15 (aligné sur MIN_PLAYER_GAMES du dataset d'entraînement, cf.
build_player_dataset.py) : en dessous, pas de rang (pas de fallback sur un autre
chemin de code — décision actée en brainstorming). Relevé depuis 5 : le modèle est
entraîné sur des agrégats mean/std/p10/p50/p90 calculés sur >= 15 games — le nourrir
à l'inférence avec un agrégat 5 games introduirait un décalage train/serve (la
dispersion mesurée sur 5 games est dominée par le bruit, pas la vraie régularité).

Hybride LP (2026-07-07) : si le rang placé est apex (master/GM/chall) et que les
regressors LP ({xgb,rf,ebm}_player_lp.pkl, cf. train_player_lp.py) sont présents,
le retour porte en plus "predicted_lp" (LP estimé sur l'échelle continue
master->challenger). Diamond n'en a jamais (divisions avec reset, hors échelle).
"""
from __future__ import annotations

import functools
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import riotlib as rl

# Voisins de src/core/ : resolus par le sys.path qui a servi a importer ce module.
import ml_features as mf

DATA_ENG = Path(__file__).resolve().parents[2] / "src" / "01_data_engineering"
if str(DATA_ENG) not in sys.path:
    sys.path.insert(0, str(DATA_ENG))
import build_dataset  # noqa: E402

MODEL_DIR = rl.DATA / "05_model"
MIN_ADC_GAMES = 15


@functools.lru_cache(maxsize=1)
def _load_models() -> dict:
    models = {}
    for name in ("xgb", "rf"):
        with open(MODEL_DIR / f"{name}_player_highelo.pkl", "rb") as f:
            models[name] = pickle.load(f)
    return models


@functools.lru_cache(maxsize=1)
def _load_features() -> list[str]:
    return json.loads((MODEL_DIR / "player_features.json").read_text())


@functools.lru_cache(maxsize=1)
def _load_calibration() -> list[dict]:
    return json.loads((MODEL_DIR / "player_rank_calibration.json").read_text())


APEX_RANKS = {"master", "grandmaster", "challenger"}


@functools.lru_cache(maxsize=1)
def _load_lp_bundle() -> tuple[list, list[str]] | None:
    """Regressors LP + ordre des features, ou None si les artefacts sont absents
    (modèle LP pas encore entraîné sur cette machine) — dégradation propre : le
    placement binaire suffit, pas de crash ni de log bruyant."""
    try:
        models = []
        for name in ("xgb", "rf", "ebm"):
            with open(MODEL_DIR / f"{name}_player_lp.pkl", "rb") as f:
                models.append(pickle.load(f))
        features = json.loads((MODEL_DIR / "player_lp_features.json").read_text())
        return models, features
    except FileNotFoundError:
        return None


def predict_lp(agg: dict) -> int | None:
    """LP estimé (moyenne de l'ensemble, arrondi, borné >= 0) depuis l'agrégat de
    features per-player déjà calculé par predict_rank. None si modèles absents.
    N'a de sens que pour un joueur placé apex (échelle LP continue master->chall,
    diamond hors échelle — divisions avec reset)."""
    bundle = _load_lp_bundle()
    if bundle is None:
        return None
    models, features = bundle
    X = pd.DataFrame([agg]).reindex(columns=features).astype(float)
    preds = [float(m.predict(X)[0]) for m in models]
    return max(0, round(sum(preds) / len(preds)))


def attach_lp(result: dict, agg: dict) -> dict:
    """Ajoute predicted_lp au retour de predict_rank quand le rang placé est apex
    ET que le modèle LP est disponible. Purement additif : ne touche à rien d'autre."""
    if result.get("predicted_rank") in APEX_RANKS:
        lp = predict_lp(agg)
        if lp is not None:
            result["predicted_lp"] = lp
    return result


def predict_rank(games: list[dict]) -> dict | None:
    """None si moins de MIN_ADC_GAMES games ADC (BOTTOM) dans l'historique fourni."""
    adc_games = [g for g in games if g.get("role") == "BOTTOM"]
    if len(adc_games) < MIN_ADC_GAMES:
        return None
    rows = pd.DataFrame([
        build_dataset.game_to_row(g, rank=None, source="inference") for g in adc_games
    ])
    agg = mf.aggregate_player_features(rows, mf.FEATURES)

    models = _load_models()
    features = _load_features()
    # astype(float) : une seule ligne agrégée avec un NaN isolé reste en dtype object
    # sans valeur de comparaison pour upcaster en float -> XGBoost refuse (même
    # raisonnement que l'ancien _game_proba, cf. historique du fichier).
    X = pd.DataFrame([agg]).reindex(columns=features).astype(float)
    probs = [m.predict_proba(X)[0, 1] for m in models.values()]
    player_proba = float(sum(probs) / len(probs))

    calibration = _load_calibration()
    closest = min(calibration, key=lambda c: abs(c["mean_proba"] - player_proba))
    return attach_lp({
        "predicted_rank": closest["rank"],
        "proba": round(player_proba, 4),
        "n_games_used": len(adc_games),
    }, agg)
