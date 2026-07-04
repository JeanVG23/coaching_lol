"""Tests de ml_rank.predict_rank — agrégation per-player (mean/std/p10/p50/p90) +
mapping calibration. Modèles/calibration mockés (pas de dépendance aux .pkl réels)."""
import numpy as np
import pytest

import ml_rank


class FakeModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]])


ADC_GAME = {
    "match_id": "EUW1_1", "role": "BOTTOM", "champion": "Zeri", "win": True,
    "lane": {"csm10": 8.0, "csm14": 7.5, "gpm10": 400, "gpm14": 420, "xppm10": 500},
    "deaths": [], "kills": [], "assists": [],
    "avg_dragon_prox": 0.5, "support_deaths_early": 0,
    "plates_diff_early": 0, "frames_in_base_early": 0,
    "position": {},
}
NON_ADC_GAME = {**ADC_GAME, "role": "MIDDLE"}

CALIBRATION = [
    {"rank": "diamond", "mean_proba": 0.2},
    {"rank": "master", "mean_proba": 0.4},
    {"rank": "grandmaster", "mean_proba": 0.65},
    {"rank": "challenger", "mean_proba": 0.85},
]


def _patch_loaders(monkeypatch, proba):
    monkeypatch.setattr(ml_rank, "_load_models",
                        lambda: {"xgb": FakeModel(proba), "rf": FakeModel(proba)})
    monkeypatch.setattr(ml_rank, "_load_features", lambda: ["csm10__mean"])
    monkeypatch.setattr(ml_rank, "_load_calibration", lambda: CALIBRATION)


def test_predict_rank_none_when_not_enough_adc_games(monkeypatch):
    _patch_loaders(monkeypatch, 0.5)
    n = ml_rank.MIN_ADC_GAMES - 1
    result = ml_rank.predict_rank([ADC_GAME] * n)
    assert result is None


def test_predict_rank_none_when_no_adc_games(monkeypatch):
    _patch_loaders(monkeypatch, 0.5)
    result = ml_rank.predict_rank([NON_ADC_GAME] * ml_rank.MIN_ADC_GAMES)
    assert result is None


def test_predict_rank_maps_to_closest_calibrated_rank(monkeypatch):
    _patch_loaders(monkeypatch, 0.8)
    n = ml_rank.MIN_ADC_GAMES
    result = ml_rank.predict_rank([ADC_GAME] * n)
    assert result["predicted_rank"] == "challenger"
    assert result["n_games_used"] == n
    assert result["proba"] == pytest.approx(0.8)


def test_predict_rank_filters_non_adc_games(monkeypatch):
    _patch_loaders(monkeypatch, 0.3)
    n = ml_rank.MIN_ADC_GAMES
    result = ml_rank.predict_rank([ADC_GAME] * n + [NON_ADC_GAME] * 2)
    assert result["n_games_used"] == n
    assert result["predicted_rank"] == "diamond"
