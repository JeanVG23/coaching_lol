"""Tests du chemin LP hybride de web/backend/ml_rank.py (helpers purs, modèles
mockés — le placement binaire existant n'est pas re-testé ici)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web" / "backend"))
import ml_rank


class _FakeReg:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value]


_AGG = {"csm10__mean": 6.0}
_FEATURES = ["csm10__mean"]


def test_predict_lp_none_when_models_missing(monkeypatch):
    monkeypatch.setattr(ml_rank, "_load_lp_bundle", lambda: None)
    assert ml_rank.predict_lp(_AGG) is None


def test_predict_lp_averages_ensemble_and_rounds(monkeypatch):
    bundle = ([_FakeReg(100.0), _FakeReg(200.0), _FakeReg(310.0)], _FEATURES)
    monkeypatch.setattr(ml_rank, "_load_lp_bundle", lambda: bundle)
    assert ml_rank.predict_lp(_AGG) == 203   # mean(100, 200, 310) = 203.33 -> round


def test_predict_lp_clamped_at_zero(monkeypatch):
    bundle = ([_FakeReg(-50.0)], _FEATURES)
    monkeypatch.setattr(ml_rank, "_load_lp_bundle", lambda: bundle)
    assert ml_rank.predict_lp(_AGG) == 0


def test_attach_lp_only_for_apex_ranks(monkeypatch):
    bundle = ([_FakeReg(250.0)], _FEATURES)
    monkeypatch.setattr(ml_rank, "_load_lp_bundle", lambda: bundle)
    apex = ml_rank.attach_lp({"predicted_rank": "master"}, _AGG)
    assert apex["predicted_lp"] == 250
    diamond = ml_rank.attach_lp({"predicted_rank": "diamond"}, _AGG)
    assert "predicted_lp" not in diamond


def test_attach_lp_graceful_without_models(monkeypatch):
    monkeypatch.setattr(ml_rank, "_load_lp_bundle", lambda: None)
    result = ml_rank.attach_lp({"predicted_rank": "challenger"}, _AGG)
    assert "predicted_lp" not in result
    assert result["predicted_rank"] == "challenger"   # rien d'autre ne change
