"""Tests des fonctions pures du train LP (sample_configs, search_best). La CV
complète et le SHAP sont vérifiés par exécution réelle (Task 6 du plan)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "02_data_science"))
import train_player_lp as tlp


def test_sample_configs_deterministic_and_within_grid():
    grid = {"a": [1, 2, 3], "b": [10, 20]}
    c1 = tlp.sample_configs(grid, n=4, seed=42)
    c2 = tlp.sample_configs(grid, n=4, seed=42)
    assert c1 == c2                      # déterministe à graine fixe
    assert len(c1) == 4
    assert len({tuple(sorted(c.items())) for c in c1}) == 4   # sans doublon
    for c in c1:
        assert c["a"] in grid["a"] and c["b"] in grid["b"]


def test_sample_configs_returns_full_product_when_small():
    grid = {"a": [1, 2], "b": [10]}
    configs = tlp.sample_configs(grid, n=50, seed=42)
    assert len(configs) == 2             # produit cartésien < n -> tout


def test_search_best_picks_highest_spearman(monkeypatch):
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fake_oof = {
        (("depth", 1),): np.array([5.0, 4.0, 3.0, 2.0, 1.0]),  # spearman -1
        (("depth", 2),): np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # spearman +1
    }
    monkeypatch.setattr(
        tlp, "oof_predictions",
        lambda name, config, folds, n_rows: fake_oof[tuple(sorted(config.items()))])
    spec = {"n_configs": 2, "grid": {"depth": [1, 2]}}
    best = tlp.search_best("xgb", spec, folds=[], y_true=y_true, n_rows=5)
    assert best["config"] == {"depth": 2}
    assert best["spearman"] == 1.0


def test_search_best_survives_degenerate_predictions(monkeypatch):
    # une config qui prédit une constante (spearman indéfini -> None) ne doit ni
    # crasher ni gagner face à une config avec un vrai spearman
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fake_oof = {
        (("depth", 1),): np.zeros(5),                            # constant -> None
        (("depth", 2),): np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    monkeypatch.setattr(
        tlp, "oof_predictions",
        lambda name, config, folds, n_rows: fake_oof[tuple(sorted(config.items()))])
    spec = {"n_configs": 2, "grid": {"depth": [1, 2]}}
    best = tlp.search_best("xgb", spec, folds=[], y_true=y_true, n_rows=5)
    assert best["config"] == {"depth": 2}
