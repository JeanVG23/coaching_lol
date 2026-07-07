"""Tests du module prod lp_metrics (src/02_data_science/), copie assumée du module
POC (le code prod n'importe pas depuis poc/, contrainte de la spec LP prod)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "02_data_science"))
import lp_metrics


def test_spearman_report_pooled_and_by_tier():
    df = pd.DataFrame({
        "rank": ["master"] * 4 + ["challenger"] * 4,
        "y_true": [10, 20, 30, 40, 500, 600, 700, 800],
        "y_pred": [12, 18, 33, 38, 510, 590, 710, 790],
    })
    report = lp_metrics.spearman_report(df)
    assert report["spearman_pooled"] > 0.9
    assert report["spearman_by_tier"]["master"]["spearman"] == 1.0
    assert report["spearman_by_tier"]["master"]["n"] == 4
    assert report["n_players_total"] == 8


def test_spearman_report_small_tier_and_degenerate_input_give_none():
    df = pd.DataFrame({
        "rank": ["master", "master", "challenger"],
        "y_true": [10, 20, 500],
        "y_pred": [15, 15, 510],
    })
    report = lp_metrics.spearman_report(df)
    assert report["spearman_by_tier"]["challenger"]["spearman"] is None  # n=1 < MIN_TIER_N
    assert report["spearman_by_tier"]["master"]["spearman"] is None      # y_pred constant on tier
    assert report["spearman_pooled"] is not None


def test_spearman_report_rmse():
    df = pd.DataFrame({"rank": ["master", "master"], "y_true": [0, 10], "y_pred": [0, 0]})
    report = lp_metrics.spearman_report(df)
    assert report["rmse_pooled"] == pytest.approx(7.07, abs=0.01)


def test_safe_spearman_none_on_constant_or_short():
    assert lp_metrics._safe_spearman([1, 1, 1], [1, 2, 3]) is None
    assert lp_metrics._safe_spearman([1, 2], [1, 2]) is None
    assert lp_metrics._safe_spearman([1, 2, 3], [10, 20, 30]) == 1.0
