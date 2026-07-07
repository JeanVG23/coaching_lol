"""Tests des fonctions pures du POC régression LP (poc/script/). Le reste (appels
API, CV complète) est vérifié par exécution réelle, cf.
docs/superpowers/specs/2026-07-07-lp-regression-poc-design.md."""
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src" / "core"))
sys.path.insert(0, str(_ROOT / "poc" / "script"))

import fetch_apex_lp
import lp_metrics


# --- build_lp_lookup ---------------------------------------------------------

def test_build_lp_lookup_merges_tiers_and_skips_missing_puuid():
    entries_by_tier = {
        "challenger": [{"puuid": "a", "leaguePoints": 800}],
        "master": [{"puuid": "b", "leaguePoints": 50},
                   {"puuid": None, "leaguePoints": 10}],
    }
    lookup = fetch_apex_lp.build_lp_lookup(entries_by_tier)
    assert lookup == {
        "a": {"tier": "challenger", "leaguePoints": 800},
        "b": {"tier": "master", "leaguePoints": 50},
    }


# --- spearman_report ----------------------------------------------------------

def test_spearman_report_pooled_and_by_tier():
    df = pd.DataFrame({
        "rank": ["master"] * 4 + ["challenger"] * 4,
        "y_true": [10, 20, 30, 40, 500, 600, 700, 800],
        "y_pred": [12, 18, 33, 38, 510, 590, 710, 790],
    })
    report = lp_metrics.spearman_report(df)
    assert report["spearman_pooled"] > 0.9
    assert report["spearman_by_tier"]["master"]["spearman"] == 1.0
    assert report["spearman_by_tier"]["challenger"]["spearman"] == 1.0
    assert report["spearman_by_tier"]["master"]["n"] == 4
    assert report["n_players_total"] == 8


def test_spearman_report_handles_small_tier_gracefully():
    df = pd.DataFrame({
        "rank": ["master", "master", "challenger"],
        "y_true": [10, 20, 500],
        "y_pred": [12, 18, 510],
    })
    report = lp_metrics.spearman_report(df)
    assert report["spearman_by_tier"]["challenger"]["spearman"] is None
    assert report["spearman_by_tier"]["challenger"]["n"] == 1


def test_spearman_report_rmse():
    df = pd.DataFrame({"rank": ["master", "master"], "y_true": [0, 10], "y_pred": [0, 0]})
    report = lp_metrics.spearman_report(df)
    assert report["rmse_pooled"] == pytest.approx(7.07, abs=0.01)
