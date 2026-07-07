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


def test_spearman_report_pooled_none_when_degenerate():
    df = pd.DataFrame({"rank": ["master", "master"], "y_true": [10, 20], "y_pred": [5, 5]})
    report = lp_metrics.spearman_report(df)
    assert report["spearman_pooled"] is None


def test_spearman_report_by_tier_none_when_y_pred_constant():
    df = pd.DataFrame({
        "rank": ["master"] * 3 + ["challenger"] * 3,
        "y_true": [10, 20, 30, 500, 600, 700],
        "y_pred": [5, 5, 5, 510, 590, 710],
    })
    report = lp_metrics.spearman_report(df)
    assert report["spearman_by_tier"]["master"]["spearman"] is None
    assert report["spearman_by_tier"]["challenger"]["spearman"] is not None


import train_lp_regression


# --- qualified_apex_players ---------------------------------------------------

def _games(puuid, rank, n, csm10):
    return pd.DataFrame({
        "puuid": [puuid] * n,
        "rank": [rank] * n,
        "win": [1] * n,
        "csm10": csm10,
    })


def test_qualified_apex_players_filters_min_games_and_excludes_diamond():
    ref = pd.concat([
        _games("p1", "master", 3, [4.0, 6.0, 8.0]),   # qualifie : master, 3>=2 games
        _games("p2", "master", 1, [5.0]),              # exclu : trop peu de games
        _games("p3", "diamond", 3, [4.0, 6.0, 8.0]),   # exclu : diamond hors scope LP
    ], ignore_index=True)
    out = train_lp_regression.qualified_apex_players(ref, min_games=2, features=["csm10"])
    assert set(out["puuid"]) == {"p1"}
    assert out.iloc[0]["csm10__mean"] == pytest.approx(6.0)


def test_qualified_apex_players_resolves_rank_by_mode_across_all_games():
    ref = pd.concat([
        _games("p4", "master", 2, [4.0, 6.0]),
        _games("p4", "diamond", 1, [5.0]),
    ], ignore_index=True)
    out = train_lp_regression.qualified_apex_players(ref, min_games=2, features=["csm10"])
    assert list(out["rank"]) == ["master"]     # mode sur tout l'historique : 2 master > 1 diamond
