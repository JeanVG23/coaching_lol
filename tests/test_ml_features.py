"""Tests de ml_features — agrégation per-player (mean/std/p10/p50/p90) et résolution
de rang, extraites de poc/per_player_hypothesis.py en module partagé (train + serve)."""
import numpy as np
import pandas as pd
import pytest

import ml_features as mf


def test_aggregate_player_features_basic_stats():
    df = pd.DataFrame({"csm10": [4.0, 6.0, 8.0]})
    agg = mf.aggregate_player_features(df, features=["csm10"])
    assert agg["csm10__mean"] == pytest.approx(6.0)
    assert agg["csm10__std"] == pytest.approx(2.0)
    assert agg["csm10__p50"] == pytest.approx(6.0)
    assert agg["n_games"] == 3


def test_aggregate_player_features_single_game_std_zero():
    df = pd.DataFrame({"csm10": [5.0]})
    agg = mf.aggregate_player_features(df, features=["csm10"])
    assert agg["csm10__std"] == 0.0
    assert agg["n_games"] == 1


def test_aggregate_player_features_missing_column_is_nan():
    df = pd.DataFrame({"csm10": [5.0, 6.0]})
    agg = mf.aggregate_player_features(df, features=["csm10", "gpm10"])
    assert np.isnan(agg["gpm10__mean"])
    assert np.isnan(agg["gpm10__std"])
    assert np.isnan(agg["gpm10__p10"])


def test_aggregate_player_features_all_nan_column_stays_nan():
    df = pd.DataFrame({"csm10": [np.nan, np.nan]})
    agg = mf.aggregate_player_features(df, features=["csm10"])
    assert np.isnan(agg["csm10__mean"])


def test_resolve_rank_mode_with_tie_break_lowest():
    group = pd.DataFrame({"rank": ["diamond", "master", "master", "diamond"]})
    assert mf.resolve_rank(group) == "diamond"


def test_resolve_rank_mode_no_tie():
    group = pd.DataFrame({"rank": ["challenger", "challenger", "diamond"]})
    assert mf.resolve_rank(group) == "challenger"


def test_player_feature_names_order():
    names = mf.player_feature_names(["csm10", "gpm10"])
    assert names == [
        "csm10__mean", "csm10__std", "csm10__p10", "csm10__p50", "csm10__p90",
        "gpm10__mean", "gpm10__std", "gpm10__p10", "gpm10__p50", "gpm10__p90",
        "win_rate", "n_games",
    ]


def test_aggregate_player_features_win_rate():
    df = pd.DataFrame({"csm10": [4.0, 6.0, 8.0], "win": [1, 0, 1]})
    agg = mf.aggregate_player_features(df, features=["csm10"])
    assert agg["win_rate"] == pytest.approx(2 / 3)


def test_aggregate_player_features_win_rate_nan_without_column():
    df = pd.DataFrame({"csm10": [4.0, 6.0]})
    agg = mf.aggregate_player_features(df, features=["csm10"])
    assert np.isnan(agg["win_rate"])
