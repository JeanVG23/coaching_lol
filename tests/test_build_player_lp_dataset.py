"""Tests du dataset per-player LP (sans balance-cap, apex tiers seulement)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "01_data_engineering"))
import build_player_lp_dataset as bld


def _games(puuid, rank, n, csm10):
    return pd.DataFrame({
        "puuid": [puuid] * n,
        "rank": [rank] * n,
        "win": [1] * n,
        "csm10": csm10,
    })


LP = {"p1": {"tier": "master", "leaguePoints": 120},
      "p4": {"tier": "master", "leaguePoints": 40}}


def test_filters_min_games_and_excludes_diamond():
    ref = pd.concat([
        _games("p1", "master", 3, [4.0, 6.0, 8.0]),   # qualifie
        _games("p2", "master", 1, [5.0]),              # exclu : trop peu de games
        _games("p3", "diamond", 3, [4.0, 6.0, 8.0]),   # exclu : diamond hors échelle LP
    ], ignore_index=True)
    out, n_dropped = bld.build_lp_player_rows(ref, LP, min_games=2, features=["csm10"])
    assert set(out["puuid"]) == {"p1"}
    assert out.iloc[0]["lp"] == 120
    assert out.iloc[0]["csm10__mean"] == pytest.approx(6.0)
    assert n_dropped == 0


def test_drops_and_counts_players_without_current_lp():
    ref = pd.concat([
        _games("p1", "master", 2, [4.0, 6.0]),
        _games("p9", "master", 2, [5.0, 7.0]),         # qualifié mais absent du lookup LP
    ], ignore_index=True)
    out, n_dropped = bld.build_lp_player_rows(ref, LP, min_games=2, features=["csm10"])
    assert set(out["puuid"]) == {"p1"}
    assert n_dropped == 1


def test_rank_resolved_by_mode_across_all_games():
    ref = pd.concat([
        _games("p4", "master", 2, [4.0, 6.0]),
        _games("p4", "diamond", 1, [5.0]),
    ], ignore_index=True)
    out, _ = bld.build_lp_player_rows(ref, LP, min_games=2, features=["csm10"])
    assert list(out["rank"]) == ["master"]  # mode : 2 master > 1 diamond


def test_no_balance_cap():
    # 3 masters vs 1 challenger : une régression garde tout le monde (pas d'undersampling)
    lp = {p: {"tier": "master", "leaguePoints": 10} for p in ("a", "b", "c", "d")}
    ref = pd.concat([
        _games("a", "master", 2, [4.0, 6.0]),
        _games("b", "master", 2, [4.0, 6.0]),
        _games("c", "master", 2, [4.0, 6.0]),
        _games("d", "challenger", 2, [4.0, 6.0]),
    ], ignore_index=True)
    out, _ = bld.build_lp_player_rows(ref, lp, min_games=2, features=["csm10"])
    assert len(out) == 4
