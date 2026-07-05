"""dataset_report : visibilité sur les datasets ML (per-game + per-player).

Fonctions pures testées sur des DataFrames synthétiques ; le main (lecture des
parquets réels) n'est pas testé ici.
"""
import json

import pandas as pd

import dataset_report as dr


def _game_rows(puuid, rank, n, win_first=1, source="referentiel"):
    """n games d'un joueur, wins alternés en commençant par win_first."""
    return pd.DataFrame({
        "match_id": [f"{puuid}_g{i}" for i in range(n)],
        "puuid": [puuid] * n,
        "source": [source] * n,
        "rank": [rank] * n,
        "win": [(win_first + i) % 2 for i in range(n)],
    })


def _per_game_df():
    return pd.concat([
        _game_rows("p1", "master", 4),
        _game_rows("p2", "challenger", 2),
        _game_rows("p3", "diamond", 1, win_first=0),
        _game_rows("me", None, 2, source="personal:spadzze"),
    ], ignore_index=True)


# --- per_game_stats -----------------------------------------------------------

def test_per_game_stats_counts():
    s = dr.per_game_stats(_per_game_df())
    assert s["rows_total"] == 9
    assert s["by_source"] == {"referentiel": 7, "personal:spadzze": 2}
    # stats référentiel uniquement (le perso n'a pas de rang)
    assert s["n_games"] == 7
    assert s["n_players"] == 3
    assert s["by_rank"] == {"master": 4, "challenger": 2, "diamond": 1}


def test_per_game_stats_win_and_class_balance():
    s = dr.per_game_stats(_per_game_df())
    assert s["win_balance"] == {"win": 3, "loss": 4}
    # high = GM+Chall, low = Master+Diamond (même définition que le train)
    assert s["class_balance"] == {"high": 2, "low": 5}


# --- depth_stats --------------------------------------------------------------

def test_depth_stats_thresholds():
    ref = pd.concat([
        _game_rows("p1", "master", 6),
        _game_rows("p2", "master", 3),
        _game_rows("p3", "diamond", 1),
    ], ignore_index=True)
    s = dr.depth_stats(ref, thresholds=(2, 5), min_games=5)
    assert s["games_per_player"]["median"] == 3
    assert s["games_per_player"]["max"] == 6
    assert s["at_least"] == {2: 2, 5: 1}
    assert s["qualified_by_rank"] == {"master": 1}


def test_depth_stats_rank_resolved_by_mode_lowest_tiebreak():
    # p1 : 2 games master + 1 challenger -> mode = master
    # p2 : égalité 2 GM / 2 challenger -> tie-break rang le plus bas = GM
    ref = pd.concat([
        _game_rows("p1", "master", 2),
        _game_rows("p1", "challenger", 1),
        _game_rows("p2", "grandmaster", 2),
        _game_rows("p2", "challenger", 2),
    ], ignore_index=True)
    ref["match_id"] = [f"g{i}" for i in range(len(ref))]  # ids uniques
    s = dr.depth_stats(ref, thresholds=(), min_games=3)
    assert s["qualified_by_rank"] == {"master": 1, "grandmaster": 1}


# --- per_player_stats ---------------------------------------------------------

def _per_player_df():
    ranks = ["challenger"] * 3 + ["grandmaster"] * 1 + ["master"] * 3 + ["diamond"] * 1
    return pd.DataFrame({
        "puuid": [f"p{i}" for i in range(len(ranks))],
        "rank": ranks,
        "high_elo": [1, 1, 1, 1, 0, 0, 0, 0],
        "n_games": [20, 30, 40, 25, 15, 35, 45, 22],
    })


def test_per_player_stats_balance_and_intra_class():
    s = dr.per_player_stats(_per_player_df())
    assert s["n_players"] == 8
    assert s["by_rank"] == {"challenger": 3, "master": 3, "grandmaster": 1, "diamond": 1}
    assert s["class_balance"] == {"high": 4, "low": 4}
    # composition intra-classe : le rang dominant et sa part
    assert s["intra_class"]["high"] == {"dominant": "challenger", "share": 0.75}
    assert s["intra_class"]["low"] == {"dominant": "master", "share": 0.75}


def test_per_player_stats_n_games_by_rank():
    s = dr.per_player_stats(_per_player_df())
    assert s["n_games_median_by_rank"]["challenger"] == 30
    assert s["n_games_median_by_rank"]["diamond"] == 22


# --- model_crosscheck ---------------------------------------------------------

def test_model_crosscheck_detects_drift():
    metrics = {"n_players": 718, "auc_cv": 0.631}
    s = dr.model_crosscheck(metrics, n_players_dataset=1040)
    assert s == {"available": True, "n_players_model": 718, "auc_cv": 0.631,
                 "n_players_dataset": 1040, "drift": True}


def test_model_crosscheck_no_drift_and_missing_metrics():
    ok = dr.model_crosscheck({"n_players": 1040, "auc_cv": 0.62}, 1040)
    assert ok["drift"] is False
    assert dr.model_crosscheck(None, 1040) == {"available": False}


# --- report / render ----------------------------------------------------------

def test_report_is_json_serializable():
    report = {
        "per_game": dr.per_game_stats(_per_game_df()),
        "depth": dr.depth_stats(_per_game_df().query("source == 'referentiel'"),
                                thresholds=(2,), min_games=2),
        "per_player": dr.per_player_stats(_per_player_df()),
        "model": dr.model_crosscheck({"n_players": 8, "auc_cv": 0.6}, 8),
    }
    json.dumps(report)  # ne doit pas lever (pas de types numpy)


def test_render_contains_key_figures():
    report = {
        "per_game": dr.per_game_stats(_per_game_df()),
        "depth": dr.depth_stats(_per_game_df().query("source == 'referentiel'"),
                                thresholds=(2,), min_games=2),
        "per_player": dr.per_player_stats(_per_player_df()),
        "model": dr.model_crosscheck({"n_players": 718, "auc_cv": 0.631}, 8),
    }
    text = dr.render(report)
    assert "7" in text                  # games référentiel
    assert "challenger" in text
    assert "0.631" in text              # AUC du modèle
    assert "⚠" in text                  # drift 718 != 8 signalé
