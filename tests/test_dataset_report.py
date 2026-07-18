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


# --- temporal_stats -----------------------------------------------------------

_DAY_MS = 86_400_000
_NOW_MS = 1_751_700_000_000


def _temporal_df():
    df = pd.concat([
        _game_rows("p1", "master", 2),
        _game_rows("p2", "challenger", 1),
    ], ignore_index=True)
    df["patch"] = ["16.13", "16.13", "16.12"]
    df["game_ts"] = [_NOW_MS - 2 * _DAY_MS, _NOW_MS - 4 * _DAY_MS,
                     _NOW_MS - 10 * _DAY_MS]
    return df


def test_temporal_stats_by_patch_and_age():
    s = dr.temporal_stats(_temporal_df(), now_ms=_NOW_MS)
    assert s["available"] is True
    assert s["by_patch"] == {"16.13": 2, "16.12": 1}
    assert s["age_days"] == {"newest": 2.0, "median": 4.0, "oldest": 10.0}
    assert s["missing_ts"] == 0


def test_temporal_stats_counts_missing_ts():
    df = _temporal_df()
    df.loc[0, "game_ts"] = None
    s = dr.temporal_stats(df, now_ms=_NOW_MS)
    assert s["missing_ts"] == 1
    assert s["age_days"]["oldest"] == 10.0  # calculé sur les ts présents


def test_temporal_stats_unavailable_on_old_parquet():
    # parquet d'avant l'ajout des colonnes temporelles -> dégradation propre
    s = dr.temporal_stats(_per_game_df().query("source == 'referentiel'"))
    assert s == {"available": False}


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


# --- model_crosscheck : nouveau schéma held-out (gold-standard-eval-protocol) --

def _held_out_metrics(test_auc=0.62, n_train=700, n_calib=150, n_test=190):
    return {
        "cv_train": {"auc": 0.635, "acc": 0.6, "per_model": {}, "n": n_train,
                     "n_pos": n_train // 2, "n_neg": n_train // 2},
        "test": {"auc": test_auc, "acc": 0.61, "n": n_test,
                  "n_pos": n_test // 2, "n_neg": n_test // 2},
        "split": {
            "proportions": {"train": 0.7, "calibration": 0.15, "test": 0.15},
            "n_by_bucket_by_rank": {
                "train": {"master": n_train // 2, "challenger": n_train - n_train // 2},
                "calibration": {"master": n_calib // 2, "challenger": n_calib - n_calib // 2},
                "test": {"master": n_test // 2, "challenger": n_test - n_test // 2},
            },
        },
        "features": [],
        "dispersion_analysis": {},
    }


def test_model_crosscheck_held_out_schema_uses_test_auc_and_split_population():
    metrics = _held_out_metrics(test_auc=0.62, n_train=700, n_calib=150, n_test=190)
    s = dr.model_crosscheck(metrics, n_players_dataset=2000)
    assert s["available"] is True
    assert s["auc_cv"] == 0.62  # headline = test.auc, pas cv_train.auc
    assert s["n_players_model"] == 700 + 150 + 190  # somme des buckets du split
    assert s["n_players_dataset"] == 2000
    assert s["drift"] is True


def test_model_crosscheck_held_out_schema_no_drift_when_split_matches_dataset():
    metrics = _held_out_metrics(n_train=700, n_calib=150, n_test=190)
    s = dr.model_crosscheck(metrics, n_players_dataset=1040)
    assert s["drift"] is False


def test_model_crosscheck_held_out_schema_falls_back_to_cv_train_auc_without_test():
    metrics = _held_out_metrics()
    del metrics["test"]
    s = dr.model_crosscheck(metrics, n_players_dataset=1040)
    assert s["auc_cv"] == 0.635  # repli sur cv_train.auc


# --- report / render ----------------------------------------------------------

def _report(metrics):
    return {
        "per_game": dr.per_game_stats(_per_game_df()),
        "depth": dr.depth_stats(_per_game_df().query("source == 'referentiel'"),
                                thresholds=(2,), min_games=2),
        "temporal": dr.temporal_stats(_temporal_df(), now_ms=_NOW_MS),
        "per_player": dr.per_player_stats(_per_player_df()),
        "model": dr.model_crosscheck(metrics, 8),
    }


def test_report_is_json_serializable():
    json.dumps(_report({"n_players": 8, "auc_cv": 0.6}))  # pas de types numpy


def test_render_contains_key_figures():
    text = dr.render(_report({"n_players": 718, "auc_cv": 0.631}))
    assert "7" in text                  # games référentiel
    assert "challenger" in text
    assert "0.631" in text              # AUC du modèle
    assert "⚠" in text                  # drift 718 != 8 signalé
    assert "16.13" in text              # section temporelle : patchs


def test_render_degrades_without_temporal():
    report = _report({"n_players": 8, "auc_cv": 0.6})
    report["temporal"] = {"available": False}
    text = dr.render(report)
    assert "temporel" in text.lower()   # la section explique l'absence au lieu de crasher
