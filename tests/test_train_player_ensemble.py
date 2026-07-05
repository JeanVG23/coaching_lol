"""train_player_ensemble vit dans src/02_data_science/ (dossier non importable tel
quel, même pattern de chargement que tests/test_build_dataset_flatten.py). Fonctions
pures testées : dispersion_share_analysis et purged_train_features — le reste (CV,
fit des modèles) est vérifié par exécution réelle."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC / "core"))
_spec = importlib.util.spec_from_file_location(
    "train_player_ensemble", _SRC / "02_data_science" / "train_player_ensemble.py")
train_player_ensemble = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_player_ensemble)


def test_dispersion_share_analysis_groups_by_stat_suffix():
    per_feature = {
        "csm10__mean": 1.0, "csm10__p50": 1.0,                        # central = 2.0
        "csm10__std": 3.0, "csm10__p10": 1.0, "csm10__p90": 0.0,      # dispersion = 4.0
        "n_games": 0.5,
    }
    result = train_player_ensemble.dispersion_share_analysis(per_feature)
    assert result["dispersion_share_of_signal"] == pytest.approx(4.0 / 6.0, abs=1e-4)
    assert result["share_by_stat"]["std"] == pytest.approx(3.0 / 6.5, abs=1e-4)
    assert result["share_by_stat"]["n_games"] == pytest.approx(0.5 / 6.5, abs=1e-4)


def test_dispersion_share_analysis_ignores_unknown_suffix():
    per_feature = {"weird__unknownstat": 5.0, "csm10__mean": 1.0}
    result = train_player_ensemble.dispersion_share_analysis(per_feature)
    # "unknownstat" n'est dans aucun bucket -> exclu du total (mean=1.0 seul compte)
    assert result["share_by_stat"]["mean"] == pytest.approx(1.0)


# --- purged_train_features ------------------------------------------------------
# 37 % des games des joueurs qualifiés opposent deux joueurs du dataset, et le
# graphe des games partagées est une composante géante (98.7 %) -> group-CV par
# composantes impossible. À la place : folds joueurs classiques, mais les agrégats
# des joueurs de TRAIN sont recalculés en excluant les matchs des joueurs de VAL
# (aucune info des games du val ne fuit dans le train).

import ml_features as mf


def _games(puuid, matches, csm10, wins):
    return pd.DataFrame({
        "puuid": [puuid] * len(matches),
        "match_id": matches,
        "rank": ["master"] * len(matches),
        "win": wins,
        "csm10": csm10,
    })


def _ref():
    return pd.concat([
        # p1 (train) : 3 games, m3 partagée avec p_val
        _games("p1", ["m1", "m2", "m3"], [4.0, 6.0, 11.0], [1, 0, 1]),
        # p2 (train) : 2 games, aucune partagée
        _games("p2", ["m4", "m5"], [8.0, 10.0], [1, 1]),
        # p3 (train) : toutes ses games partagées avec p_val
        _games("p3", ["m3", "m6"], [7.0, 9.0], [0, 1]),
        # p_val (val) : joue m3 et m6
        _games("p_val", ["m3", "m6"], [5.0, 5.0], [0, 0]),
    ], ignore_index=True)


def test_purged_excludes_shared_matches_from_train_aggregates():
    X, dropped = train_player_ensemble.purged_train_features(
        _ref(), ["p1", "p2"], {"p_val"}, features=["csm10"])
    p1 = X[X["puuid"] == "p1"].iloc[0]
    assert p1["csm10__mean"] == 5.0          # (4+6)/2, m3 exclue
    assert p1["win_rate"] == 0.5             # recalculé sur m1+m2
    assert p1["n_games"] == 2


def test_purged_keeps_untouched_player_identical():
    ref = _ref()
    X, dropped = train_player_ensemble.purged_train_features(
        ref, ["p2"], {"p_val"}, features=["csm10"])
    expected = mf.aggregate_player_features(ref[ref["puuid"] == "p2"], ["csm10"])
    row = X[X["puuid"] == "p2"].iloc[0]
    for k, v in expected.items():
        assert row[k] == v
    assert dropped == []


def test_purged_drops_player_with_no_games_left():
    X, dropped = train_player_ensemble.purged_train_features(
        _ref(), ["p1", "p3"], {"p_val"}, features=["csm10"])
    assert dropped == ["p3"]
    assert list(X["puuid"]) == ["p1"]


def test_control_drops_same_count_but_random_games():
    # contrôle : même nombre de games retirées que la purge, mais au hasard —
    # isole l'effet "moins de games" de l'effet "fuite retirée"
    X, dropped = train_player_ensemble.control_train_features(
        _ref(), ["p1", "p2"], {"p1": 1}, features=["csm10"], seed=42)
    p1 = X[X["puuid"] == "p1"].iloc[0]
    assert p1["n_games"] == 2                # 3 - 1 droppée
    p2 = X[X["puuid"] == "p2"].iloc[0]
    assert p2["n_games"] == 2                # pas dans n_drop -> intact
    assert dropped == []


def test_control_drops_player_losing_all_games():
    X, dropped = train_player_ensemble.control_train_features(
        _ref(), ["p1", "p3"], {"p3": 2}, features=["csm10"], seed=42)
    assert dropped == ["p3"]
    assert list(X["puuid"]) == ["p1"]
