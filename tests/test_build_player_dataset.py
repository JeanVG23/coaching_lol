"""build_player_dataset vit dans src/01_data_engineering/ (dossier non importable
tel quel, cf. tests/test_build_dataset_flatten.py pour le même pattern de chargement)."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "core"))
_spec = importlib.util.spec_from_file_location(
    "build_player_dataset", _SRC / "01_data_engineering" / "build_player_dataset.py")
build_player_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_player_dataset)


def _rows(puuid, rank, values):
    return pd.DataFrame({
        "puuid": [puuid] * len(values),
        "rank": [rank] * len(values),
        "csm10": values,
    })


def test_build_player_rows_filters_below_min_games():
    df = pd.concat([
        _rows("p1", "diamond", [4.0, 5.0]),                       # 2 games < min
        _rows("p2", "challenger", [6.0, 7.0, 8.0, 9.0, 10.0]),    # 5 games >= min
    ], ignore_index=True)
    out = build_player_dataset.build_player_rows(df, min_games=5)
    assert list(out["puuid"]) == ["p2"]
    assert out.iloc[0]["n_games"] == 5


def test_build_player_rows_computes_high_elo_label():
    df = _rows("p1", "challenger", [6.0] * 5)
    out = build_player_dataset.build_player_rows(df, min_games=5)
    assert out.iloc[0]["high_elo"] == 1

    df2 = _rows("p2", "diamond", [6.0] * 5)
    out2 = build_player_dataset.build_player_rows(df2, min_games=5)
    assert out2.iloc[0]["high_elo"] == 0


def test_build_player_rows_aggregates_csm10():
    df = _rows("p1", "diamond", [4.0, 6.0, 8.0, 10.0, 12.0])
    out = build_player_dataset.build_player_rows(df, min_games=5)
    assert out.iloc[0]["csm10__mean"] == 8.0


def test_build_player_rows_empty_when_no_player_meets_threshold():
    df = _rows("p1", "diamond", [4.0, 5.0])
    out = build_player_dataset.build_player_rows(df, min_games=5)
    assert out.empty
