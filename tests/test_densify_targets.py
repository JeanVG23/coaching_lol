"""densify_targets vit dans src/collection/ (dossier non importable tel quel, cf.
tests/test_build_player_dataset.py pour le même pattern de chargement)."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "core"))
_spec = importlib.util.spec_from_file_location(
    "densify_targets", _SRC / "collection" / "densify_targets.py")
densify_targets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(densify_targets)


def _rows(puuid, rank, n):
    return pd.DataFrame({"puuid": [puuid] * n, "rank": [rank] * n, "match_id": range(n)})


def test_select_targets_excludes_below_min_games():
    df = pd.concat([_rows("p1", "diamond", 3), _rows("p2", "diamond", 10)], ignore_index=True)
    out = densify_targets.select_targets(df, threshold=20, min_games=8)
    assert list(out.keys()) == ["p2"]


def test_select_targets_excludes_already_qualified():
    df = _rows("p1", "challenger", 20)
    out = densify_targets.select_targets(df, threshold=20, min_games=8)
    assert out == {}


def test_select_targets_computes_gap_and_rank():
    df = _rows("p1", "master", 12)
    out = densify_targets.select_targets(df, threshold=20, min_games=8)
    assert out["p1"] == {"rank": "master", "n_games": 12, "gap": 8}


def test_select_targets_sorted_by_ascending_gap():
    df = pd.concat([
        _rows("far", "diamond", 8),    # gap 12
        _rows("near", "diamond", 18),  # gap 2
        _rows("mid", "diamond", 13),   # gap 7
    ], ignore_index=True)
    out = densify_targets.select_targets(df, threshold=20, min_games=8)
    assert list(out.keys()) == ["near", "mid", "far"]
