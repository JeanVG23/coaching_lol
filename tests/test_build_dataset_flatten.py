"""Incrément 2 — flatten du sous-objet `position` en colonnes pos_* (build_dataset)."""
import importlib.util
import sys
from pathlib import Path

# build_dataset vit dans src/01_data_engineering/ (dossier non importable tel quel).
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))                       # riotlib
_spec = importlib.util.spec_from_file_location(
    "build_dataset", _SRC / "01_data_engineering" / "build_dataset.py")
build_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_dataset)


def _base_game(**extra):
    g = {"match_id": "EUW1_1", "puuid": "p1", "champion": "Zeri", "win": True,
         "lane": {}, "deaths": [], "kills": [], "assists": []}
    g.update(extra)
    return g


def test_position_dict_flattened_to_pos_columns():
    g = _base_game(position={"frac_base": 0.12, "avg_map_depth": 2000.5,
                             "wards_placed": 7})
    row = build_dataset.game_to_row(g, "challenger", "referentiel")
    assert row["pos_frac_base"] == 0.12
    assert row["pos_avg_map_depth"] == 2000.5
    assert row["pos_wards_placed"] == 7


def test_missing_position_yields_no_pos_columns():
    row = build_dataset.game_to_row(_base_game(), "diamond", "referentiel")
    assert not any(k.startswith("pos_") for k in row)
