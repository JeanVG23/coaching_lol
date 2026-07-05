"""densify_players vit dans src/collection/ (dossier non importable tel quel, cf.
tests/test_build_player_dataset.py pour le même pattern de chargement). Ne teste que
les fonctions pures (parsing du target-list, détection ADC) — pas main() qui appelle
l'API Riot."""
import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC / "core"))
_spec = importlib.util.spec_from_file_location(
    "densify_players", _SRC / "collection" / "densify_players.py")
densify_players = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(densify_players)


def test_parse_target_list_enriched_format_extracts_gap():
    targets = {"p1": {"rank": "master", "gap": 5}, "p2": {"rank": "diamond", "gap": 3}}
    puuids, gap_by_puuid = densify_players.parse_target_list(targets, "master")
    assert puuids == ["p1"]
    assert gap_by_puuid == {"p1": 5}


def test_parse_target_list_legacy_flat_format_no_gap():
    targets = {"p1": "master", "p2": "diamond"}
    puuids, gap_by_puuid = densify_players.parse_target_list(targets, "master")
    assert puuids == ["p1"]
    assert gap_by_puuid == {}


def test_parse_target_list_filters_by_rank():
    targets = {"p1": {"rank": "master", "gap": 5}, "p2": {"rank": "diamond", "gap": 3}}
    puuids, _ = densify_players.parse_target_list(targets, "diamond")
    assert puuids == ["p2"]


def test_closes_gap_true_when_target_is_bottom():
    games = [
        {"puuid": "p1", "role": "BOTTOM"},
        {"puuid": "p2", "role": "JUNGLE"},
    ]
    assert densify_players.closes_gap(games, "p1") is True


def test_closes_gap_false_when_target_not_bottom():
    games = [{"puuid": "p1", "role": "MIDDLE"}, {"puuid": "p2", "role": "BOTTOM"}]
    assert densify_players.closes_gap(games, "p1") is False


def test_closes_gap_false_when_target_absent():
    games = [{"puuid": "p2", "role": "BOTTOM"}]
    assert densify_players.closes_gap(games, "p1") is False
