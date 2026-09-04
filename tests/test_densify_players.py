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


def test_collect_player_stops_when_adc_gap_is_closed(monkeypatch):
    class Client:
        def match_ids(self, puuid, **kwargs):
            assert puuid == "p1"
            assert kwargs == {"count": 50, "queue": densify_players.rl.QUEUE_SOLO,
                              "start_time": 123}
            return ["seen", "wrong-patch", "missing", "good", "must-not-fetch"]

    responses = {
        "wrong-patch": ({"info": {"gameVersion": "16.12.1"}}, {}),
        "missing": None,
        "good": ({"info": {"gameVersion": "16.13.1"}}, {}),
    }
    monkeypatch.setattr(densify_players.rl, "get_match_timeline",
                        lambda _client, match_id: responses[match_id])
    extracted = [
        {"puuid": "p1", "role": "BOTTOM"},
        {"puuid": "p2", "role": "UTILITY"},
    ]
    monkeypatch.setattr(densify_players.rl, "extract_all_games",
                        lambda *_args, **_kwargs: extracted)

    seen = {"seen"}
    games, adc_games = densify_players.collect_player_games(
        Client(), "p1", rank="master", patch="16.13", max_history=50,
        start_time=123, gap=1, seen_matches=seen,
    )

    assert games == extracted
    assert adc_games == 1
    assert seen == {"seen", "wrong-patch", "missing", "good"}
