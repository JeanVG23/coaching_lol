from datetime import datetime, timezone

import live_capture as LC


def _iso(y, mo, d, h, mi, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).isoformat()


def test_find_matching_game_exact_match():
    capture_meta = {
        "start": _iso(2026, 7, 2, 18, 0, 0),
        "end": _iso(2026, 7, 2, 18, 30, 0),
        "champion": "Zeri",
    }
    candidates = [{
        "match_id": "EUW1_111",
        "champion": "Zeri",
        "game_start": _iso(2026, 7, 2, 18, 0, 5),
        "game_duration_s": 1795,
    }]
    assert LC.find_matching_game(capture_meta, candidates) == "EUW1_111"


def test_find_matching_game_no_candidate_in_tolerance():
    capture_meta = {
        "start": _iso(2026, 7, 2, 18, 0, 0),
        "end": _iso(2026, 7, 2, 18, 30, 0),
        "champion": "Zeri",
    }
    candidates = [{
        "match_id": "EUW1_222",
        "champion": "Zeri",
        "game_start": _iso(2026, 7, 2, 20, 0, 0),  # 2h plus tard, hors tolérance
        "game_duration_s": 1800,
    }]
    assert LC.find_matching_game(capture_meta, candidates) is None


def test_find_matching_game_ambiguous_picks_closest_and_warns():
    capture_meta = {
        "start": _iso(2026, 7, 2, 18, 0, 0),
        "end": _iso(2026, 7, 2, 18, 30, 0),
        "champion": "Zeri",
    }
    candidates = [
        {"match_id": "EUW1_FAR", "champion": "Zeri",
         "game_start": _iso(2026, 7, 2, 18, 2, 0), "game_duration_s": 1800},
        {"match_id": "EUW1_CLOSE", "champion": "Zeri",
         "game_start": _iso(2026, 7, 2, 18, 0, 10), "game_duration_s": 1800},
    ]
    warnings = []
    result = LC.find_matching_game(capture_meta, candidates, warn=warnings.append)
    assert result == "EUW1_CLOSE"
    assert len(warnings) == 1


def test_find_matching_game_tolerance_boundary():
    capture_meta = {
        "start": _iso(2026, 7, 2, 18, 0, 0),
        "end": _iso(2026, 7, 2, 18, 30, 0),
        "champion": "Zeri",
    }
    # pile à la limite (300s) -> inclus
    at_boundary = [{"match_id": "EUW1_AT", "champion": "Zeri",
                    "game_start": _iso(2026, 7, 2, 18, 5, 0), "game_duration_s": 1800}]
    assert LC.find_matching_game(capture_meta, at_boundary,
                                  start_tolerance_s=300) == "EUW1_AT"
    # juste au-delà (301s) -> exclu
    beyond_boundary = [{"match_id": "EUW1_BEYOND", "champion": "Zeri",
                        "game_start": _iso(2026, 7, 2, 18, 5, 1), "game_duration_s": 1800}]
    assert LC.find_matching_game(capture_meta, beyond_boundary,
                                  start_tolerance_s=300) is None
