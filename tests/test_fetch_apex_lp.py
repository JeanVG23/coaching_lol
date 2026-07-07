"""Tests des fonctions pures du fetch LP prod (src/collection/fetch_apex_lp.py).
Les appels API réels sont vérifiés à l'exécution (Task 6 du plan)."""
import fetch_apex_lp  # src/collection est sur sys.path via tests/conftest.py


def test_build_lp_lookup_merges_tiers_and_skips_missing_puuid():
    entries_by_tier = {
        "challenger": [{"puuid": "a", "leaguePoints": 800}],
        "master": [{"puuid": "b", "leaguePoints": 50},
                   {"puuid": None, "leaguePoints": 10}],
    }
    lookup = fetch_apex_lp.build_lp_lookup(entries_by_tier)
    assert lookup == {
        "a": {"tier": "challenger", "leaguePoints": 800},
        "b": {"tier": "master", "leaguePoints": 50},
    }


def test_build_payload_wraps_players_with_fetched_at():
    entries_by_tier = {"master": [{"puuid": "b", "leaguePoints": 50}]}
    payload = fetch_apex_lp.build_payload(entries_by_tier, "2026-07-07T12:00:00+00:00")
    assert payload["fetched_at"] == "2026-07-07T12:00:00+00:00"
    assert payload["players"] == {"b": {"tier": "master", "leaguePoints": 50}}
