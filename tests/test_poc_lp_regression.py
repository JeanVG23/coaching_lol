"""Tests des fonctions pures du POC régression LP (poc/script/). Le reste (appels
API, CV complète) est vérifié par exécution réelle, cf.
docs/superpowers/specs/2026-07-07-lp-regression-poc-design.md."""
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src" / "core"))
sys.path.insert(0, str(_ROOT / "poc" / "script"))

import fetch_apex_lp


# --- build_lp_lookup ---------------------------------------------------------

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
