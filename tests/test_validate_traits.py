"""Tests pour validate_traits.py — Phase 1 : chargement silver + indexation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "pipeline_ops"))
import validate_traits as vt

# On évite de mocker le filesystem : on utilise les vrais silver referentials
# (en lecture seule) pour les tests d'intégration, et des fixtures en mémoire
# pour les tests unitaires.


# ---------- Tests unitaires ----------

def test_index_games_by_champ_role_simple():
    games = [
        {"champion": "Caitlyn", "role": "BOTTOM", "win": True, "match_id": "1"},
        {"champion": "Caitlyn", "role": "BOTTOM", "win": False, "match_id": "2"},
        {"champion": "JarvanIV", "role": "JUNGLE", "win": True, "match_id": "3"},
        {"champion": "Caitlyn", "role": "MIDDLE", "win": True, "match_id": "4"},
        {"champion": "Unknown", "role": "?", "match_id": "5"},  # ignoré
    ]
    idx = vt.index_games_by_champ_role(games)
    assert len(idx) == 3
    assert len(idx[("Caitlyn", "BOTTOM")]) == 2
    assert len(idx[("JarvanIV", "JUNGLE")]) == 1
    assert len(idx[("Caitlyn", "MIDDLE")]) == 1
    assert ("Unknown", "?") not in idx


def test_index_games_skips_empty_champ_or_role():
    games = [
        {"champion": None, "role": "BOTTOM"},
        {"champion": "X", "role": None},
        {"champion": "", "role": "BOTTOM"},
        {"champion": "X", "role": "?"},
    ]
    idx = vt.index_games_by_champ_role(games)
    assert idx == {}


def test_game_count_per_axis():
    games = [
        {"champion": "JarvanIV", "role": "JUNGLE"},
        {"champion": "LeeSin", "role": "JUNGLE"},
        {"champion": "Caitlyn", "role": "BOTTOM"},
        {"champion": "Karma", "role": "UTILITY"},
        {"champion": "Ahri", "role": "MIDDLE"},
        {"champion": "Darius", "role": "TOP"},  # axe non couvert (top seul)
    ]
    idx = vt.index_games_by_champ_role(games)
    counts = vt.game_count_per_axis(idx)
    # playstyle/gank_threat = JUNGLE only : 2 games
    assert counts["playstyle"] == 2
    assert counts["gank_threat"] == 2
    # roam = MID + UTILITY : 2 games
    assert counts["roam"] == 2
    # lane_pattern = BOT + UTILITY : 2 games
    assert counts["lane_pattern"] == 2
    # power_curve = BOT + MID + UTILITY : 3 games
    assert counts["power_curve"] == 3


# ---------- Tests d'intégration (vraies données) ----------

@pytest.mark.skipif(
    not (vt.SILVER_REF_DIR.exists() and any(vt.SILVER_REF_DIR.iterdir())),
    reason="silver referentials not available"
)
def test_load_silver_referentials_returns_games():
    games = vt.load_silver_referentials()
    assert len(games) > 0
    sample = games[0]
    assert "champion" in sample
    assert "role" in sample
    assert "rank" in sample


@pytest.mark.skipif(
    not (vt.SILVER_REF_DIR.exists() and any(vt.SILVER_REF_DIR.iterdir())),
    reason="silver referentials not available"
)
def test_load_silver_referentials_filter_by_rank():
    """Filtrer par rank doit réduire le nombre de games (pas annuler).

    Note : un même match_id peut apparaître dans plusieurs ranks (chaque joueur
    du lobby a un rang différent ; le silver collecte depuis la perspective d'un
    joueur ciblé). Le filtre rank agit sur le `rank` du record, pas sur le match.
    """
    games_all = vt.load_silver_referentials()
    games_chal = vt.load_silver_referentials(ranks=["challenger"])
    games_master = vt.load_silver_referentials(ranks=["master"])
    assert len(games_chal) > 0
    assert len(games_master) > 0
    assert len(games_chal) + len(games_master) <= len(games_all)
    # Tous les records challenger ont rank='challenger'
    assert all(g.get("rank") == "challenger" for g in games_chal)
    assert all(g.get("rank") == "master" for g in games_master)


@pytest.mark.skipif(
    not (vt.SILVER_REF_DIR.exists() and any(vt.SILVER_REF_DIR.iterdir())),
    reason="silver referentials not available"
)
def test_index_real_silver_has_major_combos():
    games = vt.load_silver_referentials()
    idx = vt.index_games_by_champ_role(games)
    # Au moins 50 combos (champ, role) avec >= 20 games
    n_above = sum(1 for gs in idx.values() if len(gs) >= 20)
    assert n_above >= 50, f"only {n_above} combos >= 20 games, expected >= 50"
    # Tous les rôles "porteurs" sont représentés
    counts = vt.game_count_per_axis(idx)
    for axis, n in counts.items():
        assert n > 0, f"axis {axis} has 0 games"


# ---------- Tests Phase 2 (gank stats) ----------

def test_percentile_basic():
    assert vt._percentile([1, 2, 3, 4, 5], 50) == 3
    assert vt._percentile([1, 2, 3, 4, 5], 0) == 1
    assert vt._percentile([1, 2, 3, 4, 5], 100) == 5
    assert vt._percentile([], 50) == 0.0
    p25 = vt._percentile([1, 2, 3, 4], 25)
    assert 1 <= p25 <= 2


def test_validate_champion_axis_validated():
    group = {"score_median": 2.5, "score_p25": 2.0, "score_p75": 3.0, "score_mean": 2.5}
    assert vt.validate_champion_axis("X", "playstyle", 2.5, group, 100) == "validated"
    assert vt.validate_champion_axis("X", "playstyle", 2.2, group, 100) == "validated"


def test_validate_champion_axis_above_below():
    group = {"score_median": 2.5, "score_p25": 2.0, "score_p75": 3.0, "score_mean": 2.5}
    assert vt.validate_champion_axis("X", "playstyle", 3.5, group, 100) == "above_group"
    assert vt.validate_champion_axis("X", "playstyle", 1.5, group, 100) == "below_group"


def test_validate_champion_axis_insufficient_data():
    group = {"score_median": 2.5, "score_p25": 2.0, "score_p75": 3.0, "score_mean": 2.5}
    assert vt.validate_champion_axis("X", "playstyle", 5.0, group, 10) == "insufficient_data"


def test_gank_score_for_game_returns_none_without_position():
    game = {"champion": "LeeSin", "role": "JUNGLE", "position": {}}
    assert vt._gank_score_for_game(game) is None
    game2 = {"champion": "LeeSin", "role": "JUNGLE", "position": None}
    assert vt._gank_score_for_game(game2) is None


def test_gank_score_for_game_clamps_to_1():
    game = {
        "position": {
            "frac_enemy_half": 0.5,
            "frac_roam_mid": 0.8,
            "wards_killed": 20,
        }
    }
    score = vt._gank_score_for_game(game)
    assert abs(score - 1.0) < 0.01


def test_gank_score_for_game_zero_inputs():
    game = {"position": {"frac_enemy_half": 0, "frac_roam_mid": 0, "wards_killed": 0}}
    assert vt._gank_score_for_game(game) == 0


@pytest.mark.skipif(
    not (vt.SILVER_REF_DIR.exists() and any(vt.SILVER_REF_DIR.iterdir())),
    reason="silver referentials not available"
)
def test_detect_lane_visits_returns_dict():
    """Smoke test sur données réelles."""
    import glob
    from riotlib import _read_raw
    files = glob.glob(str(vt.SILVER_REF_DIR / "*" / "games.jsonl"))
    for fpath in files[:1]:
        with open(fpath) as f:
            for line in f:
                game = json.loads(line)
                if game.get("role") != "JUNGLE":
                    continue
                match = _read_raw(f'{game["match_id"]}_match')
                timeline = _read_raw(f'{game["match_id"]}_timeline')
                if not match or not timeline:
                    continue
                res = vt._detect_lane_visits(match, timeline, game["puuid"])
                # v2 : 6 clés (gank_kills_v2 + real_gank_frames ajoutées)
                expected_keys = {"lane_visits", "gank_frames", "gank_kills",
                                 "gank_kills_v2", "real_gank_frames", "early_deaths"}
                assert set(res.keys()) == expected_keys, f"keys mismatch: {set(res.keys())}"
                assert res["lane_visits"] >= res["gank_frames"]
                assert res["gank_frames"] >= res["gank_kills"]
                # gank_kills_v2 doit être >= gank_kills (v2 est plus large temporellement)
                assert res["gank_kills_v2"] >= res["gank_kills"]
                # real_gank_frames <= gank_frames (sous-ensemble filtré)
                assert res["real_gank_frames"] <= res["gank_frames"]
                return
    pytest.skip("no jungler game with raw data found")


def test_detect_lane_visits_counts_sustained_gank_and_followup_kill():
    match = {"metadata": {"participants": [f"p{i}" for i in range(1, 11)]}}

    def frame(minute, events=()):
        participant_frames = {
            "1": {"position": {"x": 13000, "y": 2000}},
            "6": {"position": {"x": 13000, "y": 2000}},
        }
        return {
            "timestamp": minute * 60000,
            "participantFrames": participant_frames,
            "events": list(events),
        }

    kill = {
        "type": "CHAMPION_KILL", "timestamp": 2 * 60000,
        "killerId": 1, "victimId": 6, "assistingParticipantIds": [],
    }
    followup_kill = kill | {"timestamp": 3 * 60000, "victimId": 7}
    off_lane_frame = frame(3, [followup_kill])
    off_lane_frame["participantFrames"] = {}
    timeline = {"info": {"frames": [frame(1), frame(2, [kill]), off_lane_frame]}}

    assert vt._detect_lane_visits(match, timeline, "p1") == {
        "lane_visits": 2,
        "gank_frames": 2,
        "gank_kills": 1,
        "gank_kills_v2": 4,
        "real_gank_frames": 2,
        "early_deaths": 0,
    }


# ---------- Tests Phase 6/7 (rapport + proposals) ----------

def test_verdict_from_value_validated():
    group = {"score_median": 2.5, "score_p25": 2.0, "score_p75": 3.0}
    assert vt._verdict_from_value(2.5, group, 50, 20, "above") == "validated"
    # 2.4 est à 0.1 du médian, dans 0.3*(3-2+0.01) = 0.303 → validated
    assert vt._verdict_from_value(2.4, group, 50, 20, "above") == "validated"


def test_verdict_from_value_above_below():
    group = {"score_median": 2.5, "score_p25": 2.0, "score_p75": 3.0}
    assert vt._verdict_from_value(3.5, group, 50, 20, "above") == "above_group"
    assert vt._verdict_from_value(1.5, group, 50, 20, "above") == "below_group"


def test_verdict_from_value_direction_below():
    """direction='below' : 'above_group' = champion plus bas que 75% du groupe."""
    group = {"score_median": 2.5, "score_p25": 2.0, "score_p75": 3.0}
    # value 1.5 < P25 2.0 → "above_group" (en dessous du groupe)
    assert vt._verdict_from_value(1.5, group, 50, 20, "below") == "above_group"
    # value 3.5 > P75 3.0 → "below_group" (au-dessus du groupe)
    assert vt._verdict_from_value(3.5, group, 50, 20, "below") == "below_group"


def test_verdict_from_value_insufficient():
    group = {"score_median": 2.5, "score_p25": 2.0, "score_p75": 3.0}
    assert vt._verdict_from_value(2.5, group, 5, 20, "above") == "insufficient_data"
    assert vt._verdict_from_value(None, group, 50, 20, "above") == "insufficient_data"
    assert vt._verdict_from_value(2.5, None, 50, 20, "above") == "insufficient_data"


def test_nearest_label_picks_closest_median():
    label_dists = {
        "high": {"score_median": 5.0},
        "med": {"score_median": 3.0},
        "low": {"score_median": 1.0},
    }
    assert vt._nearest_label(4.5, label_dists) == "high"
    assert vt._nearest_label(2.5, label_dists) == "med"
    assert vt._nearest_label(0.5, label_dists) == "low"


def test_nearest_label_handles_empty():
    assert vt._nearest_label(2.5, {}) is None
    assert vt._nearest_label(None, {"x": {"score_median": 1}}) is None


def test_build_report_minimal():
    """Smoke test : construit un rapport avec 1 champion."""
    traits = {"LeeSin": {"playstyle": "ganking", "gank_threat": "high"}}
    # v2 : utilise gank_kills_v2_mean (signal principal après refonte du gank detector)
    per_champ_gank = {"LeeSin": {"n": 50, "gank_kills_v2_mean": 2.6,
                                  "gank_kills_mean": 2.6,
                                  "lane_visits_mean": 7.0,
                                  "gank_frames_mean": 7.0, "raw": []}}
    by_label_gank = {
        "playstyle=ganking": {"n_champions": 5, "n_games": 5, "score_median": 2.5, "score_p25": 2.0, "score_p75": 3.0, "score_mean": 2.5},
        "gank_threat=high": {"n_champions": 5, "n_games": 5, "score_median": 2.5, "score_p25": 2.0, "score_p75": 3.0, "score_mean": 2.5},
    }
    report = vt.build_report(
        traits=traits,
        per_champ_gank=per_champ_gank,
        per_champ_roam={},
        per_champ_lp={},
        per_champ_pc={},
        by_label_gank=by_label_gank,
        by_label_roam={},
        by_label_lp={},
        by_label_pc={},
        min_games=20,
        n_total_games=100,
    )
    assert "axes" in report
    assert "champions" in report
    assert "LeeSin" in report["champions"]
    # LeeSin gank_kills_v2 2.6, group median 2.5 → validated
    assert report["champions"]["LeeSin"]["verdicts"]["playstyle"]["verdict"] == "validated"


def test_build_proposals_finds_champions_without_axes():
    """Un champion avec assez de games et sans axes curés reçoit une proposition."""
    traits = {"KnownChamp": {"playstyle": "ganking"}}  # Camille absent
    per_champ_gank = {
        "KnownChamp": {"n": 50, "gank_kills_mean": 2.5},
        "NewChamp": {"n": 50, "gank_kills_mean": 2.7},
    }
    by_label_gank = {
        "playstyle=ganking": {"score_median": 3.0, "score_p25": 2.5, "score_p75": 3.5},
        "playstyle=farming": {"score_median": 2.0, "score_p25": 1.5, "score_p75": 2.5},
    }
    proposals = vt.build_proposals(
        traits=traits,
        per_champ_gank=per_champ_gank,
        per_champ_roam={},
        per_champ_lp={},
        per_champ_pc={},
        by_label_gank=by_label_gank,
        by_label_roam={},
        by_label_lp={},
        by_label_pc={},
        min_games=20,
    )
    # NewChamp (sans axes) doit recevoir une proposition
    champs_proposed = {p["champion"] for p in proposals}
    assert "NewChamp" in champs_proposed
    assert "KnownChamp" not in champs_proposed  # déjà curé
    p = next(p for p in proposals if p["champion"] == "NewChamp")
    assert p["proposed_axes"]["playstyle"] == "ganking"  # 2.7 plus proche de 3.0 que 2.0


@pytest.mark.parametrize(
    ("champion", "inputs", "expected"),
    [
        (
            "Roamer",
            {"per_champ_roam": {"Roamer": {"n": 30, "roam_mean": 0.7}},
             "per_champ_pc": {"Roamer": {"n": 30, "winrate_long": 0.6}}},
            {"roam": "high", "power_curve": "late"},
        ),
        (
            "Carry",
            {"per_champ_lp": {"Carry": {"n": 30, "early_kp_mean": 0.8}},
             "per_champ_pc": {"Carry": {"n": 30, "winrate_long": 0.6}}},
            {"lane_pattern": "all_in", "power_curve": "late"},
        ),
    ],
)
def test_build_proposals_preserves_role_specific_axes(champion, inputs, expected):
    proposals = vt.build_proposals(
        traits={},
        per_champ_gank=inputs.get("per_champ_gank", {}),
        per_champ_roam=inputs.get("per_champ_roam", {}),
        per_champ_lp=inputs.get("per_champ_lp", {}),
        per_champ_pc=inputs.get("per_champ_pc", {}),
        by_label_gank={},
        by_label_roam={
            "low": {"score_median": 0.1},
            "high": {"score_median": 0.8},
        },
        by_label_lp={
            "poke": {"score_median": 0.2},
            "all_in": {"score_median": 0.9},
        },
        by_label_pc={
            "early": {"score_median": 0.4},
            "late": {"score_median": 0.65},
        },
        min_games=20,
    )

    assert len(proposals) == 1
    assert proposals[0]["champion"] == champion
    assert proposals[0]["proposed_axes"] == expected


def test_render_text_includes_summary():
    report = {
        "config": {"n_total_games": 100, "min_games": 20},
        "axes": {
            "playstyle": {
                "n_validated": 5, "n_above_group": 1, "n_below_group": 1,
                "n_neutral": 0, "n_insufficient_data": 0, "by_label": {},
            },
        },
        "discrepancies": [],
        "champions": {},
    }
    text = vt.render_text(report)
    assert "100 games" in text
    assert "Aucune discrepancy" in text
    assert "playstyle" in text
