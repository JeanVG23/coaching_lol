"""Tests build_sequence_dataset : extraction de frames + résolution matchup."""
import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
_spec = importlib.util.spec_from_file_location(
    "build_sequence_dataset", _SRC / "01_data_engineering" / "build_sequence_dataset.py")
bsd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsd)


def test_frame_state_returns_8dim_normalized():
    pf = {"position": {"x": 14800, "y": 7400},
          "totalGold": 600, "currentGold": 50, "xp": 100, "level": 2,
          "minionsKilled": 5, "jungleMinionsKilled": 1}
    s = bsd.frame_state(pf)
    assert len(s) == 8
    assert s[0] == 1.0          # x / MAP_SIZE
    assert s[1] == 0.5          # y / MAP_SIZE
    assert s[2] == 600.0 and s[3] == 50.0 and s[4] == 100.0
    assert s[5] == 2.0 and s[6] == 5.0 and s[7] == 1.0


def test_frame_state_missing_fields_zero():
    s = bsd.frame_state({})
    assert s == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _fake_match():
    # 2 équipes × 5 rôles ; puuids[i] ↔ participants[i]
    puuids = [f"p_{r}_{t}" for t in (100, 200)
              for r in ("BOTTOM", "UTILITY", "JUNGLE", "MIDDLE", "TOP")]
    parts = [{"teamId": t, "teamPosition": r, "championName": f"{r}_{t}"}
             for t in (100, 200) for r in ("BOTTOM", "UTILITY", "JUNGLE", "MIDDLE", "TOP")]
    return {"metadata": {"participants": puuids}, "info": {"participants": parts}}


def test_participant_and_opponent_pid():
    m = _fake_match()
    # p_BOTTOM_100 est idx 0 → pid 1
    assert bsd.participant_pid(m, "p_BOTTOM_100") == 1
    # opponent = BOTTOM de l'équipe 200 → idx 5 → pid 6
    assert bsd.opponent_pid(m, "p_BOTTOM_100") == 6
    assert bsd.opponent_pid(m, "p_BOTTOM_200") == 1


def test_opponent_pid_jungle_cross_team():
    m = _fake_match()
    assert bsd.opponent_pid(m, "p_JUNGLE_100") == 8  # jungle opp = idx 7 → pid 8


def test_opponent_pid_none_if_role_missing():
    # Adversaire de rôle BOTTOM absent de l'équipe 200 → None
    m = _fake_match()
    m["info"]["participants"][5]["teamPosition"] = "MIDDLE"  # l'ancien BOTTOM 200 devient MID
    assert bsd.opponent_pid(m, "p_BOTTOM_100") is None