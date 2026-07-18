"""Tests build_sequence_dataset : extraction de frames + résolution matchup."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
_spec = importlib.util.spec_from_file_location(
    "build_sequence_dataset", _SRC / "01_data_engineering" / "build_sequence_dataset.py")
bsd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsd)

build_dataset = bsd.build_dataset   # même objet que celui utilisé par bsd.main()


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


def _fake_timeline(n_minutes=3):
    frames = []
    for minute in range(n_minutes):
        pf = {}
        for pid in range(1, 11):
            pf[str(pid)] = {
                "position": {"x": 1000 * pid + minute, "y": 2000 * pid},
                "totalGold": 500 * pid + 100 * minute,
                "currentGold": 50 * pid,
                "xp": 100 * pid * minute if minute else 0,
                "level": 1 + minute,
                "minionsKilled": 5 * minute,
                "jungleMinionsKilled": minute,
            }
        frames.append({"timestamp": minute * 60000, "participantFrames": pf})
    return {"info": {"frames": frames}}


def _fake_timeline_with_events(n_minutes=15):
    t = _fake_timeline(n_minutes)
    # tous les events placés dans frame[0] — game_journal._events les flatten, peu importe la frame.
    t["info"]["frames"][0]["events"] = [
        # mort gank à 4:42 (minute 4) : killer=jungle(8) + assist=adc(6) -> ganked, pas solo
        {"type": "CHAMPION_KILL", "timestamp": 282000, "victimId": 1, "killerId": 8,
         "assistingParticipantIds": [6], "position": {"x": 1000, "y": 2000}},
        # mort solo à 12:30 (minute 12) : killer=adc(6), 0 assist -> pas ganked, solo
        {"type": "CHAMPION_KILL", "timestamp": 750000, "victimId": 1, "killerId": 6,
         "assistingParticipantIds": [], "position": {"x": 2000, "y": 3000}},
        # mort adverse à 8:15 (minute 8) : victimId=6 (opp)
        {"type": "CHAMPION_KILL", "timestamp": 495000, "victimId": 6, "killerId": 1,
         "assistingParticipantIds": [], "position": {"x": 3000, "y": 4000}},
        # drake tué à 7:00 par équipe 200 -> up avant, down 7-11, respawn à 12:00
        {"type": "ELITE_MONSTER_KILL", "timestamp": 420000, "monsterType": "DRAGON",
         "killerTeamId": 200, "killerId": 8},
        # achat à 2:30 (minute 2), hors opening (<90s) -> recall
        {"type": "ITEM_PURCHASED", "timestamp": 150000, "participantId": 1, "itemId": 1001},
    ]
    return t


def test_event_channels_death_gank_solo():
    t = _fake_timeline_with_events()
    ch = bsd._event_channels(t, pid=1, opp_pid=6, enemy_jungle_pid=8)
    assert ch.shape == (40, 7)
    assert ch[4, 0] == 1.0 and ch[4, 5] == 1.0 and ch[4, 6] == 0.0   # ganked, pas solo
    assert ch[12, 0] == 1.0 and ch[12, 5] == 0.0 and ch[12, 6] == 1.0  # solo, pas ganked


def test_event_channels_opp_death():
    t = _fake_timeline_with_events()
    ch = bsd._event_channels(t, pid=1, opp_pid=6, enemy_jungle_pid=8)
    assert ch[8, 1] == 1.0


def test_event_channels_drake_up_respawn():
    t = _fake_timeline_with_events()
    ch = bsd._event_channels(t, pid=1, opp_pid=6, enemy_jungle_pid=8)
    assert ch[5, 3] == 1.0 and ch[6, 3] == 1.0   # up avant le kill (first spawn 5:00)
    assert ch[9, 3] == 0.0                       # down après kill (respawn 5min)
    assert ch[12, 3] == 1.0                       # respawn à 12:00
    assert ch[0, 4] == 0.0                        # baron pas up en early (first 25:00)


def test_event_channels_recall():
    t = _fake_timeline_with_events()
    ch = bsd._event_channels(t, pid=1, opp_pid=6, enemy_jungle_pid=8)
    assert ch[2, 2] == 1.0                        # achat à 2:30 -> recall minute 2


def test_build_sequence_shapes_and_mask():
    m = _fake_match(); t = _fake_timeline(3)
    out = bsd.build_sequence(m, t, "p_BOTTOM_100")
    assert out is not None
    seq, mask = out
    assert seq.shape == (40, 27) and seq.dtype == np.float32
    assert mask.shape == (40,) and mask.dtype == bool
    assert mask.sum() == 3              # 3 minutes valides
    assert mask[0] and mask[1] and mask[2] and not mask[3]


def test_build_sequence_values_minute1():
    m = _fake_match(); t = _fake_timeline(3)
    seq, mask = bsd.build_sequence(m, t, "p_BOTTOM_100")
    # pid1 @min1 : x=1001,y=2000,gold=600,cur=50,xp=100,lvl=2,cs=5,jg=1
    self_state = [1001 / 14800, 2000 / 14800, 600.0, 50.0, 100.0, 2.0, 5.0, 1.0]
    # opp pid6 @min1 : x=6001,y=12000,gold=3100,cur=300,xp=600,lvl=2,cs=5,jg=1
    opp_state = [6001 / 14800, 12000 / 14800, 3100.0, 300.0, 600.0, 2.0, 5.0, 1.0]
    diffs = [600.0 - 3100.0,                 # gold diff
             (5.0 + 1.0) - (5.0 + 1.0),      # cs diff
             100.0 - 600.0,                  # xp diff
             2.0 - 2.0]                      # level diff
    expected = self_state + opp_state + diffs + [0.0] * 7
    np.testing.assert_allclose(seq[1], expected, rtol=1e-5)


def test_build_sequence_none_if_no_opponent_role():
    # match sans adversaire BOTTOM : opponent_pid None -> None
    puuids = ["p_bottom_100"] + [f"p_{r}_{t}" for t in (100, 200)
              for r in ("UTILITY", "JUNGLE", "MIDDLE", "TOP")]
    parts = [{"teamId": 100, "teamPosition": "BOTTOM", "championName": "a"}] + \
            [{"teamId": t, "teamPosition": r, "championName": f"{r}_{t}"}
             for t in (100, 200) for r in ("UTILITY", "JUNGLE", "MIDDLE", "TOP")]
    m = {"metadata": {"participants": puuids}, "info": {"participants": parts}}
    assert bsd.build_sequence(m, _fake_timeline(2), "p_bottom_100") is None


def test_champion_of():
    m = _fake_match()
    assert bsd.champion_of(m, "p_BOTTOM_100") == "BOTTOM_100"
    assert bsd.champion_of(m, "p_BOTTOM_200") == "BOTTOM_200"


def test_main_writes_npz(tmp_path, monkeypatch):
    # redirige DATASET_DIR vers tmp_path ; mock build_rank_map + _load_raw + adc_puuids
    monkeypatch.setattr(bsd, "DATASET_DIR", tmp_path)
    m = _fake_match(); t = _fake_timeline(3)
    monkeypatch.setattr(build_dataset, "build_rank_map",
                        lambda: ({"EUW1_1": "challenger"}, 0))
    monkeypatch.setattr(build_dataset, "_load_raw", lambda mid: (m, t))
    monkeypatch.setattr(build_dataset, "adc_puuids", lambda match: ["p_BOTTOM_100"])
    rc = bsd.main()
    assert rc == 0
    import numpy as np
    d = np.load(tmp_path / "adc_sequence_dataset.npz", allow_pickle=True)
    assert d["sequences"].shape == (1, 40, 27)
    assert d["mask"].sum() == 3
    assert list(d["rank"]) == ["challenger"]
    assert list(d["label_highelo"]) == [1]
    assert list(d["match_id"]) == ["EUW1_1"]