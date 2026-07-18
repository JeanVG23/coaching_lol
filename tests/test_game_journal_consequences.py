import game_journal as J


ME = "puuid-me"

_ROSTER = [
    ("Zeri", "BOTTOM", 100), ("Lulu", "UTILITY", 100), ("Ahri", "MIDDLE", 100),
    ("Garen", "TOP", 100), ("Vi", "JUNGLE", 100),
    ("Jinx", "BOTTOM", 200), ("Thresh", "UTILITY", 200), ("Orianna", "MIDDLE", 200),
    ("Darius", "TOP", 200), ("LeeSin", "JUNGLE", 200),
]


def _match(map_id=11, win=True):
    parts = [{"championName": c, "teamPosition": r, "teamId": t,
              "win": win if t == 100 else not win,
              "kills": 5, "deaths": 3, "assists": 7}
             for c, r, t in _ROSTER]
    return {
        "metadata": {"matchId": "EUW1_42",
                     "participants": [ME] + [f"p{i}" for i in range(2, 11)]},
        "info": {"mapId": map_id, "gameVersion": "16.13.790.6961",
                 "gameDuration": 1800, "participants": parts},
    }


def _frame(t_ms, pframes, events=()):
    return {"timestamp": t_ms, "events": list(events),
            "participantFrames": {str(pid): pf for pid, pf in pframes.items()}}


def _pf(gold_total, gold_current=1234, level=5, x=13000, y=2000):
    return {"totalGold": gold_total, "currentGold": gold_current,
            "level": level, "position": {"x": x, "y": y}}


def _kill(t_ms, victim, killer, assists=(), x=13000, y=2000):
    return {"type": "CHAMPION_KILL", "timestamp": t_ms, "victimId": victim,
            "killerId": killer, "assistingParticipantIds": list(assists),
            "position": {"x": x, "y": y}}


def _monster_kill(t_ms, killer_team, monster="BARON_NASHOR", killer_id=None):
    ev = {"type": "ELITE_MONSTER_KILL", "timestamp": t_ms, "monsterType": monster,
          "killerTeamId": killer_team}
    if killer_id is not None:
        ev["killerId"] = killer_id
    return ev


def _building_kill(t_ms, losing_team, building="TOWER_BUILDING",
                   lane="MID_LANE", tower=None):
    ev = {"type": "BUILDING_KILL", "timestamp": t_ms, "teamId": losing_team,
          "buildingType": building, "laneType": lane}
    if tower is not None:
        ev["towerType"] = tower
    return ev


def _timeline(frames):
    return {"info": {"frames": frames}}


def _basic_timeline(events_by_frame=None, minutes=40, my_gold=400, opp_gold=300):
    events_by_frame = events_by_frame or {}
    frames = []
    for minute in range(minutes + 1):
        t = minute * 60000
        frames.append(_frame(
            t,
            {1: _pf(gold_total=my_gold * minute + 500, level=min(minute + 1, 18)),
             6: _pf(gold_total=opp_gold * minute + 500, level=min(minute + 1, 18))},
            events_by_frame.get(minute, []),
        ))
    return _timeline(frames)


def _death_at(tl):
    return J.game_journal(_match(), tl, ME)["deaths"][0]


def test_objective_lost_within_window():
    # Mort à 26:04, Baron pris par l'ennemi 40 s après -> conséquence.
    tl = _basic_timeline({26: [_kill(1564000, victim=1, killer=6)],
                          27: [_monster_kill(1604000, killer_team=200)]})
    cons = _death_at(tl)["consequences"]
    assert cons["objectives_lost"] == [
        {"type": "BARON_NASHOR", "clock": "26:44", "delta_s": 40}]


def test_objective_outside_window_excluded():
    # Baron pris 70 s après la mort -> hors fenêtre 60 s.
    tl = _basic_timeline({26: [_kill(1564000, victim=1, killer=6)],
                          27: [_monster_kill(1634000, killer_team=200)]})
    assert "objectives_lost" not in _death_at(tl).get("consequences", {})


def test_objective_taken_by_my_team_excluded():
    tl = _basic_timeline({26: [_kill(1564000, victim=1, killer=6)],
                          27: [_monster_kill(1604000, killer_team=100)]})
    assert "objectives_lost" not in _death_at(tl).get("consequences", {})


def test_objective_killer_team_fallback_via_killer_id():
    # Pas de killerTeamId -> attribution via killerId (pid 10 = équipe 200).
    ev = _monster_kill(1604000, killer_team=None, killer_id=10)
    del ev["killerTeamId"]
    tl = _basic_timeline({26: [_kill(1564000, victim=1, killer=6)], 27: [ev]})
    assert _death_at(tl)["consequences"]["objectives_lost"][0]["delta_s"] == 40


def test_building_lost_within_window_mine_only():
    # teamId d'un BUILDING_KILL = équipe qui PERD le bâtiment.
    tl = _basic_timeline({
        26: [_kill(1564000, victim=1, killer=6),
             _building_kill(1590000, losing_team=100, tower="INNER_TURRET"),
             _building_kill(1595000, losing_team=200)],
    })
    cons = _death_at(tl)["consequences"]
    assert cons["buildings_lost"] == [
        {"type": "INNER_TURRET", "lane": "MID_LANE", "clock": "26:30"}]


def test_gold_swing_computed_from_team_frames():
    # Écart (moi - ennemi) : +100/min avant la mort. Frame avant = 26:00,
    # première frame >= mort+90 s = 28:00 -> swing = (28-26) * 100 = +200.
    tl = _basic_timeline({26: [_kill(1564000, victim=1, killer=6)]})
    assert _death_at(tl)["consequences"]["team_gold_swing_90s"] == 200


def test_gold_swing_null_when_no_frame_after():
    # Mort en toute fin de game : aucune frame >= mort+90 s -> pas de swing,
    # et rien d'autre dans la fenêtre -> pas de clé consequences du tout.
    tl = _basic_timeline({40: [_kill(2400000, victim=1, killer=6)]})
    assert "consequences" not in _death_at(tl)


def test_no_consequences_key_when_window_empty():
    tl = _basic_timeline({10: [_kill(600000, victim=1, killer=6)]})
    d = _death_at(tl)
    # Le swing gold existe (frames dispo) donc la clé existe, mais sans events.
    assert "objectives_lost" not in d["consequences"]
    assert "buildings_lost" not in d["consequences"]
    assert d["consequences"]["team_gold_swing_90s"] == 200
