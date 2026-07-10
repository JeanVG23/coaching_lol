import positioning as P
import game_journal as J


# ------------------------------------------------------------------ fixtures
ME = "puuid-me"

_ROSTER = [
    # (championName, teamPosition, teamId) — pid = index + 1
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


def _pf(gold_total, gold_current, level=5, x=13000, y=2000):
    return {"totalGold": gold_total, "currentGold": gold_current,
            "level": level, "position": {"x": x, "y": y}}


def _kill(t_ms, victim, killer, assists=(), x=13000, y=2000):
    return {"type": "CHAMPION_KILL", "timestamp": t_ms, "victimId": victim,
            "killerId": killer, "assistingParticipantIds": list(assists),
            "position": {"x": x, "y": y}}


def _buy(t_ms, pid, item=1055):
    return {"type": "ITEM_PURCHASED", "timestamp": t_ms,
            "participantId": pid, "itemId": item}


def _undo(t_ms, pid, before):
    return {"type": "ITEM_UNDO", "timestamp": t_ms,
            "participantId": pid, "beforeId": before}


def _dragon_kill(t_ms):
    return {"type": "ELITE_MONSTER_KILL", "timestamp": t_ms,
            "monsterType": "DRAGON", "killerId": 10}


def _timeline(frames):
    return {"info": {"frames": frames}}


def _basic_timeline(events_by_frame=None):
    """Frames toutes les 60 s de 0 à 12 min, moi pid 1 vs ADC ennemi pid 6."""
    events_by_frame = events_by_frame or {}
    frames = []
    for minute in range(13):
        t = minute * 60000
        frames.append(_frame(
            t,
            {1: _pf(gold_total=400 * minute + 500, gold_current=1234, level=minute + 1),
             6: _pf(gold_total=300 * minute + 500, gold_current=800, level=minute + 1)},
            events_by_frame.get(minute, []),
        ))
    return _timeline(frames)


# ------------------------------------------------------------------ tests
def test_returns_none_off_sr():
    assert J.game_journal(_match(map_id=12), _basic_timeline(), ME) is None


def test_meta_fields():
    j = J.game_journal(_match(win=True), _basic_timeline(), ME)
    assert j["match_id"] == "EUW1_42"
    assert j["patch"] == "16.13"
    assert j["champion"] == "Zeri" and j["role"] == "BOTTOM"
    assert j["win"] is True
    assert j["duration_min"] == 30.0
    assert j["kda"] == {"kills": 5, "deaths": 3, "assists": 7}


def test_death_anchored_fields():
    # Mort à 4:30 en BOT, solo-kill de l'ADC ennemi, gold diff +400 -> ahead.
    tl = _basic_timeline({4: [_kill(270000, victim=1, killer=6)]})
    j = J.game_journal(_match(), tl, ME)
    assert len(j["deaths"]) == 1
    d = j["deaths"][0]
    assert d["t_ms"] == 270000 and d["clock"] == "4:30"
    assert d["zone"] == "BOT" and d["phase"] == "early"
    assert d["killer_champ"] == "Jinx" and d["killer_role"] == "BOTTOM"
    assert d["is_solo"] is True and d["is_ganked_by_jungle"] is False
    assert d["gold_state"] == "ahead"
    assert d["unspent_gold"] == 1234       # currentGold de la frame 4:00
    assert d["level"] == 5                 # level de la frame 4:00


def test_death_objective_imminent_dragon():
    # Mort à 4:30 : 1er drake spawn 5:00 -> imminent dans 30 s.
    tl = _basic_timeline({4: [_kill(270000, victim=1, killer=6)]})
    d = J.game_journal(_match(), tl, ME)["deaths"][0]
    assert d["objective"] == {"type": "DRAGON", "status": "imminent", "delta_s": 30}


def test_death_objective_up_then_reset_after_kill():
    # Mort à 5:40 : drake up depuis 40 s. Drake tué à 6:00 -> respawn 11:00 :
    # mort à 8:00 -> aucun objectif (fenêtre 90 s), mort à 10:30 -> imminent 30 s.
    tl = _basic_timeline({
        5: [_kill(340000, victim=1, killer=6)],
        6: [_dragon_kill(360000)],
        8: [_kill(480000, victim=1, killer=6)],
        10: [_kill(630000, victim=1, killer=6)],
    })
    deaths = J.game_journal(_match(), tl, ME)["deaths"]
    assert deaths[0]["objective"] == {"type": "DRAGON", "status": "up", "delta_s": 40}
    assert deaths[1]["objective"] is None
    assert deaths[2]["objective"] == {"type": "DRAGON", "status": "imminent", "delta_s": 30}


def test_death_gank_flag():
    tl = _basic_timeline({4: [_kill(270000, victim=1, killer=10, assists=[6])]})
    d = J.game_journal(_match(), tl, ME)["deaths"][0]
    assert d["is_ganked_by_jungle"] is True and d["is_solo"] is False


def test_recalls_cluster_purchases_and_skip_opening_buy():
    # Achat d'ouverture (0:10) exclu ; 2 achats à 5:10/5:15 = 1 visite ;
    # 1 achat à 10:05 = 2e visite. gold_before = currentGold de la frame précédente.
    tl = _basic_timeline({
        0: [_buy(10000, 1)],
        5: [_buy(310000, 1), _buy(315000, 1)],
        10: [_buy(605000, 1)],
    })
    j = J.game_journal(_match(), tl, ME)
    assert len(j["recalls"]) == 2
    r1, r2 = j["recalls"]
    assert r1["t_ms"] == 310000 and r1["clock"] == "5:10"
    assert r1["items_bought"] == 2
    assert r1["gold_before"] == 1234
    assert r2["items_bought"] == 1


def test_recalls_ignore_other_players():
    tl = _basic_timeline({5: [_buy(310000, 6)]})
    assert J.game_journal(_match(), tl, ME)["recalls"] == []


def test_journal_never_leaks_ml_only_features():
    tl = _basic_timeline({4: [_kill(270000, victim=1, killer=6)],
                          5: [_buy(310000, 1)]})
    j = J.game_journal(_match(), tl, ME)

    def keys_of(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys_of(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys_of(v)

    assert set(keys_of(j)) & P.ML_ONLY == set()


def test_recalls_capture_item_ids_in_purchase_order():
    tl = _basic_timeline({
        5: [_buy(310000, 1, item=1038), _buy(315000, 1, item=1055)],
        10: [_buy(605000, 1, item=3031)],
    })
    r1, r2 = J.game_journal(_match(), tl, ME)["recalls"]
    assert r1["item_ids"] == [1038, 1055]
    assert r2["item_ids"] == [3031]


def test_recalls_honor_item_undo():
    # Achat 1038 annulé 5 s plus tard, puis rachat 1036 : seul 1036 subsiste.
    tl = _basic_timeline({
        5: [_buy(310000, 1, item=1038), _undo(315000, 1, before=1038),
            _buy(320000, 1, item=1036)],
    })
    (r1,) = J.game_journal(_match(), tl, ME)["recalls"]
    assert r1["item_ids"] == [1036]
    assert r1["items_bought"] == 1


def test_recalls_undo_removes_last_matching_purchase_only():
    # Deux achats du même item, un seul undo -> il en reste un.
    tl = _basic_timeline({
        5: [_buy(310000, 1, item=1036), _buy(315000, 1, item=1036),
            _undo(320000, 1, before=1036)],
    })
    (r1,) = J.game_journal(_match(), tl, ME)["recalls"]
    assert r1["item_ids"] == [1036]


def test_recalls_undo_of_other_player_ignored():
    tl = _basic_timeline({
        5: [_buy(310000, 1, item=1038), _undo(315000, 6, before=1038)],
    })
    (r1,) = J.game_journal(_match(), tl, ME)["recalls"]
    assert r1["item_ids"] == [1038]
