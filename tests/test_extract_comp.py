import riotlib as rl


def _match():
    # index 0 = moi (BOTTOM, team 100). 10 participants, 2 équipes.
    def p(team, role, champ):
        return {"teamId": team, "teamPosition": role, "championName": champ, "win": True}
    parts = [
        p(100, "BOTTOM", "Zeri"),    # moi
        p(100, "UTILITY", "Lulu"),
        p(100, "JUNGLE", "Graves"),
        p(100, "MIDDLE", "Ahri"),
        p(100, "TOP", "Aatrox"),
        p(200, "BOTTOM", "Caitlyn"),
        p(200, "UTILITY", "Leona"),
        p(200, "JUNGLE", "JarvanIV"),
        p(200, "MIDDLE", "Syndra"),
        p(200, "TOP", "Sett"),
    ]
    return {"metadata": {"matchId": "T1", "participants": [f"puuid{i}" for i in range(10)]},
            "info": {"mapId": 11, "queueId": 420, "gameVersion": "15.13.1.1", "participants": parts}}


def _timeline():
    return {"info": {"frames": []}}


def test_comp_resolves_ten_champions():
    g = rl.extract_game(_match(), _timeline(), "puuid0", rank="test")
    comp = g["comp"]
    assert comp["self_top"] == "Aatrox"
    assert comp["self_jungle"] == "Graves"
    assert comp["self_mid"] == "Ahri"
    assert comp["self_adc"] == "Zeri"
    assert comp["self_support"] == "Lulu"
    assert comp["enemy_top"] == "Sett"
    assert comp["enemy_jungle"] == "JarvanIV"
    assert comp["enemy_mid"] == "Syndra"
    assert comp["enemy_adc"] == "Caitlyn"
    assert comp["enemy_support"] == "Leona"
    assert "sides" in g
    assert "objectives" in g


def test_extract_game_characterizes_combat_objective_and_economy_metrics():
    match = _match()
    positions = {
        str(pid): {"position": {"x": 1000 + pid, "y": 1000 + pid},
                   "totalGold": 1000, "xp": 500, "minionsKilled": 20,
                   "jungleMinionsKilled": 0}
        for pid in range(1, 11)
    }
    positions["1"].update({"position": {"x": 1000, "y": 1000}, "totalGold": 2000})
    positions["6"].update({"position": {"x": 13000, "y": 2000}, "totalGold": 1500})
    combat_events = [
        {"type": "CHAMPION_KILL", "timestamp": 300000, "victimId": 1,
         "killerId": 8, "assistingParticipantIds": [6],
         "position": {"x": 13000, "y": 2000}},
        {"type": "CHAMPION_KILL", "timestamp": 300000, "victimId": 6,
         "killerId": 1, "assistingParticipantIds": [2]},
        {"type": "CHAMPION_KILL", "timestamp": 300000, "victimId": 7,
         "killerId": 2, "assistingParticipantIds": [1]},
        {"type": "TURRET_PLATE_DESTROYED", "timestamp": 300000,
         "laneType": "BOT_LANE", "teamId": 200},
        {"type": "TURRET_PLATE_DESTROYED", "timestamp": 300000,
         "laneType": "BOT_LANE", "teamId": 200},
        {"type": "TURRET_PLATE_DESTROYED", "timestamp": 300000,
         "laneType": "BOT_LANE", "teamId": 100},
    ]
    dragon_event = {
        "type": "ELITE_MONSTER_KILL", "timestamp": 360000,
        "monsterType": "DRAGON", "position": {"x": 13000, "y": 2000},
    }
    timeline = {"info": {"frames": [
        {"timestamp": 300000, "participantFrames": positions, "events": combat_events},
        {"timestamp": 360000, "participantFrames": positions | {
            "1": positions["1"] | {"position": {"x": 13000, "y": 2000}}
        }, "events": [dragon_event]},
    ]}}

    game = rl.extract_game(match, timeline, "puuid0", rank="test")

    assert game["deaths"] == [{
        "minute": 5, "phase": "early", "zone": "BOT",
        "killer_role": "JUNGLE", "killer_champ": "JarvanIV",
        "gold_state": "ahead", "is_solo": False,
        "is_ganked_by_jungle": True, "is_2v2": False,
    }]
    assert game["kills"] == [{
        "minute": 5, "phase": "early", "zone": "MID",
        "victim_role": "BOTTOM", "victim_champ": "Caitlyn",
        "is_solo": False, "is_2v2": True,
    }]
    assert game["assists"] == [{
        "minute": 5, "phase": "early", "zone": "MID",
        "killer_role": "UTILITY", "killer_champ": "Lulu",
        "victim_role": "UTILITY", "victim_champ": "Leona",
        "is_2v2": True,
    }]
    assert game["plates_diff_early"] == 1
    assert game["frames_in_base_early"] == 1
    assert game["avg_dragon_prox"] == 12042
