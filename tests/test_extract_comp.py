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


def test_comp_resolves_six_champions():
    g = rl.extract_game(_match(), _timeline(), "puuid0", rank="test")
    comp = g["comp"]
    assert comp["self_adc"] == "Zeri"
    assert comp["self_support"] == "Lulu"
    assert comp["enemy_adc"] == "Caitlyn"
    assert comp["enemy_support"] == "Leona"
    assert comp["self_jungle"] == "Graves"
    assert comp["enemy_jungle"] == "JarvanIV"
    assert comp["enemy_mid"] == "Syndra"
