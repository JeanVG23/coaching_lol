import reextract_silver as rx


def test_reextract_game_adds_comp(tmp_path, monkeypatch):
    # match minimal caché
    def p(team, role, champ):
        return {"teamId": team, "teamPosition": role, "championName": champ, "win": False}
    parts = [p(100, "BOTTOM", "Zeri"), p(100, "UTILITY", "Lulu"), p(100, "JUNGLE", "Graves"),
             p(100, "MIDDLE", "Ahri"), p(100, "TOP", "Aatrox"), p(200, "BOTTOM", "Caitlyn"),
             p(200, "UTILITY", "Leona"), p(200, "JUNGLE", "JarvanIV"),
             p(200, "MIDDLE", "Syndra"), p(200, "TOP", "Sett")]
    match = {"metadata": {"matchId": "EUW1_X", "participants": [f"p{i}" for i in range(10)]},
             "info": {"mapId": 11, "queueId": 420, "gameVersion": "15.13.1.1",
                      "participants": parts}}
    timeline = {"info": {"frames": []}}
    out = rx.reextract_one(match, timeline, "p0", rank="challenger")
    assert out is not None
    assert out["comp"]["enemy_jungle"] == "JarvanIV"
