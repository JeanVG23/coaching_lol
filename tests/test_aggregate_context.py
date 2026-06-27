import riotlib as rl

DD = {"Caitlyn": {"attackrange": 650}, "Leona": {"attackrange": 125},
      "Lux": {"attackrange": 550}}
TR = {"Leona": {"lane_pattern": "all_in"}, "Lux": {"lane_pattern": "poke"},
      "Caitlyn": {"lane_pattern": "poke"}}


def _game(enemy_support, gd10):
    return {"match_id": f"m{enemy_support}", "champion": "Zeri", "role": "BOTTOM",
            "win": True, "deaths": [], "lane": {k: None for k in rl.LANE_KEYS} | {"gd10": gd10},
            "comp": {"self_adc": "Zeri", "self_support": None, "enemy_adc": "Caitlyn",
                     "enemy_support": enemy_support, "self_jungle": None,
                     "enemy_jungle": None, "enemy_mid": None}}


def test_by_lane_context_splits_by_pattern(monkeypatch):
    monkeypatch.setattr(rl.cp, "load_ddragon", lambda: DD)
    monkeypatch.setattr(rl.cp, "load_traits", lambda: TR)
    games = [_game("Leona", -300), _game("Leona", -200), _game("Lux", 50)]
    agg = rl.aggregate(games, "adc")
    lp = agg["by_lane_context"]["lane_pattern"]
    assert lp["all_in"]["n_games"] == 2
    assert lp["poke"]["n_games"] == 1
    assert lp["all_in"]["lane"]["gd10"] == -250  # médiane de -300/-200
