import champion_profiles as cp


FAKE_DDRAGON = {
    "Caitlyn": {"attackrange": 650, "tags": ["Marksman"]},
    "Leona": {"attackrange": 125, "tags": ["Tank", "Support"]},
}
FAKE_TRAITS = {
    "Caitlyn": {"power_curve": "early", "lane_pattern": "poke"},
    "Leona": {"lane_pattern": "all_in"},
}


def test_vector_merges_ddragon_and_traits():
    v = cp.champion_vector("Caitlyn", traits=FAKE_TRAITS, ddragon=FAKE_DDRAGON)
    assert v["range_class"] == "ranged"
    assert v["tags"] == ["Marksman"]
    assert v["power_curve"] == "early"
    assert v["lane_pattern"] == "poke"


def test_vector_melee_range_class():
    v = cp.champion_vector("Leona", traits=FAKE_TRAITS, ddragon=FAKE_DDRAGON)
    assert v["range_class"] == "melee"
    assert v["lane_pattern"] == "all_in"


def test_vector_unknown_champion_degrades_cleanly():
    v = cp.champion_vector("Nobody", traits=FAKE_TRAITS, ddragon=FAKE_DDRAGON)
    assert v["range_class"] == "unknown"
    assert v["lane_pattern"] == "unknown"
    assert v["power_curve"] == "unknown"
    assert v["tags"] == []
