import champion_profiles as cp


def test_load_traits_reads_seed():
    traits = cp.load_traits()
    assert isinstance(traits, dict)
    assert "Caitlyn" in traits            # présent dans le seed
    assert traits["Caitlyn"]["lane_pattern"] in ("poke", "all_in", "sustain", "scaling")


def test_loaders_feed_real_vector():
    # Zeri doit être ranged via DDragon caché + scaling via le seed
    v = cp.champion_vector("Zeri")
    assert v["range_class"] in ("ranged", "unknown")   # ranged si DDragon fetché
    assert v["lane_pattern"] == "scaling"
