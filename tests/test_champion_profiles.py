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


def test_vector_resolves_case_insensitive_name():
    # API returns "FiddleSticks"; DDragon/traits key is "Fiddlesticks"
    dd = {"Fiddlesticks": {"attackrange": 480, "tags": ["Mage"]}}
    tr = {"Fiddlesticks": {"playstyle": "ganking", "gank_threat": "high"}}
    v = cp.champion_vector("FiddleSticks", traits=tr, ddragon=dd)
    assert v["range_class"] == "melee"        # 480 < 500
    assert v["gank_threat"] == "high"          # resolved despite casing


FAKE_ITEM_JSON = {"data": {
    "1038": {"name": "B.F. Sword", "gold": {"total": 1300}},
    "3031": {"name": "Infinity Edge", "gold": {"total": 3450}},
    "2055": {"name": "Control Ward", "gold": {"total": 75}},
}}


def test_parse_items_maps_id_to_name_and_cost():
    items = cp._parse_items(FAKE_ITEM_JSON["data"])
    assert items[1038] == {"name": "B.F. Sword", "cost": 1300}
    assert items[3031]["cost"] == 3450


def test_load_items_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "STATIC_DIR", tmp_path)
    cp.load_items.cache_clear()
    assert cp.load_items() == {}
    cp.load_items.cache_clear()


def test_load_items_reads_cached_file(tmp_path, monkeypatch):
    import json
    dest = tmp_path / "ddragon" / cp.DDRAGON_VERSION
    dest.mkdir(parents=True)
    (dest / "item.json").write_text(json.dumps(FAKE_ITEM_JSON))
    monkeypatch.setattr(cp, "STATIC_DIR", tmp_path)
    cp.load_items.cache_clear()
    assert cp.load_items()[2055] == {"name": "Control Ward", "cost": 75}
    cp.load_items.cache_clear()
