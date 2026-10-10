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


def test_jsonl_and_atomic_io(tmp_path):
    import riotlib as rl
    p_jsonl = tmp_path / "test.jsonl"
    rows = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    rl.write_jsonl(p_jsonl, rows)

    # iter_jsonl streaming
    iter_rows = list(rl.iter_jsonl(p_jsonl))
    assert iter_rows == rows

    # read_jsonl
    assert rl.read_jsonl(p_jsonl) == rows

    # append_jsonl
    rl.append_jsonl(p_jsonl, {"id": 3, "name": "c"})
    assert rl.read_jsonl(p_jsonl) == rows + [{"id": 3, "name": "c"}]

    # non existent
    assert list(rl.iter_jsonl(tmp_path / "missing.jsonl")) == []
    assert rl.read_jsonl(tmp_path / "missing.jsonl") == []

    # dump_json_atomic
    p_json = tmp_path / "test.json"
    rl.dump_json_atomic(p_json, {"ok": True}, allow_nan=False)
    assert p_json.exists()
    assert rl.read_json(p_json) == {"ok": True}

    # read_json missing
    assert rl.read_json(tmp_path / "missing.json") is None

