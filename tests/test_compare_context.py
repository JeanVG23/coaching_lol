import compare


def _agg(buckets, overall_gd10):
    # buckets: {name: (n_games, gd10)}
    lp = {name: {"n_games": n, "lane": {"gd10": gd}, "deaths_per_game": 0}
          for name, (n, gd) in buckets.items()}
    return {"overall": {"n_games": 99, "lane": {"gd10": overall_gd10}},
            "by_lane_context": {"lane_pattern": lp, "gank_exposure": {}}}


def test_uses_matching_bucket_when_ref_has_enough():
    me = _agg({"all_in": (6, -510)}, -400)
    ref = _agg({"all_in": (20, -150)}, -100)
    r = compare.context_benchmark(me, ref, "lane_pattern", "overall")
    assert r["bucket"] == "all_in"
    assert r["gd10_me"] == -510 and r["gd10_ref"] == -150
    assert r["fallback"] is False


def test_falls_back_to_global_when_ref_too_thin():
    me = _agg({"all_in": (6, -510)}, -400)
    ref = _agg({"all_in": (3, -150)}, -100)   # 3 < MIN_CONTEXT_N
    r = compare.context_benchmark(me, ref, "lane_pattern", "overall")
    assert r["fallback"] is True
    assert r["gd10_ref"] == -100              # repli sur overall
    assert r["reason"]
