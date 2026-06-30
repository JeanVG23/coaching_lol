import positioning as P


def _tl(frames):
    """Timeline minimale : frames = list de (t_ms, {pid: (x,y,level)})."""
    out = []
    for t, parts in frames:
        pf = {str(pid): {"position": {"x": x, "y": y}, "level": lvl}
              for pid, (x, y, lvl) in parts.items()}
        out.append({"timestamp": t, "participantFrames": pf, "events": []})
    return {"info": {"frames": out}}


def test_manifest_disjoint_and_proxies_ml_only():
    assert P.ML_ONLY & P.COACHING_SAFE == set()
    assert {"frac_deaths_in_fog", "avg_unaccounted_enemies",
            "overext_x_unaccounted"} <= P.ML_ONLY
    assert P.ALL_FEATURES == P.COACHING_SAFE | P.ML_ONLY
    assert len(P.ALL_FEATURES) == 17


def test_build_snaps_parses_positions_and_levels():
    tl = _tl([(0, {1: (100, 200, 1)}), (60000, {1: (300, 400, 3)})])
    snaps = P._build_snaps(tl)
    assert len(snaps) == 2
    assert snaps[0] == (0, 0, {1: (100, 200)}, {1: 1})
    assert snaps[1] == (60000, 1, {1: (300, 400)}, {1: 3})


def test_build_snaps_skips_missing_position():
    tl = {"info": {"frames": [{"timestamp": 0, "events": [],
          "participantFrames": {"1": {"level": 2}}}]}}
    snaps = P._build_snaps(tl)
    assert snaps[0][2] == {}          # pas de position
    assert snaps[0][3] == {1: 2}      # level présent
