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


def test_zone_presence_own_lane_and_river():
    # joueur ADC (BOT) : early en bot (coin bas-droit) puis en river
    # BOT ≈ x grand & y petit ; RIVER ≈ près de la diagonale, loin des bords.
    snaps = [
        (0, 0, {1: (13000, 2000)}, {}),      # early, BOT (coin bas-droit)
        (60000, 1, {1: (13000, 2000)}, {}),  # early, BOT
        (120000, 2, {1: (10000, 5000)}, {}), # early, JUNGLE/RIVER (loin des 3 lanes)
    ]
    r = P._zone_presence(snaps, 1, "BOTTOM")
    assert r["frac_own_lane_early"] == 2 / 3
    assert r["frac_river_early"] == 1 / 3


def test_zone_presence_roam_mid_counts_other_lanes():
    # mid phase (minute 15+) : 1 frame en MID (roam hors BOT), 1 en BOT
    snaps = [
        (900000, 15, {1: (7400, 7400)}, {}),   # MID (sur diagonale)
        (960000, 16, {1: (13000, 2000)}, {}),  # BOT (own lane)
    ]
    r = P._zone_presence(snaps, 1, "BOTTOM")
    assert r["frac_roam_mid"] == 1 / 2


def test_zone_presence_none_when_no_frames():
    assert P._zone_presence([], 1, "BOTTOM")["frac_own_lane_early"] is None


def test_depth_sign_and_symmetry():
    # team 100 : base bas-gauche (petit x+y) -> profondeur négative chez soi,
    # positive en terrain ennemi (grand x+y). team 200 : inverse.
    deep_enemy_for_100 = P._depth(13000, 13000, 100)
    assert deep_enemy_for_100 > 0
    assert P._depth(1000, 1000, 100) < 0
    assert P._depth(13000, 13000, 200) < 0          # même point, chez soi pour 200
    assert abs(P._depth(13000, 13000, 100) + P._depth(13000, 13000, 200)) < 1e-6


def test_territory_aggregates():
    # team 100 : 1 frame chez soi (depth<0), 1 frame deep enemy (depth>seuil)
    snaps = [
        (0, 0, {1: (1000, 1000)}, {}),
        (60000, 1, {1: (13000, 13000)}, {}),
    ]
    r = P._territory(snaps, 1, 100)
    assert r["frac_enemy_half"] == 0.5
    assert r["max_map_depth"] == P._depth(13000, 13000, 100)
    assert r["avg_map_depth"] == P._depth(13000, 13000, 100) / 2  # chez soi clampé à 0
    assert r["frac_overextended"] == 0.5


def test_in_base_box():
    assert P._in_base(2000, 2000, 100) is True
    assert P._in_base(8000, 8000, 100) is False
    assert P._in_base(13000, 13000, 200) is True
    assert P._in_base(8000, 8000, 200) is False


def test_base_and_isolation():
    # pid=1 ; allié 2 à distance connue ; allié 3 plus loin -> min retenu
    snaps = [
        (0, 0, {1: (2000, 2000), 2: (2300, 2000), 3: (9000, 9000)}, {}),  # base + allié à 300
        (60000, 1, {1: (8000, 8000), 2: (8000, 8400)}, {}),               # hors base + allié à 400
    ]
    r = P._base_and_isolation(snaps, 1, 100, [1, 2, 3])
    assert r["frac_base"] == 0.5
    assert abs(r["avg_dist_to_ally"] - (300 + 400) / 2) < 1e-6
