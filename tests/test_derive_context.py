import champion_profiles as cp

DD = {
    "Cait": {"attackrange": 650}, "Leona": {"attackrange": 125},
    "Zeri": {"attackrange": 500}, "Lux": {"attackrange": 550},
    "Jarvan": {"attackrange": 175}, "Karthus": {"attackrange": 450},
    "Ahri": {"attackrange": 550}, "Cass": {"attackrange": 550},
    "Vex": {"attackrange": 550},
}
TR = {
    "Cait": {"lane_pattern": "poke"}, "Leona": {"lane_pattern": "all_in"},
    "Zeri": {"lane_pattern": "scaling"}, "Lux": {"lane_pattern": "poke"},
    "Jarvan": {"gank_threat": "high", "playstyle": "ganking"},
    "Karthus": {"gank_threat": "low", "playstyle": "farming"},
    "Ahri": {"roam": "high"}, "Cass": {"roam": "low"},
    "Vex": {"roam": "med"},
}


def comp(**kw):
    base = dict(self_adc=None, self_support=None, enemy_adc=None,
                enemy_support=None, self_jungle=None, enemy_jungle=None, enemy_mid=None)
    base.update(kw)
    return base


def test_lane_pattern_all_in_when_enemy_support_engages():
    c = comp(enemy_adc="Cait", enemy_support="Leona")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["lane_pattern"] == "all_in"


def test_lane_pattern_poke():
    c = comp(enemy_adc="Cait", enemy_support="Lux")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["lane_pattern"] == "poke"


def test_lane_pattern_unknown_when_no_traits():
    c = comp(enemy_adc="Ghost", enemy_support="Phantom")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["lane_pattern"] == "unknown"


def test_gank_exposure_high_then_mitigated():
    # jgl ennemi high (+2) + mid roam high (+2) = 4 => high
    c = comp(enemy_jungle="Jarvan", enemy_mid="Ahri")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["gank_exposure"] == "high"
    # ton jgl ganking attenue (-1) => 3 => med
    c2 = comp(enemy_jungle="Jarvan", enemy_mid="Ahri", self_jungle="Jarvan")
    assert cp.derive_context(c2, traits=TR, ddragon=DD)["gank_exposure"] == "med"


def test_gank_exposure_low():
    c = comp(enemy_jungle="Karthus", enemy_mid="Cass")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["gank_exposure"] == "low"


def test_gank_exposure_unknown():
    c = comp()  # tout None
    assert cp.derive_context(c, traits=TR, ddragon=DD)["gank_exposure"] == "unknown"


def test_lane_pattern_scaling_when_both_enemy_scaling():
    # Zeri (scaling) + a sustain/scaling support => "scaling"
    c = comp(enemy_adc="Zeri", enemy_support="Zeri")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["lane_pattern"] == "scaling"


def test_gank_exposure_partial_unknown_still_scores():
    # enemy_jungle unknown, enemy_mid high roam (+2), self_jungle unknown => score 2 => med
    c = comp(enemy_mid="Ahri")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["gank_exposure"] == "med"


def test_gank_exposure_score_one_is_low():
    # enemy_mid med roam only (+1) => score 1 => low (guards the <=1 boundary)
    c = comp(enemy_mid="Vex")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["gank_exposure"] == "low"
