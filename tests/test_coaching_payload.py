import json

import pytest

import positioning as P
import payload as PL


def test_pos_meta_keys_match_coaching_safe():
    assert set(PL.POS_META) == P.COACHING_SAFE


def test_pos_signals_depth_always_descriptive_never_notable():
    mf = {"positioning": {"max_map_depth": 2728.0, "frac_roam_mid": 0.50}}
    rf = {"positioning": {"max_map_depth": 1633.0, "frac_roam_mid": 0.70}}
    out = {s["key"]: s for s in PL._pos_signals(mf, rf)}
    assert out["max_map_depth"]["descriptive_only"] is True
    assert out["max_map_depth"]["notable"] is False          # malgré delta énorme
    assert out["frac_roam_mid"]["notable"] is True            # |−0.20| ≥ 0.08
    assert "descriptive_only" not in out["frac_roam_mid"]


def test_pos_signals_only_coaching_safe_keys():
    mf = {"positioning": {k: 0.5 for k in P.ALL_FEATURES}}     # inclut ML_ONLY
    rf = {"positioning": {k: 0.5 for k in P.ALL_FEATURES}}
    keys = {s["key"] for s in PL._pos_signals(mf, rf)}
    assert keys <= P.COACHING_SAFE
    assert keys.isdisjoint(P.ML_ONLY)


def test_lane_signals_thresholds():
    mf = {"lane": {"gd10": 100, "csd14": 0}}
    rf = {"lane": {"gd10": -100, "csd14": -5}}
    out = {s["key"]: s for s in PL._lane_signals(mf, rf)}
    assert out["gd10"]["delta"] == 200 and out["gd10"]["notable"] is True   # >150
    assert out["csd14"]["delta"] == 5 and out["csd14"]["notable"] is True   # ≥2 cs


def test_zone_phase_signals_top_overdeaths():
    mf = {"by_zone_phase": {"BOT|mid": 0.29, "MID|late": 0.10}}
    rf = {"by_zone_phase": {"BOT|mid": 0.05, "MID|late": 0.11}}
    out = PL._zone_phase_signals(mf, rf)
    assert out[0]["key"] == "BOT|mid" and out[0]["notable"] is True         # Δ +0.24


def test_build_reads_gold_and_flags_low_sample(tmp_path):
    facet = {"n_games": 3, "deaths_per_game": 6.0,
             "lane": {"gd10": -100, "gd14": 0, "gd20": 0, "csd10": 0, "csd14": -5},
             "positioning": {k: 0.5 for k in P.COACHING_SAFE},
             "death_gold_state": {"ahead": 0.3, "even": 0.2, "behind": 0.5},
             "by_zone_phase": {"BOT|mid": 0.3}}
    agg = {"scope": "adc", "patch": "16.13", "n_games": 3, "winrate": 0.33,
           "overall": facet, "win": facet, "loss": facet, "by_lane_context": {}}
    for who, n in (("personal/spadzze", 3), ("referentiel/challenger", 1000)):
        a = dict(agg); a["n_games"] = n
        d = tmp_path / who / "adc"; d.mkdir(parents=True)
        (d / "aggregate.json").write_text(json.dumps(a))
    pl = PL.build("spadzze", "adc", "challenger", "loss", gold_dir=tmp_path)
    assert pl["meta"]["low_sample"] is True                  # 3 < 30
    assert pl["meta"]["n_games_ref"] == 1000
    # aucune feature ML_ONLY nulle part dans le payload sérialisé
    blob = json.dumps(pl)
    assert all(k not in blob for k in P.ML_ONLY)
    assert any(s["group"] == "positioning" for s in pl["signals"])


def test_build_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        PL.build("ghost", "adc", "challenger", "loss", gold_dir=tmp_path)


def test_zone_phase_ordering_is_deterministic_on_ties():
    """Régression 2026-09-04, trouvée par la parité Python/TypeScript.

    `_zone_phase_signals` itérait `set(me) | set(ref)` : à delta égal, l'ordre
    dépendait du hachage des chaînes, donc du PYTHONHASHSEED. Deux exécutions
    pouvaient donner deux payloads différents pour la même game, et le TypeScript
    (qui part de clés triées) divergeait. Seul un ex aequo rendait le défaut visible.
    """
    mf = {"by_zone_phase": {"MID|late": 0.2, "JUNGLE/RIVER|late": 0.1,
                            "JUNGLE/RIVER|early": 0.1, "BOT|early": 0.1}}
    rf = {"by_zone_phase": {}}
    keys = [s["key"] for s in PL._zone_phase_signals(mf, rf)]
    assert keys == ["MID|late", "BOT|early", "JUNGLE/RIVER|early", "JUNGLE/RIVER|late"]
