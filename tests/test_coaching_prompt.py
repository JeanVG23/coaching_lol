import json

import positioning as P
import prompt as PR


def _payload():
    return {"meta": {"player": "spadzze", "scope": "adc", "target": "challenger",
                     "outcome_focus": "loss", "n_games_me": 15, "low_sample": True},
            "signals": [{"group": "positioning", "key": "frac_roam_mid",
                         "you": 0.5, "ref": 0.7, "delta": -0.2, "notable": True}],
            "context": {}}


def test_system_encodes_asymmetry_and_depth_rules():
    s = PR.SYSTEM.lower()
    assert "asym" in s                       # règle d'asymétrie présente
    assert "profondeur" in s                 # nuance profondeur présente
    assert "descriptive_only" in PR.SYSTEM   # le LLM sait ne pas prescrire ces signaux


def test_render_returns_system_and_user_with_payload():
    system, user = PR.render(_payload())
    assert system == PR.SYSTEM
    assert "15 dernières games" in user
    assert "challenger" in user
    assert json.loads(user[user.index("{"):user.rindex("}") + 1])  # le payload JSON est inclus


def test_prompt_never_leaks_ml_only_feature_names():
    system, user = PR.render(_payload())
    for k in P.ML_ONLY:
        assert k not in system and k not in user
