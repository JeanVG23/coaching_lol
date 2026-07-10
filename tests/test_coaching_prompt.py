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


def test_system_gates_strengths_on_notable_favorable_signals():
    # Anti-filler : 1 à 3 forces, chacune adossée à un signal notable favorable —
    # jamais de remplissage pour atteindre 3.
    s = PR.SYSTEM
    assert "1 à 3" in s
    assert "remplissage" in s.lower()


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


def _game_payload():
    return {"meta": {"player": "spadzze", "scope": "adc", "target": "challenger",
                     "kind": "game", "match_id": "EUW1_42", "champion": "Zeri",
                     "opponent": "Jinx", "role": "BOTTOM", "win": False,
                     "duration_min": 30.0, "patch": "16.13",
                     "kda": {"kills": 5, "deaths": 3, "assists": 7}},
            "journal": {"deaths": [], "recalls": []},
            "benchmarks": {"outcome": "loss", "deaths_per_game": 4.2}}


def test_system_game_encodes_anchor_asymmetry_and_recall_caveat():
    s = PR.SYSTEM_GAME
    assert "asym" in s.lower()          # règle d'asymétrie présente
    assert "horodatage" in s.lower()    # chaque erreur ancrée sur un moment mm:ss
    assert "plancher" in s.lower()      # gold_before des recalls = approximation basse


def test_system_game_requires_cause_and_death_context():
    # Feedback « je sais pas pourquoi je suis mort » / « aucune idée de pourquoi » :
    # le prompt doit exiger la cause (mécanisme) + restituer le contexte de mort du
    # journal (killer/gank/zone), pas seulement l'horodatage.
    s = PR.SYSTEM_GAME.lower()
    assert "cause" in s               # champ cause exigé
    assert "pourquoi" in s            # le POURQUOI, pas seulement le moment
    assert "killer" in s              # restituer killer_champ/killer_role du journal
    assert "gank" in s                # restituer is_ganked_by_jungle
    assert "comportement" in s        # forces = comportement, pas l'issue


def test_render_game_includes_journal_and_match():
    system, user = PR.render_game(_game_payload())
    assert system == PR.SYSTEM_GAME
    assert "EUW1_42" in user and "Zeri" in user
    assert json.loads(user[user.index("{"):user.rindex("}") + 1])


def test_system_game_frames_matchup_context():
    s = PR.SYSTEM_GAME
    assert "context" in s and "champ select" in s.lower()
    assert "lane_pattern" in s and "gank_exposure" in s
    # connaissance générale des champions autorisée, mais ancrée sur le journal
    assert "connaissance générale" in s
    assert "n'invente jamais un événement" in s


def test_system_game_judges_gold_relative_to_next_buy():
    s = PR.SYSTEM_GAME
    assert "PROCHAIN ACHAT" in s
    assert "légitime" in s
