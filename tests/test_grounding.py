"""Verifications d'ancrage : le detecteur doit mordre, pas seulement passer.

Un verificateur permissif produit 100 % d'ancrage et ne prouve rien. Le test
central ici est le CONTROLE NEGATIF : on falsifie des chiffres reels et on exige
un taux de detection minimal. C'est lui qui donne sa valeur au chiffre publie.
"""
from __future__ import annotations

import json

import grounding as G


def _payload(**over):
    base = {
        "meta": {"match_id": "EUW1_1", "champion": "Zeri", "duration_min": 31.0,
                 "kda": {"kills": 4, "deaths": 3, "assists": 6}},
        "journal": {
            "deaths": [
                {"t_ms": 282384, "clock": "4:42", "minute": 4, "phase": "early",
                 "zone": "BOT", "killer_champ": "Karma", "gold_state": "even",
                 "unspent_gold": 290, "level": 3,
                 "objective": {"type": "DRAGON", "status": "imminent", "delta_s": 18}},
                {"t_ms": 913000, "clock": "15:13", "minute": 15, "phase": "mid",
                 "zone": "JUNGLE/RIVER", "killer_champ": "Katarina",
                 "gold_state": "behind", "unspent_gold": 1225, "level": 11},
            ],
            "recalls": [{"t_ms": 202944, "clock": "3:22", "minute": 3,
                         "items_bought": 2, "gold_before": 562,
                         "items": [{"name": "Doran's Blade", "cost": 450}]}],
        },
        "benchmarks": {"outcome": "loss", "n_games_ref": 1026,
                       "death_zone_phase": {"BOT|early": 0.2906},
                       "death_gold_state": {"ahead": 0.2538}},
    }
    base.update(over)
    return base


def _record(evidence: str, cause: str = "gank bot sans vision") -> dict:
    return {"ts": "t1", "kind": "game", "match_id": "EUW1_1",
            "payload": _payload(),
            "review": {"strengths": [],
                       "mistakes": [{"point": "p", "cause": cause,
                                     "evidence": evidence}],
                       "next_focus": "f", "confidence": 0.6}}


# --- parsing ------------------------------------------------------------------

def test_cited_numbers_reads_units_and_ignores_clocks():
    cites = G.cited_numbers("mort à 15:13, 1 225 g non dépensés, 29 % des morts, "
                            "drake dans 18 s, 5 cs de retard")
    assert [c[1] for c in cites] == [1225.0, 29.0, 18.0, 5.0]      # 15:13 exclu
    assert [c[2] for c in cites] == ["g", "pct", "s", "cs"]


def test_unit_of_citation_reads_full_words():
    assert G.unit_of_citation(" secondes avant") == "s"
    assert G.unit_of_citation(" minutes") == "min"
    assert G.unit_of_citation("% des morts") == "pct"
    assert G.unit_of_citation(" morts") == "morts"        # denombrement cloisonne
    assert G.unit_of_citation(" recalls") == G.ANY        # denombrement generique


# --- ancrage des chiffres ------------------------------------------------------

def test_payload_number_is_grounded():
    check = G.check_review(_record("mort à 15:13 avec 1 225 g non dépensés"))
    assert [n["status"] for n in check["numbers"]] == ["exact"]


def test_invented_number_is_caught():
    check = G.check_review(_record("mort à 15:13 avec 1 840 g non dépensés"))
    assert [n["status"] for n in check["numbers"]] == ["non_ancre"]


def test_units_are_isolated():
    """Le coeur du detecteur : 290 existe dans le payload (gold non depense),
    mais le citer comme un POURCENTAGE reste une invention. Sans cloisonnement
    par unite, n'importe quel nombre du journal ancre n'importe quelle stat."""
    assert G.check_review(_record("290 g à 4:42"))["numbers"][0]["status"] == "exact"
    assert G.check_review(_record("290 % à 4:42"))["numbers"][0]["status"] == "non_ancre"


def test_damage_unit_is_isolated_from_gold_and_percentages():
    payload = _payload()
    payload["journal"]["deaths"][0]["damage"] = {
        "total_damage": 900, "basic_share": 0.4,
    }
    record = _record("900 dégâts et 40 % d'autos à 4:42")
    record["payload"] = payload
    assert all(number["status"] != "non_ancre"
               for number in G.check_review(record)["numbers"])

    wrong_unit = _record("900 g à 4:42")
    wrong_unit["payload"] = payload
    assert G.check_review(wrong_unit)["numbers"][0]["status"] == "non_ancre"


def test_numbers_from_llm_causes_never_become_grounding_sources():
    payload = _payload(game_review_causes=[{
        "mistakes": [{"point": "ancien point", "cause": "9 999 g à 42:42"}],
    }])
    record = _record("9 999 g à 4:42")
    record["payload"] = payload
    assert G.check_review(record)["numbers"][0]["status"] == "non_ancre"


def test_derived_share_may_be_cited_rounded():
    """« 50 % de tes morts (1/2) » est derivable du journal, pas invente."""
    check = G.check_review(_record("1 mort sur 2 en BOT à 4:42, soit 50 % de tes morts"))
    assert all(n["status"] != "non_ancre" for n in check["numbers"])


def test_percentage_benchmark_is_grounded_from_a_fraction():
    """Le payload stocke 0.2906 ; le coach ecrit « 29,1 % »."""
    check = G.check_review(_record("29,1 % des morts challenger en BOT early, à 4:42"))
    assert all(n["status"] != "non_ancre" for n in check["numbers"])


# --- ancrage des horodatages ---------------------------------------------------

def test_fabricated_clock_is_caught():
    """Le schema Pydantic verifie la PRESENCE d'un mm:ss, pas sa veracite :
    une evidence citant un moment inexistant passait la validation."""
    check = G.check_review(_record("mort à 22:07 avec 1 225 g non dépensés"))
    assert [c["status"] for c in check["clocks"]] == ["non_ancre"]


def test_real_clock_is_exact():
    check = G.check_review(_record("mort à 4:42, 290 g non dépensés"))
    assert [c["status"] for c in check["clocks"]] == ["exact"]


# --- asymetrie -----------------------------------------------------------------

def test_descriptive_feature_presented_as_a_fault_is_flagged():
    """Regle 3 du prompt : profondeur et over-extension sont des observations
    neutres (elles correlent au rang INFERIEUR). Les prescrire inverse le conseil."""
    review = {"strengths": [], "mistakes": [], "habits": [
        "Ta profondeur de carte moyenne est trop élevée, réduis-la"], "next_focus": "f"}
    assert len(G.asymmetry_violations(review)) == 1


def test_same_term_in_a_neutral_section_is_not_flagged():
    review = {"strengths": [{"point": "p", "cause": "c",
                             "evidence": "profondeur moyenne 254 u"}],
              "mistakes": [], "next_focus": "f"}
    assert G.asymmetry_violations(review) == []


# --- controle negatif (le test qui donne sa valeur au chiffre) -----------------

FALSIFIED = [
    "mort à 15:13 avec 1 679 g non dépensés",          # gold invente
    "38 % des morts challenger en BOT early, à 4:42",  # benchmark deforme
    "drake dans 47 s, mort à 4:42",                    # delta_s invente
    "12 cs de retard à 4:42",                          # cs invente
    "mort à 15:13, 4 morts en BOT",                    # denombrement invente
]


def test_negative_control_catches_falsified_numbers():
    """Chaque evidence ci-dessus deforme UNE valeur du payload. Un detecteur qui
    les laisse passer rendrait le taux d'ancrage publie sans signification."""
    missed = [text for text in FALSIFIED
              if all(n["status"] != "non_ancre"
                     for n in G.check_review(_record(text))["numbers"])]
    assert missed == [], f"falsifications non détectées : {missed}"


def test_negative_control_rate_on_systematic_perturbation():
    """Sur les chiffres d'une review reelle multiplies par 1,37, le detecteur
    doit en rejeter la large majorite (mesure de puissance, pas d'exactitude)."""
    record = _record("mort à 4:42 (290 g non dépensés, drake dans 18 s), puis à "
                     "15:13 (1 225 g non dépensés) ; 29,1 % des morts challenger "
                     "en BOT early, 25,4 % en étant ahead")
    index = G.payload_index(record["payload"])
    cites = [c for text in G._insight_texts(record["review"])
             for c in G.cited_numbers(text)]
    assert len(cites) >= 5
    caught = sum(1 for _, value, unit in cites
                 if G.classify_number(value * 1.37, index, unit) == "non_ancre")
    assert caught / len(cites) >= 0.8


# --- rapport -------------------------------------------------------------------

def test_report_aggregates_over_a_reviews_file(tmp_path):
    root = tmp_path / "07_coaching"
    (root / "p").mkdir(parents=True)
    (root / "p" / "reviews.jsonl").write_text(
        json.dumps(_record("mort à 4:42, 290 g non dépensés"), ensure_ascii=False)
        + "\n"
        + json.dumps(_record("mort à 22:07, 9 999 g non dépensés"), ensure_ascii=False)
        + "\n")
    rep = G.report("p", root=root)
    assert rep["n_reviews"] == 2
    assert rep["numbers"]["grounded_rate"] == 0.5
    assert rep["clocks"]["anchored_rate"] == 0.5
    assert len(rep["offenders"]) == 1
    assert "9 999 g" in rep["offenders"][0]["numbers"]
