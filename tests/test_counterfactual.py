"""Harness contrefactuel : perturbations pures + attentes, sans appel LLM.

Le generateur est injecte : ces tests verifient la MECANIQUE (perturbation,
attente, agregation, persistance hors corpus), pas le modele.
"""
from __future__ import annotations

import json

import counterfactual as CF


def _payload():
    return {
        "meta": {"match_id": "EUW1_1", "champion": "Zeri",
                 "kda": {"kills": 4, "deaths": 2, "assists": 6}},
        "journal": {
            "deaths": [
                {"clock": "4:42", "zone": "BOT", "phase": "early",
                 "unspent_gold": 290, "killer_champ": "Karma"},
                {"clock": "15:13", "zone": "BOT", "phase": "mid",
                 "unspent_gold": 1225, "killer_champ": "Katarina"},
            ],
            "recalls": [{"clock": "3:22", "gold_before": 562, "items_bought": 2}],
        },
        "benchmarks": {"outcome": "loss"},
    }


def _review(confidence=0.8, zone="BOT", gold="1 225 g"):
    return {"strengths": [],
            "mistakes": [{"point": "meurs moins", "cause": f"exposé en {zone}",
                          "evidence": f"mort à 15:13 en {zone}, {gold} non dépensés"}],
            "next_focus": "f", "confidence": confidence}


def _record():
    return {"ts": "t1", "kind": "game", "match_id": "EUW1_1",
            "payload": _payload(), "review": _review()}


# --- perturbations (pures) -----------------------------------------------------

def test_perturbations_do_not_mutate_the_source():
    payload = _payload()
    CF.perturb_no_deaths(payload)
    CF.perturb_zone_to_top(payload)
    CF.perturb_unspent_gold_zero(payload)
    assert payload == _payload()          # la reference sert de baseline ailleurs


def test_no_deaths_empties_journal_and_kda():
    out = CF.perturb_no_deaths(_payload())
    assert out["journal"]["deaths"] == []
    assert out["meta"]["kda"]["deaths"] == 0


def test_zone_to_top_keeps_everything_but_the_zone():
    out = CF.perturb_zone_to_top(_payload())
    assert {d["zone"] for d in out["journal"]["deaths"]} == {"TOP"}
    assert [d["clock"] for d in out["journal"]["deaths"]] == ["4:42", "15:13"]


def test_unspent_gold_zero_covers_deaths_and_recalls():
    out = CF.perturb_unspent_gold_zero(_payload())
    assert {d["unspent_gold"] for d in out["journal"]["deaths"]} == {0}
    assert out["journal"]["recalls"][0]["gold_before"] == 0


# --- attentes ------------------------------------------------------------------

def test_no_deaths_expects_lower_confidence():
    assert CF.check_no_deaths(_review(0.8), _review(0.4))["passed"] is True
    assert CF.check_no_deaths(_review(0.8), _review(0.8))["passed"] is False


def test_zone_expectation_detects_a_coach_that_ignores_the_journal():
    """Le cas qui justifie le test : la sortie perturbee reste identique, donc
    le modele recite un pattern plausible au lieu de lire le journal."""
    assert CF.check_zone_to_top(_review(zone="BOT"), _review(zone="TOP"))["passed"]
    assert not CF.check_zone_to_top(_review(zone="BOT"), _review(zone="BOT"))["passed"]
    # Regression du 2026-09-04 : l'attente portait sur la disparition de « BOT »,
    # que le payload d'un ADC cite legitimement ailleurs (role, benchmarks de lane,
    # « tour BOT perdue »). Le test echouait sur un modele qui avait pourtant
    # deplace toutes ses morts. Seule la presence de la zone cible compte.
    assert CF.check_zone_to_top(_review(zone="BOT"),
                                _review(zone="TOP et tour BOT perdue"))["passed"]


def test_gold_expectation_uses_the_cited_amounts():
    assert CF.max_gold_cited(_review(gold="1 225 g")) == 1225.0
    assert CF.check_unspent_gold_zero(_review(gold="1 225 g"),
                                      _review(gold="0 g"))["passed"] is True
    assert CF.check_unspent_gold_zero(_review(gold="1 225 g"),
                                      _review(gold="1 225 g"))["passed"] is False


# --- execution -----------------------------------------------------------------

class _FakeReview(dict):
    def model_dump(self):
        return dict(self)


def _generator(mapping):
    """Rend une review differente selon la perturbation detectee dans le payload."""
    def generate(payload, model):
        if not payload["journal"]["deaths"]:
            key = "no_deaths"
        elif payload["journal"]["deaths"][0]["zone"] == "TOP":
            key = "zone_to_top"
        else:
            key = "unspent_gold_zero"
        return _FakeReview(mapping[key]), {"prompt_version": "v", "total_tokens": 10}
    return generate


SENSITIVE = {
    "no_deaths": _review(confidence=0.3, gold="0 g"),
    "zone_to_top": _review(zone="TOP", gold="1 225 g"),
    "unspent_gold_zero": _review(gold="0 g"),
}
INSENSITIVE = {name: _review() for name in SENSITIVE}


def test_run_scores_a_sensitive_model(tmp_path):
    root = tmp_path / "07_coaching"
    (root / "p").mkdir(parents=True)
    (root / "p" / "reviews.jsonl").write_text(
        json.dumps(_record(), ensure_ascii=False) + "\n")
    report = CF.run("p", n=1, model="m", root=root, generate=_generator(SENSITIVE))
    assert report["n_runs"] == 3 and report["n_errors"] == 0
    assert report["sensitivity"] == 1.0
    assert set(report["by_perturbation"]) == set(CF.PERTURBATIONS)


def test_run_catches_a_model_that_ignores_the_payload(tmp_path):
    root = tmp_path / "07_coaching"
    (root / "p").mkdir(parents=True)
    (root / "p" / "reviews.jsonl").write_text(
        json.dumps(_record(), ensure_ascii=False) + "\n")
    report = CF.run("p", n=1, model="m", root=root, generate=_generator(INSENSITIVE))
    assert report["sensitivity"] == 0.0
    # et la sortie inchangee cite un gold absent du payload perturbe : l'ancrage
    # le voit aussi, independamment de l'attente.
    gold_run = next(r for r in report["runs"]
                    if r["perturbation"] == "unspent_gold_zero")
    assert gold_run["grounded_rate"] is not None and gold_run["grounded_rate"] < 1.0


def test_llm_failure_is_reported_not_fatal(tmp_path):
    root = tmp_path / "07_coaching"
    (root / "p").mkdir(parents=True)
    (root / "p" / "reviews.jsonl").write_text(
        json.dumps(_record(), ensure_ascii=False) + "\n")

    def boom(payload, model):
        raise CF.llm_client.LLMError("api down")

    report = CF.run("p", n=1, model="m", root=root, generate=boom)
    assert report["n_errors"] == 3 and report["sensitivity"] is None


def test_report_is_persisted_outside_the_review_corpus(tmp_path):
    """Une sortie contrefactuelle n'est pas une review du joueur : elle ne doit
    ni etre annotee, ni remonter sur le site."""
    root = tmp_path / "07_coaching"
    (root / "p").mkdir(parents=True)
    path = CF.persist("p", {"player": "p"}, root=root)
    assert path == root / "p" / "eval" / "counterfactual.json"
    assert not (root / "p" / "reviews.jsonl").exists()
