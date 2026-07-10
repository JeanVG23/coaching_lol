import pytest
from pydantic import ValidationError

import schema as S


def _good():
    ins = {"point": "p", "evidence": "e"}
    return {"strengths": [ins, ins, ins], "mistakes": [ins, ins, ins],
            "habits": ["h1", "h2"], "next_focus": "focus", "confidence": 0.7}


def test_review_accepts_valid():
    r = S.Review.model_validate(_good())
    assert len(r.strengths) == 3 and len(r.habits) == 2
    assert r.strengths[0].evidence == "e"


def test_review_rejects_wrong_lengths():
    bad = _good()
    bad["strengths"].append({"point": "x", "evidence": "y"})  # 4 forces
    with pytest.raises(ValidationError):
        S.Review.model_validate(bad)


def test_review_accepts_one_strength():
    # Le schéma ne force plus 3 forces : 1 vraie force > 3 dont 2 de remplissage.
    ok = _good()
    ok["strengths"] = [{"point": "p", "evidence": "e"}]
    r = S.Review.model_validate(ok)
    assert len(r.strengths) == 1


def test_review_rejects_zero_strengths():
    bad = _good(); bad["strengths"] = []
    with pytest.raises(ValidationError):
        S.Review.model_validate(bad)


def test_review_rejects_confidence_out_of_range():
    bad = _good(); bad["confidence"] = 1.5
    with pytest.raises(ValidationError):
        S.Review.model_validate(bad)


def _game_good():
    return {"strengths": [{"point": "p", "cause": "bon reset avant drake",
                           "evidence": "bon reset à 10:04, 1451 g"}],
            "mistakes": [{"point": "m", "cause": "solo 1v1 sans flash en overextension",
                          "evidence": "mort à 17:05, drake dans 6 s"}],
            "next_focus": "f", "confidence": 0.5}


def test_game_review_accepts_valid_and_empty_strengths():
    r = S.GameReview.model_validate(_game_good())
    assert len(r.mistakes) == 1
    assert r.mistakes[0].cause.startswith("solo 1v1")
    ok = _game_good(); ok["strengths"] = []
    assert S.GameReview.model_validate(ok).strengths == []


def test_game_review_mistake_evidence_requires_timestamp():
    # L'ancrage temporel est une contrainte de SCHÉMA, pas une politesse de prompt.
    bad = _game_good()
    bad["mistakes"] = [{"point": "m", "cause": "ok",
                       "evidence": "tu meurs trop en botlane"}]
    with pytest.raises(ValidationError):
        S.GameReview.model_validate(bad)


def test_game_review_mistake_requires_cause():
    # Cause obligatoire (feedback « je sais pas pourquoi je suis mort »).
    bad = _game_good()
    bad["mistakes"] = [{"point": "m", "evidence": "mort à 17:05, drake dans 6 s"}]
    with pytest.raises(ValidationError):
        S.GameReview.model_validate(bad)


def test_game_review_rejects_empty_cause():
    bad = _game_good()
    bad["mistakes"] = [{"point": "m", "cause": "  ",
                       "evidence": "mort à 17:05, drake dans 6 s"}]
    with pytest.raises(ValidationError):
        S.GameReview.model_validate(bad)


def test_game_review_strength_requires_timestamp_and_cause():
    # Les forces sont aussi ancrées + causées (feedback « aucune idée de pourquoi »).
    bad = _game_good()
    bad["strengths"] = [{"point": "p", "cause": "ok",
                         "evidence": "bonne macro tout au long"}]   # pas de mm:ss
    with pytest.raises(ValidationError):
        S.GameReview.model_validate(bad)
    bad["strengths"] = [{"point": "p", "evidence": "bon reset à 10:04, 1451 g"}]  # pas de cause
    with pytest.raises(ValidationError):
        S.GameReview.model_validate(bad)


def test_game_review_requires_at_least_one_mistake():
    bad = _game_good(); bad["mistakes"] = []
    with pytest.raises(ValidationError):
        S.GameReview.model_validate(bad)


def test_game_review_json_schema_lengths():
    sch = S.game_review_json_schema()
    assert sch["properties"]["mistakes"]["minItems"] == 1
    assert sch["properties"]["mistakes"]["maxItems"] == 3
    assert sch["properties"]["strengths"]["maxItems"] == 2


def test_game_review_json_schema_requires_cause():
    # Le champ `cause` doit apparaître dans le JSON-schema imposé au LLM.
    sch = S.game_review_json_schema()
    defs = sch["$defs"]
    assert "cause" in defs["GameInsight"]["properties"]
    assert "cause" in defs["GameInsight"]["required"]


def test_json_schema_has_fixed_lengths():
    sch = S.review_json_schema()
    assert sch["properties"]["strengths"]["minItems"] == 1
    assert sch["properties"]["strengths"]["maxItems"] == 3
    assert sch["properties"]["mistakes"]["minItems"] == 3
    assert sch["properties"]["habits"]["maxItems"] == 2
