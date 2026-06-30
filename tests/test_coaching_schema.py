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


def test_review_rejects_confidence_out_of_range():
    bad = _good(); bad["confidence"] = 1.5
    with pytest.raises(ValidationError):
        S.Review.model_validate(bad)


def test_json_schema_has_fixed_lengths():
    sch = S.review_json_schema()
    assert sch["properties"]["strengths"]["minItems"] == 3
    assert sch["properties"]["strengths"]["maxItems"] == 3
    assert sch["properties"]["habits"]["maxItems"] == 2
