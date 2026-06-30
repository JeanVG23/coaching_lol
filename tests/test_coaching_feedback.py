import pytest
from pydantic import ValidationError

import schema as S


def _item(kind="strength", index=0, useful=True, tag=None, note=None):
    return S.FeedbackItem(kind=kind, index=index, useful=useful, tag=tag, note=note)


def test_feedback_item_useful_no_tag_ok():
    it = _item(useful=True, tag=None)
    assert it.useful is True and it.tag is None


def test_feedback_item_not_useful_without_tag_rejected():
    with pytest.raises(ValidationError):
        _item(useful=False, tag=None)


def test_feedback_item_not_useful_with_tag_ok():
    it = _item(useful=False, tag="profondeur-en-faute", note="prof prescrite")
    assert it.tag == "profondeur-en-faute"


def test_feedback_item_bad_tag_rejected():
    with pytest.raises(ValidationError):
        _item(useful=False, tag="inventé")


def test_feedback_roundtrip_and_keys():
    fb = S.Feedback(ts="2026-06-30T10:00:00", player="spadzze",
                   rated_at="2026-06-30T11:00:00", model="kimi-k2.6",
                   items=[_item(useful=True), _item(kind="mistake", index=1,
                              useful=False, tag="asymetrie", note="x")])
    d = fb.model_dump()
    assert d["ts"] == "2026-06-30T10:00:00"
    assert len(d["items"]) == 2
    assert d["items"][1]["tag"] == "asymetrie"
    # re-validation depuis dict brut
    assert S.Feedback.model_validate(d).model == "kimi-k2.6"


def test_neg_tags_matches_literal():
    assert set(S.NEG_TAGS) == {"asymetrie", "stat-inventee", "profondeur-en-faute",
                               "trop-vague", "non-actionnable", "autre"}