import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import feedback as F
import riotlib as rl
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

# --- Task 2 : helpers données ------------------------------------------------

def _review_dict():
    ins = {"point": "p", "evidence": "e"}
    return {"ts": "2026-06-30T10:00:00", "model": "kimi-k2.6",
            "scope": "adc", "target": "challenger", "outcome_focus": "loss",
            "payload": {"meta": {}},
            "review": {"strengths": [ins, ins, ins], "mistakes": [ins, ins, ins],
                       "habits": ["h1", "h2"], "next_focus": "f", "confidence": 0.6}}


def _write_reviews(tmp_path, player="spadzze", lines=None):
    out = tmp_path / player
    out.mkdir(parents=True, exist_ok=True)
    (out / "reviews.jsonl").write_text(
        "\n".join(json.dumps(l) for l in (lines or [_review_dict()])) + "\n")
    return out / "reviews.jsonl"


def test_list_reviews_empty(tmp_path):
    assert F.list_reviews("nobody", root=tmp_path) == []


def test_list_reviews_reads_lines(tmp_path):
    _write_reviews(tmp_path)
    rs = F.list_reviews("spadzze", root=tmp_path)
    assert len(rs) == 1 and rs[0]["ts"] == "2026-06-30T10:00:00"


def test_load_review_found_and_missing(tmp_path):
    _write_reviews(tmp_path)
    assert F.load_review("spadzze", "2026-06-30T10:00:00", root=tmp_path)["model"] == "kimi-k2.6"
    assert F.load_review("spadzze", "missing", root=tmp_path) is None


def test_build_feedback_skips_unanswered_and_enforces_tag():
    rev = S.Review.model_validate(_review_dict()["review"])
    responses = {
        ("strength", 0): (True, None, None),
        ("mistake", 1): (False, "profondeur-en-faute", "prof prescrite"),
        # les autres items = skip
    }
    fb = F.build_feedback(rev, ts="2026-06-30T10:00:00", player="spadzze",
                          model="kimi-k2.6", rated_at="2026-06-30T11:00:00",
                          responses=responses)
    assert isinstance(fb, S.Feedback)
    assert len(fb.items) == 2                      # skips omis
    assert fb.items[0].kind == "strength" and fb.items[0].useful is True
    assert fb.items[1].tag == "profondeur-en-faute"


def test_persist_feedback_creates_then_overwrites_same_ts(tmp_path):
    fb1 = S.Feedback(ts="t1", player="spadzze", rated_at="a", model="m",
                     items=[S.FeedbackItem(kind="strength", index=0, useful=True)])
    path, overwrote = F.persist_feedback("spadzze", fb1, root=tmp_path)
    assert overwrote is False and path.exists()
    # nouvelle annotation même ts, items différents
    fb2 = S.Feedback(ts="t1", player="spadzze", rated_at="b", model="m",
                     items=[S.FeedbackItem(kind="mistake", index=0, useful=False, tag="asymetrie")])
    path, overwrote = F.persist_feedback("spadzze", fb2, root=tmp_path)
    assert overwrote is True
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1                         # 1 ligne finale, pas 2
    assert lines[0]["rated_at"] == "b" and lines[0]["items"][0]["tag"] == "asymetrie"
