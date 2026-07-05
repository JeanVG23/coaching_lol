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


def _game_review_record():
    return {"ts": "2026-07-05T10:00:00", "model": "kimi-k2.6",
            "kind": "game", "match_id": "EUW1_42",
            "scope": "adc", "target": "challenger",
            "payload": {"meta": {}},
            "review": {"strengths": [],
                       "mistakes": [{"point": "m",
                                     "evidence": "mort à 17:05, drake dans 6 s"}],
                       "next_focus": "f", "confidence": 0.4}}


def test_annotate_handles_game_reviews(tmp_path):
    # La boucle d'éval doit accepter les reviews par-game (kind=game, sans habits).
    _write_reviews(tmp_path, lines=[_game_review_record()])
    answers = iter(["y", "n", "5", ""])       # mistake utile ; focus faux, tag 5
    rc = F.annotate("spadzze", last=True, root=tmp_path,
                    prompt=lambda _msg: next(answers))
    assert rc == 0
    fbs = F.load_feedbacks("spadzze", root=tmp_path)
    assert len(fbs) == 1
    kinds = {(it.kind, it.useful) for it in fbs[0].items}
    assert ("mistake", True) in kinds and ("focus", False) in kinds


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


# --- Task 3 : summarize + render ---------------------------------------------

def _fb(ts, model, items):
    return S.Feedback(ts=ts, player="spadzze", rated_at="r", model=model, items=items)


def _it(kind, useful, tag=None, note=None):
    return S.FeedbackItem(kind=kind, index=0, useful=useful, tag=tag, note=note)


def test_summarize_empty():
    s = F.summarize([])
    assert s["n_reviews"] == 0 and "global_rate" not in s


def test_summarize_global_and_by_kind():
    fbs = [_fb("t1", "kimi-k2.6", [
            _it("strength", True), _it("mistake", False, "profondeur-en-faute"),
            _it("habit", True)])]
    s = F.summarize(fbs)
    assert s["n_reviews"] == 1 and s["n_items"] == 3
    assert s["global_rate"] == pytest.approx(2 / 3)
    assert s["by_kind"]["strength"]["rate"] == 1.0
    assert s["by_kind"]["mistake"]["rate"] == 0.0
    assert s["by_kind"]["habit"]["rate"] == 1.0


def test_summarize_top_tags():
    fbs = [_fb("t1", "m", [_it("mistake", False, "profondeur-en-faute"),
                           _it("mistake", False, "profondeur-en-faute"),
                           _it("strength", False, "asymetrie")])]
    s = F.summarize(fbs)
    assert s["top_tags"][0] == ("profondeur-en-faute", 2)
    assert s["top_tags"][1] == ("asymetrie", 1)


def test_summarize_by_model():
    fbs = [_fb("t1", "kimi-k2.6", [_it("strength", True), _it("mistake", True)]),
           _fb("t2", "minimax-m3", [_it("mistake", False, "asymetrie")])]
    s = F.summarize(fbs)
    assert s["by_model"]["kimi-k2.6"]["rate"] == 1.0
    assert s["by_model"]["minimax-m3"]["rate"] == 0.0
    assert s["by_model"]["kimi-k2.6"]["n_reviews"] == 1


def test_summarize_low_sample_no_trend():
    fbs = [_fb(f"t{i}", "m", [_it("strength", True)]) for i in range(5)]
    s = F.summarize(fbs)
    assert s["low_sample"] is True and s["trend"] is None


def test_summarize_trend_when_enough():
    # 10 reviews : 5 premières tout faux, 5 dernières tout vrai -> tendance haussière
    # ts croissants : a* (faux) trie avant b* (vrai)
    fbs = ([_fb(f"a{i}", "m", [_it("strength", False, "trop-vague")]) for i in range(5)]
           + [_fb(f"b{i}", "m", [_it("strength", True)]) for i in range(5)])
    s = F.summarize(fbs)
    assert s["low_sample"] is False
    assert s["trend"] is not None
    assert s["trend"]["prior"] == 0.0 and s["trend"]["recent"] == 1.0


def test_summarize_collects_notes_per_tag():
    fbs = [_fb("t1", "m", [
        _it("mistake", False, "asymetrie", note="le jungle était visible pourtant"),
        _it("mistake", False, "asymetrie", note=None),   # tag sans note -> ignoré
        _it("strength", False, "trop-vague", note="pas assez concret"),
        _it("habit", True, note="peu importe, useful=True -> jamais collecté"),
    ])]
    s = F.summarize(fbs)
    assert s["tag_notes"]["asymetrie"] == ["le jungle était visible pourtant"]
    assert s["tag_notes"]["trop-vague"] == ["pas assez concret"]
    assert "habit" not in str(s["tag_notes"])  # note d'un item useful=True jamais rattachée à un tag


def test_render_summary_includes_note_verbatims():
    fbs = [_fb("t1", "kimi-k2.6", [
        _it("mistake", False, "asymetrie", note="le jungle était visible pourtant"),
    ])]
    txt = F.render_summary(F.summarize(fbs))
    assert "le jungle était visible pourtant" in txt


def test_render_summary_caps_notes_at_two_per_tag():
    fbs = [_fb("t1", "m", [_it("mistake", False, "asymetrie", note=f"note{i}")
                           for i in range(4)])]
    txt = F.render_summary(F.summarize(fbs))
    assert "note0" in txt and "note1" in txt
    assert "note2" not in txt and "note3" not in txt


def test_render_summary_has_sections():
    fbs = [_fb("t1", "kimi-k2.6", [_it("strength", True),
                                   _it("mistake", False, "profondeur-en-faute")])]
    txt = F.render_summary(F.summarize(fbs))
    assert "Taux d'utilité" in txt and "Top tags" in txt and "Par modèle" in txt
    assert "profondeur-en-faute" in txt


def test_load_feedbacks_roundtrip(tmp_path):
    fb = _fb("t1", "m", [_it("strength", True)])
    F.persist_feedback("spadzze", fb, root=tmp_path)
    loaded = F.load_feedbacks("spadzze", root=tmp_path)
    assert len(loaded) == 1 and loaded[0].ts == "t1"


# --- Task 4 : annotate interactif + main() ----------------------------------

def _reviews_file(tmp_path, n=1):
    lines = []
    for i in range(n):
        d = _review_dict()
        d["ts"] = f"2026-06-30T1{i}:00:00"
        lines.append(d)
    _write_reviews(tmp_path, lines=lines)


def test_annotate_interactive_monkeypatched_input(tmp_path, monkeypatch, capsys):
    _reviews_file(tmp_path, n=1)
    answers = iter([
        "1",                       # choix review #1
        "y",                       # strength 0 utile
        "s",                       # strength 1 skip
        "y",                       # strength 2 utile
        "n", "3", "prof prescrite", # mistake 0 faux -> tag #3 = profondeur-en-faute, note
        "y",                       # mistake 1 utile
        "y",                       # mistake 2 utile
        "y",                       # habit 0 utile
        "s",                       # habit 1 skip
        "y",                       # focus utile
    ])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    rc = F.annotate("spadzze", root=tmp_path)
    assert rc == 0
    fbs = F.load_feedbacks("spadzze", root=tmp_path)
    assert len(fbs) == 1
    fb = fbs[0]
    assert fb.ts == "2026-06-30T10:00:00"
    assert len(fb.items) == 7            # 9 - 2 skips
    mk0 = next(it for it in fb.items if it.kind == "mistake" and it.index == 0)
    assert mk0.useful is False and mk0.tag == "profondeur-en-faute"
    assert mk0.note == "prof prescrite"


def test_annotate_no_reviews(tmp_path, capsys):
    rc = F.annotate("nobody", root=tmp_path, prompt=lambda *a, **k: "")
    assert rc == 0
    assert "Aucune review" in capsys.readouterr().out


def test_annotate_ts_not_found(tmp_path, capsys):
    _reviews_file(tmp_path, n=1)
    rc = F.annotate("spadzze", ts="missing", root=tmp_path, prompt=lambda *a, **k: "")
    assert rc == 1
    assert "introuvable" in capsys.readouterr().out.lower()


def test_main_summary_subcommand(tmp_path, monkeypatch, capsys):
    # layout production : rl.DATA / "07_coaching" / player / reviews.jsonl
    _reviews_file(tmp_path / "07_coaching", n=1)
    answers = iter(["1", "y", "y", "y", "y", "y", "y", "y", "y", "y"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    monkeypatch.setattr(F.rl, "DATA", tmp_path)   # rl.DATA/07_coaching = tmp/07_coaching
    assert F.main(["annotate", "--player", "spadzze"]) == 0
    capsys.readouterr()
    assert F.main(["summary", "--player", "spadzze"]) == 0
    out = capsys.readouterr().out
    assert "Taux d'utilité" in out


# --- pending_reviews + annotate --pending --------------------------------------

def test_pending_reviews_joins_by_ts_oldest_first():
    reviews = [{"ts": "2026-07-03T10:00:00"}, {"ts": "2026-07-01T10:00:00"},
               {"ts": "2026-07-02T10:00:00"}]
    fbs = [_fb("2026-07-02T10:00:00", "m", [_it("strength", True)])]
    got = F.pending_reviews(reviews, fbs)
    assert [r["ts"] for r in got] == ["2026-07-01T10:00:00", "2026-07-03T10:00:00"]


def test_annotate_pending_iterates_oldest_first_and_quits(tmp_path):
    _write_reviews(tmp_path, lines=[_game_review_record(), _review_dict()])
    answers = iter([
        "",                                       # Entrée = annoter (l'agrégée, plus ancienne)
        "y", "y", "y", "y", "y", "y", "y", "y", "y",   # ses 9 items
        "q",                                      # quitter avant la par-game
    ])
    rc = F.annotate("spadzze", pending=True, root=tmp_path,
                    prompt=lambda _m: next(answers))
    assert rc == 0
    fbs = F.load_feedbacks("spadzze", root=tmp_path)
    assert len(fbs) == 1 and fbs[0].ts == "2026-06-30T10:00:00"


def test_annotate_pending_skip_leaves_review_pending(tmp_path):
    _write_reviews(tmp_path)                      # 1 review agrégée
    answers = iter(["n"])                         # passer -> rien persisté
    rc = F.annotate("spadzze", pending=True, root=tmp_path,
                    prompt=lambda _m: next(answers))
    assert rc == 0
    assert F.load_feedbacks("spadzze", root=tmp_path) == []


def test_annotate_pending_none_left(tmp_path, capsys):
    _write_reviews(tmp_path)
    fb = _fb("2026-06-30T10:00:00", "kimi-k2.6", [_it("strength", True)])
    F.persist_feedback("spadzze", fb, root=tmp_path)
    rc = F.annotate("spadzze", pending=True, root=tmp_path,
                    prompt=lambda _m: "")
    assert rc == 0
    assert "en attente" in capsys.readouterr().out.lower()
