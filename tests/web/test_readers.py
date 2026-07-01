# tests/web/test_readers.py
import json
from pathlib import Path

import readers
import settings


def test_load_accounts_returns_list(monkeypatch, tmp_path):
    # Point settings at a temp accounts.json
    cfg = tmp_path / "accounts.json"
    cfg.write_text(json.dumps([
        {"slug": "spadzze", "riot_id": "Spadzze#euw", "region": "euw1"},
    ]))
    monkeypatch.setattr(settings, "ACCOUNTS_FILE", cfg)
    accts = settings.load_accounts()
    assert accts == [{"slug": "spadzze", "riot_id": "Spadzze#euw", "region": "euw1"}]


def test_account_for_returns_match_or_none(monkeypatch, tmp_path):
    cfg = tmp_path / "accounts.json"
    cfg.write_text(json.dumps([
        {"slug": "spadzze", "riot_id": "Spadzze#euw", "region": "euw1"},
        {"slug": "ace", "riot_id": "Ace#euw", "region": "euw1"},
    ]))
    monkeypatch.setattr(settings, "ACCOUNTS_FILE", cfg)
    assert settings.account_for("ace")["riot_id"] == "Ace#euw"
    assert settings.account_for("nope") is None


# append to tests/web/test_readers.py
FIX = Path(__file__).resolve().parent / "fixtures"


def test_read_games_pagination():
    res = readers.read_games("spadzze", page=1, size=2, data_root=FIX)
    assert res["total"] == 3
    assert len(res["items"]) == 2
    assert res["page"] == 1 and res["size"] == 2
    res2 = readers.read_games("spadzze", page=2, size=2, data_root=FIX)
    assert len(res2["items"]) == 1


def test_read_reviews_returns_list():
    revs = readers.read_reviews("spadzze", data_root=FIX)
    assert len(revs) == 1
    assert revs[0]["scope"] == "adc"


def test_read_feedback_returns_list():
    fbs = readers.read_feedback("spadzze", data_root=FIX)
    assert len(fbs) == 1
    assert fbs[0]["player"] == "spadzze"


def test_read_shap_available():
    s = readers.read_shap("spadzze", data_root=FIX)
    assert s["available"] is True
    assert isinstance(s["drivers"], list)


def test_read_shap_unavailable_for_unknown():
    s = readers.read_shap("ghost", data_root=FIX)
    assert s == {"available": False, "drivers": []}


def test_account_summaries():
    s = readers.account_summaries(data_root=FIX)
    # account_summaries uses settings.load_accounts() — patch it to the fixture set
    by_slug = {a["slug"]: a for a in s}
    assert by_slug["spadzze"]["games_count"] == 3
    assert by_slug["spadzze"]["last_review_ts"] is not None