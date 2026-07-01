import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch):
    # Redirect JOBS_FILE + data root to tmp before importing main
    import settings
    monkeypatch.setattr(settings, "JOBS_FILE", tmp_path / "jobs.jsonl")
    monkeypatch.setattr(settings, "ACCOUNTS_FILE", tmp_path / "accounts.json")
    settings.ACCOUNTS_FILE.write_text('[{"slug":"spadzze","riot_id":"Spadzze#euw","region":"euw1"}]')
    # Point readers at fixture data
    import readers
    fix = Path(__file__).resolve().parent / "fixtures"
    monkeypatch.setattr(readers, "rl", _rl_proxy(fix), raising=False)
    import main
    return TestClient(main.app)


class _RlProxy:
    """Shim so readers' `data_root or rl.DATA` picks the fixture dir."""
    def __init__(self, data):
        self.DATA = data


def _rl_proxy(fix):
    return _RlProxy(fix)


def test_health(client=None):
    import main
    c = TestClient(main.app)
    r = c.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_accounts_endpoint(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/accounts")
    assert r.status_code == 200
    body = r.json()
    assert body[0]["slug"] == "spadzze"
    assert body[0]["games_count"] == 3


def test_games_endpoint(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/c/spadzze/games?size=2")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3 and len(body["items"]) == 2


def test_games_bad_params_422(tmp_path, monkeypatch):
    """Validation route-layer : page>=1 et size in [1,200]."""
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/c/spadzze/games?page=0").status_code == 422
    assert c.get("/api/c/spadzze/games?size=0").status_code == 422
    assert c.get("/api/c/spadzze/games?size=201").status_code == 422


def test_shap_endpoint(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/c/spadzze/shap")
    assert r.status_code == 200 and r.json()["available"] is True


def test_rank_endpoint_returns_cached_rank(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/c/spadzze/rank")
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == "DIAMOND" and body["league_points"] == 42


def test_rank_endpoint_never_fetched_returns_null_shape(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/c/ghost/rank")
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] is None and body["fetched_at"] is None


def test_predicted_rank_endpoint(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    with patch("routers.predicted_rank.ml_rank.predict_rank",
               return_value={"predicted_rank": "master", "proba": 0.5, "n_games_used": 3}):
        r = c.get("/api/c/spadzze/predicted-rank")
    assert r.status_code == 200
    assert r.json() == {"predicted_rank": "master", "proba": 0.5, "n_games_used": 3}


def test_predicted_rank_endpoint_insufficient_data(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    with patch("routers.predicted_rank.ml_rank.predict_rank", return_value=None):
        r = c.get("/api/c/spadzze/predicted-rank")
    assert r.status_code == 200
    assert r.json()["predicted_rank"] is None


def test_fetch_creates_job(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    with patch("routers.jobs.pipeline.fetch_games", return_value={"n_games": 5}):
        r = c.post("/api/fetch", json={"slug": "spadzze", "n": 5})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    r2 = c.get(f"/api/jobs/{jid}")
    assert r2.status_code == 200 and r2.json()["id"] == jid


def test_jobs_missing_404(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/jobs/nope").status_code == 404


def test_coach_creates_job(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    with patch("routers.jobs.pipeline.run_coach", return_value={"ts": "t"}):
        r = c.post("/api/coach", json={"slug": "spadzze", "scope": "adc",
                                       "outcome": "loss", "target": "challenger"})
    assert r.status_code == 200 and "job_id" in r.json()


def test_feedback_post(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    with patch("routers.feedback.feedback.build_feedback", return_value="FB"), \
         patch("routers.feedback.feedback.persist_feedback", return_value=("p", False)):
        r = c.post("/api/feedback", json={"slug": "spadzze", "ts": "2026-06-30T17:53:39",
                "responses": {"strength,0": {"useful": True}}})
    assert r.status_code == 200 and r.json()["ok"] is True

def test_fetch_writes_to_patched_jobs_path(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    patched = tmp_path / "jobs.jsonl"
    assert patched.exists() is False or patched.stat().st_size == 0
    with patch("routers.jobs.pipeline.fetch_games", return_value={"n_games": 1}):
        r = c.post("/api/fetch", json={"slug": "spadzze", "n": 1})
    jid = r.json()["job_id"]
    for _ in range(100):
        if c.get(f"/api/jobs/{jid}").json()["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    # The job MUST be persisted in the patched (tmp) path, not the real data dir.
    assert patched.exists()
    assert jid in patched.read_text()


def test_fetch_job_progresses_to_done(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    calls = []

    def fake_work(account, n=1, on_progress=None):
        if on_progress:
            on_progress("1/1")
        calls.append("done")
        return {"n_games": 1}

    with patch("routers.jobs.pipeline.fetch_games", fake_work):
        r = c.post("/api/fetch", json={"slug": "spadzze", "n": 1})
    jid = r.json()["job_id"]
    for _ in range(100):
        got = c.get(f"/api/jobs/{jid}").json()
        if got["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert c.get(f"/api/jobs/{jid}").json()["status"] == "done"
    assert calls == ["done"]


def test_unknown_account_404(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.post("/api/fetch", json={"slug": "ghost"}).status_code == 404
    assert c.post("/api/coach", json={"slug": "ghost"}).status_code == 404


def test_feedback_missing_useful_422(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/feedback", json={"slug": "spadzze", "ts": "2026-06-30T17:53:39",
            "responses": {"strength,0": {"tag": "asymetrie"}}})  # no "useful"
    assert r.status_code == 422


def test_feedback_not_useful_without_tag_422(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    with patch("routers.feedback.feedback.load_review", return_value={
            "ts": "2026-06-30T17:53:39", "model": "kimi-k2.6",
            "review": {"strengths": [{"point": "x", "evidence": "e"}]*3,
                       "mistakes": [{"point": "y", "evidence": "e"}]*3,
                       "habits": ["h1", "h2"], "next_focus": "f",
                       "confidence": 0.6}}):
        r = c.post("/api/feedback", json={"slug": "spadzze",
                "ts": "2026-06-30T17:53:39",
                "responses": {"strength,0": {"useful": False}}})  # no tag
    assert r.status_code == 422
