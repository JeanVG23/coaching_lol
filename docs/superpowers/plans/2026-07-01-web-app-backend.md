# Web App V1 — Backend API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FastAPI backend that exposes the coaching_lol pipeline over HTTP — accounts, games history, async fetch + coaching jobs (threadpool + polling), reviews, feedback, and per-account SHAP — fully testable via pytest + Starlette TestClient.

**Architecture:** No DB. Pure functions read the existing JSONL/JSON files in `data/` (silver, gold, reviews, feedback, shap). A `JobStore` persists async job state to `data/08_jobs/jobs.jsonl`. Long-running pipeline work (Riot pull, LLM call) is blocking and runs in a `ThreadPoolExecutor` via the job runner; the frontend polls `GET /api/jobs/{id}`. Pipeline wrappers reuse existing `riotlib` / `coach` / `feedback` functions and accept a progress callback.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2 (already a dep), `httpx` (for TestClient), existing `src/riotlib.py`, `src/aggregate_games.py` (SCOPES), `src/04_coaching/{coach,payload,schema,feedback}.py`.

## Scope of this plan

This plan covers the **backend only**. It produces a fully working, testable HTTP API.
Two follow-up plans will cover: (1) the Alpine/Tailwind/Chart.js frontend, (2) Fly.io
volume + secrets + seed deploy. The backend is the testable foundation; the frontend
consumes these endpoints.

## Global Constraints

- Python source lives under `src/`; the web backend under `web/backend/`. `tests/conftest.py`
  already puts `src/` and `src/04_coaching` on `sys.path`, so web tests can `import riotlib`,
  `import coach`, `import feedback`, `import payload`, `import schema` directly.
- `riotlib.DATA` resolves to `<repo>/data` (`ROOT = src.parent`). All data reads go through
  `rl.DATA` (or an overrideable `data_root` arg for tests). Never hardcode `data/` paths.
- Secrets come from env (`RIOT_API_ID`, `OLLAMA_API_KEY`, `OLLAMA_MODEL`), falling back to
  `.env` via `rl.load_env()`. Never log secrets.
- Pipeline calls are **blocking** and **must** run inside the threadpool (never in the event
  loop directly). Every pipeline wrapper takes an `on_progress: Callable[[str], None] | None`
  callback.
- Commit after each task. Conventional commits, French-scoped (`feat(web): …`).
- Run tests with `.venv/bin/python -m pytest tests/web -v` (project convention).
- `httpx` must be added to `requirements.txt` (TestClient dependency).

---

## File Structure

**Create:**
- `web/backend/settings.py` — paths (`ACCOUNTS_FILE`, `JOBS_FILE`), secret/model getters, account loader + puuid resolver.
- `web/backend/accounts.json` — server-side config: list of `{slug, riot_id, region}`.
- `web/backend/readers.py` — pure data readers (silver games, gold aggregate, reviews, feedback, shap, account summaries). All take optional `data_root`.
- `web/backend/jobs.py` — `JobStore` (jobs.jsonl CRUD + threading.Lock) + `submit_job` runner (ThreadPoolExecutor).
- `web/backend/pipeline.py` — `fetch_games(account, n, on_progress)` and `run_coach(player, scope, outcome, target, model, on_progress)` wrappers reusing existing pipeline.
- `web/backend/routers/accounts.py`, `games.py`, `jobs.py`, `reviews.py`, `feedback.py`, `shap.py` — FastAPI APIRouters.
- `web/backend/main.py` — modified: add `src/04_coaching` to path, include routers, SPA catch-all.
- `web/backend/__init__.py` (empty), `web/backend/routers/__init__.py` (empty).
- `tests/web/__init__.py` (empty), `tests/web/conftest.py` (fixtures root), `tests/web/fixtures/…` (small JSONL samples).
- `tests/web/test_readers.py`, `test_jobs.py`, `test_pipeline.py`, `test_api.py`.

**Modify:**
- `web/backend/main.py` — include routers, add 04_coaching to path, catch-all route.
- `requirements.txt` — add `httpx>=0.27`.

---

## Task 1: Settings, accounts config, and data readers

**Files:**
- Create: `web/backend/settings.py`
- Create: `web/backend/accounts.json`
- Create: `web/backend/readers.py`
- Create: `tests/web/__init__.py`, `tests/web/conftest.py`
- Create: `tests/web/fixtures/spadzze/silver_games.jsonl` (3 lines)
- Create: `tests/web/fixtures/spadzze/reviews.jsonl` (1 line)
- Create: `tests/web/fixtures/spadzze/feedback.jsonl` (1 line)
- Create: `tests/web/fixtures/spadzze/shap_drivers.json`
- Create: `tests/web/test_readers.py`
- Modify: `tests/web/__init__.py` empty marker

**Interfaces:**
- Consumes: `riotlib` (`rl.DATA`, `rl.SILVER_DIR`, `rl.GOLD_DIR`, `rl.load_env`), `feedback` (`list_reviews`, `load_feedbacks`).
- Produces (exact signatures later tasks rely on):
  - `settings.ACCOUNTS_FILE: Path`, `settings.JOBS_FILE: Path`
  - `settings.riot_api_key() -> str | None`, `settings.ollama_model() -> str`
  - `settings.load_accounts() -> list[dict]`, `settings.account_for(slug) -> dict | None`
  - `settings.resolve_puuid(account: dict) -> str` (in-memory cached, calls RiotClient)
  - `readers.read_games(slug, page=1, size=20, data_root=None) -> dict` → `{"items": [...], "page": int, "size": int, "total": int}`
  - `readers.read_aggregate(slug, scope="all", data_root=None) -> dict | None`
  - `readers.read_reviews(slug, data_root=None) -> list[dict]`
  - `readers.read_feedback(slug, data_root=None) -> list[dict]`
  - `readers.read_shap(slug, data_root=None) -> dict` → `{"available": bool, "drivers": [...]}`
  - `readers.account_summaries(data_root=None) -> list[dict]` → `{slug, riot_id, region, games_count, last_review_ts}`

- [ ] **Step 1: Write the failing test (settings + account lookup)**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/web/test_readers.py::test_load_accounts_returns_list -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'readers'` (or `settings`).

- [ ] **Step 3: Write settings.py + accounts.json**

```python
# web/backend/settings.py
"""Config web : comptes préconfigurés, chemins, secrets lus via env puis .env."""
from __future__ import annotations

import json
import os
from pathlib import Path

import riotlib as rl

HERE = Path(__file__).resolve().parent
ACCOUNTS_FILE = HERE / "accounts.json"
JOBS_FILE = rl.DATA / "08_jobs" / "jobs.jsonl"

_PUUID_CACHE: dict[str, str] = {}


def _env() -> dict[str, str]:
    return rl.load_env()


def riot_api_key() -> str | None:
    return os.environ.get("RIOT_API_ID") or _env().get("RIOT_API_ID")


def ollama_key() -> str | None:
    return os.environ.get("OLLAMA_API_KEY") or _env().get("OLLAMA_API_KEY")


def ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL") or _env().get("OLLAMA_MODEL", "kimi-k2.6")


def load_accounts() -> list[dict]:
    return json.loads(ACCOUNTS_FILE.read_text())


def account_for(slug: str) -> dict | None:
    for a in load_accounts():
        if a["slug"] == slug:
            return a
    return None


def resolve_puuid(account: dict) -> str:
    """Résout le puuid via account-v1 (cache en mémoire). Lève si introuvable."""
    slug = account["slug"]
    if slug in _PUUID_CACHE:
        return _PUUID_CACHE[slug]
    regional = rl.PLATFORM_TO_REGIONAL[account["region"]]
    client = rl.RiotClient(riot_api_key(), regional, account["region"])
    game_name, tag_line = account["riot_id"].split("#", 1)
    puuid = client.puuid_from_riot_id(game_name, tag_line)
    if not puuid:
        raise RuntimeError(f"Riot ID introuvable : {account['riot_id']}")
    _PUUID_CACHE[slug] = puuid
    return puuid
```

```json
[
  {"slug": "spadzze", "riot_id": "Spadzze#euw", "region": "euw1"}
]
```
(`web/backend/accounts.json` — un compte pour démarrer ; tu ajouteras les smurfs/amis plus tard.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/web/test_readers.py::test_load_accounts_returns_list tests/web/test_readers.py::test_account_for_returns_match_or_none -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test (read_games pagination + read_reviews + read_shap)**

```python
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
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/web/test_readers.py -v`
Expected: FAIL (readers functions missing / fixtures absent).

- [ ] **Step 7: Create fixtures**

```jsonl
{"match_id":"EUW1_1","puuid":"p1","rank":"emerald","patch":"16.13","champion":"Zeri","role":"BOTTOM","win":true,"queue":420,"lane":"BOT","comp":{},"deaths":3,"kills":8,"assists":5,"support_deaths_early":0,"plates_diff_early":1,"frames_in_base_early":2,"avg_dragon_prox":0.4,"position":{}}
{"match_id":"EUW1_2","puuid":"p1","rank":"emerald","patch":"16.13","champion":"Jinx","role":"BOTTOM","win":false,"queue":420,"lane":"BOT","comp":{},"deaths":6,"kills":2,"assists":3,"support_deaths_early":1,"plates_diff_early":-1,"frames_in_base_early":3,"avg_dragon_prox":0.3,"position":{}}
{"match_id":"EUW1_3","puuid":"p1","rank":"emerald","patch":"16.13","champion":"Caitlyn","role":"BOTTOM","win":true,"queue":420,"lane":"BOT","comp":{},"deaths":2,"kills":5,"assists":7,"support_deaths_early":0,"plates_diff_early":2,"frames_in_base_early":1,"avg_dragon_prox":0.5,"position":{}}
```
(`tests/web/fixtures/spadzze/silver_games.jsonl` — 3 lines.)

```json
{"ts":"2026-06-30T17:53:39","model":"kimi-k2.6","scope":"adc","target":"challenger","outcome_focus":"loss","payload":{"meta":{"player":"spadzze","scope":"adc"}},"review":{"strengths":[{"point":"x","evidence":"e"}],"mistakes":[{"point":"y","evidence":"e"}],"habits":["h1","h2"],"next_focus":"f","confidence":0.6}}
```
(`tests/web/fixtures/spadzze/reviews.jsonl` — 1 line.)

```json
{"ts":"2026-06-30T17:53:39","player":"spadzze","rated_at":"2026-06-30T18:00:00","model":"kimi-k2.6","overall_useful":null,"items":[{"kind":"strength","index":0,"useful":true,"tag":null,"note":null}]}
```
(`tests/web/fixtures/spadzze/feedback.jsonl` — 1 line.)

```json
{"features":["avg_dist_to_ally","wards_killed"],"values":[1888.0,1.5],"global_mean":[1700.0,1.0]}
```
(`tests/web/fixtures/spadzze/shap_drivers.json`.)

```python
# tests/web/conftest.py
import sys
from pathlib import Path

# Reuse the project conftest path setup (src + 04_coaching already on path via
# tests/conftest.py). Add web/backend so `import readers`, `import settings` work.
_WEB = Path(__file__).resolve().parent.parent.parent / "web" / "backend"
sys.path.insert(0, str(_WEB))
```

```python
# tests/web/__init__.py
```

- [ ] **Step 8: Write readers.py**

```python
# web/backend/readers.py
"""Lecteurs purs des données existantes (silver/gold/reviews/feedback/shap).

Aucune DB. Toutes les fonctions acceptent `data_root` (défaut rl.DATA) pour les tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import riotlib as rl

import settings


def _silver_games_path(slug: str, data_root: Path | None) -> Path:
    base = data_root if data_root is not None else rl.DATA
    return Path(base) / "02_silver" / "personal" / slug / "games.jsonl"


def _aggregate_path(slug: str, scope: str, data_root: Path | None) -> Path:
    base = data_root if data_root is not None else rl.DATA
    return Path(base) / "03_gold" / "personal" / slug / scope / "aggregate.json"


def _coaching_dir(slug: str, data_root: Path | None) -> Path:
    base = data_root if data_root is not None else rl.DATA
    return Path(base) / "07_coaching" / slug


def _shap_drivers_path(slug: str, data_root: Path | None) -> Path:
    base = data_root if data_root is not None else rl.DATA
    return Path(base) / "06_shap" / f"{slug}_drivers.json"


def read_games(slug: str, page: int = 1, size: int = 20,
               data_root: Path | None = None) -> dict:
    path = _silver_games_path(slug, data_root)
    rows: list[dict] = []
    if path.exists():
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = list(reversed(rows))  # plus récentes d'abord
    total = len(rows)
    start = (page - 1) * size
    items = rows[start:start + size]
    return {"items": items, "page": page, "size": size, "total": total}


def read_aggregate(slug: str, scope: str = "all",
                   data_root: Path | None = None) -> dict | None:
    path = _aggregate_path(slug, scope, data_root)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_reviews(slug: str, data_root: Path | None = None) -> list[dict]:
    path = _coaching_dir(slug, data_root) / "reviews.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def read_feedback(slug: str, data_root: Path | None = None) -> list[dict]:
    path = _coaching_dir(slug, data_root) / "feedback.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def read_shap(slug: str, data_root: Path | None = None) -> dict:
    path = _shap_drivers_path(slug, data_root)
    if not path.exists():
        return {"available": False, "drivers": []}
    return {"available": True, "drivers": json.loads(path.read_text())}


def account_summaries(data_root: Path | None = None) -> list[dict]:
    out = []
    for a in settings.load_accounts():
        games = read_games(a["slug"], page=1, size=1, data_root=data_root)
        revs = read_reviews(a["slug"], data_root=data_root)
        out.append({
            "slug": a["slug"],
            "riot_id": a["riot_id"],
            "region": a["region"],
            "games_count": games["total"],
            "last_review_ts": revs[-1]["ts"] if revs else None,
        })
    return out
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/web/test_readers.py -v`
Expected: PASS (6 tests). Note: `test_account_summaries` uses `settings.load_accounts()` which reads the real `accounts.json` — the fixture dir only contains `spadzze`, so ensure `accounts.json` lists only `spadzze` (it does by default). If you later add accounts to `accounts.json`, the test reads `games_count` from the fixture for `spadzze` and `0` for others — acceptable.

- [ ] **Step 10: Commit**

```bash
git add web/backend/settings.py web/backend/accounts.json web/backend/readers.py \
        web/backend/__init__.py tests/web/
git commit -m "feat(web): settings + lecteurs données (silver/gold/reviews/feedback/shap)"
```

---

## Task 2: Job store + threadpool runner

**Files:**
- Create: `web/backend/jobs.py`
- Create: `tests/web/test_jobs.py`

**Interfaces:**
- Consumes: `settings.JOBS_FILE` (path).
- Produces:
  - `jobs.JobStore(path: Path)` with `.create(type, slug) -> dict`, `.get(id) -> dict | None`, `.list(slug=None) -> list[dict]`, `.set_progress(id, progress)`, `.set_done(id, result_ref=None)`, `.set_error(id, error)`.
  - `jobs.submit_job(store, fn, *, type, slug) -> dict` — runs `fn(on_progress=...)` in a `ThreadPoolExecutor`; updates the store on done/error. Returns the created job dict.
  - Job dict shape: `{"id": str, "type": "fetch"|"coach", "slug": str, "status": "pending"|"running"|"done"|"error", "progress": str|None, "ts_start": str, "ts_end": str|None, "error": str|None, "result_ref": dict|None}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/web/test_jobs.py
import time

from jobs import JobStore, submit_job


def test_create_and_get(tmp_path):
    store = JobStore(tmp_path / "jobs.jsonl")
    j = store.create("fetch", "spadzze")
    assert j["type"] == "fetch" and j["slug"] == "spadzze"
    assert j["status"] == "pending"
    assert store.get(j["id"])["id"] == j["id"]
    assert store.get("nope") is None


def test_set_progress_then_done(tmp_path):
    store = JobStore(tmp_path / "jobs.jsonl")
    j = store.create("fetch", "spadzze")
    store.set_progress(j["id"], "5/20")
    assert store.get(j["id"])["progress"] == "5/20"
    assert store.get(j["id"])["status"] == "running"
    store.set_done(j["id"], result_ref={"n_games": 20})
    got = store.get(j["id"])
    assert got["status"] == "done"
    assert got["result_ref"] == {"n_games": 20}
    assert got["ts_end"] is not None


def test_set_error(tmp_path):
    store = JobStore(tmp_path / "jobs.jsonl")
    j = store.create("coach", "spadzze")
    store.set_error(j["id"], "boom")
    got = store.get(j["id"])
    assert got["status"] == "error" and got["error"] == "boom"


def test_list_filters_by_slug(tmp_path):
    store = JobStore(tmp_path / "jobs.jsonl")
    store.create("fetch", "spadzze")
    store.create("fetch", "ace")
    assert len(store.list()) == 2
    assert len(store.list("spadzze")) == 1


def test_submit_job_runs_and_marks_done(tmp_path):
    store = JobStore(tmp_path / "jobs.jsonl")

    def work(on_progress=None):
        if on_progress:
            on_progress("1/1")
        return {"ok": True}

    j = submit_job(store, work, type="fetch", slug="spadzze")
    # poll until done
    for _ in range(100):
        got = store.get(j["id"])
        if got["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert store.get(j["id"])["status"] == "done"
    assert store.get(j["id"])["result_ref"] == {"ok": True}


def test_submit_job_captures_error(tmp_path):
    store = JobStore(tmp_path / "jobs.jsonl")

    def boom(on_progress=None):
        raise RuntimeError("nope")

    j = submit_job(store, boom, type="coach", slug="spadzze")
    for _ in range(100):
        got = store.get(j["id"])
        if got["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert store.get(j["id"])["status"] == "error"
    assert "nope" in store.get(j["id"])["error"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/web/test_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jobs'`.

- [ ] **Step 3: Write jobs.py**

```python
# web/backend/jobs.py
"""État des jobs async : store JSONL (1 ligne/job) + runner threadpool.

Le pipeline existant est bloquant -> il tourne dans un ThreadPoolExecutor.
L'état est persisté dans data/08_jobs/jobs.jsonl (survit aux redémarrages).
"""
from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_EXECUTOR = ThreadPoolExecutor(max_workers=2)
# Le store global partagé entre l'API et le runner.
from settings import JOBS_FILE  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else JOBS_FILE
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def _write(self, jobs: list[dict]) -> None:
        self.path.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n"
                                    for j in jobs))

    def create(self, type: str, slug: str) -> dict:
        job = {"id": uuid.uuid4().hex, "type": type, "slug": slug,
               "status": "pending", "progress": None, "ts_start": _now(),
               "ts_end": None, "error": None, "result_ref": None}
        with self._lock:
            jobs = self._read()
            jobs.append(job)
            self._write(jobs)
        return job

    def get(self, id: str) -> dict | None:
        with self._lock:
            for j in self._read():
                if j["id"] == id:
                    return j
        return None

    def list(self, slug: str | None = None) -> list[dict]:
        with self._lock:
            jobs = self._read()
        if slug is None:
            return jobs
        return [j for j in jobs if j["slug"] == slug]

    def _update(self, id: str, **fields) -> dict:
        with self._lock:
            jobs = self._read()
            for j in jobs:
                if j["id"] == id:
                    j.update(fields)
                    self._write(jobs)
                    return j
        raise KeyError(id)

    def set_progress(self, id: str, progress: str) -> None:
        self._update(id, status="running", progress=progress)

    def set_done(self, id: str, result_ref: dict | None = None) -> None:
        self._update(id, status="done", ts_end=_now(), result_ref=result_ref)

    def set_error(self, id: str, error: str) -> None:
        self._update(id, status="error", ts_end=_now(), error=error)


def submit_job(store: JobStore, fn: Callable, *, type: str, slug: str) -> dict:
    """Lance fn(on_progress=cb) dans le threadpool, met à jour le store en fin."""
    job = store.create(type, slug)

    def _wrapper():
        try:
            res = fn(on_progress=lambda p: store.set_progress(job["id"], p))
            store.set_done(job["id"], result_ref=res if isinstance(res, dict) else None)
        except Exception as e:  # noqa: BLE001
            store.set_error(job["id"], repr(e))

    _EXECUTOR.submit(_wrapper)
    return job
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/web/test_jobs.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add web/backend/jobs.py tests/web/test_jobs.py
git commit -m "feat(web): job store JSONL + runner threadpool (progress/done/error)"
```

---

## Task 3: Pipeline wrappers (fetch + coach) with progress callback

**Files:**
- Create: `web/backend/pipeline.py`
- Create: `tests/web/test_pipeline.py`

**Interfaces:**
- Consumes: `riotlib` (`RiotClient`, `match_ids`, `get_match_timeline`, `extract_game`, `merge_jsonl`, `write_gold`, `QUEUE_SOLO`, `PLATFORM_TO_REGIONAL`, `load_env`), `aggregate_games.SCOPES`, `coach` (`generate_review`, `persist`), `payload` (`build`), `settings` (`riot_api_key`, `ollama_model`, `resolve_puuid`).
- Produces:
  - `pipeline.fetch_games(account: dict, n: int = 20, on_progress: Callable[[str],None] | None = None) -> dict` → `{"n_games": int, "player": str}`
  - `pipeline.run_coach(player: str, scope="adc", outcome="loss", target="challenger", model: str | None = None, on_progress=None) -> dict` → `{"ts": str}`

- [ ] **Step 1: Write the failing test (fetch_games drives progress + writes silver/gold)**

```python
# tests/web/test_pipeline.py
import json
from pathlib import Path
from unittest.mock import patch

import pipeline
import riotlib as rl


def test_fetch_games_progress_and_writes_silver_gold(tmp_path):
    account = {"slug": "spadzze", "riot_id": "Spadzze#euw", "region": "euw1"}
    progresses: list[str] = []

    fake_match = {"info": {"participants": [{"puuid": "p1", "championName": "Zeri"}]}}
    fake_timeline = {"info": {"frames": []}}

    def fake_puuid(self, game_name, tag_line):
        return "p1"

    def fake_match_ids(self, count, queue):
        return ["m1", "m2"]

    def fake_get_timeline(client, mid):
        return (fake_match, fake_timeline)

    def fake_extract(match, timeline, puuid):
        return {"match_id": mid_global[0], "puuid": puuid, "rank": "emerald",
                "patch": "16.13", "champion": "Zeri", "role": "BOTTOM", "win": True,
                "queue": 420, "lane": "BOT", "comp": {}, "deaths": 3, "kills": 8,
                "assists": 5, "support_deaths_early": 0, "plates_diff_early": 1,
                "frames_in_base_early": 2, "avg_dragon_prox": 0.4, "position": {}}

    mid_global = ["m1"]
    calls = []

    def fake_extract_factory():
        def f(match, timeline, puuid):
            g = fake_extract(match, timeline, puuid)
            g["match_id"] = calls.pop(0) if calls else "m1"
            return g
        return f

    with patch("riotlib.RiotClient.puuid_from_riot_id", fake_puuid), \
         patch("riotlib.RiotClient.match_ids", fake_match_ids), \
         patch("riotlib.get_match_timeline", fake_get_timeline), \
         patch("riotlib.extract_game",
               lambda m, t, p: {"match_id": "m1", "puuid": p, "rank": "emerald",
                                "patch": "16.13", "champion": "Zeri",
                                "role": "BOTTOM", "win": True, "queue": 420,
                                "lane": "BOT", "comp": {}, "deaths": 3, "kills": 8,
                                "assists": 5, "support_deaths_early": 0,
                                "plates_diff_early": 1, "frames_in_base_early": 2,
                                "avg_dragon_prox": 0.4, "position": {}}), \
         patch("riotlib.merge_jsonl",
               lambda path, new: new), \
         patch("riotlib.write_gold") as wg, \
         patch("pipeline.settings.riot_api_key", lambda: "k"):
        res = pipeline.fetch_games(account, n=2,
                                   on_progress=lambda p: progresses.append(p))
    assert res["n_games"] == 2
    assert progresses[-1] == "2/2"
    assert wg.called


def test_run_coach_calls_payload_build_and_persist(tmp_path):
    with patch("pipeline.payload.build", return_value={"meta": {"scope": "adc"}}) as pb, \
         patch("pipeline.coach.generate_review", return_value="REVIEW") as gr, \
         patch("pipeline.coach.persist", return_value=tmp_path / "r.jsonl") as pe:
        res = pipeline.run_coach("spadzze", scope="adc", outcome="loss",
                                 target="challenger", model="kimi-k2.6")
    assert pb.called and pb.call_args.args == ("spadzze", "adc", "challenger", "loss")
    assert gr.called and gr.call_args.args[1] == "kimi-k2.6"
    assert pe.called
    assert "ts" in res
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/web/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline'`.

- [ ] **Step 3: Write pipeline.py**

```python
# web/backend/pipeline.py
"""Wrappers autour du pipeline existant pour le web, avec callback de progression.

Réutilise riotlib (pull/silver/gold) et coach (payload/generate/persist). Ne réécrit
rien : juste un point d'entrée callable avec on_progress.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Callable

import riotlib as rl
from aggregate_games import SCOPES

import coach as coach_mod
import payload as payload_mod
import settings


def fetch_games(account: dict, n: int = 20,
                on_progress: Callable[[str], None] | None = None) -> dict:
    """Pull Riot -> silver -> gold pour un compte. Bloquant : lancer via threadpool."""
    key = settings.riot_api_key()
    if not key:
        raise RuntimeError("RIOT_API_ID manquant")
    platform = account["region"]
    regional = rl.PLATFORM_TO_REGIONAL[platform]
    client = rl.RiotClient(key, regional, platform)
    game_name, tag_line = account["riot_id"].split("#", 1)
    player = game_name.lower()
    puuid = settings.resolve_puuid(account)
    if not puuid:
        raise RuntimeError("Riot ID introuvable")

    games: list[dict] = []
    for i, mid in enumerate(client.match_ids(puuid, count=n, queue=rl.QUEUE_SOLO), 1):
        got = rl.get_match_timeline(client, mid)
        if not got:
            continue
        g = rl.extract_game(got[0], got[1], puuid)
        if g:
            games.append(g)
        if on_progress:
            on_progress(f"{i}/{n}")

    if not games:
        raise RuntimeError("Aucune game exploitable")
    merged = rl.merge_jsonl(rl.SILVER_DIR / "personal" / player / "games.jsonl", games)
    rl.write_gold(rl.GOLD_DIR / "personal" / player, merged, SCOPES, player=player)
    return {"n_games": len(merged), "player": player}


def run_coach(player: str, scope: str = "adc", outcome: str = "loss",
              target: str = "challenger", model: str | None = None,
              on_progress: Callable[[str], None] | None = None) -> dict:
    """Payload -> LLM -> persist. Bloquant : lancer via threadpool."""
    if model is None:
        model = settings.ollama_model()
    if on_progress:
        on_progress("payload")
    pl = payload_mod.build(player, scope, target, outcome)
    if on_progress:
        on_progress("llm")
    review = coach_mod.generate_review(pl, model)
    ts = datetime.now().isoformat(timespec="seconds")
    coach_mod.persist(player, model, pl, review, ts)
    return {"ts": ts}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/web/test_pipeline.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add web/backend/pipeline.py tests/web/test_pipeline.py
git commit -m "feat(web): wrappers pipeline fetch+coach avec callback progression"
```

---

## Task 4: API routers (accounts, games, jobs, reviews, feedback, shap)

**Files:**
- Create: `web/backend/routers/__init__.py` (empty)
- Create: `web/backend/routers/accounts.py`, `games.py`, `jobs.py`, `reviews.py`, `feedback.py`, `shap.py`
- Create: `tests/web/test_api.py`
- Modify: `requirements.txt` (add `httpx>=0.27`)

**Interfaces:**
- Consumes: `readers`, `jobs` (`JobStore`, `submit_job`), `pipeline` (`fetch_games`, `run_coach`), `settings` (`account_for`), `feedback` (`build_feedback`, `persist_feedback`, `load_review`).
- Produces (HTTP):
  - `GET /api/accounts` -> `list[dict]` (account_summaries)
  - `GET /api/c/{slug}/games?page=&size=` -> `{items,page,size,total}`
  - `GET /api/c/{slug}/reviews` -> `list[dict]`
  - `GET /api/c/{slug}/feedback` -> `list[dict]`
  - `GET /api/c/{slug}/shap` -> `{available, drivers}`
  - `POST /api/fetch` body `{slug, n?}` -> `{job_id}`
  - `POST /api/coach` body `{slug, scope, outcome, target, model?}` -> `{job_id}`
  - `GET /api/jobs/{id}` -> job dict (404 if missing)
  - `POST /api/feedback` body `{slug, ts, responses}` -> `{ok: true}`

- [ ] **Step 1: Add httpx to requirements**

Append to `requirements.txt`:
```
httpx>=0.27
```
Run: `.venv/bin/pip install -r requirements.txt`

- [ ] **Step 2: Write the failing test (TestClient smoke + fetch/coach job creation)**

```python
# tests/web/test_api.py
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


class _rl_proxy:
    """Shim so readers' `data_root or rl.DATA` picks the fixture dir."""
    def __init__(self, data):
        self.DATA = data


def _rl_proxy(fix):
    return _rl_proxy(fix)


def test_health(client := None):
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


def test_shap_endpoint(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/c/spadzze/shap")
    assert r.status_code == 200 and r.json()["available"] is True


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/web/test_api.py -v`
Expected: FAIL (routers / main not wired; imports missing).

- [ ] **Step 4: Write the routers**

```python
# web/backend/routers/__init__.py
```

```python
# web/backend/routers/accounts.py
from fastapi import APIRouter

import readers

router = APIRouter()


@router.get("/api/accounts")
def list_accounts():
    return readers.account_summaries()
```

```python
# web/backend/routers/games.py
from fastapi import APIRouter

import readers

router = APIRouter()


@router.get("/api/c/{slug}/games")
def games(slug: str, page: int = 1, size: int = 20):
    return readers.read_games(slug, page=page, size=size)
```

```python
# web/backend/routers/reviews.py
from fastapi import APIRouter

import readers

router = APIRouter()


@router.get("/api/c/{slug}/reviews")
def reviews(slug: str):
    return readers.read_reviews(slug)


@router.get("/api/c/{slug}/feedback")
def feedback_list(slug: str):
    return readers.read_feedback(slug)
```

```python
# web/backend/routers/shap.py
from fastapi import APIRouter

import readers

router = APIRouter()


@router.get("/api/c/{slug}/shap")
def shap(slug: str):
    return readers.read_shap(slug)
```

```python
# web/backend/routers/jobs.py
"""Jobs async : POST /api/fetch, POST /api/coach, GET /api/jobs/{id}."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import pipeline
import settings
from jobs import JobStore, submit_job

router = APIRouter()

# Store partagé pour l'instance.
_store = JobStore()


class FetchReq(BaseModel):
    slug: str
    n: int = 20


class CoachReq(BaseModel):
    slug: str
    scope: str = "adc"
    outcome: str = "loss"
    target: str = "challenger"
    model: str | None = None


@router.post("/api/fetch")
def fetch(req: FetchReq):
    account = settings.account_for(req.slug)
    if not account:
        raise HTTPException(404, "compte inconnu")
    job = submit_job(_store, lambda on_progress=None: pipeline.fetch_games(
        account, n=req.n, on_progress=on_progress), type="fetch", slug=req.slug)
    return {"job_id": job["id"]}


@router.post("/api/coach")
def coach(req: CoachReq):
    if not settings.account_for(req.slug):
        raise HTTPException(404, "compte inconnu")
    job = submit_job(_store, lambda on_progress=None: pipeline.run_coach(
        req.slug, scope=req.scope, outcome=req.outcome, target=req.target,
        model=req.model, on_progress=on_progress), type="coach", slug=req.slug)
    return {"job_id": job["id"]}


@router.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = _store.get(job_id)
    if not j:
        raise HTTPException(404, "job inconnu")
    return j
```

```python
# web/backend/routers/feedback.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import feedback as F
import settings

router = APIRouter()


class FeedbackReq(BaseModel):
    slug: str
    ts: str
    responses: dict[str, dict]  # "kind,index" -> {useful, tag?, note?}


@router.post("/api/feedback")
def post_feedback(req: FeedbackReq):
    if not settings.account_for(req.slug):
        raise HTTPException(404, "compte inconnu")
    review_dict = F.load_review(req.slug, req.ts)
    if not review_dict:
        raise HTTPException(404, "review introuvable")
    review = F.schema_mod.Review.model_validate(review_dict["review"])
    responses = {}
    for k, v in req.responses.items():
        kind, idx = k.split(",")
        responses[(kind, int(idx))] = (v["useful"], v.get("tag"), v.get("note"))
    from datetime import datetime
    fb = F.build_feedback(review, req.ts, req.slug, review_dict["model"],
                          datetime.now().isoformat(timespec="seconds"), responses)
    F.persist_feedback(req.slug, fb)
    return {"ok": True}
```

- [ ] **Step 5: Wire routers + SPA catch-all into main.py**

```python
# web/backend/main.py
"""coaching_lol — backend FastAPI. Un seul process Fly.io sert l'API (/api/*)
et le frontend statique."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
COACH = SRC / "04_coaching"
if str(COACH) not in sys.path:
    sys.path.insert(0, str(COACH))
BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

from routers import accounts, games, jobs, reviews, feedback, shap  # noqa: E402

app = FastAPI(title="coaching_lol", version="0.1.0")

app.include_router(accounts.router)
app.include_router(games.router)
app.include_router(reviews.router)
app.include_router(jobs.router)
app.include_router(feedback.router)
app.include_router(shap.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "coaching-lol",
            "server_time": datetime.now(timezone.utc).isoformat()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/c/{slug}")
@app.get("/readme")
def spa(slug: str | None = None) -> FileResponse:
    """Catch-all SPA : sert le même index.html, le routeur Alpine lit l'URL."""
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/web/test_api.py -v`
Expected: PASS (7 tests). If `test_health`/`test_accounts_endpoint` import-order issues arise, ensure `tests/web/conftest.py` (Task 1) puts `web/backend` on the path before `import main`.

- [ ] **Step 7: Commit**

```bash
git add web/backend/routers/ web/backend/main.py requirements.txt tests/web/test_api.py
git commit -m "feat(web): routers API (accounts/games/jobs/reviews/feedback/shap) + SPA catch-all"
```

---

## Task 5: Full backend smoke test + job progress integration

**Files:**
- Modify: `tests/web/test_api.py` (append an integration test that runs a real short job end-to-end through the threadpool and observes progress transitions).

**Interfaces:**
- Consumes: everything from Tasks 1-4.

- [ ] **Step 1: Write the integration test**

```python
# append to tests/web/test_api.py
import time


def test_fetch_job_progresses_to_done(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    calls = []

    def fake_work(on_progress=None):
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
```

- [ ] **Step 2: Run the full web test suite**

Run: `.venv/bin/python -m pytest tests/web -v`
Expected: PASS (all readers + jobs + pipeline + api tests).

- [ ] **Step 3: Manual smoke — boot the server**

Run: `.venv/bin/python -m uvicorn main:app --app-dir web/backend --reload`
Then in another shell:
```bash
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/accounts
curl -s http://127.0.0.1:8000/api/c/spadzze/games?size=5
```
Expected: health JSON, accounts list with `spadzze`, games list from real `data/`.

- [ ] **Step 4: Commit**

```bash
git add tests/web/test_api.py
git commit -m "test(web): intégration job fetch (progress -> done) + 404 compte inconnu"
```

---

## Self-Review notes

**Spec coverage:**
- Pages (home/account/readme) → frontend plan (out of scope here). Backend provides the data each page needs: `/api/accounts` (home grid), `/api/c/{slug}/games` + `/reviews` + `/shap` + `POST /api/feedback` + `POST /api/coach` (account page), README is static (no endpoint).
- Data model (read files directly, no DB) → Task 1 readers. ✅
- `jobs.jsonl` new entity → Task 2. ✅
- Async: threadpool + polling → Tasks 2, 4 (`POST /api/fetch`, `GET /api/jobs/{id}`). ✅
- Pipeline wrappers with progress callback → Task 3. ✅
- Endpoints table (spec § Endpoints) → all 9 endpoints present in Task 4. ✅
- SHAP limited to precomputed slugs → `read_shap` returns `{available:false}` when file absent. ✅
- Secrets via env + .env fallback → `settings.riot_api_key`/`ollama_model`. ✅

**Placeholder scan:** none — all code blocks are complete.

**Type consistency:** `JobStore.set_progress(id, progress)`, `submit_job(store, fn, type=, slug=)`, `fetch_games(account, n, on_progress)`, `run_coach(player, scope, outcome, target, model, on_progress)` signatures match across tasks 2-4. `readers.read_games` returns `{items,page,size,total}` used by both router and tests. ✅

**Gaps deferred to follow-up plans:** frontend (Alpine/Tailwind/Chart.js views), Fly volume + seed + secrets deploy. The backend is independently testable and shippable behind a curl-able API.