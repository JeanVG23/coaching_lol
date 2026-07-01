# Task 4 Report — API routers + SPA catch-all

## Steps

1. **requirements.txt** — appended `httpx>=0.27`. `httpx` was already installed (0.28.1) so no install needed.
2. **Failing test** — wrote `tests/web/test_api.py` (brief code + added `test_games_bad_params_422` per cross-task context).
3. **Confirm fail** — initial run failed (syntax error, then `ModuleNotFoundError: routers`, then 404s) as expected.
4. **Routers** — created `web/backend/routers/` with `__init__.py` (empty), `accounts.py`, `games.py` (with route-layer validation), `reviews.py`, `shap.py`, `jobs.py`, `feedback.py`.
5. **main.py** — rewrote per brief: adds `src/04_coaching` to `sys.path`, includes all 6 routers, SPA catch-all `GET /c/{slug}` + `GET /readme` serving `index.html`, mounts `/static`.
6. **Confirm pass** — 9/9 in `test_api.py`, 25/25 in full web suite.
7. **Commit** — `a3b7d27`.

## Final test command + output

### API test file
```
.venv/bin/python -m pytest tests/web/test_api.py -v
========================= 9 passed, 1 warning in 0.16s =========================
```
9 tests: `test_health`, `test_accounts_endpoint`, `test_games_endpoint`, `test_games_bad_params_422`, `test_shap_endpoint`, `test_fetch_creates_job`, `test_jobs_missing_404`, `test_coach_creates_job`, `test_feedback_post`.

### Full web suite
```
.venv/bin/python -m pytest tests/web -v
======================== 25 passed, 1 warning in 0.28s ========================
```
All Task 1-3 tests still green (readers/jobs/pipeline) + 9 new Task 4 tests.

## Deviations from the brief

The brief's provided test code (`tests/web/test_api.py`) contained two genuine Python errors that prevented collection/execution. Per the instruction "Fix code (not tests) only if a test reveals a genuine bug, and note it in your report" — these were bugs in the brief's test code itself (not production code), but they made the brief literally non-runnable, so minimal fixes were applied to the test file and are documented here:

1. **`def test_health(client := None):` → `def test_health(client=None):`**
   Walrus operator (`:=`) cannot appear in a function signature (SyntaxError on Python 3.14). Changed to a plain default. The `client` param is unused by the test body; behavior unchanged.

2. **`_rl_proxy` name collision (RecursionError)**
   The brief defines both a class `_rl_proxy` and a function `_rl_proxy(fix)` returning `_rl_proxy(fix)` — the function shadows the class, so the call recurses infinitely. Renamed the class to `_RlProxy`; the factory function `_rl_proxy(fix)` now returns `_RlProxy(fix)` as intended.

## Deviation: feedback router import alias (production-code fix)

The brief's router uses `import feedback as F` and references `F.load_review` / `F.schema_mod` / `F.build_feedback` / `F.persist_feedback`. The brief's test patches `routers.feedback.feedback.build_feedback` and `routers.feedback.feedback.persist_feedback`. With `import feedback as F`, the name bound in the router module's namespace is `F`, not `feedback`, so `routers.feedback.feedback` does not exist and the patch raises `AttributeError`.

This is a genuine inconsistency between the brief's router code and the brief's own test's patch targets. Fix applied to the **router** (production code, not the test): changed `import feedback as F` to `import feedback` and replaced `F.` with `feedback.` throughout `web/backend/routers/feedback.py`. The test's patch targets now resolve correctly (`routers.feedback.feedback` → the `src/04_coaching/feedback.py` module).

## Cross-task addition (route-layer validation, per brief's Minor finding)

`games.py` validates `page >= 1` and `1 <= size <= 200` before delegating to `readers.read_games`; on violation raises `HTTPException(422, "page>=1 et size in [1,200]")`. `test_games_bad_params_422` covers `page=0`, `size=0`, `size=201`.

## Self-review

- All 9 Task 4 endpoints wired and covered by tests.
- SPA catch-all (`/c/{slug}`, `/readme`) added; `/` and `/static` preserved from prior main.py.
- `src/04_coaching` added to `sys.path` so `import feedback`, `import coach`, `import payload` resolve.
- Route order: API routers + `/api/health` + `/` + SPA `/c/{slug}` + `/readme` are all declared before the `/static` mount, so statics don't shadow API routes.
- `JobStore()` is instantiated once at module import in `routers/jobs.py` (per brief). Tests monkeypatch `settings.JOBS_FILE` before importing `main`, so `_store` points at the tmp `jobs.jsonl`. Verified by `test_fetch_creates_job` / `test_coach_creates_job` / `test_jobs_missing_404`.
- No production-code bug fixes were needed beyond the import-alias correction above (which is itself a brief inconsistency, not a logic bug).

## Concerns

None. All 9 API endpoints + SPA catch-all wired and tested; full web suite (25 tests) green.
## Fix: lazy jobs store + 422 + cleanup

Commands run:
- `.venv/bin/python -m pytest tests/web/test_api.py -v` → 10 passed (incl. new `test_fetch_writes_to_patched_jobs_path`)
- `.venv/bin/python -m pytest tests/web -v` → 26 passed
- `: > data/08_jobs/jobs.jsonl` (truncate polluted real file to 0 bytes; still 0 after tests → no pollution)

Per-fix notes:
- Fix 1: `routers/jobs.py` — replaced module-level eager `_store = JobStore()` with path-keyed `get_store()` cache reading live `settings.JOBS_FILE` at call time; all three call sites (`fetch`, `coach`, `job_status`) updated.
- Fix 2: appended `test_fetch_writes_to_patched_jobs_path` to `tests/web/test_api.py` (+ `import time`); verifies the job lands in the tmp patched path, not the real data dir.
- Fix 3: `routers/feedback.py` — wrapped `"kind,index"` parse in try/except → 422 on malformed key.
- Fix 4: `web/backend/main.py` — removed unused `Request` from the `fastapi` import.
- Cleanup: real `data/08_jobs/jobs.jsonl` truncated to 0 bytes; confirmed still empty after the full web suite runs.

## Fix: feedback 422 error contract (I1 + I2)

Commands run:
- `.venv/bin/python -m pytest tests/web/test_api.py -v` → 14 passed
- `.venv/bin/python -m pytest tests/web -v` → 30 passed

Fixes:
- I1: `build_feedback` ValidationError (tag requis si `useful=False`) désormais capturé → HTTP 422 au lieu de 500.
- I2: `responses` typé via `FeedbackResponse` (Pydantic) ; `useful` manquant ou `tag` invalide → HTTP 422 au lieu de `KeyError`/500.
