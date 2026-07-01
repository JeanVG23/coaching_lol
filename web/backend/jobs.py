# web/backend/jobs.py
"""État des jobs async : store JSONL (1 ligne/job) + runner threadpool.

Le pipeline existant est bloquant -> il tourne dans un ThreadPoolExecutor.
L'état est persisté dans data/08_jobs/jobs.jsonl (survit aux redémarrages).
"""
from __future__ import annotations

import json
import os
import tempfile
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".jobs_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.replace(tmp, self.path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

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