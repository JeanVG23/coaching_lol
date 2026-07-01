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