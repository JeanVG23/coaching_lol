# Capture Live Client Data API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture, store and re-attach to a real `matchId` the local, live-only data from
Riot's Live Client Data API — no feature extraction, no coaching wiring (v1 scope only).

**Architecture:** One self-contained script, `src/live_capture.py`, with two independent
modes: `capture` (stdlib-only polling loop, runs on any PC with just Python installed)
and `--match` (post-hoc reconciliation against Match-V5, runs only on the main machine
where the full repo/`.env`/API key live). The two modes never share an import path —
`--match` lazily imports `riotlib` only when invoked, so plain `capture` mode has zero
third-party dependencies.

**Tech Stack:** Python 3 stdlib only for capture (`urllib.request`, `ssl`, `json`,
`pathlib`, `datetime`, `platform`); existing `riotlib.RiotClient` for the `--match` mode;
pytest for tests (existing project convention, see `tests/conftest.py`).

## Global Constraints

- Zero third-party dependency in capture mode — must run on a fresh Python install with
  no `pip install` step (spec: "Contrainte fondamentale : localhost uniquement, multi-PC").
- No CLI framework (`argparse`) — this codebase's scripts use a manual `arg()` flag helper
  (see `src/aggregate_games.py:24`); follow the same convention for consistency.
- No feature extraction, no `states_timeline`/`events` wiring, no coaching payload changes
  — out of scope for this plan (spec: "Hors scope (différé)").
- Storage layout is fixed by the spec: `data/01_raw_live/pending/` and
  `data/01_raw_live/matched/`, files named `<matchId>_live.jsonl` /
  `<matchId>_live_meta.json` once matched.
- `data/` is gitignored project-wide (per `CLAUDE.md`) — no directories need to be
  pre-created or committed; both are created at runtime via `mkdir(parents=True,
  exist_ok=True)`.

---

## Task 1: Matching function `find_matching_game` (pure, TDD)

**Files:**
- Create: `src/live_capture.py`
- Create: `tests/test_live_capture.py`

**Interfaces:**
- Produces: `find_matching_game(capture_meta: dict, candidates: list[dict], *, start_tolerance_s: float = 300, duration_tolerance_s: float = 90, warn=lambda msg: None) -> str | None`
  - `capture_meta`: `{"start": <ISO8601 str>, "end": <ISO8601 str>, "champion": <str>}`
  - `candidates`: list of `{"match_id": <str>, "champion": <str>, "game_start": <ISO8601 str>, "game_duration_s": <int>}`
  - Returns the matched `match_id`, or `None` if nothing qualifies.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_live_capture.py`:

```python
from datetime import datetime, timezone

import live_capture as LC


def _iso(y, mo, d, h, mi, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).isoformat()


def test_find_matching_game_exact_match():
    capture_meta = {
        "start": _iso(2026, 7, 2, 18, 0, 0),
        "end": _iso(2026, 7, 2, 18, 30, 0),
        "champion": "Zeri",
    }
    candidates = [{
        "match_id": "EUW1_111",
        "champion": "Zeri",
        "game_start": _iso(2026, 7, 2, 18, 0, 5),
        "game_duration_s": 1795,
    }]
    assert LC.find_matching_game(capture_meta, candidates) == "EUW1_111"


def test_find_matching_game_no_candidate_in_tolerance():
    capture_meta = {
        "start": _iso(2026, 7, 2, 18, 0, 0),
        "end": _iso(2026, 7, 2, 18, 30, 0),
        "champion": "Zeri",
    }
    candidates = [{
        "match_id": "EUW1_222",
        "champion": "Zeri",
        "game_start": _iso(2026, 7, 2, 20, 0, 0),  # 2h plus tard, hors tolérance
        "game_duration_s": 1800,
    }]
    assert LC.find_matching_game(capture_meta, candidates) is None


def test_find_matching_game_ambiguous_picks_closest_and_warns():
    capture_meta = {
        "start": _iso(2026, 7, 2, 18, 0, 0),
        "end": _iso(2026, 7, 2, 18, 30, 0),
        "champion": "Zeri",
    }
    candidates = [
        {"match_id": "EUW1_FAR", "champion": "Zeri",
         "game_start": _iso(2026, 7, 2, 18, 2, 0), "game_duration_s": 1800},
        {"match_id": "EUW1_CLOSE", "champion": "Zeri",
         "game_start": _iso(2026, 7, 2, 18, 0, 10), "game_duration_s": 1800},
    ]
    warnings = []
    result = LC.find_matching_game(capture_meta, candidates, warn=warnings.append)
    assert result == "EUW1_CLOSE"
    assert len(warnings) == 1


def test_find_matching_game_tolerance_boundary():
    capture_meta = {
        "start": _iso(2026, 7, 2, 18, 0, 0),
        "end": _iso(2026, 7, 2, 18, 30, 0),
        "champion": "Zeri",
    }
    # pile à la limite (300s) -> inclus
    at_boundary = [{"match_id": "EUW1_AT", "champion": "Zeri",
                    "game_start": _iso(2026, 7, 2, 18, 5, 0), "game_duration_s": 1800}]
    assert LC.find_matching_game(capture_meta, at_boundary,
                                  start_tolerance_s=300) == "EUW1_AT"
    # juste au-delà (301s) -> exclu
    beyond_boundary = [{"match_id": "EUW1_BEYOND", "champion": "Zeri",
                        "game_start": _iso(2026, 7, 2, 18, 5, 1), "game_duration_s": 1800}]
    assert LC.find_matching_game(capture_meta, beyond_boundary,
                                  start_tolerance_s=300) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_live_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'live_capture'`

- [ ] **Step 3: Write minimal implementation**

Create `src/live_capture.py`:

```python
#!/usr/bin/env python3
"""
live_capture — capture locale de la Live Client Data API (Riot) pendant une game.

Zéro dépendance hors stdlib en mode capture : ce fichier seul est copiable sur
n'importe quel PC où Python est installé, même sans le reste du repo.

Usage :
    python3 live_capture.py                              # capture (Ctrl+C pour annuler)
    python3 live_capture.py --out /chemin                 # capture, sortie dans /chemin
    python3 live_capture.py --match "Riot#Id" euw1        # relie les captures en attente
                                                           # (nécessite le repo complet)
"""
from __future__ import annotations

from datetime import datetime


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def find_matching_game(capture_meta: dict, candidates: list[dict], *,
                        start_tolerance_s: float = 300, duration_tolerance_s: float = 90,
                        warn=lambda msg: None) -> str | None:
    """Fonction pure : relie une capture (meta) à une game candidate (Match-V5).

    capture_meta : {"start", "end", "champion"} (ISO 8601 pour start/end).
    candidates   : liste de {"match_id", "champion", "game_start", "game_duration_s"}.
    """
    capture_start = _parse_iso(capture_meta["start"])
    capture_end = _parse_iso(capture_meta["end"])
    capture_duration = (capture_end - capture_start).total_seconds()
    champion = capture_meta.get("champion")

    qualifying = []
    for c in candidates:
        if champion and champion != "unknown" and c["champion"] != champion:
            continue
        start_diff = abs((_parse_iso(c["game_start"]) - capture_start).total_seconds())
        if start_diff > start_tolerance_s:
            continue
        duration_diff = abs(c["game_duration_s"] - capture_duration)
        if duration_diff > duration_tolerance_s:
            continue
        qualifying.append((start_diff, c["match_id"]))

    if not qualifying:
        return None
    qualifying.sort(key=lambda t: t[0])
    if len(qualifying) > 1:
        warn(f"{len(qualifying)} games candidates dans la tolérance, "
             f"choix du plus proche en heure de début ({qualifying[0][1]})")
    return qualifying[0][1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_live_capture.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/live_capture.py tests/test_live_capture.py
git commit -m "feat(live-capture): add matchId reconciliation logic

Fonction pure find_matching_game : relie une capture Live Client à sa
game Match-V5 par champion + heure de début + durée, avec tolérance et
résolution d'ambiguïté déterministe (plus proche en heure de début)."
```

---

## Task 2: Capture loop (`_fetch_snapshot`, `_extract_champion`, `capture`)

**Files:**
- Modify: `src/live_capture.py` (append after `find_matching_game`)
- Modify: `tests/test_live_capture.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1 (independent code path in the same file).
- Produces:
  - `capture(out_dir: Path, interval: float = POLL_INTERVAL_S, fail_threshold: int = FAIL_THRESHOLD, url: str = LIVE_CLIENT_URL) -> tuple[Path, Path] | None`
  - `LIVE_CLIENT_URL: str`, `POLL_INTERVAL_S: float`, `FAIL_THRESHOLD: int` (module constants)
  - Task 3 will call `capture`'s sibling helpers indirectly only via `main()`; no other
    task depends on `_fetch_snapshot`/`_extract_champion` directly.

- [ ] **Step 1: Write the failing test**

The Live Client endpoint only exists during a real game, but the capture loop's logic
(poll → write snapshot → detect end-of-game via consecutive failures) is deterministic
and can be exercised against a local mock HTTP server standing in for
`127.0.0.1:2999`. Append to `tests/test_live_capture.py`:

```python
import http.server
import json
import threading
from pathlib import Path


class _FakeLiveClientHandler(http.server.BaseHTTPRequestHandler):
    responses = []
    call_count = 0

    def do_GET(self):
        cls = type(self)
        if cls.call_count >= len(cls.responses):
            self.send_response(500)
            self.end_headers()
            return
        payload = cls.responses[cls.call_count]
        cls.call_count += 1
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # silence les logs HTTP pendant les tests


def _run_fake_server(responses):
    _FakeLiveClientHandler.responses = responses
    _FakeLiveClientHandler.call_count = 0
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeLiveClientHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_capture_writes_snapshots_until_game_ends(tmp_path):
    snapshot = {
        "activePlayer": {"riotIdGameName": "Spadzze"},
        "allPlayers": [{"riotIdGameName": "Spadzze", "championName": "Zeri"}],
    }
    server, thread = _run_fake_server([snapshot, snapshot, snapshot])
    try:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/liveclientdata/allgamedata"
        result = LC.capture(Path(tmp_path), interval=0.05, fail_threshold=2, url=url)
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert result is not None
    jsonl_path, meta_path = result
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        row = json.loads(line)
        assert row["data"] == snapshot

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["champion"] == "Zeri"
    assert "start" in meta and "end" in meta and "machine" in meta
```

Add `import live_capture as LC` at the top of `tests/test_live_capture.py` if not already
present from Task 1 (Task 1 already imports `live_capture as LC` — reuse it, don't
duplicate the import).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_live_capture.py::test_capture_writes_snapshots_until_game_ends -v`
Expected: FAIL with `AttributeError: module 'live_capture' has no attribute 'capture'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/live_capture.py` (after `find_matching_game`, and add the new imports to
the top of the file alongside the existing `from datetime import datetime` line):

```python
import json
import platform
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import timezone
from pathlib import Path

LIVE_CLIENT_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"
POLL_INTERVAL_S = 2.5
FAIL_THRESHOLD = 5  # échecs consécutifs après le début de capture -> fin de game détectée

_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.check_hostname = False
_SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def _fetch_snapshot(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, context=_SSL_CONTEXT, timeout=3) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError,
            json.JSONDecodeError, OSError):
        return None


def _extract_champion(snapshot: dict) -> str:
    """Best-effort : le schéma Live Client a bougé avec la migration Riot ID.
    Ne lève jamais -> 'unknown' si non identifiable, le matching s'en passe."""
    try:
        active = snapshot.get("activePlayer", {})
        my_name = active.get("riotIdGameName") or active.get("summonerName")
        if not my_name:
            return "unknown"
        for p in snapshot.get("allPlayers", []):
            p_name = p.get("riotIdGameName") or p.get("summonerName")
            if p_name == my_name:
                return p.get("championName", "unknown")
    except (AttributeError, TypeError):
        pass
    return "unknown"


def capture(out_dir: Path, interval: float = POLL_INTERVAL_S,
            fail_threshold: int = FAIL_THRESHOLD,
            url: str = LIVE_CLIENT_URL) -> tuple[Path, Path] | None:
    """Boucle bloquante : attend une game, capture jusqu'à sa fin (ou Ctrl+C).
    Retourne (jsonl_path, meta_path), ou None si rien n'a été capturé."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("En attente d'une game (Live Client Data API)... Ctrl+C pour annuler.")

    start_time = None
    champion = "unknown"
    jsonl_path = None
    meta_path = None
    fh = None
    consecutive_fails = 0

    try:
        while True:
            snapshot = _fetch_snapshot(url)
            if snapshot is None:
                if start_time is not None:
                    consecutive_fails += 1
                    if consecutive_fails >= fail_threshold:
                        break
                time.sleep(interval)
                continue

            consecutive_fails = 0
            if start_time is None:
                start_time = datetime.now(timezone.utc)
                champion = _extract_champion(snapshot)
                stamp = start_time.strftime("%Y%m%dT%H%M%SZ")
                jsonl_path = out_dir / f"{stamp}_{champion}.jsonl"
                meta_path = out_dir / f"{stamp}_{champion}_meta.json"
                fh = jsonl_path.open("a", encoding="utf-8")
                print(f"Game détectée ({champion}) — capture vers {jsonl_path.name}")

            fh.write(json.dumps({"t": datetime.now(timezone.utc).isoformat(),
                                  "data": snapshot}) + "\n")
            fh.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nInterruption manuelle.")
    finally:
        if fh is not None:
            fh.close()

    if start_time is None:
        print("Aucune game détectée, rien capturé.")
        return None

    end_time = datetime.now(timezone.utc)
    meta = {
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "champion": champion,
        "machine": platform.node(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    duration = (end_time - start_time).total_seconds()
    print(f"Capture terminée : {jsonl_path.name} ({duration:.0f}s)")
    return jsonl_path, meta_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_live_capture.py -v`
Expected: PASS (5 tests total: 4 from Task 1 + this one)

- [ ] **Step 5: Commit**

```bash
git add src/live_capture.py tests/test_live_capture.py
git commit -m "feat(live-capture): add stdlib-only Live Client polling loop

capture() poll /liveclientdata/allgamedata, écrit un snapshot JSONL par
tick + un sidecar meta (champion/heure/machine) à la fin. Zéro dépendance
tierce : copiable seul sur un PC où seul Python est installé. Testé via
un serveur HTTP local simulant l'endpoint (le vrai Live Client n'existe
qu'en game réelle)."
```

---

## Task 3: Reconciliation against Match-V5 (`_candidate_games`, `match_pending_captures`)

**Files:**
- Modify: `src/live_capture.py` (append)
- Modify: `tests/test_live_capture.py` (append)

**Interfaces:**
- Consumes: `find_matching_game` (Task 1).
- Produces:
  - `_candidate_games(client, rl_module, puuid: str, count: int = 15) -> list[dict]`
  - `match_pending_captures(pending_dir: Path, matched_dir: Path, client, rl_module, puuid: str) -> None`
  - `rl_module` is passed as a parameter (not imported at module scope) so this code has
    no import-time dependency on `riotlib`/`requests`/`zstandard` — only `main()` (Task 4)
    imports `riotlib` for real, lazily, when `--match` is actually used.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_capture.py`:

```python
class _FakeClient:
    def match_ids(self, puuid, count=15):
        return ["EUW1_111"]


class _FakeRiotLib:
    def get_match_timeline(self, client, match_id):
        match = {
            "info": {
                "gameStartTimestamp": int(datetime(2026, 7, 2, 18, 0, 0,
                                                    tzinfo=timezone.utc).timestamp() * 1000),
                "gameDuration": 1800,
                "participants": [{"puuid": "PUUID1", "championName": "Zeri"}],
            }
        }
        return match, {}


def _write_capture(pending_dir, stem, champion, start, end):
    pending_dir.mkdir(parents=True, exist_ok=True)
    (pending_dir / f"{stem}.jsonl").write_text('{"t": "x", "data": {}}\n', encoding="utf-8")
    meta = {"start": start, "end": end, "champion": champion, "machine": "TEST-PC"}
    (pending_dir / f"{stem}_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_match_pending_captures_moves_matched_files(tmp_path):
    pending = tmp_path / "pending"
    matched = tmp_path / "matched"
    _write_capture(pending, "20260702T180000Z_Zeri", "Zeri",
                    _iso(2026, 7, 2, 18, 0, 2), _iso(2026, 7, 2, 18, 30, 0))

    LC.match_pending_captures(pending, matched, _FakeClient(), _FakeRiotLib(), "PUUID1")

    assert not (pending / "20260702T180000Z_Zeri.jsonl").exists()
    assert (matched / "EUW1_111_live.jsonl").exists()
    assert (matched / "EUW1_111_live_meta.json").exists()


def test_match_pending_captures_skips_corrupt_meta(tmp_path, capsys):
    pending = tmp_path / "pending"
    matched = tmp_path / "matched"
    pending.mkdir(parents=True)
    (pending / "bad.jsonl").write_text("{}\n", encoding="utf-8")
    (pending / "bad_meta.json").write_text("{not json", encoding="utf-8")

    LC.match_pending_captures(pending, matched, _FakeClient(), _FakeRiotLib(), "PUUID1")

    assert (pending / "bad.jsonl").exists()  # laissé en place, pas déplacé
    assert not matched.exists() or not any(matched.iterdir())


def test_match_pending_captures_skips_missing_jsonl(tmp_path):
    pending = tmp_path / "pending"
    matched = tmp_path / "matched"
    pending.mkdir(parents=True)
    meta = {"start": _iso(2026, 7, 2, 18, 0, 0), "end": _iso(2026, 7, 2, 18, 30, 0),
            "champion": "Zeri", "machine": "TEST-PC"}
    (pending / "orphan_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    LC.match_pending_captures(pending, matched, _FakeClient(), _FakeRiotLib(), "PUUID1")

    assert (pending / "orphan_meta.json").exists()  # toujours là, jamais traité
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_live_capture.py -k match_pending_captures -v`
Expected: FAIL with `AttributeError: module 'live_capture' has no attribute 'match_pending_captures'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/live_capture.py`:

```python
def _candidate_games(client, rl_module, puuid: str, count: int = 15) -> list[dict]:
    """Résume les games récentes du joueur pour le matching (cache raw réutilisé
    par rl_module.get_match_timeline)."""
    candidates = []
    for mid in client.match_ids(puuid, count=count):
        match, _ = rl_module.get_match_timeline(client, mid)
        info = match["info"]
        participant = next(p for p in info["participants"] if p["puuid"] == puuid)
        game_start = datetime.fromtimestamp(info["gameStartTimestamp"] / 1000,
                                             tz=timezone.utc)
        candidates.append({
            "match_id": mid,
            "champion": participant["championName"],
            "game_start": game_start.isoformat(),
            "game_duration_s": info["gameDuration"],
        })
    return candidates


def match_pending_captures(pending_dir: Path, matched_dir: Path, client, rl_module,
                            puuid: str) -> None:
    matched_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_games(client, rl_module, puuid)

    for meta_path in sorted(pending_dir.glob("*_meta.json")):
        jsonl_path = meta_path.with_name(meta_path.name.replace("_meta.json", ".jsonl"))
        if not jsonl_path.exists():
            print(f"⚠ {meta_path.name}: .jsonl manquant, ignoré.")
            continue
        try:
            capture_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"⚠ {meta_path.name}: meta corrompu, ignoré.")
            continue

        match_id = find_matching_game(
            capture_meta, candidates,
            warn=lambda msg, f=meta_path.name: print(f"⚠ {f}: {msg}"))
        if match_id is None:
            print(f"… {meta_path.name}: aucune game correspondante pour l'instant.")
            continue

        jsonl_path.rename(matched_dir / f"{match_id}_live.jsonl")
        meta_path.rename(matched_dir / f"{match_id}_live_meta.json")
        print(f"✓ {meta_path.name} → {match_id}_live.jsonl")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_live_capture.py -v`
Expected: PASS (8 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/live_capture.py tests/test_live_capture.py
git commit -m "feat(live-capture): reconcile pending captures against Match-V5

match_pending_captures relie chaque capture en attente à son matchId réel
(champion + heure + durée via find_matching_game), déplace le couple
jsonl/meta vers matched/ une fois résolu. Idempotent : un fichier sans
correspondance reste en pending, relançable sans risque."
```

---

## Task 4: CLI entry point + manual verification

**Files:**
- Modify: `src/live_capture.py` (append `arg()`, `main()`, `if __name__ == "__main__"`)

**Interfaces:**
- Consumes: `capture` (Task 2), `match_pending_captures` (Task 3).
- Produces: the executable script itself — no further task depends on `main()`.

- [ ] **Step 1: Write the implementation**

No new automated test for this step: `main()` is thin argument-parsing glue over
already-tested functions (`capture`, `match_pending_captures`), and its `--match` branch
needs a real `.env` + network access to verify end-to-end — that final check is a manual
step below, matching how the rest of this codebase's CLI scripts are verified (see
`src/aggregate_games.py`, which also has no dedicated test file).

Append to `src/live_capture.py`:

```python
def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> int:
    if "--match" in sys.argv:
        import riotlib as rl  # import différé : uniquement nécessaire en mode --match

        idx = sys.argv.index("--match")
        pos = [a for a in sys.argv[idx + 1:] if not a.startswith("-")]
        env = rl.load_env()
        riot_id = pos[0] if pos else env.get("RIOT_ID")
        platform_name = (pos[1] if len(pos) > 1 else env.get("RIOT_REGION", "")).lower()
        api_key = env.get("RIOT_API_ID")

        if not (api_key and riot_id and "#" in riot_id and platform_name):
            print("✗ Usage: live_capture.py --match \"Riot#Id\" <platform> "
                  "(ou RIOT_API_ID/RIOT_ID/RIOT_REGION dans .env)", file=sys.stderr)
            return 1
        regional = rl.PLATFORM_TO_REGIONAL.get(platform_name)
        if not regional:
            print(f"✗ Région inconnue: {platform_name!r}", file=sys.stderr)
            return 1

        game_name, tag_line = riot_id.split("#", 1)
        client = rl.RiotClient(api_key, regional, platform_name)
        puuid = client.puuid_from_riot_id(game_name, tag_line)
        if not puuid:
            print("✗ Riot ID introuvable.", file=sys.stderr)
            return 1

        pending_dir = Path(arg("--pending", "data/01_raw_live/pending"))
        matched_dir = Path(arg("--matched", "data/01_raw_live/matched"))
        match_pending_captures(pending_dir, matched_dir, client, rl, puuid)
        return 0

    capture(Path(arg("--out", ".")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the full test suite to confirm nothing broke**

Run: `.venv/bin/python -m pytest tests/test_live_capture.py -v`
Expected: PASS (8 tests, unchanged from Task 3 — this step adds no new automated tests)

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (full existing suite still green — confirms `src/live_capture.py` doesn't
collide with any existing module name or break `conftest.py`'s `sys.path` setup)

- [ ] **Step 3: Manual smoke check — capture mode needs no running game**

Run: `.venv/bin/python src/live_capture.py`
Expected: prints `En attente d'une game (Live Client Data API)... Ctrl+C pour annuler.`
and hangs (no League game running, so the endpoint never responds). Press `Ctrl+C`.
Expected: prints `Interruption manuelle.` then `Aucune game détectée, rien capturé.` and
the process exits with code 0.

This confirms the script starts, waits correctly, and shuts down cleanly without ever
needing riotlib/requests/zstandard — verify no import error is raised before reaching the
"En attente" line, which would indicate an accidental module-level dependency leak.

- [ ] **Step 4: Commit**

```bash
git add src/live_capture.py
git commit -m "feat(live-capture): add CLI entry point (capture / --match modes)

main() suit la convention du repo (arg() manuel, pas d'argparse, cf.
aggregate_games.py). Import de riotlib différé au seul chemin --match :
le mode capture par défaut reste 100% stdlib."
```

- [ ] **Step 5: Real-world verification (you, not the implementer)**

Once this plan is fully executed, the actual end-to-end validation needs a real League
game and is yours to run, per the spec's success criteria:
1. On your main PC: `python3 src/live_capture.py`, launch a real game, let it play out.
   Check a `.jsonl` + `_meta.json` pair landed next to the script.
2. Move that pair into `data/01_raw_live/pending/`.
3. Run `python3 src/live_capture.py --match "<YourRiotId>#<tag>" <platform>` (or rely on
   `.env`'s `RIOT_ID`/`RIOT_REGION` if set) — confirm the pair is renamed into
   `data/01_raw_live/matched/<realMatchId>_live.jsonl`.
4. (Optional, proves the multi-PC path) Copy just `src/live_capture.py` to a secondary
   machine with only Python installed, run it there during a game, bring the two output
   files back, and repeat steps 2–3.

---

## Plan Self-Review Notes

- **Spec coverage:** capture loop ✅ (Task 2), storage layout `pending/`/`matched/` ✅
  (Tasks 2–3, created at runtime), matching by champion+start+duration with tolerance ✅
  (Task 1), CLI ✅ (Task 4), zero-dependency multi-PC constraint ✅ (enforced by lazy
  `riotlib` import + stdlib-only capture path, verified manually in Task 4 Step 3), error
  handling for corrupt/missing files ✅ (Task 3 tests). Feature extraction and coaching
  wiring are explicitly out of scope per the spec and are not present in any task.
- **Type consistency:** `capture()` returns `tuple[Path, Path] | None` (Task 2) and is
  never consumed by another task's code — only by the manual verification step and the
  test. `find_matching_game` (Task 1) is consumed as-is by `match_pending_captures`
  (Task 3) with the same signature. `rl_module` parameter name is consistent across
  `_candidate_games` and `match_pending_captures` (Task 3) and matches how `main()`
  (Task 4) passes the real `riotlib` module.
- **Testing beyond the spec's original assumption:** the spec assumed the capture loop
  was unverifiable outside a real game ("pas de test automatisé possible"). Task 2 adds
  an automated test using a local mock HTTP server standing in for the Live Client
  endpoint, which covers the deterministic state machine (poll → write → detect
  end-of-game) without needing League running. This is a pure quality improvement — it
  doesn't change any scope or user-facing behavior from the spec.
