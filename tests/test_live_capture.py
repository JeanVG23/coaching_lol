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


class _FlakyFakeClient:
    def match_ids(self, puuid, count=15):
        return ["EUW1_FAILED", "EUW1_111"]


class _FlakyFakeRiotLib:
    """Simule un fetch qui échoue pour une game (comme le vrai get_match_timeline
    -> None sur erreur réseau/404) mêlé à une game valide."""

    def get_match_timeline(self, client, match_id):
        if match_id == "EUW1_FAILED":
            return None
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


def test_match_pending_captures_skips_candidate_with_failed_fetch(tmp_path):
    pending = tmp_path / "pending"
    matched = tmp_path / "matched"
    _write_capture(pending, "20260702T180000Z_Zeri", "Zeri",
                    _iso(2026, 7, 2, 18, 0, 2), _iso(2026, 7, 2, 18, 30, 0))

    LC.match_pending_captures(pending, matched, _FlakyFakeClient(), _FlakyFakeRiotLib(),
                               "PUUID1")

    assert (matched / "EUW1_111_live.jsonl").exists()
