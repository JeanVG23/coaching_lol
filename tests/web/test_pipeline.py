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

    def fake_match_ids(self, puuid, count, queue):
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
         patch("riotlib.RiotClient.entries_by_puuid", lambda self, puuid: []), \
         patch("pipeline.rl.SILVER_DIR", tmp_path), \
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


def test_fetch_games_caches_current_rank(tmp_path):
    account = {"slug": "spadzze", "riot_id": "Spadzze#euw", "region": "euw1"}
    fake_match = {"info": {"participants": [{"puuid": "p1", "championName": "Zeri"}]}}
    fake_timeline = {"info": {"frames": []}}

    def fake_entries(self, puuid):
        return [
            {"queueType": "RANKED_FLEX_SR", "tier": "GOLD", "rank": "I",
             "leaguePoints": 10, "wins": 1, "losses": 1},
            {"queueType": "RANKED_SOLO_5x5", "tier": "DIAMOND", "rank": "II",
             "leaguePoints": 42, "wins": 120, "losses": 100},
        ]

    with patch("riotlib.RiotClient.puuid_from_riot_id", lambda self, g, t: "p1"), \
         patch("riotlib.RiotClient.match_ids", lambda self, puuid, count, queue: ["m1"]), \
         patch("riotlib.RiotClient.entries_by_puuid", fake_entries), \
         patch("riotlib.get_match_timeline", lambda client, mid: (fake_match, fake_timeline)), \
         patch("riotlib.extract_game",
               lambda m, t, p: {"match_id": "m1", "puuid": p, "rank": "emerald",
                                "patch": "16.13", "champion": "Zeri", "role": "BOTTOM",
                                "win": True, "queue": 420, "lane": "BOT", "comp": {},
                                "deaths": 3, "kills": 8, "assists": 5,
                                "support_deaths_early": 0, "plates_diff_early": 1,
                                "frames_in_base_early": 2, "avg_dragon_prox": 0.4,
                                "position": {}}), \
         patch("riotlib.merge_jsonl", lambda path, new: new), \
         patch("riotlib.write_gold"), \
         patch("pipeline.rl.SILVER_DIR", tmp_path), \
         patch("pipeline.settings.riot_api_key", lambda: "k"):
        pipeline.fetch_games(account, n=1)

    rank_path = tmp_path / "personal" / "spadzze" / "rank.json"
    assert rank_path.exists()
    data = json.loads(rank_path.read_text())
    assert data["tier"] == "DIAMOND"
    assert data["division"] == "II"
    assert data["league_points"] == 42
    assert data["wins"] == 120 and data["losses"] == 100
    assert "fetched_at" in data


def test_fetch_games_writes_unranked_when_no_solo_entry(tmp_path):
    account = {"slug": "spadzze", "riot_id": "Spadzze#euw", "region": "euw1"}
    fake_match = {"info": {"participants": [{"puuid": "p1", "championName": "Zeri"}]}}
    fake_timeline = {"info": {"frames": []}}

    with patch("riotlib.RiotClient.puuid_from_riot_id", lambda self, g, t: "p1"), \
         patch("riotlib.RiotClient.match_ids", lambda self, puuid, count, queue: ["m1"]), \
         patch("riotlib.RiotClient.entries_by_puuid", lambda self, puuid: []), \
         patch("riotlib.get_match_timeline", lambda client, mid: (fake_match, fake_timeline)), \
         patch("riotlib.extract_game",
               lambda m, t, p: {"match_id": "m1", "puuid": p, "rank": "emerald",
                                "patch": "16.13", "champion": "Zeri", "role": "BOTTOM",
                                "win": True, "queue": 420, "lane": "BOT", "comp": {},
                                "deaths": 3, "kills": 8, "assists": 5,
                                "support_deaths_early": 0, "plates_diff_early": 1,
                                "frames_in_base_early": 2, "avg_dragon_prox": 0.4,
                                "position": {}}), \
         patch("riotlib.merge_jsonl", lambda path, new: new), \
         patch("riotlib.write_gold"), \
         patch("pipeline.rl.SILVER_DIR", tmp_path), \
         patch("pipeline.settings.riot_api_key", lambda: "k"):
        pipeline.fetch_games(account, n=1)

    data = json.loads((tmp_path / "personal" / "spadzze" / "rank.json").read_text())
    assert data["tier"] is None
    assert "fetched_at" in data


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