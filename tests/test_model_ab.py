from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "04_coaching" / "model_ab.py"
SPEC = importlib.util.spec_from_file_location("model_ab", MODULE_PATH)
AB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AB)


class Review(dict):
    def model_dump(self):
        return dict(self)


def _review(payload, model):
    deaths = payload["journal"]["deaths"]
    zone = deaths[0]["zone"] if deaths else "BOT"
    confidence = 0.3 if not deaths else 0.8
    gold = max([d.get("unspent_gold", 0) for d in deaths] or [0])
    if model == "weak":
        zone, confidence, gold = "BOT", 0.8, 1225
    return Review({
        "strengths": [],
        "mistakes": [{"point": "p", "cause": f"position {zone}",
                      "evidence": f"mort à 15:13 en {zone}, {gold} g"}],
        "next_focus": "f", "confidence": confidence,
    }), {"latency_ms": 100 if model == "strong" else 50,
         "total_tokens": 10, "schema_retries": 0}


def test_ab_regenerates_baseline_per_model_and_recommends_quality(tmp_path):
    root = tmp_path / "07_coaching"
    (root / "p").mkdir(parents=True)
    source = {
        "ts": "t", "kind": "game", "match_id": "m1",
        "payload": {
            "meta": {"kda": {"deaths": 1}},
            "journal": {"deaths": [{"clock": "15:13", "zone": "BOT",
                                      "unspent_gold": 1225}],
                        "recalls": []},
        },
        "review": {},
    }
    (root / "p" / "reviews.jsonl").write_text(json.dumps(source) + "\n")
    report = AB.run("p", ["weak", "strong"], n=1, root=root, generate=_review)
    assert report["protocol"]["calls_planned"] == 8
    assert report["recommended_model"] == "strong"
    by_model = {row["model"]: row for row in report["models"]}
    assert by_model["strong"]["sensitivity"] == 1.0
    assert by_model["weak"]["sensitivity"] < 1.0
    assert by_model["strong"]["latency_ms_total"] == 400


def test_ab_report_persists_outside_reviews(tmp_path):
    path = AB.persist("p", {"player": "p"}, root=tmp_path)
    assert path == tmp_path / "p" / "eval" / "model_ab.json"
    assert not (tmp_path / "p" / "reviews.jsonl").exists()
