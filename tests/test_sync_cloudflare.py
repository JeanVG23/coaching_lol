"""Tests du sync KV : dossiers temporaires, fake KV, zéro réseau."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for module_path in (
    ROOT / "src" / "core",
    ROOT / "src" / "collection",
    ROOT / "web" / "backend",
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import ml_rank  # noqa: E402
import riotlib as rl  # noqa: E402
import sync_cloudflare as sc  # noqa: E402


class FakeKV(sc.KV):
    def __init__(self):
        super().__init__("account", "namespace", "token")
        self.store: dict[str, str] = {}

    def put(self, key: str, value: str) -> None:
        self.puts.append(key)
        self.store[key] = value

    def get(self, key: str) -> str | None:
        return self.store.get(key)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "DATA", tmp_path)
    return tmp_path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_sync_account_pushes_keys(data_root, monkeypatch):
    _write(
        data_root / "02_silver" / "personal" / "p" / "games.jsonl",
        json.dumps({"match_id": "EUW1_10", "role": "BOTTOM"}) + "\n",
    )
    _write(
        data_root / "02_silver" / "personal" / "p" / "rank.json",
        json.dumps({"tier": "MASTER"}),
    )
    _write(
        data_root / "03_gold" / "personal" / "p" / "all" / "aggregate.json",
        json.dumps({"n_games": 10}),
    )
    _write(
        data_root / "03_gold" / "personal" / "p" / "adc" / "aggregate.json",
        json.dumps({"n_games": 8}),
    )
    _write(data_root / "06_shap" / "p_drivers.json", json.dumps([{"feature": "gd10"}]))
    monkeypatch.setattr(
        ml_rank,
        "predict_rank",
        lambda games: {"predicted_rank": "master", "proba": 0.6, "n_games_used": 20},
    )

    kv = FakeKV()
    sc.sync_account(kv, "p")

    assert kv.store["silver:p:games"] == (
        data_root / "02_silver" / "personal" / "p" / "games.jsonl"
    ).read_text()
    assert json.loads(kv.store["silver:p:rank"]) == {"tier": "MASTER"}
    assert json.loads(kv.store["gold:p:all"]) == {"n_games": 10}
    assert json.loads(kv.store["gold:p:adc"]) == {"n_games": 8}
    assert json.loads(kv.store["pred:p"])["predicted_rank"] == "master"
    assert json.loads(kv.store["shap:p:drivers"]) == [{"feature": "gd10"}]
    assert not [key for key in kv.store if key.startswith("coaching:")]


def test_sync_account_slices_last_20_games(data_root, monkeypatch):
    seen: list[int] = []

    def fake_predict(games):
        seen.append(len(games))
        return None

    monkeypatch.setattr(ml_rank, "predict_rank", fake_predict)
    games = data_root / "02_silver" / "personal" / "p" / "games.jsonl"
    games.parent.mkdir(parents=True, exist_ok=True)
    games.write_text("\n".join(json.dumps({"match_id": f"EUW1_{i}"}) for i in range(30)) + "\n")

    kv = FakeKV()
    sc.sync_account(kv, "p")
    assert seen == [20]
    assert "pred:p" not in kv.store


def test_seed_reviews_only_when_kv_key_absent(data_root, monkeypatch):
    monkeypatch.setattr(ml_rank, "predict_rank", lambda games: None)
    _write(
        data_root / "07_coaching" / "p" / "reviews.jsonl",
        json.dumps({"ts": "t1"}) + "\n",
    )
    kv = FakeKV()
    sc.sync_account(kv, "p", seed_reviews=True)
    assert kv.store.get("coaching:p:reviews") == (
        data_root / "07_coaching" / "p" / "reviews.jsonl"
    ).read_text()

    kv2 = FakeKV()
    kv2.store["coaching:p:reviews"] = json.dumps({"ts": "web"}) + "\n"
    sc.sync_account(kv2, "p", seed_reviews=True)
    assert json.loads(kv2.store["coaching:p:reviews"])["ts"] == "web"


def test_sync_referential(data_root):
    _write(
        data_root / "03_gold" / "referentiel" / "challenger" / "adc" / "aggregate.json",
        json.dumps({"n_games": 100}),
    )
    kv = FakeKV()
    sc.sync_referential(kv)
    assert json.loads(kv.store["ref:challenger:adc"]) == {"n_games": 100}


def test_dry_kv_never_calls_network(monkeypatch):
    monkeypatch.setattr(sc.requests, "put", lambda *args, **kwargs: pytest.fail("network PUT"))
    monkeypatch.setattr(sc.requests, "get", lambda *args, **kwargs: pytest.fail("network GET"))
    kv = sc.DryKV()
    kv.put("key", "value")
    assert kv.get("key") is None
    assert kv.puts == ["key"]
