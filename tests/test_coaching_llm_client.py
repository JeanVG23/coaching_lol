import json

import pytest

import llm_client as LC


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}
    def json(self):
        return self._payload
    def raise_for_status(self):
        pass


def test_generate_json_posts_and_parses(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        captured["headers"] = headers
        content = '{"ok": true}'
        return _Resp(200, {"message": {"content": content}})

    monkeypatch.setattr(LC.rl, "load_env", lambda: {"OLLAMA_API_KEY": "k123"})
    monkeypatch.setattr(LC.requests, "post", fake_post)
    out = LC.generate_json("deepseek-v4-pro", "sys", "usr", {"type": "object"})
    assert out == {"ok": True}
    assert captured["url"] == LC.OLLAMA_URL
    assert captured["headers"]["Authorization"] == "Bearer k123"
    assert captured["body"]["format"] == {"type": "object"}
    assert captured["body"]["stream"] is False
    assert captured["body"]["model"] == "deepseek-v4-pro"


def test_generate_json_missing_key_raises(monkeypatch):
    monkeypatch.setattr(LC.rl, "load_env", lambda: {})
    with pytest.raises(LC.LLMError):
        LC.generate_json("m", "s", "u", {})


def test_generate_json_401_raises_clear(monkeypatch):
    monkeypatch.setattr(LC.rl, "load_env", lambda: {"OLLAMA_API_KEY": "k"})
    monkeypatch.setattr(LC.requests, "post", lambda *a, **k: _Resp(401))
    with pytest.raises(LC.LLMError) as e:
        LC.generate_json("m", "s", "u", {})
    assert "OLLAMA_API_KEY" in str(e.value)


def test_generate_json_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(LC.rl, "load_env", lambda: {"OLLAMA_API_KEY": "k"})
    monkeypatch.setattr(LC.time, "sleep", lambda _s: None)   # pas d'attente réelle
    seq = [_Resp(429), _Resp(200, {"message": {"content": '{"ok": true}'}})]

    def fake_post(*a, **k):
        return seq.pop(0)

    monkeypatch.setattr(LC.requests, "post", fake_post)
    out = LC.generate_json("m", "s", "u", {})
    assert out == {"ok": True}            # retry a réussi sur 2e appel


def test_generate_json_timeout_exhausts_attempts(monkeypatch):
    monkeypatch.setattr(LC.rl, "load_env", lambda: {"OLLAMA_API_KEY": "k"})
    monkeypatch.setattr(LC.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise LC.requests.exceptions.Timeout("boom")

    monkeypatch.setattr(LC.requests, "post", boom)
    with pytest.raises(LC.LLMError):
        LC.generate_json("m", "s", "u", {})
    assert calls["n"] == LC._MAX_ATTEMPTS   # a épuisé les retries


def test_generate_json_4xx_other_raises_immediately(monkeypatch):
    monkeypatch.setattr(LC.rl, "load_env", lambda: {"OLLAMA_API_KEY": "k"})
    monkeypatch.setattr(LC.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        return _Resp(403)                   # 4xx non-401/non-429 -> échec sec

    monkeypatch.setattr(LC.requests, "post", fake_post)
    with pytest.raises(LC.LLMError):
        LC.generate_json("m", "s", "u", {})
    assert calls["n"] == 1                 # pas de retry sur 4xx dur