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