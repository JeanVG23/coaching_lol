"""Tests de câblage du frontend statique (servi par FastAPI).

Pas de test d'interactivité Alpine (pas de Playwright) — on verrouille le câblage :
assets vendored servis, index.html référence les assets et porte les hooks clés,
app.js définit le routeur et les helpers. L'interactivité est vérifiée manuellement.
"""
from fastapi.testclient import TestClient

import main as main_mod


def _client():
    return TestClient(main_mod.app)


def test_index_served_at_root():
    r = _client().get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert 'x-data="app()"' in body
    assert '/static/vendor/alpine.min.js' in body
    assert '/static/vendor/chart.umd.min.js' in body
    assert '/static/style.css' in body
    assert '/static/app.js' in body


def test_spa_catch_all_serves_index():
    c = _client()
    for path in ("/c/spadzze", "/readme"):
        r = c.get(path)
        assert r.status_code == 200
        assert 'x-data="app()"' in r.text


def test_assets_served():
    c = _client()
    for path, ct in [
        ("/static/style.css", "text/css"),
        ("/static/app.js", "javascript"),
        ("/static/vendor/alpine.min.js", "javascript"),
        ("/static/vendor/chart.umd.min.js", "javascript"),
    ]:
        r = c.get(path)
        assert r.status_code == 200, path
        assert ct in r.headers["content-type"], path
        assert len(r.content) > 1000, path


def test_style_css_has_tokens():
    css = _client().get("/static/style.css").text
    for token in ("--bg:#0e1116", "--panel:#16181d", "--gold:#c8aa6e",
                  "--win:#3fb950", "--loss:#f85149", "tabular-nums"):
        assert token in css, token


def test_app_js_has_router_and_helpers():
    js = _client().get("/static/app.js").text
    assert "function app()" in js
    assert "function api(" in js
    assert "function pollJob(" in js
    assert "location.pathname" in js


def test_home_page_wired():
    body = _client().get("/").text
    js = _client().get("/static/app.js").text
    assert "function homePage()" in js
    assert "/api/accounts" in js
    # marqueurs propres au template home (absents de la nav switcher F1)
    assert 'class="accounts-grid"' in body
    assert 'class="account-card"' in body


def test_account_page_history_wired():
    body = _client().get("/c/spadzze").text
    js = _client().get("/static/app.js").text
    assert "function accountPage(" in js
    assert "/api/c/" in js and "/games" in js
    assert "/api/fetch" in js
    assert "/api/jobs/" in js
    assert 'class="game-row"' in body or "game-row" in body
    assert "job-banner" in body


def test_coaching_tab_wired():
    body = _client().get("/c/spadzze").text
    js = _client().get("/static/app.js").text
    assert "/api/coach" in js
    assert "/api/c/" in js and "/reviews" in js
    assert "/api/feedback" in js
    assert "NEG_TAGS" in js
    assert "insight-card" in body or "evidence-chip" in body


def test_feedback_sends_full_map():
    """Régression F4 : submitFb doit envoyer la fbMap complète (le backend
    écrase la ligne par ts — un envoi par insight perdait les notations précédentes)."""
    js = _client().get("/static/app.js").text
    assert "Object.entries(newMap)" in js
    # l'ancien pattern mono-insight ne doit plus être présent
    assert "responses = { [key]" not in js