"""Tests de la commande Riot -> local -> Cloudflare, sans acces reseau."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for module_path in (
    ROOT / "src" / "core",
    ROOT / "src" / "collection",
    ROOT / "src" / "04_coaching",
    ROOT / "web" / "backend",
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import refresh_cloudflare as refresh  # noqa: E402


ACCOUNTS = [
    {"slug": "one", "riot_id": "One#EUW", "region": "euw1"},
    {"slug": "two", "riot_id": "Two#EUW", "region": "euw1"},
]


@pytest.fixture(autouse=True)
def configured_cloudflare(monkeypatch):
    monkeypatch.setattr(refresh, "ensure_cloudflare_configured", lambda: None)


def test_main_refreshes_selected_account_then_syncs_without_ref(monkeypatch, capsys):
    calls: list[tuple[str, int]] = []
    sync_calls: list[list[str]] = []
    monkeypatch.setattr(refresh.sync_cloudflare, "load_accounts", lambda: ACCOUNTS)

    def fake_fetch(account, n, on_progress):
        calls.append((account["slug"], n))
        on_progress("1/7")
        return {"n_games": 42, "player": account["slug"]}

    monkeypatch.setattr(refresh.pipeline, "fetch_games", fake_fetch)
    monkeypatch.setattr(refresh.sync_cloudflare, "main", lambda args: sync_calls.append(args))

    refresh.main(["--slug", "two", "-n", "7"])

    assert calls == [("two", 7)]
    assert sync_calls == [["--slug", "two", "--skip-ref"]]
    assert "[two] 1/7" in capsys.readouterr().out


def test_main_refreshes_all_accounts_and_can_publish_referential(monkeypatch):
    calls: list[str] = []
    sync_calls: list[list[str]] = []
    monkeypatch.setattr(refresh.sync_cloudflare, "load_accounts", lambda: ACCOUNTS)

    def fake_fetch(account, n, on_progress):
        del n, on_progress
        calls.append(account["slug"])
        return {"n_games": 20, "player": account["slug"]}

    monkeypatch.setattr(refresh.pipeline, "fetch_games", fake_fetch)
    monkeypatch.setattr(refresh.sync_cloudflare, "main", lambda args: sync_calls.append(args))

    refresh.main(["--with-ref"])

    assert calls == ["one", "two"]
    assert sync_calls == [[]]


def test_unknown_slug_stops_before_fetch_and_sync(monkeypatch):
    monkeypatch.setattr(refresh.sync_cloudflare, "load_accounts", lambda: ACCOUNTS)
    monkeypatch.setattr(
        refresh.pipeline,
        "fetch_games",
        lambda *args, **kwargs: pytest.fail("fetch ne doit pas etre appele"),
    )
    monkeypatch.setattr(
        refresh.sync_cloudflare,
        "main",
        lambda *args, **kwargs: pytest.fail("sync ne doit pas etre appele"),
    )

    with pytest.raises(SystemExit, match="compte inconnu : absent"):
        refresh.main(["--slug", "absent"])


def test_fetch_failure_prevents_cloudflare_sync(monkeypatch):
    monkeypatch.setattr(refresh.sync_cloudflare, "load_accounts", lambda: ACCOUNTS[:1])

    def failed_fetch(*args, **kwargs):
        raise RuntimeError("quota Riot")

    monkeypatch.setattr(refresh.pipeline, "fetch_games", failed_fetch)
    monkeypatch.setattr(
        refresh.sync_cloudflare,
        "main",
        lambda *args, **kwargs: pytest.fail("sync ne doit pas etre appele"),
    )

    with pytest.raises(SystemExit, match="rafraichissement Riot interrompu : quota Riot"):
        refresh.main([])


@pytest.mark.parametrize("games", ["0", "101"])
def test_games_bounds_are_validated_before_network(monkeypatch, games):
    monkeypatch.setattr(
        refresh.sync_cloudflare,
        "load_accounts",
        lambda: pytest.fail("configuration ne doit pas etre chargee"),
    )

    with pytest.raises(SystemExit, match="--games doit etre compris entre 1 et 100"):
        refresh.main(["--games", games])


def test_missing_cloudflare_config_stops_before_riot(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(refresh.rl, "load_env", lambda: {})
    for key in refresh.CF_CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(SystemExit, match="CF_API_TOKEN, CF_ACCOUNT_ID, CF_NAMESPACE_ID"):
        refresh.ensure_cloudflare_configured()
