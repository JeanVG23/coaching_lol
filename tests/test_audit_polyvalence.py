"""Tests pour l'audit polyvalence : on injecte un dict positions_by_alias
simulé, pas besoin de mocker requests/CDragon.
"""
from __future__ import annotations

import sys
from pathlib import Path

# conftest fait déjà le path setup. On importe direct.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "pipeline_ops"))
import audit_polyvalence as ap


TR = {
    # ADC classique : 1 position, 1 axe OK
    "Caitlyn": {"lane_pattern": "poke", "power_curve": "early"},
    # ADC + support : polyvalent, axes cohérents sur les 2 positions
    "Karma": {"lane_pattern": "poke", "roam": "med"},
    # ADC avec lane_pattern: scaling mais SEUL build officiel = MIDDLE
    # => axe "scaling" n'est attendu que sur BOTTOM, alerte axe_orphelin
    "MidOnly": {"lane_pattern": "scaling"},
    # Jungler curé "ganking" sans build JUNGLE dans CDragon => axe_orphelin
    "NotAJungler": {"playstyle": "ganking", "gank_threat": "high"},
    # Champion CDragon ne le connaît pas
    "Zaahen": {},
}


# Dict {alias: {positions}} tel que renvoyé par parse_positions (lowercase).
POS = {
    "caitlyn": {"BOTTOM"},
    "karma": {"MIDDLE", "UTILITY"},
    "midonly": {"MIDDLE"},
    "notajungler": {"TOP"},
    # Zaahen absent => alias_introuvable
}


def test_audit_summary_counts():
    rep = ap.audit(TR, POS)
    s = rep["summary"]
    assert s["n_traits"] == 5
    assert s["n_in_cdragon"] == 4  # Zaahen absent
    assert s["n_introuvables"] == 1
    assert s["n_polyvalents"] == 1  # Karma
    # 3 entrées : MidOnly (lane_pattern) + NotAJungler (playstyle) + (gank_threat)
    assert s["n_axe_orphelin"] == 3


def test_polyvalence_detected():
    rep = ap.audit(TR, POS)
    polyvalents = {r["champion"] for r in rep["polyvalence"]}
    assert polyvalents == {"Karma"}


def test_axe_orphelin_for_mid_only_scaling():
    rep = ap.audit(TR, POS)
    orphelins = {r["champion"] for r in rep["axe_orphelin"]}
    assert "MidOnly" in orphelins
    # Le message doit mentionner l'axe et l'absence de position porteuse
    msg = next(r["detail"] for r in rep["axe_orphelin"] if r["champion"] == "MidOnly")
    assert "lane_pattern=scaling" in msg
    assert "MIDDLE" in msg


def test_axe_orphelin_for_ganker_without_jungle_build():
    rep = ap.audit(TR, POS)
    orphelins = {r["champion"] for r in rep["axe_orphelin"]}
    assert "NotAJungler" in orphelins
    # axe playstyle=ganking n'a du sens qu'en JUNGLE
    msg = next(r["detail"] for r in rep["axe_orphelin"] if r["champion"] == "NotAJungler")
    assert "playstyle=ganking" in msg


def test_karma_axes_consistent_across_positions():
    """Karma a lane_pattern: poke (cohérent BOTTOM+UTILITY) et roam: med
    (cohérent MIDDLE). Donc 0 axe orphelin pour elle, juste la polyvalence."""
    rep = ap.audit(TR, POS)
    orphelins = {r["champion"] for r in rep["axe_orphelin"]}
    assert "Karma" not in orphelins


def test_alias_introuvable_listed():
    rep = ap.audit(TR, POS)
    assert rep["alias_introuvable"] == ["Zaahen"]


def test_no_alert_when_clean():
    """Caitlyn pure (BOTTOM seul, axes compatibles) ne doit générer aucun signal."""
    rep = ap.audit({"Caitlyn": {"lane_pattern": "poke"}}, {"caitlyn": {"BOTTOM"}})
    assert rep["summary"]["n_polyvalents"] == 0
    assert rep["summary"]["n_axe_orphelin"] == 0
    assert rep["summary"]["n_introuvables"] == 0


def test_parse_positions_filters_aram():
    """parse_positions(map_id=11) doit exclure les recos ARAM (mapId=12)."""
    fake_recs = [
        {
            "championId": 100,
            "runeRecommendations": [
                {"mapId": 12, "position": "NONE"},
                {"mapId": 11, "position": "MIDDLE"},
                {"mapId": 11, "position": "BOTTOM"},
            ],
        }
    ]
    id_to_alias = {100: "TestChamp"}
    out = ap.parse_positions(fake_recs, id_to_alias, map_id=11)
    assert out == {"testchamp": {"MIDDLE", "BOTTOM"}}


def test_parse_positions_skips_unknown_champion_id():
    """Les recos dont championId n'est pas dans id_to_alias sont ignorées."""
    fake_recs = [{"championId": 999, "runeRecommendations": [
        {"mapId": 11, "position": "MIDDLE"}]}]
    out = ap.parse_positions(fake_recs, {1: "Annie"}, map_id=11)
    assert out == {}


def test_render_text_contains_summary():
    rep = ap.audit(TR, POS)
    text = ap.render_text(rep)
    assert "5 champions curés" in text
    assert "Karma" in text
    assert "Zaahen" in text
