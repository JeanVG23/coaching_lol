"""Incrément 2 — agrégation gold des features positionnelles (COACHING_SAFE only)."""
import positioning as P
import riotlib as rl


def test_fmedian_preserves_fractions():
    # _median arrondit à l'entier (écraserait une fraction à 0/1) ; _fmedian non.
    assert rl._median([0.0, 1.0]) == 0          # round(0.5) -> 0
    assert rl._fmedian([0.0, 1.0]) == 0.5       # médiane flottante préservée
    assert rl._fmedian([]) is None
    assert rl._fmedian([0.3333333]) == 0.3333   # arrondi 4 décimales


def test_facet_positioning_only_coaching_safe():
    # position contient TOUTES les features (safe + proxys ML_ONLY) ; l'agrégat
    # ne doit retenir QUE les COACHING_SAFE (garde-fou asymétrie côté gold).
    pos = {k: 1.0 for k in P.ALL_FEATURES}
    subset = [
        {"win": True, "deaths": [], "lane": {}, "position": dict(pos)},
        {"win": False, "deaths": [], "lane": {}, "position": {k: 3.0 for k in P.ALL_FEATURES}},
    ]
    facet = rl._facet(subset)
    assert "positioning" in facet
    assert set(facet["positioning"].keys()) == P.COACHING_SAFE
    assert set(facet["positioning"]).isdisjoint(P.ML_ONLY)
    # médiane de [1.0, 3.0] = 2.0 (flottant, pas arrondi entier)
    assert facet["positioning"]["frac_base"] == 2.0


def test_facet_positioning_tolerates_missing_position():
    # vieux record silver sans sous-objet "position" -> None partout, pas de crash.
    facet = rl._facet([{"win": True, "deaths": [], "lane": {}}])
    assert set(facet["positioning"].keys()) == P.COACHING_SAFE
    assert all(v is None for v in facet["positioning"].values())
