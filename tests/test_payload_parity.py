"""Contrat golden du payload agrégé Python vers TypeScript."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for module_path in (
    ROOT / "src" / "core",
    ROOT / "src" / "04_coaching",
    ROOT / "src" / "reporting",
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import payload as payload_mod  # noqa: E402
import riotlib as rl  # noqa: E402

GOLDEN_DIR = ROOT / "web" / "cf" / "test" / "golden"
PLAYER, TARGET = "spadzze", "challenger"
CASES = [
    ("all", "overall"),
    ("all", "win"),
    ("all", "loss"),
    ("adc", "overall"),
    ("adc", "win"),
    ("adc", "loss"),
    ("zeri", "loss"),
]


def _aggregate(kind: str, name: str, scope: str):
    path = rl.GOLD_DIR / kind / name / scope / "aggregate.json"
    return json.loads(path.read_text()) if path.exists() else None


@pytest.mark.parametrize(("scope", "outcome"), CASES)
def test_payload_parity_golden(scope, outcome, demo_data):
    """Contrat entre les deux runtimes, mesuré sur le jeu de démo.

    Les goldens dérivaient auparavant des agrégats locaux, absents du dépôt : en
    CI le test se sautait, donc il ne protégeait que le poste qui collecte. Et il
    cassait à chaque densification, pour une divergence de DONNÉES et non de code.
    Adossé aux fixtures, il ne bouge que si `payload.py` bouge.
    """
    me = _aggregate("personal", PLAYER, scope)
    ref = _aggregate("referentiel", TARGET, scope)
    assert me is not None and ref is not None, "fixtures démo non construites"
    expected_file = GOLDEN_DIR / f"payload_{scope}_{outcome}.json"
    built = payload_mod.build(PLAYER, scope, TARGET, outcome)
    if os.environ.get("GOLDEN_REGEN") == "1" or not expected_file.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        expected_file.write_text(json.dumps({
            "args": {
                "player": PLAYER,
                "scope": scope,
                "target": TARGET,
                "outcome": outcome,
            },
            "me": me,
            "ref": ref,
            "expected": built,
        }, ensure_ascii=False, indent=1) + "\n")
        return
    golden = json.loads(expected_file.read_text())
    assert built == golden["expected"], (
        f"payload.py a divergé de {expected_file.name} — "
        "régénérer avec GOLDEN_REGEN=1 si l'évolution est voulue"
    )
