"""Parité de la métrique d'éval entre la CLI Python et le Worker TypeScript.

Le taux d'utilité est calculé deux fois : par `feedback.py summary` (local, sert
la page CV) et par `web/cf/src/evaluation.ts` (le site, qui doit compter aussi
les annotations laissées depuis le web). Deux runtimes, donc deux implémentations,
mais un seuil déclaré à deux endroits dérive en silence : publier « cible 70 % »
d'un côté et mesurer 60 % de l'autre invaliderait le chiffre affiché.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "core"))
sys.path.insert(0, str(ROOT / "src" / "04_coaching"))

import feedback as fb  # noqa: E402

EVAL_TS = ROOT / "web" / "cf" / "src" / "evaluation.ts"


def _ts_const(name: str) -> float:
    m = re.search(rf"export const {name} = ([\d.]+);", EVAL_TS.read_text())
    assert m, f"{name} introuvable dans {EVAL_TS}"
    return float(m.group(1))


def test_objective_thresholds_match():
    assert _ts_const("TARGET_N") == fb._OBJECTIVE_N
    assert _ts_const("TARGET_RATE") == fb._OBJECTIVE_RATE


def test_both_report_the_same_shape():
    """Le frontend lit une seule forme de rapport, quelle que soit sa source."""
    ts = EVAL_TS.read_text()
    for field in ("n_game_reviews", "objective", "target_met", "global_rate",
                  "by_kind", "top_tags", "n_reviews_annotated", "n_items"):
        assert f"{field}" in ts, field
    report = fb.eval_report("nobody", root=ROOT / "tests" / "nonexistent")
    assert set(report) >= {"n_game_reviews", "objective", "target_met",
                           "global_rate", "by_kind", "top_tags",
                           "n_reviews_annotated", "n_items"}
