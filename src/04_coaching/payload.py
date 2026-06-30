"""Gold (perso + référentiel) -> payload de coaching compact et déterministe.

PUR (sauf lecture des aggregate.json). Les features ont déjà conclu : on
sélectionne ici les signaux saillants (flag `notable`) selon des seuils fixes.
N'expose QUE des métriques asymétrie-safe : positioning ⊂ COACHING_SAFE, et
les features de profondeur sont `descriptive_only` (jamais une erreur).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # accès src/
import riotlib as rl
import positioning
import compare

LANE_SIGNALS = ["gd10", "gd14", "gd20", "csd10", "csd14"]
LANE_LABELS = {"gd10": "gold diff @10", "gd14": "gold diff @14", "gd20": "gold diff @20",
               "csd10": "cs diff @10", "csd14": "cs diff @14"}

# clé -> (label, unit, notable_threshold | None, descriptive_only)
# Profondeur (avg/max_map_depth, frac_overextended) : threshold=None + descriptive_only=True
# -> jamais `notable`, observable mais jamais prescrit.
POS_META = {
    "frac_own_lane_early": ("% lane (early)", "pct", 0.08, False),
    "frac_river_early":    ("% river (early)", "pct", 0.08, False),
    "frac_roam_mid":       ("% roam (mid)", "pct", 0.08, False),
    "frac_enemy_half":     ("% moitié ennemie", "pct", 0.08, False),
    "frac_base":           ("% en base", "pct", 0.08, False),
    "frac_overextended":   ("% over-extended", "pct", None, True),
    "avg_map_depth":       ("profondeur moy.", "u", None, True),
    "max_map_depth":       ("profondeur max", "u", None, True),
    "avg_dist_to_ally":    ("isolement (allié)", "u", 200.0, False),
    "gold_dead_time":      ("temps mort (s)", "s", 20.0, False),
    "wards_placed":        ("wards posées", "ward", 2.0, False),
    "wards_placed_early":  ("wards early", "ward", 1.0, False),
    "control_wards_placed": ("control wards", "ward", 1.0, False),
    "wards_killed":        ("wards détruites", "ward", 2.0, False),
}
# Garde-fou asymétrie : la table doit couvrir EXACTEMENT les features safe, ni plus ni moins.
assert set(POS_META) == positioning.COACHING_SAFE, \
    "POS_META doit refléter exactement positioning.COACHING_SAFE"

LOW_SAMPLE_THRESHOLD = 30


def _lane_signals(mf: dict, rf: dict) -> list[dict]:
    out = []
    lane_me, lane_rf = mf.get("lane", {}), rf.get("lane", {})
    for key in LANE_SIGNALS:
        you, ref = lane_me.get(key), lane_rf.get(key)
        if you is None or ref is None:
            continue
        delta = you - ref
        unit = "cs" if key.startswith("cs") else "g"
        notable = abs(delta) >= 2 if unit == "cs" else abs(delta) > 150
        out.append({"group": "lane", "key": key, "label": LANE_LABELS[key],
                    "you": you, "ref": ref, "delta": delta, "unit": unit,
                    "notable": notable})
    return out


def _pos_signals(mf: dict, rf: dict) -> list[dict]:
    out = []
    pos_me, pos_rf = mf.get("positioning", {}), rf.get("positioning", {})
    for key in sorted(POS_META):
        you, ref = pos_me.get(key), pos_rf.get(key)
        if you is None or ref is None:
            continue
        label, unit, thr, descriptive = POS_META[key]
        delta = round(you - ref, 4)
        notable = (thr is not None) and abs(delta) >= thr
        sig = {"group": "positioning", "key": key, "label": label,
               "you": you, "ref": ref, "delta": delta, "unit": unit,
               "notable": notable}
        if descriptive:
            sig["descriptive_only"] = True
        out.append(sig)
    return out


def _zone_phase_signals(mf: dict, rf: dict, top: int = 5) -> list[dict]:
    me_zp, rf_zp = mf.get("by_zone_phase", {}), rf.get("by_zone_phase", {})
    rows = []
    for key in set(me_zp) | set(rf_zp):
        you, ref = me_zp.get(key, 0.0), rf_zp.get(key, 0.0)
        delta = round(you - ref, 4)
        rows.append({"group": "deaths_zone_phase", "key": key,
                     "label": f"morts {key}", "you": you, "ref": ref,
                     "delta": delta, "unit": "pct", "notable": delta >= 0.08})
    rows.sort(key=lambda s: s["delta"], reverse=True)   # où tu sur-meurs d'abord
    return rows[:top]


def _gold_state_signals(mf: dict, rf: dict) -> list[dict]:
    me_gs, rf_gs = mf.get("death_gold_state", {}), rf.get("death_gold_state", {})
    labels = {"ahead": "morts en avance", "even": "morts à égalité", "behind": "morts en retard"}
    out = []
    for key, label in labels.items():
        you, ref = me_gs.get(key), rf_gs.get(key)
        if you is None or ref is None:
            continue
        delta = round(you - ref, 4)
        out.append({"group": "death_gold_state", "key": key, "label": label,
                    "you": you, "ref": ref, "delta": delta, "unit": "pct",
                    "notable": abs(delta) >= 0.10})
    return out


def _load(gold_dir: Path, *parts) -> dict:
    path = gold_dir.joinpath(*parts) / "aggregate.json"
    if not path.exists():
        raise FileNotFoundError(f"gold manquant : {path}")
    return json.loads(path.read_text())


def build(player: str, scope: str = "adc", target: str = "challenger",
          outcome: str = "loss", gold_dir=None) -> dict:
    gold_dir = Path(gold_dir) if gold_dir is not None else rl.GOLD_DIR
    me = _load(gold_dir, "personal", player, scope)
    ref = _load(gold_dir, "referentiel", target, scope)
    mf, rf = me[outcome], ref[outcome]

    meta = {
        "player": player, "scope": scope, "target": target,
        "outcome_focus": outcome, "patch": me.get("patch", "?"),
        "n_games_me": me["n_games"], "n_games_ref": ref["n_games"],
        "winrate_me": me["winrate"],
        "low_sample": me["n_games"] < LOW_SAMPLE_THRESHOLD,
        "deaths_per_game": {oc: {"you": me[oc]["deaths_per_game"],
                                 "ref": ref[oc]["deaths_per_game"]}
                            for oc in ("overall", "win", "loss")},
    }
    signals = (_lane_signals(mf, rf) + _pos_signals(mf, rf)
               + _zone_phase_signals(mf, rf) + _gold_state_signals(mf, rf))

    context = {}
    for axis in ("lane_pattern", "gank_exposure"):
        cb = compare.context_benchmark(me, ref, axis, outcome)
        if cb:
            context[axis] = cb

    return {"meta": meta, "signals": signals, "context": context}
