#!/usr/bin/env python3
"""
compare — situe une slice perso face aux référentiels par rang (livrable coaching).

Compare à ISSUE ÉGALE pour neutraliser le biais win/lose (en win on meurt moins) :
  - métriques de tête déclinées overall / win / loss ;
  - écarts de morts ZONE×PHASE pour une issue donnée (défaut: loss, le plus diagnostique).

Usage :
    python3 compare.py                                   # spadzze, adc, loss, vs tous rangs
    python3 compare.py --scope adc --outcome loss --target challenger
    python3 compare.py --scope zeri --outcome overall
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import positioning as pos
import riotlib as rl
from ranks import RANKS
from cli import arg

MIN_N = 5  # seuil d'effectif sous lequel on prévient
MIN_CONTEXT_N = 8  # sous ce seuil de games référentiel dans le bucket, on retombe sur global

# Lignes du benchmark positionnement (libellé, clé, format). "pct" pour les fractions
# 0..1, "num" pour profondeurs/distances/temps/comptes. ORDRE = lecture coaching.
POS_ROWS = [
    ("% lane (early)", "frac_own_lane_early", "pct"),
    ("% river (early)", "frac_river_early", "pct"),
    ("% roam (mid)", "frac_roam_mid", "pct"),
    ("% moitié ennemie", "frac_enemy_half", "pct"),
    ("% en base", "frac_base", "pct"),
    ("% over-extended", "frac_overextended", "pct"),
    ("profondeur moy.", "avg_map_depth", "num"),
    ("profondeur max", "max_map_depth", "num"),
    ("isolement (allié)", "avg_dist_to_ally", "num"),
    ("temps mort (s)", "gold_dead_time", "num"),
    ("wards posées", "wards_placed", "num"),
    ("wards early", "wards_placed_early", "num"),
    ("control wards", "control_wards_placed", "num"),
    ("wards détruites", "wards_killed", "num"),
]
# Garde-fou asymétrie (mécanique) : le coaching ne benchmarke QUE des features
# exactes/asymétrie-safe. Si un proxy ML_ONLY se glisse dans POS_ROWS, on crashe net.
assert {k for _, k, _ in POS_ROWS} <= pos.COACHING_SAFE, \
    "POS_ROWS contient une feature non COACHING_SAFE — violation d'asymétrie"
assert {k for _, k, _ in POS_ROWS}.isdisjoint(pos.ML_ONLY), \
    "POS_ROWS contient un proxy ML_ONLY — interdit en prescription"

# ⚠ Sens contre-intuitif des features de PROFONDEUR (avg_map_depth / max_map_depth).
# Le modèle EBM dia_chall les classe "valeur haute → diamond" (max_map_depth :
# monotonic_rho=-0.98, swing 0.495, crossover ~3775u). Autrement dit : plus tu
# t'enfonces en terrain ennemi (pic ou moyenne), plus tu ressembles à un DIAMOND
# (rang INFÉRIEUR), pas à un challenger. Ne JAMAIS prescrire « prends plus d'espace » :
# la profondeur est un marqueur de RISQUE, pas de force. Un perso qui over-extend
# MOINS que les challengers n'a donc aucune correction à faire de ce côté. Ces deux
# lignes sont descriptives, pas prescriptives.


def context_benchmark(me_agg, ref_agg, axis, outcome):
    """Compare le bucket de contexte DOMINANT côté perso au même bucket référentiel.

    Repli explicite et loggué sur 'overall' si le référentiel a < MIN_CONTEXT_N games
    dans ce bucket (échantillon trop fin pour un benchmark honnête).
    """
    # NOTE: `outcome` est réservé (interface du plan) mais pas encore utilisé :
    # by_lane_context n'est pas tranché par issue (calculé sur toutes les games).
    me_buckets = me_agg.get("by_lane_context", {}).get(axis, {})
    ref_buckets = ref_agg.get("by_lane_context", {}).get(axis, {})
    if not me_buckets:
        return None
    candidates = {b: v for b, v in me_buckets.items() if b != "unknown"} or me_buckets
    bucket = max(candidates, key=lambda b: candidates[b].get("n_games", 0))
    n_me = me_buckets[bucket].get("n_games", 0)
    gd10_me = me_buckets[bucket].get("lane", {}).get("gd10")
    ref_b = ref_buckets.get(bucket, {})
    n_ref = ref_b.get("n_games", 0)
    if n_ref < MIN_CONTEXT_N:
        glob = ref_agg.get("overall", {})
        return {"bucket": bucket, "n_me": n_me, "n_ref": n_ref,
                "gd10_me": gd10_me, "gd10_ref": glob.get("lane", {}).get("gd10"),
                "fallback": True,
                "reason": f"réf. {bucket}={n_ref}<{MIN_CONTEXT_N} games → repli global"}
    return {"bucket": bucket, "n_me": n_me, "n_ref": n_ref, "gd10_me": gd10_me,
            "gd10_ref": ref_b.get("lane", {}).get("gd10"), "fallback": False, "reason": None}


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def dpg(agg, outcome):
    """Morts par game d'une facette (le compteur d'effectif n'était lu par personne)."""
    return agg.get(outcome, {}).get("deaths_per_game", 0)


def main() -> int:
    player = arg("--player", "spadzze")
    scope = arg("--scope", "adc")
    outcome = arg("--outcome", "loss")
    target = arg("--target", "challenger")

    me = load(rl.gold_aggregate(rl.KIND_PERSONAL, player, scope))
    if not me or not me["n_games"]:
        print(f"✗ Pas de données perso pour {player}/{scope}.", file=sys.stderr)
        return 1
    refs = {r: load(rl.gold_aggregate(rl.KIND_REF, r, scope))
            for r in RANKS}
    refs = {r: a for r, a in refs.items() if a and a["n_games"]}
    if not refs:
        print(f"✗ Aucun référentiel pour scope {scope}.", file=sys.stderr)
        return 1
    cols = list(refs.keys())

    print(f"\nCOMPARAISON — {player} ({scope})  vs référentiels  [patch {me.get('patch','?')}]")
    print("=" * 68)

    # --- métriques de tête, à issue égale ---
    def row(label, fn):
        return f"  {label:<20}{fn(me):>9}" + "".join(f"{fn(refs[r]):>11}" for r in cols)

    print(f"  {'':<20}{'TOI':>9}" + "".join(f"{r[:9]:>11}" for r in cols))
    print(row("winrate", lambda a: f"{a['winrate']:.0%}"))
    print(row("n_games", lambda a: a["n_games"]))
    print("  " + "-" * 64)
    for oc in ("overall", "win", "loss"):
        print(row(f"morts/game [{oc}]",
                  lambda a, oc=oc: f"{a[oc]['deaths_per_game']}"
                                   + ("" if a[oc]["n_games"] >= MIN_N else "*")))
    print("  (* effectif faible <5 games — prudence)")

    # --- tables de benchmark (une seule mécanique d'impression) --------------
    # Les 3 sections ci-dessous partageaient le même imprimeur recopié : lire
    # a[outcome][section][clé], formater, poser TOI + une colonne par rang.
    # Ajouter une ligne se payait en 2 endroits (la table ET son imprimeur).
    def section_table(title, section, rows, width=16, footer=None):
        def cell(a, key, fmt):
            v = a.get(outcome, {}).get(section, {}).get(key)
            if v is None:
                return "—"
            if fmt == "pct":
                return f"{v:.0%}"
            if fmt == "signed":
                return f"{v:+d}"
            return f"{v:.1f}"

        print(f"\n  {title}")
        print(f"    {'':<{width}}{'TOI':>9}" + "".join(f"{r[:9]:>11}" for r in cols))
        for lbl, key, fmt in rows:
            print(f"    {lbl:<{width}}{cell(me, key, fmt):>9}"
                  + "".join(f"{cell(refs[r], key, fmt):>11}" for r in cols))
        if footer:
            print(footer)

    section_table(
        f"Benchmark de lane vs adversaire — {outcome.upper()} (médianes) :", "lane",
        [("gold diff @10", "gd10", "signed"), ("gold diff @14", "gd14", "signed"),
         ("cs diff @10", "csd10", "signed"), ("cs diff @14", "csd14", "signed"),
         ("gold diff @20", "gd20", "signed")])

    # positionnement : timeline, 0 CV ; COACHING_SAFE uniquement (cf. assert POS_ROWS)
    section_table(
        f"Benchmark positionnement — {outcome.upper()} (médianes) :", "positioning",
        POS_ROWS, width=18,
        footer="  (profondeur ↑ = plus diamond/risqué, PAS un objectif — cf. note compare.py)")

    section_table(
        f"Contexte éco. des morts — {outcome.upper()} (part des morts) :",
        "death_gold_state",
        [("mort en avance", "ahead", "pct"), ("mort à égalité", "even", "pct"),
         ("mort en retard", "behind", "pct")])

    # --- écarts ZONE×PHASE pour l'issue choisie ---
    if target not in refs:
        target = cols[-1]
    me_f, ref_f = me.get(outcome, {}), refs[target].get(outcome, {})
    me_zp, ref_zp = me_f.get("by_zone_phase", {}), ref_f.get("by_zone_phase", {})
    keys = set(me_zp) | set(ref_zp)
    deltas = sorted(((k, me_zp.get(k, 0), ref_zp.get(k, 0)) for k in keys),
                    key=lambda t: t[1] - t[2], reverse=True)

    warn = ""
    if me_f.get("n_games", 0) < MIN_N or ref_f.get("n_games", 0) < MIN_N:
        warn = f"  ⚠ effectif faible (toi={me_f.get('n_games',0)}, {target}={ref_f.get('n_games',0)})"
    print(f"\n  Où tu SUR-meurs vs {target} — issue = {outcome.upper()}{warn}")
    print(f"    {'zone × phase':<20}{'toi':>7}{target[:5]:>9}{'écart':>9}")
    for k, mv, rv in deltas[:6]:
        gap = mv - rv
        flag = "  ←" if gap >= 0.08 else ""
        print(f"    {k:<20}{mv:>7.0%}{rv:>9.0%}{gap:>+8.0%}{flag}")

    # --- benchmark conditionné sur le contexte de lane ---
    print(f"\n  Benchmark conditionné sur le contexte de lane (toutes issues, vs {target}) :")
    for axis in ("lane_pattern", "gank_exposure"):
        r = context_benchmark(me, refs[target], axis, outcome)
        if not r:
            print(f"    {axis}: pas de contexte côté perso (comp manquant ?)")
            continue
        gm = f"{r['gd10_me']:+d}" if r["gd10_me"] is not None else "—"
        gr = f"{r['gd10_ref']:+d}" if r["gd10_ref"] is not None else "—"
        tag = f"  ⚠ {r['reason']}" if r["fallback"] else ""
        print(f"    {axis} = {r['bucket']:<10} gd10 toi {gm} vs {target} {gr} "
              f"(toi n={r['n_me']}, réf n={r['n_ref']}){tag}")

    # --- verdict ---
    print("\n  ⚑ Verdict :")
    mo = dpg(me, outcome)
    ro = dpg(refs[target], outcome)
    if mo - ro > 0.5:
        print(f"    • En {outcome}, tu meurs {mo - ro:.1f} de plus/game qu'un {target} "
              f"({mo} vs {ro}) → marge réelle, biais d'issue neutralisé.")
    if deltas and deltas[0][1] - deltas[0][2] >= 0.08:
        k, mv, rv = deltas[0]
        print(f"    • Écart n°1 ({outcome}) = {k} : {mv:.0%} de tes morts vs {rv:.0%}. LE focus.")
    else:
        print("    • Écarts diffus à cette issue — pas de pattern dominant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
