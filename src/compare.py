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

import riotlib as rl

RANKS = ["diamond", "master", "grandmaster", "challenger"]
MIN_N = 5  # seuil d'effectif sous lequel on prévient
MIN_CONTEXT_N = 8  # sous ce seuil de games référentiel dans le bucket, on retombe sur global


def context_benchmark(me_agg, ref_agg, axis, outcome):
    """Compare le bucket de contexte DOMINANT côté perso au même bucket référentiel.

    Repli explicite et loggué sur 'overall' si le référentiel a < MIN_CONTEXT_N games
    dans ce bucket (échantillon trop fin pour un benchmark honnête).
    """
    me_buckets = me_agg.get("by_lane_context", {}).get(axis, {})
    ref_buckets = ref_agg.get("by_lane_context", {}).get(axis, {})
    if not me_buckets:
        return None
    bucket = max(me_buckets, key=lambda b: me_buckets[b].get("n_games", 0))
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


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def dpg(agg, outcome):
    f = agg.get(outcome, {})
    n = f.get("n_games", 0)
    return f.get("deaths_per_game", 0), n


def main() -> int:
    player = arg("--player", "spadzze")
    scope = arg("--scope", "adc")
    outcome = arg("--outcome", "loss")
    target = arg("--target", "challenger")

    me = load(rl.GOLD_DIR / "personal" / player / scope / "aggregate.json")
    if not me or not me["n_games"]:
        print(f"✗ Pas de données perso pour {player}/{scope}.", file=sys.stderr)
        return 1
    refs = {r: load(rl.GOLD_DIR / "referentiel" / r / scope / "aggregate.json")
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

    # --- benchmark de lane (issue choisie, médianes) ---
    def lane_cell(a, key):
        v = a.get(outcome, {}).get("lane", {}).get(key)
        return (f"{v:+d}" if v is not None else "—")

    def lane_row(label, key):
        print(f"    {label:<16}{lane_cell(me, key):>9}"
              + "".join(f"{lane_cell(refs[r], key):>11}" for r in cols))

    print(f"\n  Benchmark de lane vs adversaire — {outcome.upper()} (médianes) :")
    print(f"    {'':<16}{'TOI':>9}" + "".join(f"{r[:9]:>11}" for r in cols))
    for lbl, key in [("gold diff @10", "gd10"), ("gold diff @14", "gd14"),
                     ("cs diff @10", "csd10"), ("cs diff @14", "csd14"),
                     ("gold diff @20", "gd20")]:
        lane_row(lbl, key)

    # --- contexte économique des morts ---
    def gs_cell(a, key):
        v = a.get(outcome, {}).get("death_gold_state", {}).get(key)
        return f"{v:.0%}" if v is not None else "—"

    print(f"\n  Contexte éco. des morts — {outcome.upper()} (part des morts) :")
    print(f"    {'':<16}{'TOI':>9}" + "".join(f"{r[:9]:>11}" for r in cols))
    for lbl, key in [("mort en avance", "ahead"), ("mort à égalité", "even"),
                     ("mort en retard", "behind")]:
        print(f"    {lbl:<16}{gs_cell(me, key):>9}"
              + "".join(f"{gs_cell(refs[r], key):>11}" for r in cols))

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
    print(f"\n  Benchmark conditionné sur le contexte de lane (vs {target}) :")
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
    mo, _ = dpg(me, outcome)
    ro, _ = dpg(refs[target], outcome)
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
