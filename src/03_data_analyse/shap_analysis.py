#!/usr/bin/env python3
"""
03_data_analyse — analyse EBM-primary (glass-box) + cross-check SHAP-sur-arbres.

INVERSION DE RÔLE (cf. décision EBM) : l'EBM (GA²M) est désormais la SOURCE PRIMAIRE
d'explication, pas un simple validateur. Justification mesurée : en CV out-of-fold
honnête (groupage puuid), AUC(ebm) ≈ AUC(ensemble) — l'EBM ne sacrifie aucun pouvoir
prédictif ET son explication est EXACTE par construction (ses shape functions SONT le
modèle), là où le SHAP-sur-arbres est une attribution post-hoc qui se dégrade sous
features corrélées (et les nôtres le sont : blocs csm/gpm/xppm, frac_behind/ahead,
familles deaths_*).

Sorties (par cible, sous data/06_shap/<target>/) :
  - ebm_shape_functions.json : LE livrable prescriptif. Par feature : direction, seuil
    de bascule (valeur où le score log-odds croise 0), amplitude d'effet, monotonie.
    Robuste : résumé restreint au cœur des données [p5, p95] (les bins extrêmes
    low-density de l'EBM sont bruités).
  - ebm_ranking.json : importance globale des main effects (ranking primaire).
  - ebm_interactions.json : top interactions par paires (structure que l'additif rate).
  - crosscheck_tree_vs_ebm.json : SHAP moyen (xgb+rf) vs contributions EBM par feature
    (Spearman + accord de signe) — le SHAP valide maintenant l'EBM, pas l'inverse.
  - spadzze_ebm_drivers.json : drivers EBM des games de Spadzze.
  - shap_bar.png / shap_beeswarm.png : visuels SHAP-sur-arbres (cross-check).
  - diagnostics.json : auto-diagnostic LOWESS sur contributions EBM.

Usage : poetry run python3 src/03_data_analyse/shap_analysis.py [--target high_elo|dia_chall]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import riotlib as rl
import shap
import plotter
from scipy.stats import spearmanr

DATASET = rl.DATA / "04_dataset" / "adc_dataset.parquet"
MODEL_DIR = rl.DATA / "05_model"

# Noms des deux classes (neg=0, pos=1) par cible — pour une lecture prescriptive
# orientée ("valeur haute -> pousse vers <pos>").
TARGET_NAMES = {
    "high_elo": ("low(M/D)", "high(GM/C)"),
    "dia_chall": ("diamond", "challenger"),
}
# Rangs de la population d'analyse (doit matcher l'entraînement du modèle) : None =
# tout le référentiel. Sinon, les rows hors-scope sont hors-distribution pour ce
# modèle et fausseraient cross-check / diagnostics / quantiles des shape functions.
TARGET_RANKS = {
    "high_elo": None,
    "dia_chall": ["diamond", "challenger"],
}


def ebm_term_index(ebm, feature: str) -> int | None:
    """Index du terme main-effect (univarié) correspondant à `feature`, ou None."""
    for i, name in enumerate(ebm.term_names_):
        if name == feature:
            return i
    return None


def extract_shape(ebm, ti: int, vals: pd.Series, neg: str, pos: str) -> dict:
    """Shape function exacte d'un main effect -> résumé prescriptif robuste.

    score log-odds par bin (>0 pousse vers `pos`). On restreint le résumé au cœur des
    données [p5, p95] : les bins extrêmes low-density de l'EBM sont bruités et donnent
    des seuils trompeurs."""
    d = ebm.explain_global().data(ti)
    edges, scores = list(d["names"]), list(d["scores"])
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(scores))]
    clean = vals.dropna()
    p5, p95 = (float(clean.quantile(0.05)), float(clean.quantile(0.95))) if len(clean) else (mids[0], mids[-1])
    core = [(m, s) for m, s in zip(mids, scores) if p5 <= m <= p95] or list(zip(mids, scores))
    cmids = [m for m, _ in core]
    cscores = [s for _, s in core]

    swing = max(cscores) - min(cscores)
    rho = float(spearmanr(cmids, cscores)[0]) if len(set(cscores)) > 1 else 0.0
    lo, hi = cscores[0], cscores[-1]

    # seuil de bascule : 1re valeur où le score change de signe par rapport au bas.
    crossover = None
    s0 = np.sign(lo) if lo != 0 else 0
    for i in range(1, len(cscores)):
        if s0 and np.sign(cscores[i]) == -s0:
            crossover = round(cmids[i], 2)
            break

    direction = (f"valeur haute → {pos}" if hi > lo else f"valeur haute → {neg}")
    return {
        "swing_logodds": round(float(swing), 3),
        "monotonic_rho": round(rho, 2),
        "score_low": round(float(lo), 3),
        "score_high": round(float(hi), 3),
        "crossover_value": crossover,
        "direction": direction,
        "core_range": [round(p5, 2), round(p95, 2)],
    }


def tree_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP (classe 1) d'un modèle arbre, robuste aux structures RF/XGB."""
    sv = shap.TreeExplainer(model)(X)
    if isinstance(sv.values, list):          # RF binaire -> liste par classe
        return sv.values[1]
    if len(sv.values.shape) == 3:            # autre structure RF (n, f, classes)
        return sv.values[:, :, 1]
    return sv.values


def main(target: str = "dia_chall") -> int:
    neg, pos = TARGET_NAMES[target]
    sfx = "highelo" if target == "high_elo" else target
    FEATURES = json.loads((MODEL_DIR / "features.json").read_text())

    models = {}
    for name in ["xgb", "rf", "ebm"]:
        with open(MODEL_DIR / f"{name}_{sfx}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    ebm = models["ebm"]

    df = pd.read_parquet(DATASET)
    ref = df[df["source"] == "referentiel"].copy()
    ranks = TARGET_RANKS[target]
    if ranks is not None:                       # restreindre à la population du modèle
        ref = ref[ref["rank"].isin(ranks)]
    Xref = ref[FEATURES]

    OUT = rl.DATA / "06_shap" / target
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"  cible='{target}' (neg={neg} / pos={pos}) | {len(Xref)} games référentiel\n")

    # ============================================================ PRIMARY : EBM
    eg = ebm.explain_global().data()
    term_names, term_scores = list(eg["names"]), [float(s) for s in eg["scores"]]
    main_imp = {n: s for n, s in zip(term_names, term_scores) if " & " not in n}
    interactions = sorted(
        [(n, s) for n, s in zip(term_names, term_scores) if " & " in n],
        key=lambda t: -t[1])

    # --- ranking primaire (importance des main effects) ---
    ranking = sorted(((f, main_imp.get(f, 0.0)) for f in FEATURES), key=lambda t: -t[1])
    print("  ⭐ EBM main effects — importance globale (ranking primaire) :")
    for f, v in ranking:
        print(f"    {f:<20} {v:.3f}")
    (OUT / "ebm_ranking.json").write_text(json.dumps(
        [{"feature": f, "importance": round(v, 4)} for f, v in ranking], indent=2))

    # --- shape functions : LE livrable prescriptif ---
    shapes = {}
    for f in FEATURES:
        ti = ebm_term_index(ebm, f)
        if ti is not None:
            shapes[f] = extract_shape(ebm, ti, Xref[f], neg, pos)
    by_swing = sorted(shapes.items(), key=lambda kv: -kv[1]["swing_logodds"])
    print("\n  📐 EBM shape functions (prescriptif, trié par amplitude d'effet) :")
    print(f"    {'feature':<20} {'swing':>6} {'mono':>5} {'seuil':>9}  sens")
    for f, s in by_swing:
        seuil = "—" if s["crossover_value"] is None else f"{s['crossover_value']:.2f}"
        print(f"    {f:<20} {s['swing_logodds']:>6.2f} {s['monotonic_rho']:>+5.2f} "
              f"{seuil:>9}  {s['direction']}")
    (OUT / "ebm_shape_functions.json").write_text(json.dumps(shapes, indent=2))

    # --- interactions par paires (la structure que l'additif pur ne voit pas) ---
    print("\n  🔗 EBM — top interactions par paires :")
    inter_rows = []
    for name, score in interactions[:10]:
        print(f"    {name:<32} {score:.3f}")
        inter_rows.append({"pair": name, "score": round(score, 4)})
    (OUT / "ebm_interactions.json").write_text(json.dumps(inter_rows, indent=2))

    # contributions EBM par sample (pour cross-check + diagnostics + Spadzze)
    def ebm_contribs(X: pd.DataFrame) -> np.ndarray:
        loc = ebm.explain_local(X)._internal_obj["specific"]
        out = np.zeros((len(X), len(FEATURES)))
        for i in range(len(X)):
            n2s = dict(zip(loc[i]["names"], loc[i]["scores"]))
            for j, f in enumerate(FEATURES):
                out[i, j] = float(n2s.get(f, 0.0))
        return out

    ebm_ref = ebm_contribs(Xref)

    # ====================================================== CROSS-CHECK : SHAP arbres
    # Le SHAP moyen (xgb+rf) VALIDE maintenant l'EBM (rôle inversé). Si direction
    # d'accord -> le signal primaire EBM n'est pas un artefact de l'additif.
    sv_trees = [tree_shap_values(models[n], Xref) for n in ("xgb", "rf")]
    sv_ensemble_vals = np.mean(sv_trees, axis=0)
    sv_ensemble = shap.Explanation(values=sv_ensemble_vals, data=Xref.values,
                                   feature_names=FEATURES)

    print("\n  📈 Cross-check SHAP-arbres vs EBM (validation de la primaire) :")
    cross = []
    for j, f in enumerate(FEATURES):
        rho = float(spearmanr(ebm_ref[:, j], sv_ensemble_vals[:, j])[0])
        sign_agree = float(np.mean(np.sign(ebm_ref[:, j]) == np.sign(sv_ensemble_vals[:, j])))
        cross.append({"feature": f, "spearman": round(rho, 3), "sign_agree": round(sign_agree, 3)})
    cross.sort(key=lambda d: -abs(d["spearman"]))
    for c in cross:
        flag = "✓" if c["spearman"] > 0.3 else ("⚠" if c["spearman"] < -0.2 else "·")
        print(f"    {flag} {c['feature']:<20} rho={c['spearman']:+.2f}  sign_agree={c['sign_agree']:.0%}")
    agree = np.mean([1 for c in cross if c["spearman"] > 0.3])
    print(f"    → {int(sum(1 for c in cross if c['spearman'] > 0.3))}/{len(cross)} features en accord direction (rho>0.3)")
    (OUT / "crosscheck_tree_vs_ebm.json").write_text(json.dumps(cross, indent=2))

    plt.figure()
    shap.plots.bar(sv_ensemble, max_display=len(FEATURES), show=False)
    plt.tight_layout(); plt.savefig(OUT / "shap_bar.png", dpi=130); plt.close()
    plt.figure()
    shap.plots.beeswarm(sv_ensemble, max_display=len(FEATURES), show=False)
    plt.tight_layout(); plt.savefig(OUT / "shap_beeswarm.png", dpi=130); plt.close()

    # ============================================================ Spadzze via EBM
    spad = df[df["source"].str.startswith("personal:spadzze", na=False)].copy()
    if len(spad):
        contrib = ebm_contribs(spad[FEATURES])
        mean_signed = contrib.mean(axis=0)
        drivers = sorted(zip(FEATURES, mean_signed), key=lambda t: t[1])
        print(f"\n  🎯 Drivers EBM — Spadzze ({len(spad)} games) :")
        for f, v in drivers:
            print(f"    {f:<20} {v:+.3f}  {'→ ' + neg if v < 0 else '→ ' + pos}")
        (OUT / "spadzze_ebm_drivers.json").write_text(json.dumps(
            [{"feature": f, "mean_ebm_contrib": round(float(v), 4)} for f, v in drivers], indent=2))

    # ============================================================ diagnostics LOWESS
    print("\n  🔍 Auto-diagnostic (LOWESS) sur contributions EBM :")
    diagnostics = plotter.generate_lol_diagnostics(Xref, ebm_ref, FEATURES)
    for d in diagnostics[:15]:
        print(f"    {d['feature']:<20} | {d['diagnostic']}")
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    print(f"\n✓ Analyse EBM-primary écrite dans {OUT}/")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGET_NAMES), default="dia_chall")
    sys.exit(main(ap.parse_args().target))
