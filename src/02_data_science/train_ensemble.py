#!/usr/bin/env python3
"""
DÉPRÉCIÉ — arrêté le 2026-07-18. Modèle per-game (1 ligne = 1 ADC d'une game) NON servi
en prod : le web (web/backend/ml_rank.py) tourne sur le per-player « constance/plancher ».
AUC trop basse (~0.63 dia_chall / ~0.59 high_elo) — 1 game porte un signal quasi aléatoire
(RNG matchmaking/stomps), sans valeur prédictive utile. Conservé pour l'historique et la
reproductibilité (aucune suppression). Ne pas migrer au protocole gold standard.
Voir docs/superpowers/specs/2026-07-18-gold-standard-eval-protocol-design.md.

02_data_science — entraîne un Ensemble (XGBoost, Random Forest + EBM) pour séparer
high-elo (GM+Chall) de low (master+diam).

But : Obtenir un modèle très robuste dont les valeurs SHAP moyennes
représentent de vraies tendances de fond, indépendantes de l'algorithme choisi.

Composition (3 biais inductifs distincts, zéro redondance) :
- xgb : GBDT (splits axe-parallèles, interactions, boosting).
- rf  : bagging (splits axe-parallèles, interactions, arbres décorélés) —
         famille différente des boostés, vraie diversité dans la niche arbre.
- ebm : GA²M glass-box (Explainable Boosting Machine) — main effects = splines
         additives par feature + interactions par paires, boosting cyclique avec
         bagging. Membre de validation ET d'analyse d'interactions.

L'EBM joue un double rôle : (1) ses main effects (splines additives) ont un biais
inductif radicalement différent des arbres — contre-point de la monoculture arbre,
comme l'était le GAM auparavant ; (2) ses interactions par paires exposent la
structure que les arbres capturent mais qu'un additif pur ne peut pas voir — c'est
l'explication des divergences arbre-vs-additif observées sur n_deaths /
deaths_teamfight / deaths_early_jungle (games high-elo plus longues => plus de
morts : interaction durée×morts que l'additif pur rate).

Retraits de redondance : LightGBM (jumeau GBDT de XGBoost), LogisticRegression
(subsumé par un additif) et GAM/pyGAM (subsumé par l'EBM dont les main effects SONT
un GAM, augmenté des interactions). L'EBM remplace donc le GAM : stricte
généralisation, même rôle de validation + interactions en bonus.

Anti-fuite : on EXCLUT `win` et toute colonne dérivée du rang.
Sorties : data/05_model/{xgb,rf,ebm}_highelo.pkl, metrics.json, features.json
Usage : poetry run python3 src/02_data_science/train_ensemble.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import numpy as np
import pandas as pd
import riotlib as rl
import ml_features
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from interpret.glassbox import ExplainableBoostingClassifier

DATASET = rl.DATA / "04_dataset" / "adc_dataset.parquet"
MODEL_DIR = rl.DATA / "05_model"

RANKS = ["diamond", "master", "grandmaster", "challenger"]
HIGH_ELO = {"grandmaster", "challenger"}

FEATURES = ml_features.FEATURES  # canonique, cf. src/core/ml_features.py


def make_models() -> dict:
    # Conservateur pour eviter l'overfitting. 3 biais d'inductifs distincts :
    # GBDT (xgb) / bagging (rf) / GA²M glass-box (ebm).
    return {
        "xgb": xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.0, eval_metric="logloss", tree_method="hist",
            random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=5,
            max_features="sqrt", bootstrap=True, n_jobs=-1,
            random_state=42
        ),
        # EBM (GA²M, Explainable Boosting Machine) : main effects (splines additives)
        # + interactions par paires, glass-box native. Remplace le GAM : mêmes main
        # effects (rôle de validation additif) + interactions qui exposent la structure
        # que les arbres capturent et que l'additif pur ne voit pas. Gère les NaN
        # nativement (bin "missing"), pas d'imputer nécessaire.
        "ebm": ExplainableBoostingClassifier(
            interactions=10, random_state=42,
        ),
    }


# Cibles binaires : la coupe par défaut (high_elo) tombe à la frontière Master|GM,
# deux tiers adjacents quasi-indistinguables. La variante dia_chall retire le milieu
# (Master+GM) et oppose les EXTRÊMES (Diamond vs Challenger) — teste si le signal de
# rang existe tout court dans les features macro.
TARGETS = {
    "high_elo": {"ranks": RANKS, "pos": HIGH_ELO,
                 "names": ["low(M/D)", "high(GM/C)"]},
    "dia_chall": {"ranks": ["diamond", "challenger"], "pos": {"challenger"},
                  "names": ["diamond", "challenger"]},
}


def main(target: str = "high_elo") -> int:
    spec = TARGETS[target]
    df = pd.read_parquet(DATASET)
    train = df[(df["source"] == "referentiel") & df["rank"].isin(spec["ranks"])].copy()
    X = train.reindex(columns=FEATURES)   # colonnes pos_* absentes (silver non ré-extrait) -> NaN, pas KeyError
    y = train["rank"].isin(spec["pos"]).astype(int)
    print(f"  cible='{target}' | {len(train)} games référentiel | "
          f"pos={int(y.sum())} / neg={int((1-y).sum())}")

    # --- évaluation honnête en CV stratifiée groupée (out-of-fold) ---
    # Anti-fuite : on groupe par PUUID (un joueur entier reste dans un seul fold).
    # On a ABANDONNÉ l'union connexe puuid∪match_id : depuis qu'on extrait les 2 ADC
    # par game (dataset densifié), les games multi-rang + la récurrence des joueurs
    # fusionnent ~84% des rows en UNE composante connexe géante (le high-elo est un
    # seul graphe social) → CV dégénérée (un fold s'entraîne sur 8% de données 92% low
    # → AUC OOF < 0.5, ininterprétable). Un split player-disjoint propre n'existe pas
    # à cette densité.
    # Choix : protéger la fuite FORTE — joueur→rang mémorisé (57% des rows viennent de
    # joueurs à ≥5 games) — en groupant par puuid. La fuite FAIBLE restante (les 2 ADC
    # miroir d'une même game tombant dans 2 folds) est négligeable ICI : les features
    # de lane sont des DIFFS signées OPPOSÉES entre les 2 ADC, et les stats de morts
    # sont par-joueur → rien de copiable trivialement.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = train["puuid"].astype("category").cat.codes.to_numpy()
    gsz = pd.Series(groups).value_counts()
    print(f"  {len(gsz)} groupes puuid (anti-fuite) | taille max={gsz.max()} "
          f"({gsz.max()/len(train):.1%} des rows)")

    oof_preds = {name: np.zeros(len(X)) for name in make_models().keys()}

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        models = make_models()
        for name, model in models.items():
            model.fit(X_train, y_train)
            oof_preds[name][val_idx] = model.predict_proba(X_val)[:, 1]
            
    # --- perf par modèle (out-of-fold) ---
    # Décisif pour la question EBM : si AUC(ebm) ≈ AUC(ensemble), l'EBM (glass-box,
    # explication exacte par construction) peut devenir la source PRIMAIRE d'insights
    # prescriptifs, le SHAP-sur-arbres passant en cross-check. Sinon il reste validateur.
    per_model = {}
    print("\n  Perf par modèle (CV out-of-fold) :")
    for name, preds in oof_preds.items():
        m_auc = roc_auc_score(y, preds)
        m_acc = accuracy_score(y, (preds >= 0.5).astype(int))
        per_model[name] = {"auc": round(m_auc, 4), "acc": round(m_acc, 4)}
        print(f"    {name:<4} AUC={m_auc:.3f}  accuracy={m_acc:.3f}")

    # Soft Voting
    ensemble_proba = np.mean(list(oof_preds.values()), axis=0)

    auc = roc_auc_score(y, ensemble_proba)
    acc = accuracy_score(y, (ensemble_proba >= 0.5).astype(int))
    print(f"\n  Ensemble CV out-of-fold : AUC={auc:.3f}  accuracy={acc:.3f}")
    # Écart EBM vs ensemble : < ~0.01 AUC => promotion EBM en source primaire défendable.
    ebm_gap = auc - per_model.get("ebm", {}).get("auc", auc)
    print(f"  → écart AUC ensemble−ebm = {ebm_gap:+.3f} "
          f"({'EBM ~= ensemble, promotion défendable' if ebm_gap < 0.01 else 'ensemble nettement devant, EBM reste validateur'})")
    print("  (L'objectif est d'obtenir le signal le plus robuste pour le SHAP)")
    print(classification_report(y, (ensemble_proba >= 0.5).astype(int),
                                target_names=spec["names"], digits=3))

    # --- modèle final sur toutes les données (pour SHAP) ---
    # Suffixe par cible : la cible par défaut (high_elo) garde les noms historiques
    # ({name}_highelo.pkl, metrics.json) que shap_analysis.py lit ; les variantes
    # écrivent à part pour ne pas écraser le pipeline principal.
    sfx = "highelo" if target == "high_elo" else target
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    final_models = make_models()
    for name, model in final_models.items():
        model.fit(X, y)
        with open(MODEL_DIR / f"{name}_{sfx}.pkl", "wb") as f:
            pickle.dump(model, f)

    metrics_name = "metrics.json" if target == "high_elo" else f"metrics_{target}.json"
    (MODEL_DIR / "features.json").write_text(json.dumps(FEATURES, indent=2))
    (MODEL_DIR / metrics_name).write_text(json.dumps({
        "target": target,
        "auc_cv": round(auc, 4), "acc_cv": round(acc, 4),
        "per_model_cv": per_model,
        "ebm_gap_vs_ensemble": round(ebm_gap, 4),
        "n_train": len(train), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "features": FEATURES,
    }, indent=2))

    print(f"\n✓ {len(final_models)} modèles (Ensemble) écrits dans {MODEL_DIR}/ (suffixe '{sfx}')")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS), default="high_elo",
                    help="high_elo (GM/C vs M/D) | dia_chall (Challenger vs Diamond, extrêmes)")
    sys.exit(main(ap.parse_args().target))
