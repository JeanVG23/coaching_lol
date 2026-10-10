"""src/core/ebm_explain.py : moteur d'analyse EBM glass-box, unifié sur deux niveaux.

L'EBM (GA²M) est la SOURCE PRIMAIRE d'explication (inversion de rôle, décision
mesurée : en CV out-of-fold honnête AUC(ebm) ≈ AUC(ensemble), et son explication
est EXACTE par construction : ses shape functions SONT le modèle, là où le
SHAP-sur-arbres est une attribution post-hoc qui se dégrade sous features
corrélées). Le SHAP-sur-arbres devient le cross-check.

Deux niveaux déclarés dans LEVELS, aux rôles explicites :
  - "player" : LE modèle servi. Explication du placement d'un joueur (features
    agrégées mean/std/p10/p50/p90). Le sync publie la décomposition exacte
    « pourquoi le modèle te place master » sous la clé KV shap:{slug}:drivers.
  - "game" : modèle d'explication du jeu-type (dia_chall), re-entraîné dans le
    DAG pour l'analyse (seuils de bascule in-game), JAMAIS servi pour le rang.
    La frontière per-game high_elo est abandonnée : illisible à N=1 (tabular
    0.609, transformer séquentiel 0.546, MLP 0.504 ; cf. spec 2026-09-08).

Bibliothèque pure : 0 écriture disque ; les fonctions d'analyse reçoivent l'EBM déjà
chargé (seule load_level touche aux pkl : déserialiser un EBM importe interpret via
pickle, c'est assumé : on ne peut pas expliquer un objet sans le charger). Import
shap différé dans tree_shap_values : seul le cross-check en a besoin, pas le sync
(drivers per-player). Consommée par le CLI d'analyse
(03_data_analyse/shap_analysis.py) ET par le sync (collection/sync_cloudflare.py),
qui a déjà src/core dans sys.path ; le wildcard $(CORE) du Makefile périme
automatiquement l'aval quand ce moteur change.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import riotlib as rl
from scipy.stats import spearmanr

LEVELS = {
    "player": {   # LE MODÈLE SERVI : explication du placement
        "models": ("xgb_player_highelo", "rf_player_highelo", "ebm_player_highelo"),
        "features_file": "player_features.json",
        "dataset": "04_dataset/adc_player_dataset.parquet",
        "names": ("low(M/D)", "high(GM/C)"),
        "source": None,           # tout le dataset (rows 100% référentiel)
        "ranks": None,            # population entière = population d'entraînement
        "interactions": False,    # interactions=0 au training, rien à lire
        "out_dir": "06_shap/player/high_elo",
    },
    "game": {     # MODÈLE D'EXPLICATION DU JEU-TYPE : jamais servi
        "models": ("xgb_dia_chall", "rf_dia_chall", "ebm_dia_chall"),
        "features_file": "features.json",
        "dataset": "04_dataset/adc_dataset.parquet",
        "names": ("diamond", "challenger"),
        "source": "referentiel",  # les rows perso ne font PAS partie de la population
        "ranks": ["diamond", "challenger"],   # sinon hors-distribution pour ce modèle
        "interactions": True,
        "out_dir": "06_shap/game/dia_chall",
    },
}


def load_level(level: str) -> dict:
    """Charge un niveau du registre : 3 modèles, features, dataset complet et
    population filtrée.

    Le filtre de population (source puis ranks) fait partie du contrat : une row
    hors scope est hors-distribution pour ce modèle et fausserait cross-check,
    diagnostics et quantiles des shape functions. `df_all` garde le dataset non
    filtré pour les rows perso (drivers Spadzze au niveau game)."""
    cfg = LEVELS[level]
    model_dir = rl.DATA / "05_model"   # résolu à l'APPEL : les tests substituent rl.DATA
    features = rl.read_json(model_dir / cfg["features_file"]) or []
    models = {}
    for name in cfg["models"]:
        with open(model_dir / f"{name}.pkl", "rb") as f:
            models[name.split("_", 1)[0]] = pickle.load(f)
    df_all = pd.read_parquet(rl.DATA / cfg["dataset"])
    df = df_all
    if cfg["source"] is not None:
        df = df[df["source"] == cfg["source"]]
    if cfg["ranks"] is not None:
        df = df[df["rank"].isin(cfg["ranks"])]
    return {"models": models, "features": features, "df": df, "df_all": df_all,
            "config": cfg}


def term_index(ebm, feature: str) -> int | None:
    """Index du terme main-effect (univarié) correspondant à `feature`, ou None."""
    for i, name in enumerate(ebm.term_names_):
        if name == feature:
            return i
    return None


def shape_summary(ebm, ti: int, vals: pd.Series, neg: str, pos: str) -> dict:
    """Shape function exacte d'un main effect -> résumé prescriptif robuste.

    score log-odds par bin (>0 pousse vers `pos`). On restreint le résumé au cœur
    des données [p5, p95] : les bins extrêmes low-density de l'EBM sont bruités et
    donnent des seuils trompeurs. La clé `degenerate` porte la raison quand le
    résumé n'est PAS prescriptible (`constant` : colonne sans contraste, refusée ;
    `core_hors_bins` : repli sur tous les bins faute de cœur mesurable)."""
    d = ebm.explain_global().data(ti)
    edges, scores = list(d["names"]), list(d["scores"])
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(scores))]
    clean = vals.dropna()
    # Une colonne CONSTANTE n'a pas de forme : la population n'offre aucun contraste,
    # et les bins que l'EBM a quand même appris ne décrivent que du binning. La classer
    # produirait un seuil de bascule inventé (9 features __p10 du niveau player sont
    # dans ce cas). On refuse de la ranger : swing 0 -> bas du tri prescriptif, et le
    # JSON dit POURQUOI plutôt que de livrer un chiffre indistinguable d'un vrai.
    # Série VIDE ≠ constante : pas de données n'est pas l'absence de contraste, on
    # garde là le repli historique sur tous les bins.
    if clean.nunique() == 1:
        v = round(float(clean.iloc[0]), 2)
        return {
            "swing_logodds": 0.0, "monotonic_rho": 0.0,
            "score_low": 0.0, "score_high": 0.0, "crossover_value": None,
            "direction": "indéterminé (colonne constante)",
            "core_range": [v, v], "degenerate": "constant",
        }
    p5, p95 = (float(clean.quantile(0.05)), float(clean.quantile(0.95))) if len(clean) else (mids[0], mids[-1])
    core = [(m, s) for m, s in zip(mids, scores) if p5 <= m <= p95]
    degenerate = None
    if not core:   # [p5, p95] plus étroit qu'un bin : aucun cœur mesurable
        core, degenerate = list(zip(mids, scores)), "core_hors_bins"
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
        "degenerate": degenerate,
    }


def term_contributions(ebm, X: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Contributions exactes par terme et par sample : matrice (n_samples,
    n_features) ordonnée sur `features`, 0.0 pour une feature sans terme
    main-effect. Reprise de l'ancien ebm_contribs de shap_analysis.py."""
    loc = ebm.explain_local(X)._internal_obj["specific"]
    out = np.zeros((len(X), len(features)))
    for i in range(len(X)):
        n2s = dict(zip(loc[i]["names"], loc[i]["scores"]))
        for j, f in enumerate(features):
            out[i, j] = float(n2s.get(f, 0.0))
    return out


def tree_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP (classe 1) d'un modèle arbre, robuste aux structures RF/XGB.
    Import différé : shap ne sert qu'au cross-check ; le sync (drivers
    per-player) ne doit pas charger un second runtime OpenMP pour rien."""
    import shap
    sv = shap.TreeExplainer(model)(X)
    if isinstance(sv.values, list):          # RF binaire -> liste par classe
        return sv.values[1]
    if len(sv.values.shape) == 3:            # autre structure RF (n, f, classes)
        return sv.values[:, :, 1]
    return sv.values


def crosscheck(models: dict, X: pd.DataFrame, features: list[str],
               ebm_contribs: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """SHAP moyen (xgb+rf) vs contributions EBM par feature : Spearman + accord
    de signe. Le SHAP-sur-arbres VALIDE l'EBM primaire ; au niveau player, xgb+rf
    sont les deux modèles servis, ce qui légitime les drivers EBM publiés alors
    que le rang vient de l'ensemble.

    Retourne (lignes par feature triées par |spearman| décroissant, valeurs
    SHAP moyennes brutes) : le CLI resert les brutes pour shap_bar/beeswarm."""
    sv_trees = [tree_shap_values(models[n], X) for n in ("xgb", "rf")]
    sv_vals = np.mean(sv_trees, axis=0)
    rows = []
    for j, f in enumerate(features):
        rho = float(spearmanr(ebm_contribs[:, j], sv_vals[:, j])[0])
        # NaN scipy = au moins une des deux séries est constante : la corrélation
        # n'est pas définie, ce qui n'est PAS « corrélation nulle ». On garde 0.0
        # pour que le tri par |rho| reste total, mais `degenerate` dit que ce 0.0
        # est un « non mesurable » : sans ce drapeau, une feature dont l'EBM ne dit
        # rien serait indistinguable d'une feature où EBM et arbres se contredisent.
        degenerate = None
        if not np.isfinite(rho):
            rho, degenerate = 0.0, "non_mesurable"
        sign_agree = float(np.mean(np.sign(ebm_contribs[:, j]) == np.sign(sv_vals[:, j])))
        rows.append({"feature": f, "spearman": round(rho, 3),
                     "sign_agree": round(sign_agree, 3), "degenerate": degenerate})
    rows.sort(key=lambda d: -abs(d["spearman"]))
    return rows, sv_vals


def explain_player_row(ebm, agg: dict, features: list[str]) -> list[dict]:
    """Décomposition exacte d'UNE ligne de features per-player : l'EBM additif
    rend la somme des contributions identique au score, par construction.
    Retourne [{"feature", "contribution"}, ...] sur TOUTES les features, signe
    conservé ; l'appelant rogne (top_drivers)."""
    X = pd.DataFrame([agg]).reindex(columns=features).astype(float)
    contribs = term_contributions(ebm, X, features)[0]
    return [{"feature": f, "contribution": float(c)}
            for f, c in zip(features, contribs)]


def top_drivers(contribs: list[dict], n: int = 20) -> list[dict]:
    """Top-n par |contribution|, signe conservé (125 features per-player :
    l'onglet n'affichera pas 125 barres)."""
    return sorted(contribs, key=lambda d: -abs(d["contribution"]))[:n]
