#!/usr/bin/env python3
"""Décide l'ouverture publique de chaque rôle à partir du held-out."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import cli  # noqa: E402
import riotlib as rl  # noqa: E402
import role_features as rf  # noqa: E402

MODEL_DIR = rl.DATA / "05_model"
THRESHOLD = 0.70
DEFAULT_CORPUS = "research"
"""Seul `production` autorise l'ouverture : c'est une certification, pas un
état par défaut. Rien ne borne plus l'écart entre les parties agrégées et la
capture du rang (`label_age_days` le mesure sans le refuser) ; annoncer
production doit donc rester un geste écrit à la main."""

CORPUS_FILE = rl.ROOT / "config" / "role_corpus.json"
"""Où vit ce geste. Pas dans `role_readiness.json`, que `make roles` réécrit en
entier : une certification posée là disparaîtrait au premier réentraînement,
sans bruit et sans trace. Ici elle est versionnée, donc datée et attribuée par
git. Sous `ROOT` et non sous `DATA` : c'est une décision du dépôt, pas une
couche du médaillon, et `COACHING_DATA_DIR` n'a pas à la déplacer."""


def load_corpus_table(path=None) -> dict:
    """Certifications par rôle, ou une table vide si le fichier manque.

    Un clone frais n'a pas à porter ce fichier pour que `make roles` tourne, et
    un JSON cassé ferme les rôles au lieu de faire tomber la régénération : les
    deux échecs rendent la table vide, donc `research` partout.
    """
    path = Path(path) if path is not None else CORPUS_FILE
    raw = rl.read_json(path)
    if not isinstance(raw, dict):
        return {}
    return {rf.normalize_role(role): entry for role, entry in raw.items()
            if isinstance(entry, dict)}


def resolve_corpus(role: str, model_id: str | None, table: dict) -> str:
    """Corpus certifié pour ce rôle CE modèle-ci, sinon `research`.

    La certification porte le `model_id` qu'elle certifie. Un réentraînement en
    produit un autre, donc périme la certification et referme le rôle de
    lui-même. Sans ce lien, la certification voyagerait vers le modèle suivant
    et ouvrirait au public des chiffres que personne n'a lus.
    """
    entry = table.get(rf.normalize_role(role)) or {}
    certified = entry.get("model_id")
    if entry.get("corpus") == "production" and certified and certified == model_id:
        return "production"
    return DEFAULT_CORPUS


def _stderr(auc: float, n_players: int) -> float:
    """Approximation de Hanley-McNeil pour un échantillon équilibré."""
    n_pos = n_neg = max(n_players / 2, 1)
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    variance = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc**2)
        + (n_neg - 1) * (q2 - auc**2)
    ) / (n_pos * n_neg)
    return max(variance, 0.0) ** 0.5


def readiness(metrics: dict, *, threshold: float = THRESHOLD,
              corpus: str = DEFAULT_CORPUS) -> dict:
    """Résume la marge prudente ; un corpus research ne peut jamais ouvrir.

    La marge se lit sur la MÉDIANE des tirages du held-out, jamais sur un
    tirage. Mesuré le 2026-09-18 : à 28 joueurs de held-out, le même modèle sur
    les mêmes données rend 0.663 ou 0.918 selon les personnes tirées, et la
    certification passait cinq fois sur dix.

    Le bruit retranché est la PLUS GRANDE des deux estimations disponibles,
    l'approximation paramétrique et la dispersion réellement observée. Ce n'est
    pas un double comptage : les deux mesurent le même bruit d'échantillonnage
    par deux chemins, et à faible effectif chacune peut sous-estimer l'autre.
    """
    auc = float(metrics["auc_heldout_median"]["ebm"])
    draws = [float(row["ebm"]) for row in metrics["auc_heldout_seeds"]]
    if not draws:
        raise KeyError("auc_heldout_seeds vide")
    n_players = int(metrics["n_players"])
    parametric = _stderr(auc, n_players)
    observed = statistics.pstdev(draws) if len(draws) > 1 else 0.0
    stderr = max(parametric, observed)
    margin = auc - stderr - threshold
    passed = sum(1 for draw in draws
                 if draw - _stderr(draw, n_players) - threshold > 0)
    return {
        "role": rf.normalize_role(metrics["role"]),
        # Reporté tel quel : le service refuse une image où l'export embarqué
        # et la marge publiée ne viennent pas du même entraînement.
        "model_id": metrics["model_id"],
        "auc_ebm": round(auc, 4),
        "n_seeds": len(draws),
        "stderr": round(stderr, 4),
        "stderr_parametric": round(parametric, 4),
        "stderr_observed": round(observed, 4),
        # Diagnostic, pas un critère : dit à quelle fréquence la décision
        # aurait basculé si l'on s'était contenté d'un tirage.
        "pass_rate": round(passed / len(draws), 2),
        "margin": round(margin, 4),
        "corpus": corpus,
        # None, jamais zéro : zéro affirmerait un label contemporain des parties.
        "label_age_days": (metrics.get("dataset") or {}).get(
            "label_age_days", {}).get("max"),
        "open": bool(margin > 0 and corpus == "production"),
    }


def main() -> int:
    threshold = float(cli.arg("--threshold", THRESHOLD))
    # Aucun `--corpus` : un drapeau certifierait sans laisser de trace dans un
    # diff, et ferait une seconde source de vérité à côté du fichier.
    table = load_corpus_table(CORPUS_FILE)
    rows = []
    for role in rf.ROLES:
        # Aucun repli sur `utility_player_metrics.json` : ces métriques ne sont
        # plus régénérables et les servir sous le nom SUPPORT publierait l'AUC
        # d'un modèle pour celle d'un autre.
        path = MODEL_DIR / f"{role.lower()}_player_metrics.json"
        if not path.exists():
            print(f"  ⚠ {role} : métriques absentes ({path.name})")
            continue
        metrics = rl.read_json(path)
        # Le rôle du NOM DE FICHIER, pas celui des métriques : c'est le fichier
        # que `make roles` régénère, et donc l'entrée que l'on certifie.
        corpus = resolve_corpus(role, metrics.get("model_id"), table)
        # Des métriques d'avant le held-out répété n'ont pas de médiane.
        # Prendre leur `auc_heldout` pour une servirait un tirage unique sous
        # le nom d'une statistique robuste : on saute le rôle et on le dit.
        try:
            rows.append(readiness(metrics, threshold=threshold, corpus=corpus))
        except KeyError:
            print(f"  ⚠ {role} : métriques d'un seul tirage, relancer "
                  f"train_role_ensemble.py --role {role.lower()}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rl.dump_json_atomic(MODEL_DIR / "role_readiness.json", rows, indent=2)
    print(f"{'rôle':<9}{'AUC méd.':>9}{'bruit':>8}{'marge':>9}"
          f"{'tirages>0':>11}{'âge label':>11}  ouverture")
    for row in sorted(rows, key=lambda item: -item["margin"]):
        age = row["label_age_days"]
        print(f"{row['role']:<9}{row['auc_ebm']:>9.4f}"
              f"{row['stderr']:>8.4f}{row['margin']:>9.4f}"
              f"{f'{row['pass_rate']:.0%} de {row['n_seeds']}':>11}"
              f"{'?' if age is None else f'{age} j':>11}  "
              f"{'oui' if row['open'] else 'non'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
