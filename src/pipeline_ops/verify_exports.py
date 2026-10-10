#!/usr/bin/env python3
"""Vérifie la parité entre les cinq exports servis et leurs pickles.

Le test unitaire valide l'évaluateur sur un modèle synthétique. Ce script
contrôle les artefacts réels avant la construction de l'image de production.

Usage : poetry run python3 src/pipeline_ops/verify_exports.py [--rows 50]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import pandas as pd  # noqa: E402

import cli  # noqa: E402
import ebm_lookup  # noqa: E402
import riotlib as rl  # noqa: E402
import role_features as rf  # noqa: E402

MODEL_DIR = rl.DATA / "05_model"
DATASET_DIR = rl.DATA / "04_dataset"
TOLERANCE = 1e-12


def verify_role(role: str, n_rows: int) -> list[str]:
    """Retourne les écarts constatés pour un rôle."""
    slug = role.lower()
    export = rl.read_json(MODEL_DIR / f"{slug}_ebm_export.json") or {}
    with (MODEL_DIR / f"{slug}_player_ebm.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    ebm, features = bundle["model"], bundle["features"]

    problems = []
    if export["features"] != features:
        problems.append(
            "le manifeste de features de l'export diffère de celui du pickle")
        return problems

    frame = pd.read_parquet(
        DATASET_DIR / f"{slug}_player_dataset.parquet")
    frame = frame.reindex(columns=features).astype(float).head(n_rows)
    reference = ebm.eval_terms(frame)
    for index in range(len(frame)):
        row = frame.iloc[index].to_dict()
        got = ebm_lookup.contributions(export, row)
        for position, name in enumerate(features):
            gap = abs(got[name] - float(reference[index][position]))
            if gap > TOLERANCE:
                problems.append(
                    f"ligne {index}, {name} : écart {gap:.3e}")
    return problems


def main() -> int:
    n_rows = int(cli.arg("--rows", 50))
    failed = False
    for role in rf.ROLES:
        try:
            problems = verify_role(role, n_rows)
        except FileNotFoundError as exc:
            print(f"  ✗ {role} : artefact absent ({exc})")
            failed = True
            continue
        if problems:
            failed = True
            print(f"  ✗ {role} : {len(problems)} écart(s)")
            for line in problems[:5]:
                print(f"      {line}")
        else:
            print(f"  ✓ {role} : parité exacte sur {n_rows} lignes")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
