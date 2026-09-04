"""src/core/ranks.py — source unique des rangs et des cibles binaires ML.

`RANKS`, `HIGH_ELO`, `RANK_ORD` et `APEX` étaient recopiés dans une dizaine de
scripts (collecte, datasets, entraînements, calibrations, rapports). La frontière
de rang est un paramètre de recherche actif (cf. CLAUDE.md : `high_elo` plafonne
à ~0.59 d'AUC, `dia_chall` monte à ~0.72) : la déplacer doit se faire ici, pas en
retrouvant N littéraux. Volontairement stdlib-only pour rester importable par les
scripts de collecte (qui ne dépendent ni de pandas ni de numpy).
"""
from __future__ import annotations

# Ordre croissant de skill. RANK_ORD sert au tie-break "rang le plus bas" quand
# un joueur a des games sur plusieurs rangs (ne pas gonfler high_elo aux frontières).
RANKS = ["diamond", "master", "grandmaster", "challenger"]
RANK_ORD = {r: i for i, r in enumerate(RANKS)}

# Ordre de collecte : du plus rare au plus peuplé (on remplit le haut d'abord).
COLLECT_ORDER = list(reversed(RANKS))

# Tiers apex : LP comparables entre eux (diamond exclu de la régression LP).
APEX = {"master", "grandmaster", "challenger"}

HIGH_ELO = {"grandmaster", "challenger"}

# Cibles binaires de classification. `high_elo` coupe à la frontière Master|GM,
# deux tiers adjacents quasi-indistinguables sur features macro ; `dia_chall`
# retire le milieu et oppose les extrêmes (teste si le signal existe tout court).
TARGETS = {
    "high_elo": {"ranks": RANKS, "pos": HIGH_ELO,
                 "names": ["low(M/D)", "high(GM/C)"]},
    "dia_chall": {"ranks": ["diamond", "challenger"], "pos": {"challenger"},
                  "names": ["diamond", "challenger"]},
}


def lowest(ranks) -> str:
    """Rang le plus bas d'un itérable (tie-break canonique du projet)."""
    return sorted(ranks, key=lambda r: RANK_ORD[r])[0]
