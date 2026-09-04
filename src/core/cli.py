"""src/core/cli.py — parseur argv minimal partagé par les scripts de collecte.

`arg()` / `flag()` étaient recopiés à l'identique dans 7 scripts (`compare`,
`densify_targets`, `densify_sweet_spot`, `build_referential`, `densify_players`,
`fetch_apex_lp`, `aggregate_games`). Le reste du dépôt utilise `argparse` ; ce
module ne cherche pas à le remplacer, seulement à supprimer la recopie sur les
scripts déjà écrits dans ce style.

Note : `src/collection/live_capture.py` garde sa propre copie — contrainte
assumée « copiable seul sur une machine sans le reste du repo ».
"""
from __future__ import annotations

import sys


def arg(name: str, default=None, argv: list[str] | None = None):
    """Valeur suivant `name` dans argv (`--rank challenger` -> 'challenger')."""
    argv = sys.argv if argv is None else argv
    if name not in argv:
        return default
    i = argv.index(name)
    return argv[i + 1] if i + 1 < len(argv) else default


def flag(name: str, argv: list[str] | None = None) -> bool:
    """Présence d'un drapeau booléen (`--dry-run`)."""
    return name in (sys.argv if argv is None else argv)


def int_arg(name: str, default: int, argv: list[str] | None = None) -> int:
    return int(arg(name, default, argv))


def csv_arg(name: str, default: list[str], argv: list[str] | None = None) -> list[str]:
    """Liste séparée par des virgules (`--rank master,challenger`)."""
    raw = arg(name, None, argv)
    return [p.strip() for p in raw.split(",") if p.strip()] if raw else list(default)
