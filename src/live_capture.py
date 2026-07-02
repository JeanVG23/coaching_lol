#!/usr/bin/env python3
"""
live_capture — capture locale de la Live Client Data API (Riot) pendant une game.

Zéro dépendance hors stdlib en mode capture : ce fichier seul est copiable sur
n'importe quel PC où Python est installé, même sans le reste du repo.

Usage :
    python3 live_capture.py                              # capture (Ctrl+C pour annuler)
    python3 live_capture.py --out /chemin                 # capture, sortie dans /chemin
    python3 live_capture.py --match "Riot#Id" euw1        # relie les captures en attente
                                                           # (nécessite le repo complet)
"""
from __future__ import annotations

from datetime import datetime


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def find_matching_game(capture_meta: dict, candidates: list[dict], *,
                        start_tolerance_s: float = 300, duration_tolerance_s: float = 90,
                        warn=lambda msg: None) -> str | None:
    """Fonction pure : relie une capture (meta) à une game candidate (Match-V5).

    capture_meta : {"start", "end", "champion"} (ISO 8601 pour start/end).
    candidates   : liste de {"match_id", "champion", "game_start", "game_duration_s"}.
    """
    capture_start = _parse_iso(capture_meta["start"])
    capture_end = _parse_iso(capture_meta["end"])
    capture_duration = (capture_end - capture_start).total_seconds()
    champion = capture_meta.get("champion")

    qualifying = []
    for c in candidates:
        if champion and champion != "unknown" and c["champion"] != champion:
            continue
        start_diff = abs((_parse_iso(c["game_start"]) - capture_start).total_seconds())
        if start_diff > start_tolerance_s:
            continue
        duration_diff = abs(c["game_duration_s"] - capture_duration)
        if duration_diff > duration_tolerance_s:
            continue
        qualifying.append((start_diff, c["match_id"]))

    if not qualifying:
        return None
    qualifying.sort(key=lambda t: t[0])
    if len(qualifying) > 1:
        warn(f"{len(qualifying)} games candidates dans la tolérance, "
             f"choix du plus proche en heure de début ({qualifying[0][1]})")
    return qualifying[0][1]
