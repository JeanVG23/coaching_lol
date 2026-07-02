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

import json
import platform
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


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


LIVE_CLIENT_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"
POLL_INTERVAL_S = 2.5
FAIL_THRESHOLD = 5  # échecs consécutifs après le début de capture -> fin de game détectée

_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.check_hostname = False
_SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def _fetch_snapshot(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, context=_SSL_CONTEXT, timeout=3) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError,
            json.JSONDecodeError, OSError):
        return None


def _extract_champion(snapshot: dict) -> str:
    """Best-effort : le schéma Live Client a bougé avec la migration Riot ID.
    Ne lève jamais -> 'unknown' si non identifiable, le matching s'en passe."""
    try:
        active = snapshot.get("activePlayer", {})
        my_name = active.get("riotIdGameName") or active.get("summonerName")
        if not my_name:
            return "unknown"
        for p in snapshot.get("allPlayers", []):
            p_name = p.get("riotIdGameName") or p.get("summonerName")
            if p_name == my_name:
                return p.get("championName", "unknown")
    except (AttributeError, TypeError):
        pass
    return "unknown"


def capture(out_dir: Path, interval: float = POLL_INTERVAL_S,
            fail_threshold: int = FAIL_THRESHOLD,
            url: str = LIVE_CLIENT_URL) -> tuple[Path, Path] | None:
    """Boucle bloquante : attend une game, capture jusqu'à sa fin (ou Ctrl+C).
    Retourne (jsonl_path, meta_path), ou None si rien n'a été capturé."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("En attente d'une game (Live Client Data API)... Ctrl+C pour annuler.")

    start_time = None
    champion = "unknown"
    jsonl_path = None
    meta_path = None
    fh = None
    consecutive_fails = 0

    try:
        while True:
            snapshot = _fetch_snapshot(url)
            if snapshot is None:
                if start_time is not None:
                    consecutive_fails += 1
                    if consecutive_fails >= fail_threshold:
                        break
                time.sleep(interval)
                continue

            consecutive_fails = 0
            if start_time is None:
                start_time = datetime.now(timezone.utc)
                champion = _extract_champion(snapshot)
                stamp = start_time.strftime("%Y%m%dT%H%M%SZ")
                jsonl_path = out_dir / f"{stamp}_{champion}.jsonl"
                meta_path = out_dir / f"{stamp}_{champion}_meta.json"
                fh = jsonl_path.open("a", encoding="utf-8")
                print(f"Game détectée ({champion}) — capture vers {jsonl_path.name}")

            fh.write(json.dumps({"t": datetime.now(timezone.utc).isoformat(),
                                  "data": snapshot}) + "\n")
            fh.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nInterruption manuelle.")
    finally:
        if fh is not None:
            fh.close()

    if start_time is None:
        print("Aucune game détectée, rien capturé.")
        return None

    end_time = datetime.now(timezone.utc)
    meta = {
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "champion": champion,
        "machine": platform.node(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    duration = (end_time - start_time).total_seconds()
    print(f"Capture terminée : {jsonl_path.name} ({duration:.0f}s)")
    return jsonl_path, meta_path
