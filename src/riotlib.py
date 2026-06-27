#!/usr/bin/env python3
"""
riotlib — socle partagé du coach LoL.

Regroupe le client API Riot, les helpers d'extraction (silver) et d'agrégation
(gold), et les chemins de l'architecture médaillon (raw / silver / gold).
Importé par phase1_pull, aggregate_games, build_referential, compare.
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import requests

# --------------------------------------------------------------------- chemins
# riotlib vit dans src/ ; la racine projet (data/, .env) est le dossier parent.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# Couches médaillon numérotées pour matérialiser l'ordre du pipeline.
RAW_DIR = DATA / "01_raw"       # JSON API brut, immuable, partagé
SILVER_DIR = DATA / "02_silver" # 1 ligne JSONL = 1 game nettoyée
GOLD_DIR = DATA / "03_gold"     # agrégats prêts conso (benchmarks)

# ----------------------------------------------------------------- constantes
MAP_W, MAP_H = 14870, 14980     # dimensions de la Faille de l'invocateur
SR_MAP_ID = 11                  # Summoner's Rift (12 = ARAM, etc.)
RANKED_SOLO = "RANKED_SOLO_5x5"
QUEUE_SOLO = 420                # ranked solo/duo

PHASES = [("early", 0, 14), ("mid", 15, 24), ("late", 25, 999)]

# rôle/champion → filtre de scope (gold)
ROLE_SCOPES = {
    "all": None, "top": "TOP", "jungle": "JUNGLE",
    "mid": "MIDDLE", "adc": "BOTTOM", "support": "UTILITY",
}

PLATFORM_TO_REGIONAL = {
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe", "me1": "europe",
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "ph2": "sea", "sg2": "sea", "th2": "sea", "tw2": "sea", "vn2": "sea",
}


# -------------------------------------------------------------------- helpers
def load_env(path: Path = ROOT / ".env") -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def patch_of(game_version: str) -> str:
    """'16.13.790.6961' -> '16.13'."""
    parts = (game_version or "").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else game_version


def phase_of(minute: int) -> str:
    for name, lo, hi in PHASES:
        if lo <= minute <= hi:
            return name
    return "late"


def approx_zone(x: int, y: int) -> str:
    """Classification grossière d'une position en lane/zone (PoC)."""
    d_mid = abs(x - y) / (2 ** 0.5)      # distance à la diagonale (mid)
    d_top = min(x, MAP_H - y)             # bord gauche / haut
    d_bot = min(y, MAP_W - x)             # bord bas / droit
    lanes = {"MID": d_mid, "TOP": d_top, "BOT": d_bot}
    lane = min(lanes, key=lanes.get)
    return "JUNGLE/RIVER" if lanes[lane] > 2000 else lane


# --------------------------------------------------------------------- client
class RiotClient:
    """Client API Riot. Routing régional (account/match) vs plateforme (league)."""

    def __init__(self, api_key: str, regional: str, platform: str,
                 min_interval: float = 1.3):
        self.session = requests.Session()
        self.session.headers["X-Riot-Token"] = api_key
        self.regional = regional
        self.platform = platform
        # Espacement régulier : ~1.3s/appel reste sous 100 req/2min (tier dev),
        # ce qui évite les stalls 429 de 77-108s. Surchargeable si clé production.
        self.min_interval = min_interval
        self._last = 0.0

    def _throttle(self):
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _get(self, base: str, path: str, **params):
        url = f"https://{base}.api.riotgames.com{path}"
        for _ in range(6):
            self._throttle()
            r = self.session.get(url, params=params, timeout=20)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "2"))
                print(f"  429, attente {wait}s…", file=sys.stderr)
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Échec après retries: {url}")

    # account-v1 (régional)
    def puuid_from_riot_id(self, game_name: str, tag_line: str) -> str | None:
        d = self._get(self.regional,
                      f"/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}")
        return d["puuid"] if d else None

    # match-v5 (régional)
    def match_ids(self, puuid: str, count: int = 20, queue: int | None = None,
                  start: int = 0) -> list[str]:
        params = {"count": count, "start": start}
        if queue is not None:
            params["queue"] = queue
        return self._get(self.regional,
                         f"/lol/match/v5/matches/by-puuid/{puuid}/ids", **params) or []

    def match(self, match_id: str) -> dict | None:
        return self._get(self.regional, f"/lol/match/v5/matches/{match_id}")

    def timeline(self, match_id: str) -> dict | None:
        return self._get(self.regional, f"/lol/match/v5/matches/{match_id}/timeline")

    # league-v4 / league-exp-v4 (plateforme)
    def apex_league(self, tier: str, queue: str = RANKED_SOLO) -> list[dict]:
        """tier ∈ {challenger, grandmaster, master}."""
        d = self._get(self.platform,
                      f"/lol/league/v4/{tier}leagues/by-queue/{queue}")
        return d.get("entries", []) if d else []

    def league_exp_entries(self, tier: str, division: str, page: int = 1,
                           queue: str = RANKED_SOLO) -> list[dict]:
        return self._get(self.platform,
                         f"/lol/league-exp/v4/entries/{queue}/{tier}/{division}",
                         page=page) or []


# ----------------------------------------------------------- raw (cache brut)
def get_match_timeline(client: RiotClient, match_id: str) -> tuple[dict, dict] | None:
    """Charge (match, timeline) depuis raw/ si présents, sinon fetch et cache."""
    mp = RAW_DIR / f"{match_id}_match.json"
    tp = RAW_DIR / f"{match_id}_timeline.json"
    if mp.exists() and tp.exists():
        return json.loads(mp.read_text()), json.loads(tp.read_text())
    try:
        match = client.match(match_id)
        timeline = client.timeline(match_id)
    except Exception as e:
        print(f"  ⚠ skip {match_id}: {e}", file=sys.stderr)
        return None
    if not match or not timeline:
        return None
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(match))
    tp.write_text(json.dumps(timeline))
    return match, timeline


# ---------------------------------------------------------- silver (1 game)
LANE_KEYS = ["gd10", "gd14", "gd20", "csd10", "csd14", "xpd10"]  # diffs vs adversaire de lane


def _cs(pf: dict) -> int:
    return pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0)


def _frames_by_minute(timeline: dict, pid: int) -> dict[int, dict]:
    out = {}
    for fr in timeline["info"]["frames"]:
        pf = fr["participantFrames"].get(str(pid))
        if pf:
            out[round(fr["timestamp"] / 60000)] = pf
    return out


def _gold_state(gd: int | None) -> str | None:
    """Avance/retard/égalité économique vs adversaire de lane (seuil ±300g)."""
    if gd is None:
        return None
    return "ahead" if gd > 300 else "behind" if gd < -300 else "even"


def extract_game(match: dict, timeline: dict, puuid: str,
                 rank: str | None = None) -> dict | None:
    """Une game -> record silver (morts + benchmark de lane). None si hors Faille."""
    info = match["info"]
    if info.get("mapId") != SR_MAP_ID:
        return None
    meta = match["metadata"]
    if puuid not in meta["participants"]:
        return None
    pidx = meta["participants"].index(puuid)
    participant_id = pidx + 1
    parts = info["participants"]
    me = parts[pidx]

    pid_role = {i + 1: p.get("teamPosition") or "?" for i, p in enumerate(parts)}
    pid_champ = {i + 1: p["championName"] for i, p in enumerate(parts)}

    # Adversaire de lane = ennemi de même teamPosition
    my_role, my_team = me.get("teamPosition") or "", me["teamId"]
    opp_pid = next((i + 1 for i, p in enumerate(parts)
                    if p["teamId"] != my_team and (p.get("teamPosition") or "") == my_role
                    and my_role), None)

    my_fr = _frames_by_minute(timeline, participant_id)
    opp_fr = _frames_by_minute(timeline, opp_pid) if opp_pid else {}

    def gold_diff_at(minute: int) -> int | None:
        a, b = my_fr.get(minute), opp_fr.get(minute)
        return (a.get("totalGold", 0) - b.get("totalGold", 0)) if a and b else None

    # Snapshots de lane aux minutes clés
    lane = {
        "gd10": gold_diff_at(10), "gd14": gold_diff_at(14), "gd20": gold_diff_at(20),
        "csd10": (_cs(my_fr[10]) - _cs(opp_fr[10])) if 10 in my_fr and 10 in opp_fr else None,
        "csd14": (_cs(my_fr[14]) - _cs(opp_fr[14])) if 14 in my_fr and 14 in opp_fr else None,
        "xpd10": (my_fr[10].get("xp", 0) - opp_fr[10].get("xp", 0)) if 10 in my_fr and 10 in opp_fr else None,
    }
    if opp_pid:
        lane["opponent"] = pid_champ[opp_pid]

    def gold_state_at(minute: int) -> str | None:
        for m in range(minute, -1, -1):  # frame la plus récente ≤ minute
            if m in my_fr and m in opp_fr:
                return _gold_state(my_fr[m].get("totalGold", 0) - opp_fr[m].get("totalGold", 0))
        return None

    deaths = []
    for frame in timeline["info"]["frames"]:
        for ev in frame.get("events", []):
            if ev.get("type") == "CHAMPION_KILL" and ev.get("victimId") == participant_id:
                pos = ev.get("position", {})
                minute = round(ev["timestamp"] / 60000)
                kpid = ev.get("killerId")
                deaths.append({
                    "minute": minute,
                    "phase": phase_of(minute),
                    "zone": approx_zone(pos.get("x", 0), pos.get("y", 0)),
                    "killer_role": pid_role.get(kpid, "?"),
                    "killer_champ": pid_champ.get(kpid, "?"),
                    "gold_state": gold_state_at(minute),
                })
    return {
        "match_id": meta["matchId"],
        "puuid": puuid,                          # pour ré-extraction depuis raw sans API
        "rank": rank,
        "patch": patch_of(info.get("gameVersion", "")),
        "champion": me["championName"],
        "role": my_role or "?",
        "win": me["win"],
        "queue": info.get("queueId"),
        "lane": lane,
        "deaths": deaths,
    }


# ------------------------------------------------------------- gold (agrégat)
def filter_scope(games: list[dict], scope: str) -> list[dict]:
    """all / role (adc, mid…) / nom de champion (zeri…)."""
    s = scope.lower()
    if s == "all":
        return games
    if s in ROLE_SCOPES:
        role = ROLE_SCOPES[s]
        return [g for g in games if g["role"] == role]
    return [g for g in games if g["champion"].lower() == s]


def _norm(counter: collections.Counter, total: int) -> dict:
    return {k: round(v / total, 4) for k, v in counter.most_common()} if total else {}


def _median(vals) -> int | None:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    m = len(vals) // 2
    return vals[m] if len(vals) % 2 else round((vals[m - 1] + vals[m]) / 2)


def _facet(subset: list[dict]) -> dict:
    """Bloc de métriques pour un sous-ensemble de games (à issue fixée ou non)."""
    deaths = [d for g in subset for d in g["deaths"]]
    n, total = len(subset), len(deaths)
    by_zone = collections.Counter(d["zone"] for d in deaths)
    by_phase = collections.Counter(d["phase"] for d in deaths)
    by_killer = collections.Counter(d["killer_role"] for d in deaths)
    by_zone_phase = collections.Counter(f'{d["zone"]}|{d["phase"]}' for d in deaths)
    # benchmark de lane : médiane des diffs vs adversaire (robuste aux outliers)
    lane = {k: _median([g.get("lane", {}).get(k) for g in subset]) for k in LANE_KEYS}
    # contexte économique des morts (avance/retard/égalité)
    gs = collections.Counter(d.get("gold_state") for d in deaths if d.get("gold_state"))
    gs_total = sum(gs.values())
    return {
        "n_games": n,
        "deaths_total": total,
        "deaths_per_game": round(total / n, 2) if n else 0,
        "lane": lane,
        "death_gold_state": _norm(gs, gs_total),
        "by_zone": _norm(by_zone, total),
        "by_phase": _norm(by_phase, total),
        "by_killer_role": _norm(by_killer, total),
        "by_zone_phase": _norm(by_zone_phase, total),
        "raw_counts": {
            "by_zone": dict(by_zone), "by_phase": dict(by_phase),
            "by_killer_role": dict(by_killer), "by_zone_phase": dict(by_zone_phase),
        },
    }


def aggregate(games: list[dict], scope: str, **labels) -> dict:
    """Agrège en record gold, à facettes par issue (overall / win / loss).

    L'issue de game est un confondant majeur (en win on meurt moins) : on calcule
    donc des facettes séparées pour comparer à issue égale (tes loses vs leurs loses).
    """
    subset = filter_scope(games, scope)
    wins = [g for g in subset if g["win"]]
    losses = [g for g in subset if not g["win"]]
    return {
        "scope": scope,
        **labels,
        "n_games": len(subset),
        "winrate": round(len(wins) / len(subset), 3) if subset else 0,
        "overall": _facet(subset),
        "win": _facet(wins),
        "loss": _facet(losses),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_gold(base: Path, games: list[dict], scopes: list[str], **labels) -> None:
    """Écrit base/<scope>/aggregate.json pour chaque scope."""
    for scope in scopes:
        agg = aggregate(games, scope, **labels)
        out = base / scope / "aggregate.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(agg, indent=2))
