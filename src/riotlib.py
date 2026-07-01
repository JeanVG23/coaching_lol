#!/usr/bin/env python3
"""
riotlib — socle partagé du coach LoL.

Regroupe le client API Riot, les helpers d'extraction (silver) et d'agrégation
(gold), et les chemins de l'architecture médaillon (raw / silver / gold).
Importé par phase1_pull, aggregate_games, build_referential, compare.
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
import time
from pathlib import Path

import requests
import zstandard as zstd

import champion_profiles as cp

# --------------------------------------------------------------------- chemins
# riotlib vit dans src/ ; la racine projet (data/, .env) est le dossier parent.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# Couches médaillon numérotées pour matérialiser l'ordre du pipeline.
RAW_DIR = DATA / "01_raw"       # JSON API brut, immuable, partagé (compressé .json.zst)
SILVER_DIR = DATA / "02_silver" # 1 ligne JSONL = 1 game nettoyée
GOLD_DIR = DATA / "03_gold"     # agrégats prêts conso (benchmarks)

# Niveau de compression zstd du cache raw : 6 = bon ratio + rapide (les timelines
# JSON sont très répétitives, le gain vient surtout de la compression elle-même).
ZSTD_LEVEL = 6
_RAW_EXTS = (".json.zst", ".json.gz", ".json")  # ordre de recherche à la lecture

# ----------------------------------------------------------------- constantes
MAP_W, MAP_H = 14870, 14980     # dimensions de la Faille de l'invocateur
SR_MAP_ID = 11                  # Summoner's Rift (12 = ARAM, etc.)
RANKED_SOLO = "RANKED_SOLO_5x5"
QUEUE_SOLO = 420                # ranked solo/duo
QUEUE_FLEX = 440                # ranked flex

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
            if r.status_code in (500, 502, 503, 504):
                print(f"  Erreur serveur {r.status_code}, nouvelle tentative dans 5s...", file=sys.stderr)
                time.sleep(5)
                continue
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

    def entries_by_puuid(self, puuid: str) -> list[dict]:
        """Rang(s) d'un joueur, un élément par file (solo/flex). [] si unranked."""
        return self._get(self.platform, f"/lol/league/v4/entries/by-puuid/{puuid}") or []


# ----------------------------------------------------------- raw (cache brut)
# Le cache raw est compressé en zstd (.json.zst) pour gagner ~8× de stockage.
# La lecture est tolérante : elle cherche .json.zst, puis .json.gz, puis .json
# brut, de façon à rester lisible pendant/après la migration des fichiers existants.
def _read_raw(base: str) -> dict | None:
    """Lit un document raw par son préfixe (ex: '<matchId>_match')."""
    for ext in _RAW_EXTS:
        p = RAW_DIR / (base + ext)
        if not p.exists():
            continue
        data = p.read_bytes()
        if ext == ".json.zst":
            data = zstd.ZstdDecompressor().decompress(data)
        elif ext == ".json.gz":
            data = gzip.decompress(data)
        return json.loads(data)
    return None


def _write_raw(base: str, obj: dict) -> None:
    """Écrit un document raw compressé en zstd (.json.zst)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    p = RAW_DIR / (base + ".json.zst")
    p.write_bytes(zstd.ZstdCompressor(level=ZSTD_LEVEL).compress(json.dumps(obj).encode()))


def get_match_timeline(client: RiotClient, match_id: str) -> tuple[dict, dict] | None:
    """Charge (match, timeline) depuis raw/ si présents, sinon fetch et cache."""
    match = _read_raw(f"{match_id}_match")
    timeline = _read_raw(f"{match_id}_timeline")
    if match is not None and timeline is not None:
        return match, timeline
    try:
        match = client.match(match_id)
        timeline = client.timeline(match_id)
    except Exception as e:
        print(f"  ⚠ skip {match_id}: {e}", file=sys.stderr)
        return None
    if not match or not timeline:
        return None
    _write_raw(f"{match_id}_match", match)
    _write_raw(f"{match_id}_timeline", timeline)
    return match, timeline


# ---------------------------------------------------------- silver (1 game)
LANE_KEYS = ["gd10", "gd14", "gd20", "csd10", "csd14", "xpd10", "csm10", "csm14", "gpm10", "gpm14", "xppm10"]  # diffs + absolus


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

    my_role, my_team = me.get("teamPosition") or "", me["teamId"]
    enemy_team = 100 if my_team == 200 else 200
    opp_pid = next((i + 1 for i, p in enumerate(parts)
                    if p["teamId"] != my_team and (p.get("teamPosition") or "") == my_role
                    and my_role), None)
    
    enemy_jungle_pid = next((i + 1 for i, p in enumerate(parts)
                             if p["teamId"] != my_team and p.get("teamPosition") == "JUNGLE"), None)
                             
    support_pid = next((i + 1 for i, p in enumerate(parts)
                        if p["teamId"] == my_team and p.get("teamPosition") == "UTILITY"), None)
                        
    enemy_adc_pid = next((i + 1 for i, p in enumerate(parts)
                          if p["teamId"] != my_team and p.get("teamPosition") == "BOTTOM"), None)
    enemy_supp_pid = next((i + 1 for i, p in enumerate(parts)
                           if p["teamId"] != my_team and p.get("teamPosition") == "UTILITY"), None)
    enemy_bot_pids = {enemy_adc_pid, enemy_supp_pid} - {None}

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
        "csm10": _cs(my_fr[10]) / 10.0 if 10 in my_fr else None,
        "csm14": _cs(my_fr[14]) / 14.0 if 14 in my_fr else None,
        "gpm10": my_fr[10].get("totalGold", 0) / 10.0 if 10 in my_fr else None,
        "gpm14": my_fr[14].get("totalGold", 0) / 14.0 if 14 in my_fr else None,
        "xppm10": my_fr[10].get("xp", 0) / 10.0 if 10 in my_fr else None,
    }
    if opp_pid:
        lane["opponent"] = pid_champ[opp_pid]

    def gold_state_at(minute: int) -> str | None:
        for m in range(minute, -1, -1):  # frame la plus récente ≤ minute
            if m in my_fr and m in opp_fr:
                return _gold_state(my_fr[m].get("totalGold", 0) - opp_fr[m].get("totalGold", 0))
        return None

    deaths = []
    kills = []
    assists = []
    support_deaths = []
    dragon_distances = []
    my_plates = 0
    enemy_plates = 0
    frames_in_base = 0
    
    for frame in timeline["info"]["frames"]:
        minute = round(frame["timestamp"] / 60000)
        
        if minute < 14:
            p_frame = frame["participantFrames"].get(str(participant_id))
            if p_frame and "position" in p_frame:
                px, py = p_frame["position"].get("x"), p_frame["position"].get("y")
                if px is not None and py is not None:
                    if my_team == 100 and px < 3500 and py < 3500:
                        frames_in_base += 1
                    elif my_team == 200 and px > 11300 and py > 11300:
                        frames_in_base += 1
                        
        for ev in frame.get("events", []):
            if ev.get("type") == "CHAMPION_KILL":
                ev_minute = round(ev["timestamp"] / 60000)
                kpid = ev.get("killerId")
                assisters = ev.get("assistingParticipantIds", [])
                involved = {kpid}.union(set(assisters)) - {None}
                
                if ev.get("victimId") == participant_id:
                    pos = ev.get("position", {})
                    is_2v2 = len(involved) > 0 and involved.issubset(enemy_bot_pids)
                    
                    deaths.append({
                        "minute": ev_minute,
                        "phase": phase_of(ev_minute),
                        "zone": approx_zone(pos.get("x", 0), pos.get("y", 0)),
                        "killer_role": pid_role.get(kpid, "?"),
                        "killer_champ": pid_champ.get(kpid, "?"),
                        "gold_state": gold_state_at(ev_minute),
                        "is_solo": len(assisters) == 0,
                        "is_ganked_by_jungle": (enemy_jungle_pid is not None) and (enemy_jungle_pid in involved),
                        "is_2v2": is_2v2,
                    })
                elif ev.get("victimId") == support_pid:
                    support_deaths.append(ev_minute)
                
                if kpid == participant_id:
                    my_bot_pids = {participant_id, support_pid} - {None}
                    is_kill_2v2 = len(involved) > 0 and involved.issubset(my_bot_pids) and ev.get("victimId") in enemy_bot_pids
                    kills.append({
                        "minute": ev_minute,
                        "phase": phase_of(ev_minute),
                        "is_solo": len(assisters) == 0,
                        "is_2v2": is_kill_2v2,
                    })
                elif participant_id in assisters:
                    my_bot_pids = {participant_id, support_pid} - {None}
                    is_assist_2v2 = len(involved) > 0 and involved.issubset(my_bot_pids) and ev.get("victimId") in enemy_bot_pids
                    assists.append({
                        "minute": ev_minute,
                        "phase": phase_of(ev_minute),
                        "is_2v2": is_assist_2v2,
                    })
            elif ev.get("type") == "ELITE_MONSTER_KILL" and ev.get("monsterType") == "DRAGON":
                minute_before = max(0, round((ev["timestamp"] - 60000) / 60000))
                p_frame = my_fr.get(minute_before)
                if p_frame and "position" in p_frame and "position" in ev:
                    px, py = p_frame["position"].get("x"), p_frame["position"].get("y")
                    mx, my = ev["position"].get("x"), ev["position"].get("y")
                    if px is not None and py is not None and mx is not None and my is not None:
                        dragon_distances.append(((px - mx)**2 + (py - my)**2)**0.5)
            elif ev.get("type") == "TURRET_PLATE_DESTROYED" and ev.get("laneType") == "BOT_LANE":
                ev_minute = round(ev["timestamp"] / 60000)
                if ev_minute < 14:
                    if ev.get("teamId") == enemy_team:
                        my_plates += 1
                    elif ev.get("teamId") == my_team:
                        enemy_plates += 1

    avg_dragon_prox = round(sum(dragon_distances) / len(dragon_distances)) if dragon_distances else None

    def champ_at(team_is_mine: bool, role: str) -> str | None:
        for i, p in enumerate(parts):
            same = (p["teamId"] == my_team)
            if same == team_is_mine and (p.get("teamPosition") or "") == role:
                return pid_champ[i + 1]
        return None

    comp = {
        "self_adc": me["championName"],
        "self_support": champ_at(True, "UTILITY"),
        "enemy_adc": champ_at(False, "BOTTOM"),
        "enemy_support": champ_at(False, "UTILITY"),
        "self_jungle": champ_at(True, "JUNGLE"),
        "enemy_jungle": champ_at(False, "JUNGLE"),
        "enemy_mid": champ_at(False, "MIDDLE"),
    }

    import positioning  # import paresseux : évite le cycle riotlib<->positioning
    pid_team = {i + 1: p["teamId"] for i, p in enumerate(parts)}
    position = positioning.positioning_features(
        timeline, participant_id, pid_team, my_role or "BOTTOM")

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
        "comp": comp,
        "deaths": deaths,
        "kills": kills,
        "assists": assists,
        "support_deaths_early": sum(1 for m in support_deaths if m < 14),
        "plates_diff_early": my_plates - enemy_plates,
        "frames_in_base_early": frames_in_base,
        "avg_dragon_prox": avg_dragon_prox,
        "position": position,
    }


def extract_all_games(match: dict, timeline: dict, rank: str | None = None) -> list[dict]:
    """Extrait les statistiques pour les 10 joueurs de la partie."""
    meta = match.get("metadata", {})
    puuids = meta.get("participants", [])
    results = []
    for p in puuids:
        g = extract_game(match, timeline, p, rank)
        if g:
            results.append(g)
    return results


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


def _fmedian(vals) -> float | None:
    """Médiane SANS arrondi entier (préserve fractions 0..1, profondeurs, comptes).

    `_median` arrondit à l'entier (ok pour gold/cs diffs) ; appliqué à une fraction
    comme frac_overextended ça l'écrase à 0/1. Les features positionnelles ont des
    échelles mixtes (0..1, ~milliers, comptes) → médiane flottante, arrondie à 4 déc."""
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    m = len(vals) // 2
    med = vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2
    return round(med, 4)


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
    # benchmark positionnement : médiane des features COACHING_SAFE de la timeline.
    # Asymétrie : on n'agrège QUE les features exactes/safe (jamais les proxys ML_ONLY)
    # — la couche de benchmark coaching ne doit pouvoir prescrire que de l'asymétrie-safe.
    import positioning  # import paresseux : évite le cycle riotlib<->positioning
    positioning_med = {k: _fmedian([(g.get("position") or {}).get(k) for g in subset])
                       for k in sorted(positioning.COACHING_SAFE)}
    # contexte économique des morts (avance/retard/égalité)
    gs = collections.Counter(d.get("gold_state") for d in deaths if d.get("gold_state"))
    gs_total = sum(gs.values())
    return {
        "n_games": n,
        "deaths_total": total,
        "deaths_per_game": round(total / n, 2) if n else 0,
        "lane": lane,
        "positioning": positioning_med,
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


def _by_lane_context(subset: list[dict]) -> dict:
    """Facettes par bucket de contexte dérivé (lane_pattern, gank_exposure)."""
    axes = {"lane_pattern": collections.defaultdict(list),
            "gank_exposure": collections.defaultdict(list)}
    for g in subset:
        comp = g.get("comp")
        if not comp:
            continue
        ctx = cp.derive_context(comp)
        for axis, bucket in ctx.items():
            axes[axis][bucket].append(g)
    return {axis: {bucket: _facet(games) for bucket, games in buckets.items()}
            for axis, buckets in axes.items()}


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
        "by_lane_context": _by_lane_context(subset),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_jsonl(path: Path, new_rows: list[dict]) -> list[dict]:
    """Lit l'existant, fusionne en ignorant les doublons (match_id, puuid), et sauvegarde."""
    existing = read_jsonl(path)
    seen = {(r.get("match_id"), r.get("puuid")) for r in existing}
    merged = existing[:]
    for r in new_rows:
        key = (r.get("match_id"), r.get("puuid"))
        if key not in seen:
            merged.append(r)
            seen.add(key)
    write_jsonl(path, merged)
    return merged


def write_gold(base: Path, games: list[dict], scopes: list[str], **labels) -> None:
    """Écrit base/<scope>/aggregate.json pour chaque scope."""
    for scope in scopes:
        agg = aggregate(games, scope, **labels)
        out = base / scope / "aggregate.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(agg, indent=2))
