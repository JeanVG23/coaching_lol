"""Journal structuré d'UNE game depuis match + timeline (0 CV, module pur).

Événements ancrés (timestamp exact) du joueur ciblé : morts et recalls, chacun
avec son contexte — gold-state, gold non dépensé, objectif up/imminent. C'est la
granularité qui manque aux médianes agrégées : le coach peut citer « mort à 17:42
en BOT, drake dans 30 s, 1 450 g non dépensés » au lieu de « tu meurs trop ».

ASYMÉTRIE : tout champ du journal est une information que le joueur AVAIT —
sa propre mort (killer affiché dans le death recap), son propre gold, les timers
d'objectifs (HUD public). Aucun proxy de vision (fog, unaccounted) : ces features
restent ML_ONLY dans positioning.py et ne doivent jamais entrer ici.
"""
from __future__ import annotations

import bisect
from collections import defaultdict

from riotlib import (SR_MAP_ID, approx_zone, enemy_team_of, find_pid,
                     frames_by_minute, iter_events, participant_id, patch_of,
                     phase_of, _gold_state)

# Timers d'objectifs (ms) — v1 approximative, À AJUSTER PAR PATCH.
# Elder (après l'âme) et Atakhan ignorés en v1 : le drake standard porte
# l'essentiel du signal coaching ADC (botside, récurrent).
OBJECTIVES = {
    "DRAGON": {"first": 5 * 60000, "respawn": 5 * 60000},
    "BARON_NASHOR": {"first": 25 * 60000, "respawn": 6 * 60000},
}
IMMINENT_WINDOW_S = 90        # objectif "imminent" si spawn dans <= 90 s
OPENING_BUY_MS = 90000        # achats avant 1:30 = shopping de départ, pas un recall
RECALL_CLUSTER_GAP_MS = 30000 # achats espacés de <= 30 s = même visite de shop

# Fenêtres de conséquences post-mort — approximation v1 : la timeline ne donne
# pas le death timer réel, on attribue à la mort ce que l'ennemi prend juste après.
CONSEQUENCE_WINDOW_S = 60     # objectifs + bâtiments pris dans les 60 s
GOLD_SWING_WINDOW_S = 90      # swing de gold d'équipe mesuré à ~90 s


def _clock(t_ms: int) -> str:
    return f"{t_ms // 60000}:{(t_ms % 60000) // 1000:02d}"


def _events(timeline: dict):
    """Alias historique de `riotlib.iter_events` (conservé : utilisé hors module)."""
    return iter_events(timeline)


def _objective_kills(timeline: dict) -> dict[str, list[int]]:
    kills = {name: [] for name in OBJECTIVES}
    for ev in iter_events(timeline):
        if ev.get("type") == "ELITE_MONSTER_KILL" and ev.get("monsterType") in kills:
            kills[ev["monsterType"]].append(ev["timestamp"])
    return {name: sorted(ts) for name, ts in kills.items()}


def _objective_at(kills: dict[str, list[int]], t_ms: int) -> dict | None:
    """Objectif up ou imminent à t_ms. Up prioritaire, sinon le plus proche."""
    up, imminent = [], []
    for name, cfg in OBJECTIVES.items():
        past = [k for k in kills[name] if k <= t_ms]
        next_spawn = past[-1] + cfg["respawn"] if past else cfg["first"]
        if t_ms >= next_spawn:
            up.append((t_ms - next_spawn, name))
        elif next_spawn - t_ms <= IMMINENT_WINDOW_S * 1000:
            imminent.append((next_spawn - t_ms, name))
    if up:
        delta, name = min(up)
        return {"type": name, "status": "up", "delta_s": round(delta / 1000)}
    if imminent:
        delta, name = min(imminent)
        return {"type": name, "status": "imminent", "delta_s": round(delta / 1000)}
    return None


def _frame_before(timeline: dict, pid: int, t_ms: int) -> dict | None:
    """Dernière participantFrame du joueur pid avec timestamp <= t_ms.

    Conservée pour les appels hors journal ; le journal passe par `_Timeline`, qui
    indexe une fois au lieu de rebalayer les frames à chaque mort/recall.
    """
    best = None
    for fr in timeline["info"]["frames"]:
        if fr["timestamp"] > t_ms:
            break
        pf = fr["participantFrames"].get(str(pid))
        if pf:
            best = pf
    return best


class _Timeline:
    """Index construit UNE fois par game : events triés + frames repérées par temps.

    Avant, chaque mort déclenchait une traversée complète des events (`_consequences`)
    plus deux balayages de frames (`_frame_before`, swing de gold) : ~8-10 traversées
    redondantes par game, multipliées par la taille du lot et par les 2 ADC extraits
    depuis le raw. Objet volontairement court-durée de vie, ne retenant que les
    tableaux d'index (pas une closure sur tout le scope appelant).
    """

    def __init__(self, timeline: dict, pid: int):
        events = sorted(iter_events(timeline), key=lambda ev: ev.get("timestamp", 0))
        self._events = events
        self._event_ts = [ev.get("timestamp", 0) for ev in events]
        frames = timeline["info"]["frames"]
        self._frames = frames
        self._frame_ts = [fr["timestamp"] for fr in frames]
        # Frames du joueur ciblé uniquement : seul pid interrogé par le journal.
        self._my_ts: list[int] = []
        self._my_frames: list[dict] = []
        for fr in frames:
            pf = fr["participantFrames"].get(str(pid))
            if pf:
                self._my_ts.append(fr["timestamp"])
                self._my_frames.append(pf)

    def events_between(self, t0: int, t1: int):
        """Events dans ]t0, t1] — bornes identiques à la fenêtre d'origine."""
        lo = bisect.bisect_right(self._event_ts, t0)
        hi = bisect.bisect_right(self._event_ts, t1)
        return self._events[lo:hi]

    def my_frame_before(self, t_ms: int) -> dict | None:
        i = bisect.bisect_right(self._my_ts, t_ms)
        return self._my_frames[i - 1] if i else None

    def frame_before(self, t_ms: int) -> dict | None:
        i = bisect.bisect_right(self._frame_ts, t_ms)
        return self._frames[i - 1] if i else None

    def frame_at_or_after(self, t_ms: int) -> dict | None:
        i = bisect.bisect_left(self._frame_ts, t_ms)
        return self._frames[i] if i < len(self._frames) else None


def _team_gold_diff(frame: dict, pid_team: dict[int, int], my_team: int) -> int:
    """Écart de gold d'équipe (mon équipe - ennemie) sur une frame."""
    diff = 0
    for pid_str, pf in frame["participantFrames"].items():
        g = pf.get("totalGold", 0)
        diff += g if pid_team.get(int(pid_str)) == my_team else -g
    return diff


def _consequences(tl: _Timeline, t_ms: int, my_team: int,
                  pid_team: dict[int, int]) -> dict:
    """Conséquences mécaniques d'une mort : ce que l'ennemi prend dans la
    fenêtre post-mort + swing de gold d'équipe. ASYMÉTRIE : tout est de l'info
    que le joueur avait (annonces objectif/tour, gold d'équipe au scoreboard).

    ⚠️ Sémantique timeline : BUILDING_KILL.teamId = équipe qui PERD le bâtiment.
    """
    win_end = t_ms + CONSEQUENCE_WINDOW_S * 1000
    objectives, buildings = [], []
    for ev in tl.events_between(t_ms, win_end):
        et = ev.get("timestamp", 0)
        if ev.get("type") == "ELITE_MONSTER_KILL":
            team = ev.get("killerTeamId") or pid_team.get(ev.get("killerId"))
            if team != my_team:
                objectives.append({"type": ev.get("monsterType"),
                                   "clock": _clock(et),
                                   "delta_s": round((et - t_ms) / 1000)})
        elif ev.get("type") == "BUILDING_KILL" and ev.get("teamId") == my_team:
            buildings.append({"type": ev.get("towerType") or ev.get("buildingType"),
                              "lane": ev.get("laneType"),
                              "clock": _clock(et)})
    before = tl.frame_before(t_ms)
    after = tl.frame_at_or_after(t_ms + GOLD_SWING_WINDOW_S * 1000)
    out: dict = {}
    if objectives:
        out["objectives_lost"] = objectives
    if buildings:
        out["buildings_lost"] = buildings
    if before is not None and after is not None:
        out["team_gold_swing_90s"] = (_team_gold_diff(after, pid_team, my_team)
                                      - _team_gold_diff(before, pid_team, my_team))
    return out


def _damage_value(row: dict) -> int:
    """Dégâts toutes mitigations appliquées d'une ligne du death recap."""
    return sum(max(0, int(row.get(key) or 0))
               for key in ("physicalDamage", "magicDamage", "trueDamage"))


def _damage_summary(event: dict) -> dict | None:
    """Résume le death recap sans exposer sa liste verbeuse au LLM.

    Riot fournit deux fenêtres : ``victimDamageReceived`` pour la séquence
    fatale et ``victimTeamfightDamageReceived`` pour le combat entier. Leur
    différence rend visible le poke encaissé avant l'engage qui a rendu la mort
    possible. Sur certains matchs les deux listes sont identiques : on restitue
    alors honnêtement 0 dégât pré-engage au lieu d'inventer une chronologie.
    """
    fatal_rows = event.get("victimDamageReceived") or []
    fight_rows = event.get("victimTeamfightDamageReceived") or fatal_rows
    if not isinstance(fight_rows, list):
        return None

    total = sum(_damage_value(row) for row in fight_rows if isinstance(row, dict))
    if total <= 0:
        return None
    fatal = min(total, sum(_damage_value(row) for row in fatal_rows
                           if isinstance(row, dict)))
    before = max(0, total - fatal)
    basic = sum(_damage_value(row) for row in fight_rows
                if isinstance(row, dict) and row.get("basic") is True)

    sources: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"damage": 0, "basic_damage": 0, "spell_damage": 0})
    fatal_sources: dict[tuple[str, str], int] = defaultdict(int)
    for row in fatal_rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name") or row.get("type") or "source inconnue"
        fatal_sources[(str(name), str(row.get("type") or "OTHER"))] += _damage_value(row)
    for row in fight_rows:
        if not isinstance(row, dict):
            continue
        damage = _damage_value(row)
        if damage <= 0:
            continue
        name = row.get("name") or row.get("type") or "source inconnue"
        source_type = str(row.get("type") or "OTHER")
        source = sources[(str(name), source_type)]
        source["damage"] += damage
        source["basic_damage" if row.get("basic") is True else "spell_damage"] += damage

    top_sources = []
    for (name, source_type), values in sorted(
            sources.items(), key=lambda item: item[1]["damage"], reverse=True)[:3]:
        during_source = min(values["damage"], fatal_sources[(name, source_type)])
        top_sources.append({
            "source": name,
            "type": source_type,
            **values,
            "before_engage_damage": values["damage"] - during_source,
            "during_engage_damage": during_source,
            "share": round(values["damage"] / total, 4),
        })

    return {
        "total_damage": total,
        "before_engage_damage": before,
        "during_engage_damage": fatal,
        "before_engage_share": round(before / total, 4),
        "during_engage_share": round(fatal / total, 4),
        "basic_damage": basic,
        "spell_damage": total - basic,
        "basic_share": round(basic / total, 4),
        "spell_share": round((total - basic) / total, 4),
        "top_sources": top_sources,
    }


def _deaths(ctx: "_GameContext") -> list[dict]:
    """Morts du joueur ciblé, horodatées et contextualisées.

    Les 9 paramètres positionnels d'origine (tables participants, closure de gold,
    kills d'objectifs, équipes) sont devenus les champs d'un seul contexte de game.
    """
    tl, pid = ctx.tl, ctx.pid
    pid_champ, pid_role = ctx.pid_champ, ctx.pid_role
    out = []
    for ev in tl.events_between(-1, ctx.end_ms):
        if ev.get("type") != "CHAMPION_KILL" or ev.get("victimId") != pid:
            continue
        t = ev["timestamp"]
        kpid = ev.get("killerId")
        assisters = ev.get("assistingParticipantIds", [])
        involved = ({kpid} | set(assisters)) - {None}
        pos = ev.get("position", {})
        pf = tl.my_frame_before(t)
        entry = {
            "t_ms": t, "clock": _clock(t),
            "minute": t // 60000, "phase": phase_of(t // 60000),
            "zone": approx_zone(pos.get("x", 0), pos.get("y", 0)),
            "killer_champ": pid_champ.get(kpid, "?"),
            "killer_role": pid_role.get(kpid, "?"),
            "is_solo": len(assisters) == 0,
            "is_ganked_by_jungle": (ctx.enemy_jungle_pid is not None
                                    and ctx.enemy_jungle_pid in involved),
            "gold_state": ctx.gold_state_at(t // 60000),
            "unspent_gold": pf.get("currentGold") if pf else None,
            "level": pf.get("level") if pf else None,
            "objective": _objective_at(ctx.obj_kills, t),
        }
        cons = _consequences(tl, t, ctx.my_team, ctx.pid_team)
        if cons:
            entry["consequences"] = cons
        damage = _damage_summary(ev)
        if damage:
            entry["damage"] = damage
        out.append(entry)
    out.sort(key=lambda d: d["t_ms"])
    return out


def _recalls(tl: "_Timeline", pid: int, obj_kills: dict, end_ms: int) -> list[dict]:
    """Visites de shop (clusters d'achats), hors shopping de départ.

    Approximation v1 : un achat implique la présence au shop (recall ou reset
    après mort — les deux sont des « resets » à coacher). gold_before = currentGold
    de la dernière frame avant la visite (léger plancher, frames espacées de 60 s).
    ITEM_UNDO honoré (retire le dernier achat correspondant) ; ITEM_SOLD ignoré.
    """
    # Un seul balayage des events triés au lieu de deux (achats + undos).
    buys, undos = [], []
    for ev in tl.events_between(-1, end_ms):
        if ev.get("participantId") != pid:
            continue
        et = ev["timestamp"]
        if ev.get("type") == "ITEM_PURCHASED" and et >= OPENING_BUY_MS:
            buys.append((et, ev.get("itemId")))
        elif ev.get("type") == "ITEM_UNDO":
            undos.append((et, ev.get("beforeId")))
    for undo_t, before in undos:
        for i in range(len(buys) - 1, -1, -1):
            if buys[i][1] == before and buys[i][0] <= undo_t:
                del buys[i]
                break
    visits: list[list[tuple[int, int | None]]] = []
    for t, item in buys:
        if visits and t - visits[-1][-1][0] <= RECALL_CLUSTER_GAP_MS:
            visits[-1].append((t, item))
        else:
            visits.append([(t, item)])
    out = []
    for visit in visits:
        t0 = visit[0][0]
        pf = tl.my_frame_before(t0)
        out.append({
            "t_ms": t0, "clock": _clock(t0),
            "minute": t0 // 60000, "phase": phase_of(t0 // 60000),
            "items_bought": len(visit),
            "item_ids": [item for _, item in visit if item is not None],
            "gold_before": pf.get("currentGold") if pf else None,
            "objective": _objective_at(obj_kills, t0),
        })
    return out


def recalls_for(timeline: dict, pid: int, obj_kills: dict | None = None) -> list[dict]:
    """Visites de shop d'un joueur, sans construire tout le contexte de game.

    Point d'entrée des consommateurs hors journal (ex. build_sequence_dataset), qui
    appelaient `_recalls(timeline, pid, obj_kills)` avant l'indexation.
    """
    if obj_kills is None:
        obj_kills = _objective_kills(timeline)
    frames = timeline["info"]["frames"]
    end_ms = (frames[-1]["timestamp"] if frames else 0) + 60000
    return _recalls(_Timeline(timeline, pid), pid, obj_kills, end_ms)


class _GameContext:
    """Tout ce dont le journal a besoin sur une game, résolu une fois.

    Les résolutions de participants (adversaire de lane, jungler ennemi, tables
    pid->champion/rôle/équipe) passent par les primitives de `riotlib` : elles y
    étaient recopiées verbatim, avec un ordre de conditions déjà subtilement
    différent sur la définition de « adversaire de lane ».
    """

    def __init__(self, match: dict, timeline: dict, pid: int):
        info, parts = match["info"], match["info"]["participants"]
        self.pid = pid
        self.me = parts[pid - 1]
        self.my_team = self.me["teamId"]
        self.my_role = self.me.get("teamPosition") or ""
        enemy_team = enemy_team_of(self.my_team)
        self.pid_champ = {i + 1: p["championName"] for i, p in enumerate(parts)}
        self.pid_role = {i + 1: p.get("teamPosition") or "?" for i, p in enumerate(parts)}
        self.pid_team = {i + 1: p["teamId"] for i, p in enumerate(parts)}
        self.opp_pid = (find_pid(match, team=enemy_team, role=self.my_role)
                        if self.my_role else None)
        self.enemy_jungle_pid = find_pid(match, team=enemy_team, role="JUNGLE")
        self.tl = _Timeline(timeline, pid)
        self.obj_kills = _objective_kills(timeline)
        # Borne supérieure des fenêtres d'events : durée de game + une marge de frame.
        self.end_ms = max(info.get("gameDuration", 0) * 1000,
                          timeline["info"]["frames"][-1]["timestamp"]) + 60000
        self._my_fr = frames_by_minute(timeline, pid)
        self._opp_fr = frames_by_minute(timeline, self.opp_pid) if self.opp_pid else {}

    def gold_state_at(self, minute: int) -> str | None:
        """Avance/retard vs adversaire de lane à la frame la plus récente <= minute."""
        for m in range(minute, -1, -1):
            if m in self._my_fr and m in self._opp_fr:
                return _gold_state(self._my_fr[m].get("totalGold", 0)
                                   - self._opp_fr[m].get("totalGold", 0))
        return None


def game_journal(match: dict, timeline: dict, puuid: str) -> dict | None:
    """Une game -> journal d'événements ancrés du joueur. None si hors Faille."""
    info = match["info"]
    if info.get("mapId") != SR_MAP_ID:
        return None
    pid = participant_id(match, puuid)
    if pid is None:
        return None
    ctx = _GameContext(match, timeline, pid)
    me = ctx.me
    return {
        "match_id": match["metadata"]["matchId"],
        "patch": patch_of(info.get("gameVersion", "")),
        "champion": me["championName"],
        "role": ctx.my_role,
        "win": me.get("win"),
        "duration_min": round(info.get("gameDuration", 0) / 60, 1),
        "kda": {"kills": me.get("kills", 0), "deaths": me.get("deaths", 0),
                "assists": me.get("assists", 0)},
        "opponent": ctx.pid_champ.get(ctx.opp_pid),
        "deaths": _deaths(ctx),
        "recalls": _recalls(ctx.tl, ctx.pid, ctx.obj_kills, ctx.end_ms),
    }
