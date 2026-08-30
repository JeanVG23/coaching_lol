# Conséquences des morts (chaîne causale intra-game) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attacher à chaque mort du journal par-game ses conséquences mécaniques (objectifs/tours pris par l'ennemi juste après, swing de gold d'équipe) et forcer le LLM à restituer cette chaîne causale.

**Architecture:** Tout est calculé de façon déterministe dans `src/core/game_journal.py` (module pur, 0 API) depuis la timeline Riot déjà en raw ; le payload par-game (`payload.build_game`) embarque `journal["deaths"]` verbatim (`src/04_coaching/payload.py:229`) donc aucun changement payload ; seule s'ajoute une règle dans `prompt.SYSTEM_GAME`. Le champ `cause` du schéma existe déjà.

**Tech Stack:** Python 3, pytest (`poetry run pytest tests/`), convention flat-import (`import game_journal as J` via `tests/conftest.py`).

**Spec:** `docs/superpowers/specs/2026-07-18-death-consequences-design.md`

## Global Constraints

- Asymétrie : uniquement de l'info que le joueur avait (annonces objectif/tour, scoreboard). Aucun proxy `ML_ONLY` de `positioning.py` (test garde-fou existant `test_journal_never_leaks_ml_only_features`).
- `CONSEQUENCE_WINDOW_S = 60` (objectifs + bâtiments), `GOLD_SWING_WINDOW_S = 90` — constantes en tête de module, documentées comme approximation v1 (pas de death timer réel dans la timeline).
- Sémantique timeline : `teamId` d'un `BUILDING_KILL` = équipe qui PERD le bâtiment ; `ELITE_MONSTER_KILL` porte `killerTeamId` (fallback : `killerId` → team).
- Pas de clé `consequences` sur une mort sans rien dans la fenêtre (pas de bruit payload).
- Commits en français, style conventionnel du repo (`feat(journal): …`).

---

### Task 1: `_consequences` dans `game_journal.py`

**Files:**
- Modify: `src/core/game_journal.py` (constantes en tête ; nouvelle fonction `_consequences` + helper `_team_gold_diff` ; `_deaths` et `game_journal` passent `my_team`/`pid_team`)
- Test: `tests/test_game_journal_consequences.py` (nouveau)

**Interfaces:**
- Consumes: `_events(timeline)`, `_clock(t_ms)` existants dans `game_journal.py`.
- Produces: `_consequences(timeline, t_ms, my_team, pid_team) -> dict` — dict avec clés optionnelles `objectives_lost` (list de `{"type": str, "clock": str, "delta_s": int}`), `buildings_lost` (list de `{"type": str, "lane": str, "clock": str}`), `team_gold_swing_90s` (int). Dict vide si rien. Chaque mort du journal porte `consequences` UNIQUEMENT si non vide. `pid_team` = `{pid: teamId}`.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_game_journal_consequences.py` (mêmes fixtures que `tests/test_game_journal.py`, dupliquées ici car le fichier existant ne les exporte pas — DRY toléré entre fichiers de test du repo) :

```python
import game_journal as J


ME = "puuid-me"

_ROSTER = [
    ("Zeri", "BOTTOM", 100), ("Lulu", "UTILITY", 100), ("Ahri", "MIDDLE", 100),
    ("Garen", "TOP", 100), ("Vi", "JUNGLE", 100),
    ("Jinx", "BOTTOM", 200), ("Thresh", "UTILITY", 200), ("Orianna", "MIDDLE", 200),
    ("Darius", "TOP", 200), ("LeeSin", "JUNGLE", 200),
]


def _match(map_id=11, win=True):
    parts = [{"championName": c, "teamPosition": r, "teamId": t,
              "win": win if t == 100 else not win,
              "kills": 5, "deaths": 3, "assists": 7}
             for c, r, t in _ROSTER]
    return {
        "metadata": {"matchId": "EUW1_42",
                     "participants": [ME] + [f"p{i}" for i in range(2, 11)]},
        "info": {"mapId": map_id, "gameVersion": "16.13.790.6961",
                 "gameDuration": 1800, "participants": parts},
    }


def _frame(t_ms, pframes, events=()):
    return {"timestamp": t_ms, "events": list(events),
            "participantFrames": {str(pid): pf for pid, pf in pframes.items()}}


def _pf(gold_total, gold_current=1234, level=5, x=13000, y=2000):
    return {"totalGold": gold_total, "currentGold": gold_current,
            "level": level, "position": {"x": x, "y": y}}


def _kill(t_ms, victim, killer, assists=(), x=13000, y=2000):
    return {"type": "CHAMPION_KILL", "timestamp": t_ms, "victimId": victim,
            "killerId": killer, "assistingParticipantIds": list(assists),
            "position": {"x": x, "y": y}}


def _monster_kill(t_ms, killer_team, monster="BARON_NASHOR", killer_id=None):
    ev = {"type": "ELITE_MONSTER_KILL", "timestamp": t_ms, "monsterType": monster,
          "killerTeamId": killer_team}
    if killer_id is not None:
        ev["killerId"] = killer_id
    return ev


def _building_kill(t_ms, losing_team, building="TOWER_BUILDING",
                   lane="MID_LANE", tower=None):
    ev = {"type": "BUILDING_KILL", "timestamp": t_ms, "teamId": losing_team,
          "buildingType": building, "laneType": lane}
    if tower is not None:
        ev["towerType"] = tower
    return ev


def _timeline(frames):
    return {"info": {"frames": frames}}


def _basic_timeline(events_by_frame=None, minutes=40, my_gold=400, opp_gold=300):
    events_by_frame = events_by_frame or {}
    frames = []
    for minute in range(minutes + 1):
        t = minute * 60000
        frames.append(_frame(
            t,
            {1: _pf(gold_total=my_gold * minute + 500, level=min(minute + 1, 18)),
             6: _pf(gold_total=opp_gold * minute + 500, level=min(minute + 1, 18))},
            events_by_frame.get(minute, []),
        ))
    return _timeline(frames)


def _death_at(tl):
    return J.game_journal(_match(), tl, ME)["deaths"][0]


def test_objective_lost_within_window():
    # Mort à 26:04, Baron pris par l'ennemi 40 s après -> conséquence.
    tl = _basic_timeline({26: [_kill(1564000, victim=1, killer=6)],
                          27: [_monster_kill(1604000, killer_team=200)]})
    cons = _death_at(tl)["consequences"]
    assert cons["objectives_lost"] == [
        {"type": "BARON_NASHOR", "clock": "26:44", "delta_s": 40}]


def test_objective_outside_window_excluded():
    # Baron pris 70 s après la mort -> hors fenêtre 60 s.
    tl = _basic_timeline({26: [_kill(1564000, victim=1, killer=6)],
                          27: [_monster_kill(1634000, killer_team=200)]})
    assert "objectives_lost" not in _death_at(tl).get("consequences", {})


def test_objective_taken_by_my_team_excluded():
    tl = _basic_timeline({26: [_kill(1564000, victim=1, killer=6)],
                          27: [_monster_kill(1604000, killer_team=100)]})
    assert "objectives_lost" not in _death_at(tl).get("consequences", {})


def test_objective_killer_team_fallback_via_killer_id():
    # Pas de killerTeamId -> attribution via killerId (pid 10 = équipe 200).
    ev = _monster_kill(1604000, killer_team=None, killer_id=10)
    del ev["killerTeamId"]
    tl = _basic_timeline({26: [_kill(1564000, victim=1, killer=6)], 27: [ev]})
    assert _death_at(tl)["consequences"]["objectives_lost"][0]["delta_s"] == 40


def test_building_lost_within_window_mine_only():
    # teamId d'un BUILDING_KILL = équipe qui PERD le bâtiment.
    tl = _basic_timeline({
        26: [_kill(1564000, victim=1, killer=6),
             _building_kill(1590000, losing_team=100, tower="INNER_TURRET"),
             _building_kill(1595000, losing_team=200)],
    })
    cons = _death_at(tl)["consequences"]
    assert cons["buildings_lost"] == [
        {"type": "INNER_TURRET", "lane": "MID_LANE", "clock": "26:30"}]


def test_gold_swing_computed_from_team_frames():
    # Écart (moi - ennemi) : +100/min avant la mort. Frame avant = 26:00,
    # première frame >= mort+90 s = 28:00 -> swing = (28-26) * 100 = +200.
    tl = _basic_timeline({26: [_kill(1564000, victim=1, killer=6)]})
    assert _death_at(tl)["consequences"]["team_gold_swing_90s"] == 200


def test_gold_swing_null_when_no_frame_after():
    # Mort en toute fin de game : aucune frame >= mort+90 s -> pas de swing,
    # et rien d'autre dans la fenêtre -> pas de clé consequences du tout.
    tl = _basic_timeline({40: [_kill(2400000, victim=1, killer=6)]})
    assert "consequences" not in _death_at(tl)


def test_no_consequences_key_when_window_empty():
    tl = _basic_timeline({10: [_kill(600000, victim=1, killer=6)]})
    d = _death_at(tl)
    # Le swing gold existe (frames dispo) donc la clé existe, mais sans events.
    assert "objectives_lost" not in d["consequences"]
    assert "buildings_lost" not in d["consequences"]
    assert d["consequences"]["team_gold_swing_90s"] == 200
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `poetry run pytest tests/test_game_journal_consequences.py -v`
Expected: FAIL — `KeyError: 'consequences'` sur chaque test.

- [ ] **Step 3: Implémenter dans `src/core/game_journal.py`**

Ajouter les constantes après `RECALL_CLUSTER_GAP_MS` (ligne 27) :

```python
# Fenêtres de conséquences post-mort — approximation v1 : la timeline ne donne
# pas le death timer réel, on attribue à la mort ce que l'ennemi prend juste après.
CONSEQUENCE_WINDOW_S = 60     # objectifs + bâtiments pris dans les 60 s
GOLD_SWING_WINDOW_S = 90      # swing de gold d'équipe mesuré à ~90 s
```

Ajouter après `_frame_before` :

```python
def _team_gold_diff(frame: dict, pid_team: dict[int, int], my_team: int) -> int:
    """Écart de gold d'équipe (mon équipe - ennemie) sur une frame."""
    diff = 0
    for pid_str, pf in frame["participantFrames"].items():
        g = pf.get("totalGold", 0)
        diff += g if pid_team.get(int(pid_str)) == my_team else -g
    return diff


def _consequences(timeline: dict, t_ms: int, my_team: int,
                  pid_team: dict[int, int]) -> dict:
    """Conséquences mécaniques d'une mort : ce que l'ennemi prend dans la
    fenêtre post-mort + swing de gold d'équipe. ASYMÉTRIE : tout est de l'info
    que le joueur avait (annonces objectif/tour, gold d'équipe au scoreboard).

    ⚠️ Sémantique timeline : BUILDING_KILL.teamId = équipe qui PERD le bâtiment.
    """
    win_end = t_ms + CONSEQUENCE_WINDOW_S * 1000
    objectives, buildings = [], []
    for ev in _events(timeline):
        et = ev.get("timestamp", 0)
        if not t_ms < et <= win_end:
            continue
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
    before = after = None
    for fr in timeline["info"]["frames"]:
        if fr["timestamp"] <= t_ms:
            before = fr
        elif after is None and fr["timestamp"] >= t_ms + GOLD_SWING_WINDOW_S * 1000:
            after = fr
    out: dict = {}
    if objectives:
        out["objectives_lost"] = objectives
    if buildings:
        out["buildings_lost"] = buildings
    if before is not None and after is not None:
        out["team_gold_swing_90s"] = (_team_gold_diff(after, pid_team, my_team)
                                      - _team_gold_diff(before, pid_team, my_team))
    return out
```

Modifier `_deaths` : signature `def _deaths(timeline, pid, pid_champ, pid_role, enemy_jungle_pid, gold_state_at, obj_kills, my_team, pid_team)` et, après la construction du dict de mort (avant `out.append`), remplacer l'append par :

```python
        entry = {
            "t_ms": t, "clock": _clock(t),
            "minute": t // 60000, "phase": phase_of(t // 60000),
            "zone": approx_zone(pos.get("x", 0), pos.get("y", 0)),
            "killer_champ": pid_champ.get(kpid, "?"),
            "killer_role": pid_role.get(kpid, "?"),
            "is_solo": len(assisters) == 0,
            "is_ganked_by_jungle": (enemy_jungle_pid is not None
                                    and enemy_jungle_pid in involved),
            "gold_state": gold_state_at(t // 60000),
            "unspent_gold": pf.get("currentGold") if pf else None,
            "level": pf.get("level") if pf else None,
            "objective": _objective_at(obj_kills, t),
        }
        cons = _consequences(timeline, t, my_team, pid_team)
        if cons:
            entry["consequences"] = cons
        out.append(entry)
```

Dans `game_journal`, ajouter après `pid_role` :

```python
    pid_team = {i + 1: p["teamId"] for i, p in enumerate(parts)}
```

et passer les nouveaux arguments à l'appel :

```python
        "deaths": _deaths(timeline, pid, pid_champ, pid_role,
                          enemy_jungle_pid, gold_state_at, obj_kills,
                          my_team, pid_team),
```

- [ ] **Step 4: Vérifier que tout passe (nouveaux tests + non-régression)**

Run: `poetry run pytest tests/test_game_journal_consequences.py tests/test_game_journal.py tests/test_coaching_payload_game.py -v`
Expected: PASS partout (les tests existants ne posent pas de mort avec événements dans la fenêtre, sauf swing gold éventuel — s'ils cassent sur une clé `consequences` inattendue, c'est qu'ils font une égalité stricte de dict : adapter le test EXISTANT est interdit sans le lire ; les tests existants d'`test_game_journal.py` font des asserts par champ, pas d'égalité stricte, donc ils passent).

- [ ] **Step 5: Commit**

```bash
git add src/core/game_journal.py tests/test_game_journal_consequences.py
git commit -m "feat(journal): conséquences post-mort (objectifs/tours perdus, swing gold équipe)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Règle chaîne causale dans `SYSTEM_GAME`

**Files:**
- Modify: `src/04_coaching/prompt.py` (règle 2 de `SYSTEM_GAME`)
- Test: `tests/test_coaching_prompt.py` (nouveau test)

**Interfaces:**
- Consumes: le bloc `consequences` produit par Task 1 (clés `objectives_lost`, `buildings_lost`, `team_gold_swing_90s`).
- Produces: rien (prompt statique).

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `tests/test_coaching_prompt.py` :

```python
def test_system_game_requires_consequence_chain():
    s = PR.SYSTEM_GAME
    assert "consequences" in s            # le LLM sait où chercher
    assert "chaîne" in s.lower()          # restituer la chaîne causale
    assert "pendant que tu étais mort" in s.lower()  # formulation prudente
    assert "corrélation" in s.lower()     # fenêtre = corrélation, pas preuve
```

- [ ] **Step 2: Vérifier qu'il échoue**

Run: `poetry run pytest tests/test_coaching_prompt.py::test_system_game_requires_consequence_chain -v`
Expected: FAIL sur `assert "consequences" in s`.

- [ ] **Step 3: Étendre la règle 2 de `SYSTEM_GAME`**

Dans `src/04_coaching/prompt.py`, à la fin de la règle 2 (après « Regroupe les morts similaires en une seule erreur qui cite 2-3 horodatages. »), ajouter :

```
Quand une mort porte un bloc `consequences` (objectifs/tours pris par l'ennemi \
juste après ta mort, `team_gold_swing_90s`), RESTITUE la CHAÎNE causale complète \
dans la cause et l'evidence : « mort à 26:04 → Baron perdu 40 s après, -1 840 g \
d'écart d'équipe en 90 s » — c'est le COÛT réel de la mort, pas juste l'événement. \
Formule prudemment : « pendant que tu étais mort / juste après ta mort, l'ennemi a \
pris X » — la fenêtre est une corrélation temporelle forte, pas une preuve absolue, \
et n'invente jamais de lien absent du journal.
```

(l'insérer dans la chaîne Python avec les continuations `\` du style existant).

- [ ] **Step 4: Vérifier que tout passe**

Run: `poetry run pytest tests/test_coaching_prompt.py -v`
Expected: PASS (le nouveau + les 8 existants).

- [ ] **Step 5: Commit**

```bash
git add src/04_coaching/prompt.py tests/test_coaching_prompt.py
git commit -m "feat(coaching): SYSTEM_GAME restitue la chaîne causale des consequences

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Validation bout en bout + doc

**Files:**
- Modify: `CLAUDE.md` (section `game_journal.py` + état d'avancement)

**Interfaces:**
- Consumes: Task 1 + Task 2 committées.
- Produces: review par-game régénérée avec chaîne causale visible.

- [ ] **Step 1: Suite complète de tests**

Run: `poetry run pytest tests/`
Expected: PASS (aucune régression).

- [ ] **Step 2: Vérifier le journal enrichi sur une game réelle (0 LLM)**

```bash
poetry run python3 -c "
import sys; sys.path.insert(0, 'src/core'); sys.path.insert(0, 'src/04_coaching')
import payload
p = payload.build_game('spadzze', 'EUW1_7900379457', scope='adc')
import json
for d in p['journal']['deaths']:
    print(d['clock'], d.get('consequences'))
"
```

Expected: la mort à 26:04 (ou une autre) porte un bloc `consequences` non vide (cette game a des morts pré-Baron/pré-drake). Si la signature de `build_game` diffère, lire `src/04_coaching/payload.py:151-230` et adapter l'appel (le smoke test est l'objectif, pas la commande exacte).

- [ ] **Step 3: Review LLM bout en bout**

Run: `poetry run python3 src/04_coaching/coach.py --player spadzze --scope adc --game EUW1_7900379457`
Expected: la sortie JSON validée contient au moins une `cause`/`evidence` qui restitue la chaîne (« pendant que tu étais mort… », coût gold ou objectif/tour perdu). Nécessite `OLLAMA_API_KEY` ; si absent, le signaler et s'arrêter là (Tasks 1-2 restent valides).

- [ ] **Step 4: Mettre à jour `CLAUDE.md`**

Dans la section `game_journal.py`, après « **Asymétrie** : uniquement de l'info que le joueur avait — aucun proxy ML_ONLY. », ajouter :

```
**Conséquences post-mort** (chaîne causale, 2026-07-18) : chaque mort porte un bloc
`consequences` calculé mécaniquement — objectifs (`ELITE_MONSTER_KILL` ennemi) et
bâtiments (`BUILDING_KILL`, ⚠️ `teamId` = équipe qui PERD) pris dans les
`CONSEQUENCE_WINDOW_S=60` s post-mort, + `team_gold_swing_90s` (écart de gold
d'équipe avant vs ~90 s après). Clé omise si fenêtre vide. `SYSTEM_GAME` impose de
restituer la chaîne (« mort → Baron perdu → -1 840 g ») en formulation corrélationnelle.
```

- [ ] **Step 5: Commit final**

```bash
git add CLAUDE.md
git commit -m "docs: conséquences post-mort dans game_journal (chaîne causale coaching)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
