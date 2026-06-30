# Macro-Positionnement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter une couche de features macro-positionnement (17 features) dérivées de la timeline Riot (0 CV), alimentant le classifieur EBM `dia_chall` ET le coaching benchmarké.

**Architecture:** Module pur `src/positioning.py` avec une fonction orchestratrice `positioning_features` et des helpers testables par famille. `extract_game` (riotlib) l'appelle en import paresseux et niche le résultat sous `record["position"]`. `build_dataset` aplatit ces clés en colonnes ML.

**Tech Stack:** Python 3.14, pytest, pandas, riotlib (helpers `approx_zone`/`phase_of`, constantes `MAP_W`/`MAP_H`).

## Global Constraints

- **0 appel API** : tout dérive de la timeline déjà cachée en raw.
- **Module pur** : `positioning.py` n'a pas d'I/O ; il reçoit `timeline`/`pid_team`/`participant_id`/`my_role` et retourne un dict plat `{str: float|int|None}`. `None` si non calculable → laissé NaN.
- **Import paresseux** : `riotlib.extract_game` fait `import positioning` DANS le corps de la fonction (pas au top du module) pour éviter le cycle `riotlib → positioning → riotlib`.
- **Garde-fou asymétrie** : `positioning.py` exporte `COACHING_SAFE` et `ML_ONLY` ; `ML_ONLY ∩ COACHING_SAFE == ∅`. Les 3 proxys vision ∈ `ML_ONLY`.
- **Constantes** : `SIGHT = 1350.0`, `OVEREXT_THRESHOLD = 2000.0`. Base = box existante (`team 100 : x<3500 & y<3500` ; `team 200 : x>11300 & y>11300`), cohérent avec `frames_in_base` actuel.
- **Tests** : `.venv/bin/python -m pytest tests/` doit rester vert (23 tests actuels + nouveaux).

**Catalogue (17) :** A — `frac_own_lane_early`, `frac_river_early`, `frac_roam_mid`, `frac_enemy_half`, `frac_base`. B — `avg_map_depth`, `max_map_depth`, `frac_overextended`, `avg_dist_to_ally`, `gold_dead_time`. C exact — `wards_placed`, `wards_placed_early`, `control_wards_placed`, `wards_killed`. C proxy (ML_ONLY) — `frac_deaths_in_fog`, `avg_unaccounted_enemies`, `overext_x_unaccounted`.

---

### Task 1: Scaffold du module — constantes, manifeste, `_build_snaps`

**Files:**
- Create: `src/positioning.py`
- Test: `tests/test_positioning.py`

**Interfaces:**
- Produces: `SIGHT`, `OVEREXT_THRESHOLD`, `COACHING_SAFE: set[str]`, `ML_ONLY: set[str]`, `ALL_FEATURES: set[str]`, `_build_snaps(timeline: dict) -> list[tuple[int,int,dict,dict]]` (chaque snap = `(t_ms, minute, {pid:(x,y)}, {pid:level})`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_positioning.py
import positioning as P


def _tl(frames):
    """Timeline minimale : frames = list de (t_ms, {pid: (x,y,level)})."""
    out = []
    for t, parts in frames:
        pf = {str(pid): {"position": {"x": x, "y": y}, "level": lvl}
              for pid, (x, y, lvl) in parts.items()}
        out.append({"timestamp": t, "participantFrames": pf, "events": []})
    return {"info": {"frames": out}}


def test_manifest_disjoint_and_proxies_ml_only():
    assert P.ML_ONLY & P.COACHING_SAFE == set()
    assert {"frac_deaths_in_fog", "avg_unaccounted_enemies",
            "overext_x_unaccounted"} <= P.ML_ONLY
    assert P.ALL_FEATURES == P.COACHING_SAFE | P.ML_ONLY
    assert len(P.ALL_FEATURES) == 17


def test_build_snaps_parses_positions_and_levels():
    tl = _tl([(0, {1: (100, 200, 1)}), (60000, {1: (300, 400, 3)})])
    snaps = P._build_snaps(tl)
    assert len(snaps) == 2
    assert snaps[0] == (0, 0, {1: (100, 200)}, {1: 1})
    assert snaps[1] == (60000, 1, {1: (300, 400)}, {1: 3})


def test_build_snaps_skips_missing_position():
    tl = {"info": {"frames": [{"timestamp": 0, "events": [],
          "participantFrames": {"1": {"level": 2}}}]}}
    snaps = P._build_snaps(tl)
    assert snaps[0][2] == {}          # pas de position
    assert snaps[0][3] == {1: 2}      # level présent
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'positioning'`)

- [ ] **Step 3: Write the scaffold**

```python
# src/positioning.py
"""Features macro-positionnement depuis la timeline Riot (0 CV, module pur).

extract_game (riotlib) appelle positioning_features en import PARESSEUX et niche
le retour sous record["position"]. Voir docs/superpowers/specs/2026-06-30-macro-positioning-design.md.
"""
from __future__ import annotations

from riotlib import approx_zone, phase_of, MAP_W, MAP_H

SIGHT = 1350.0                 # portée de vue d'un champion (proxy vision)
OVEREXT_THRESHOLD = 2000.0     # profondeur en terrain ennemi = "over-extended"
_MAP_MID = (MAP_W + MAP_H) / 2.0
_SQRT2 = 2 ** 0.5

# Base respawn wait (s) par niveau 1..18 (patch 16.x). v1 : facteur temps late-game
# ignoré (négligeable <30 min, sous-estimation conservatrice) -> raffinement v2.
_BRW = {1: 10, 2: 10, 3: 12, 4: 12, 5: 14, 6: 16, 7: 20, 8: 25, 9: 28, 10: 32.5,
        11: 35, 12: 37.5, 13: 40, 14: 42.5, 15: 45, 16: 47.5, 17: 50, 18: 52.5}

# Zone "lane" attendue par rôle (pour frac_own_lane / roam).
_ROLE_ZONE = {"TOP": "TOP", "MIDDLE": "MID", "BOTTOM": "BOT",
              "UTILITY": "BOT", "JUNGLE": "JUNGLE/RIVER"}

COACHING_SAFE = {
    "frac_own_lane_early", "frac_river_early", "frac_roam_mid", "frac_enemy_half",
    "frac_base", "avg_map_depth", "max_map_depth", "frac_overextended",
    "avg_dist_to_ally", "gold_dead_time",
    "wards_placed", "wards_placed_early", "control_wards_placed", "wards_killed",
}
ML_ONLY = {"frac_deaths_in_fog", "avg_unaccounted_enemies", "overext_x_unaccounted"}
ALL_FEATURES = COACHING_SAFE | ML_ONLY


def _build_snaps(timeline: dict) -> list[tuple[int, int, dict, dict]]:
    """Une passe : [(t_ms, minute, {pid:(x,y)}, {pid:level})] pour les 10 joueurs."""
    snaps = []
    for fr in timeline["info"]["frames"]:
        t = fr["timestamp"]
        pos, lvl = {}, {}
        for pid_s, pf in fr["participantFrames"].items():
            pid = int(pid_s)
            p = pf.get("position")
            if p and p.get("x") is not None and p.get("y") is not None:
                pos[pid] = (p["x"], p["y"])
            lvl[pid] = pf.get("level", 1)
        snaps.append((t, round(t / 60000), pos, lvl))
    return snaps
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/positioning.py tests/test_positioning.py
git commit -m "feat(positioning): module scaffold + asymmetry manifest + _build_snaps"
```

---

### Task 2: Famille A — présence carte & roam

**Files:**
- Modify: `src/positioning.py`
- Test: `tests/test_positioning.py`

**Interfaces:**
- Consumes: `_build_snaps` output, `approx_zone`, `phase_of`, `_ROLE_ZONE`.
- Produces: `_zone_presence(snaps, pid: int, my_role: str) -> dict` avec clés `frac_own_lane_early`, `frac_river_early`, `frac_roam_mid`.

- [ ] **Step 1: Write the failing tests**

```python
def test_zone_presence_own_lane_and_river():
    # joueur ADC (BOT) : early en bot (coin bas-droit) puis en river
    # BOT ≈ x grand & y petit ; RIVER ≈ près de la diagonale, loin des bords.
    snaps = [
        (0, 0, {1: (13000, 2000)}, {}),      # early, BOT (coin bas-droit)
        (60000, 1, {1: (13000, 2000)}, {}),  # early, BOT
        (120000, 2, {1: (10000, 5000)}, {}), # early, JUNGLE/RIVER (loin des 3 lanes)
    ]
    r = P._zone_presence(snaps, 1, "BOTTOM")
    assert r["frac_own_lane_early"] == 2 / 3
    assert r["frac_river_early"] == 1 / 3


def test_zone_presence_roam_mid_counts_other_lanes():
    # mid phase (minute 15+) : 1 frame en MID (roam hors BOT), 1 en BOT
    snaps = [
        (900000, 15, {1: (7400, 7400)}, {}),   # MID (sur diagonale)
        (960000, 16, {1: (13000, 2000)}, {}),  # BOT (own lane)
    ]
    r = P._zone_presence(snaps, 1, "BOTTOM")
    assert r["frac_roam_mid"] == 1 / 2


def test_zone_presence_none_when_no_frames():
    assert P._zone_presence([], 1, "BOTTOM")["frac_own_lane_early"] is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k zone_presence -v`
Expected: FAIL (`AttributeError: module 'positioning' has no attribute '_zone_presence'`)

- [ ] **Step 3: Implement `_zone_presence`**

```python
def _zone_presence(snaps: list, pid: int, my_role: str) -> dict:
    own = _ROLE_ZONE.get(my_role, "BOT")
    e_own = e_river = e_tot = m_roam = m_tot = 0
    for _t, minute, pos, _lvl in snaps:
        if pid not in pos:
            continue
        z = approx_zone(*pos[pid])
        ph = phase_of(minute)
        if ph == "early":
            e_tot += 1
            if z == own:
                e_own += 1
            if z == "JUNGLE/RIVER":
                e_river += 1
        elif ph == "mid":
            m_tot += 1
            # roam = autre lane que la sienne (pas river/jungle, pas sa lane)
            if z in ("TOP", "MID", "BOT") and z != own:
                m_roam += 1
    return {
        "frac_own_lane_early": e_own / e_tot if e_tot else None,
        "frac_river_early": e_river / e_tot if e_tot else None,
        "frac_roam_mid": m_roam / m_tot if m_tot else None,
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k zone_presence -v`
Expected: PASS (3 tests). Si les zones attendues ne matchent pas `approx_zone`, ajuster les coordonnées de test (pas la prod) pour cibler la zone voulue.

- [ ] **Step 5: Commit**

```bash
git add src/positioning.py tests/test_positioning.py
git commit -m "feat(positioning): family A presence & roam (_zone_presence)"
```

---

### Task 3: Famille B — territoire & over-extension

**Files:**
- Modify: `src/positioning.py`
- Test: `tests/test_positioning.py`

**Interfaces:**
- Produces: `_territory(snaps, pid: int, my_team: int) -> dict` avec clés `frac_enemy_half`, `avg_map_depth`, `max_map_depth`, `frac_overextended`. Helper interne `_depth(x, y, my_team) -> float` (>0 = terrain ennemi).

- [ ] **Step 1: Write the failing tests**

```python
def test_depth_sign_and_symmetry():
    # team 100 : base bas-gauche (petit x+y) -> profondeur négative chez soi,
    # positive en terrain ennemi (grand x+y). team 200 : inverse.
    deep_enemy_for_100 = P._depth(13000, 13000, 100)
    assert deep_enemy_for_100 > 0
    assert P._depth(1000, 1000, 100) < 0
    assert P._depth(13000, 13000, 200) < 0          # même point, chez soi pour 200
    assert abs(P._depth(13000, 13000, 100) + P._depth(13000, 13000, 200)) < 1e-6


def test_territory_aggregates():
    # team 100 : 1 frame chez soi (depth<0), 1 frame deep enemy (depth>seuil)
    snaps = [
        (0, 0, {1: (1000, 1000)}, {}),
        (60000, 1, {1: (13000, 13000)}, {}),
    ]
    r = P._territory(snaps, 1, 100)
    assert r["frac_enemy_half"] == 0.5
    assert r["max_map_depth"] == P._depth(13000, 13000, 100)
    assert r["avg_map_depth"] == P._depth(13000, 13000, 100) / 2  # chez soi clampé à 0
    assert r["frac_overextended"] == 0.5
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k "depth or territory" -v`
Expected: FAIL (`_depth` / `_territory` introuvables)

- [ ] **Step 3: Implement `_depth` and `_territory`**

```python
def _depth(x: float, y: float, my_team: int) -> float:
    """Profondeur signée dans le terrain ennemi (>0 = chez l'ennemi)."""
    raw = (x + y - _MAP_MID) / _SQRT2
    return raw if my_team == 100 else -raw


def _territory(snaps: list, pid: int, my_team: int) -> dict:
    depths, n, enemy_half, overext = [], 0, 0, 0
    for _t, _m, pos, _lvl in snaps:
        if pid not in pos:
            continue
        n += 1
        d = _depth(pos[pid][0], pos[pid][1], my_team)
        depths.append(max(0.0, d))
        if d > 0:
            enemy_half += 1
        if d > OVEREXT_THRESHOLD:
            overext += 1
    if not n:
        return {k: None for k in ("frac_enemy_half", "avg_map_depth",
                                  "max_map_depth", "frac_overextended")}
    return {
        "frac_enemy_half": enemy_half / n,
        "avg_map_depth": sum(depths) / n,
        "max_map_depth": max(depths),
        "frac_overextended": overext / n,
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k "depth or territory" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/positioning.py tests/test_positioning.py
git commit -m "feat(positioning): family B territory & over-extension (_territory)"
```

---

### Task 4: Famille B — base & isolement

**Files:**
- Modify: `src/positioning.py`
- Test: `tests/test_positioning.py`

**Interfaces:**
- Produces: `_base_and_isolation(snaps, pid: int, my_team: int, allies: list[int]) -> dict` avec clés `frac_base`, `avg_dist_to_ally`. Helper `_in_base(x, y, my_team) -> bool` (box existante).

- [ ] **Step 1: Write the failing tests**

```python
def test_in_base_box():
    assert P._in_base(2000, 2000, 100) is True
    assert P._in_base(8000, 8000, 100) is False
    assert P._in_base(13000, 13000, 200) is True
    assert P._in_base(8000, 8000, 200) is False


def test_base_and_isolation():
    # pid=1 ; allié 2 à distance connue ; allié 3 plus loin -> min retenu
    snaps = [
        (0, 0, {1: (2000, 2000), 2: (2300, 2000), 3: (9000, 9000)}, {}),  # base + allié à 300
        (60000, 1, {1: (8000, 8000), 2: (8000, 8400)}, {}),               # hors base + allié à 400
    ]
    r = P._base_and_isolation(snaps, 1, 100, [1, 2, 3])
    assert r["frac_base"] == 0.5
    assert abs(r["avg_dist_to_ally"] - (300 + 400) / 2) < 1e-6
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k "in_base or base_and_isolation" -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
def _in_base(x: float, y: float, my_team: int) -> bool:
    if my_team == 100:
        return x < 3500 and y < 3500
    return x > 11300 and y > 11300


def _base_and_isolation(snaps: list, pid: int, my_team: int, allies: list) -> dict:
    n = base = 0
    dists = []
    others = [a for a in allies if a != pid]
    for _t, _m, pos, _lvl in snaps:
        if pid not in pos:
            continue
        n += 1
        x, y = pos[pid]
        if _in_base(x, y, my_team):
            base += 1
        near = [((pos[a][0] - x) ** 2 + (pos[a][1] - y) ** 2) ** 0.5
                for a in others if a in pos]
        if near:
            dists.append(min(near))
    if not n:
        return {"frac_base": None, "avg_dist_to_ally": None}
    return {
        "frac_base": base / n,
        "avg_dist_to_ally": sum(dists) / len(dists) if dists else None,
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k "in_base or base_and_isolation" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/positioning.py tests/test_positioning.py
git commit -m "feat(positioning): family B base time & isolation (_base_and_isolation)"
```

---

### Task 5: Famille C exact — comptes de wards

**Files:**
- Modify: `src/positioning.py`
- Test: `tests/test_positioning.py`

**Interfaces:**
- Produces: `_ward_counts(timeline: dict, pid: int) -> dict` avec clés `wards_placed`, `wards_placed_early`, `control_wards_placed`, `wards_killed`.

- [ ] **Step 1: Write the failing tests**

```python
def _tl_events(events):
    return {"info": {"frames": [{"timestamp": 0, "participantFrames": {},
                                 "events": events}]}}


def test_ward_counts_only_mine():
    tl = _tl_events([
        {"type": "WARD_PLACED", "creatorId": 1, "wardType": "YELLOW_TRINKET", "timestamp": 60000},
        {"type": "WARD_PLACED", "creatorId": 1, "wardType": "CONTROL_WARD", "timestamp": 900000},
        {"type": "WARD_PLACED", "creatorId": 2, "wardType": "YELLOW_TRINKET", "timestamp": 60000},  # pas moi
        {"type": "WARD_KILL", "killerId": 1, "wardType": "YELLOW_TRINKET", "timestamp": 120000},
        {"type": "WARD_KILL", "killerId": 5, "wardType": "YELLOW_TRINKET", "timestamp": 120000},   # pas moi
    ])
    r = P._ward_counts(tl, 1)
    assert r["wards_placed"] == 2
    assert r["wards_placed_early"] == 1          # seul le 1er est en early (<14 min)
    assert r["control_wards_placed"] == 1
    assert r["wards_killed"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k ward_counts -v`
Expected: FAIL

- [ ] **Step 3: Implement `_ward_counts`**

```python
def _ward_counts(timeline: dict, pid: int) -> dict:
    placed = early = control = killed = 0
    for fr in timeline["info"]["frames"]:
        for ev in fr.get("events", []):
            t = ev.get("type")
            if t == "WARD_PLACED" and ev.get("creatorId") == pid:
                placed += 1
                if round(ev["timestamp"] / 60000) < 14:
                    early += 1
                if ev.get("wardType") == "CONTROL_WARD":
                    control += 1
            elif t == "WARD_KILL" and ev.get("killerId") == pid:
                killed += 1
    return {"wards_placed": placed, "wards_placed_early": early,
            "control_wards_placed": control, "wards_killed": killed}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k ward_counts -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/positioning.py tests/test_positioning.py
git commit -m "feat(positioning): family C exact ward counts (_ward_counts)"
```

---

### Task 6: Famille C proxy — ennemis non-vus (ML_ONLY)

**Files:**
- Modify: `src/positioning.py`
- Test: `tests/test_positioning.py`

**Interfaces:**
- Produces: `_vision_frames(snaps, pid: int, allies: list[int], enemies: list[int], my_team: int) -> dict` avec clés `avg_unaccounted_enemies`, `overext_x_unaccounted`.

- [ ] **Step 1: Write the failing tests**

```python
def test_unaccounted_enemies_one_seen():
    # allié 1 et 2 ; ennemis 6,7,8,9,10. 6 est collé à l'allié 1 (vu), les 4 autres loin.
    snaps = [(0, 0, {1: (1000, 1000), 2: (1100, 1000),
                     6: (1200, 1000),               # à ~200 de l'allié 1 -> vu
                     7: (14000, 14000), 8: (14000, 13000),
                     9: (13000, 14000), 10: (13500, 13500)}, {})]
    r = P._vision_frames(snaps, 1, [1, 2], [6, 7, 8, 9, 10], 100)
    assert r["avg_unaccounted_enemies"] == 4.0


def test_overext_x_unaccounted_zero_when_home():
    # joueur chez lui (depth<=0) -> overext_x_unaccounted = 0 quel que soit unaccounted
    snaps = [(0, 0, {1: (1000, 1000), 7: (14000, 14000)}, {})]
    r = P._vision_frames(snaps, 1, [1], [7], 100)
    assert r["avg_unaccounted_enemies"] == 1.0
    assert r["overext_x_unaccounted"] == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k "unaccounted or overext_x" -v`
Expected: FAIL

- [ ] **Step 3: Implement `_vision_frames`**

```python
def _vision_frames(snaps: list, pid: int, allies: list, enemies: list,
                   my_team: int) -> dict:
    unacc_per_frame, overext_unacc = [], []
    for _t, _m, pos, _lvl in snaps:
        if pid not in pos:
            continue
        seen = [pos[a] for a in allies if a in pos]
        unacc = 0
        for e in enemies:
            if e not in pos:
                continue
            ex, ey = pos[e]
            if not any(((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5 <= SIGHT
                       for sx, sy in seen):
                unacc += 1
        unacc_per_frame.append(unacc)
        depth = max(0.0, _depth(pos[pid][0], pos[pid][1], my_team))
        overext_unacc.append(depth * unacc)
    if not unacc_per_frame:
        return {"avg_unaccounted_enemies": None, "overext_x_unaccounted": None}
    return {
        "avg_unaccounted_enemies": sum(unacc_per_frame) / len(unacc_per_frame),
        "overext_x_unaccounted": sum(overext_unacc) / len(overext_unacc),
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k "unaccounted or overext_x" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/positioning.py tests/test_positioning.py
git commit -m "feat(positioning): family C vision proxy unaccounted enemies (_vision_frames)"
```

---

### Task 7: Morts en fog & gold dead time

**Files:**
- Modify: `src/positioning.py`
- Test: `tests/test_positioning.py`

**Interfaces:**
- Produces: `_death_features(timeline, snaps, pid: int, allies: list[int], my_team: int) -> dict` avec clés `frac_deaths_in_fog`, `gold_dead_time`. Helper `_interp(snaps, apid: int, t_ms: int) -> tuple|None` (position interpolée linéairement de l'allié `apid` au temps `t_ms`).

- [ ] **Step 1: Write the failing tests**

```python
def test_interp_linear_between_frames():
    snaps = [(0, 0, {2: (0, 0)}, {}), (60000, 1, {2: (6000, 0)}, {})]
    assert P._interp(snaps, 2, 30000) == (3000.0, 0.0)   # mi-chemin
    assert P._interp(snaps, 2, 0) == (0, 0)


def test_death_features_fog_vs_vision_and_dead_time():
    # 2 morts. Mort A (t=30000) : allié 2 interpolé à (3000,0), mort en (3000,0) -> VISION.
    # Mort B (t=90000) : allié 2 à (12000,0) interp, mort en (0,12000) -> FOG (loin).
    snaps = [
        (0, 0, {1: (0, 0), 2: (0, 0)}, {1: 3}),
        (60000, 1, {1: (0, 0), 2: (6000, 0)}, {1: 6}),
        (120000, 2, {1: (0, 0), 2: (12000, 0)}, {1: 8}),
    ]
    events = [
        {"type": "CHAMPION_KILL", "victimId": 1, "timestamp": 30000, "position": {"x": 3000, "y": 0}},
        {"type": "CHAMPION_KILL", "victimId": 1, "timestamp": 90000, "position": {"x": 0, "y": 12000}},
    ]
    tl = {"info": {"frames": [{"timestamp": 0, "participantFrames": {}, "events": events}]}}
    r = P._death_features(tl, snaps, 1, [1, 2], 100)
    assert r["frac_deaths_in_fog"] == 0.5
    # dead time : level au frame le plus proche. Mort A t=30000 -> frame 0 (level 3) = BRW 12 ;
    # Mort B t=90000 -> frame 60000 (level 6) = BRW 16. Total = 28.
    assert r["gold_dead_time"] == P._BRW[3] + P._BRW[6]


def test_death_features_none_when_no_death():
    snaps = [(0, 0, {1: (0, 0)}, {1: 1})]
    tl = {"info": {"frames": [{"timestamp": 0, "participantFrames": {}, "events": []}]}}
    r = P._death_features(tl, snaps, 1, [1], 100)
    assert r["frac_deaths_in_fog"] is None
    assert r["gold_dead_time"] == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k "interp or death_features" -v`
Expected: FAIL

- [ ] **Step 3: Implement `_interp` and `_death_features`**

```python
def _interp(snaps: list, apid: int, t_ms: int):
    """Position interpolée linéairement de l'allié apid au temps t_ms (ou None)."""
    prev = None
    for t, _m, pos, _lvl in snaps:
        if apid not in pos:
            continue
        if t <= t_ms:
            prev = (t, pos[apid])
        else:
            if prev is None:
                return pos[apid]
            t0, (x0, y0) = prev
            x1, y1 = pos[apid]
            if t == t0:
                return (x0, y0)
            f = (t_ms - t0) / (t - t0)
            return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
    return prev[1] if prev else None


def _level_at(snaps: list, pid: int, t_ms: int) -> int:
    """Niveau du joueur pid au frame le plus proche (≤ t_ms si possible)."""
    best_lvl, best_dt = 1, None
    for t, _m, _pos, lvl in snaps:
        if pid not in lvl:
            continue
        dt = abs(t - t_ms)
        if best_dt is None or dt < best_dt:
            best_dt, best_lvl = dt, lvl[pid]
    return best_lvl


def _death_features(timeline: dict, snaps: list, pid: int, allies: list,
                    my_team: int) -> dict:
    others = [a for a in allies if a != pid]
    deaths, fog, dead_time = 0, 0, 0.0
    for fr in timeline["info"]["frames"]:
        for ev in fr.get("events", []):
            if ev.get("type") != "CHAMPION_KILL" or ev.get("victimId") != pid:
                continue
            deaths += 1
            t = ev["timestamp"]
            dead_time += _BRW.get(min(18, max(1, _level_at(snaps, pid, t))), 0)
            dx, dy = ev["position"]["x"], ev["position"]["y"]
            in_vision = False
            for a in others:
                ap = _interp(snaps, a, t)
                if ap and ((ap[0] - dx) ** 2 + (ap[1] - dy) ** 2) ** 0.5 <= SIGHT:
                    in_vision = True
                    break
            if not in_vision:
                fog += 1
    return {
        "frac_deaths_in_fog": fog / deaths if deaths else None,
        "gold_dead_time": dead_time,
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k "interp or death_features" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/positioning.py tests/test_positioning.py
git commit -m "feat(positioning): fog deaths + gold dead time (_death_features)"
```

---

### Task 8: Orchestrateur `positioning_features` + test d'intégration

**Files:**
- Modify: `src/positioning.py`
- Test: `tests/test_positioning.py`

**Interfaces:**
- Consumes: tous les helpers ci-dessus.
- Produces: `positioning_features(timeline: dict, participant_id: int, pid_team: dict[int,int], my_role: str) -> dict` — dict plat des 17 clés (`ALL_FEATURES`).

- [ ] **Step 1: Write the failing tests**

```python
def test_positioning_features_returns_all_keys():
    snaps_tl = _tl([(0, {1: (1000, 1000, 1), 2: (1100, 1000, 1), 7: (14000, 14000, 1)})])
    pid_team = {1: 100, 2: 100, 7: 200}
    r = P.positioning_features(snaps_tl, 1, pid_team, "BOTTOM")
    assert set(r.keys()) == P.ALL_FEATURES
    assert len(r) == 17


def test_positioning_features_on_real_raw():
    import sys, glob, os
    sys.path.insert(0, "src")
    import riotlib as rl
    mid = os.path.basename(glob.glob("data/01_raw/*_timeline.json.zst")[0])[:-len("_timeline.json.zst")]
    match = rl._read_raw(f"{mid}_match")
    tl = rl._read_raw(f"{mid}_timeline")
    parts = match["info"]["participants"]
    pid_team = {i + 1: p["teamId"] for i, p in enumerate(parts)}
    r = P.positioning_features(tl, 1, pid_team, parts[0].get("teamPosition") or "BOTTOM")
    assert set(r.keys()) == P.ALL_FEATURES
    # types : tout est float/int ou None
    assert all(v is None or isinstance(v, (int, float)) for v in r.values())
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k positioning_features -v`
Expected: FAIL (`positioning_features` introuvable)

- [ ] **Step 3: Implement `positioning_features`**

```python
def positioning_features(timeline: dict, participant_id: int,
                         pid_team: dict, my_role: str) -> dict:
    my_team = pid_team[participant_id]
    allies = [p for p, t in pid_team.items() if t == my_team]
    enemies = [p for p, t in pid_team.items() if t != my_team]
    snaps = _build_snaps(timeline)
    out = {}
    out.update(_zone_presence(snaps, participant_id, my_role))
    out.update(_territory(snaps, participant_id, my_team))
    out.update(_base_and_isolation(snaps, participant_id, my_team, allies))
    out.update(_ward_counts(timeline, participant_id))
    out.update(_vision_frames(snaps, participant_id, allies, enemies, my_team))
    out.update(_death_features(timeline, snaps, participant_id, allies, my_team))
    return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -v`
Expected: PASS (tous, dont l'intégration sur raw réel)

- [ ] **Step 5: Commit**

```bash
git add src/positioning.py tests/test_positioning.py
git commit -m "feat(positioning): positioning_features orchestrator + real-raw integration test"
```

---

### Task 9: Brancher dans `extract_game`

**Files:**
- Modify: `src/riotlib.py` (dans `extract_game`, avant le `return`, ~ligne 404)
- Test: `tests/test_positioning.py` (test d'intégration extract_game)

**Interfaces:**
- Consumes: `positioning.positioning_features`.
- Produces: `extract_game(...)` retourne désormais un record avec une clé `"position": dict` (17 clés).

- [ ] **Step 1: Write the failing test**

```python
def test_extract_game_includes_position():
    import sys, glob, os
    sys.path.insert(0, "src")
    import riotlib as rl
    mid = os.path.basename(glob.glob("data/01_raw/*_timeline.json.zst")[0])[:-len("_timeline.json.zst")]
    match = rl._read_raw(f"{mid}_match")
    tl = rl._read_raw(f"{mid}_timeline")
    puuid = match["metadata"]["participants"][0]
    rec = rl.extract_game(match, tl, puuid)
    assert "position" in rec
    assert set(rec["position"].keys()) == P.ALL_FEATURES
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_positioning.py -k extract_game_includes_position -v`
Expected: FAIL (`KeyError: 'position'`)

- [ ] **Step 3: Modify `extract_game`**

Dans `src/riotlib.py`, juste avant le `return {` final d'`extract_game` (~ligne 404), ajouter :

```python
    import positioning  # import paresseux : évite le cycle riotlib<->positioning
    pid_team = {i + 1: p["teamId"] for i, p in enumerate(parts)}
    position = positioning.positioning_features(
        timeline, participant_id, pid_team, my_role or "BOTTOM")
```

Puis ajouter la clé dans le dict retourné (après `"avg_dragon_prox": avg_dragon_prox,`) :

```python
        "position": position,
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS (tous, dont les 23 tests existants + nouveaux)

- [ ] **Step 5: Commit**

```bash
git add src/riotlib.py tests/test_positioning.py
git commit -m "feat(positioning): wire positioning_features into extract_game (position sub-object)"
```

---

### Task 10: Pipeline ML — colonnes, features, ré-extraction & verdict AUC

**Files:**
- Modify: `src/01_data_engineering/build_dataset.py` (`game_to_row`)
- Modify: `src/02_data_science/train_ensemble.py` (`FEATURES`)

**Interfaces:**
- Consumes: `record["position"]` (17 clés).
- Produces: dataset Parquet avec 17 colonnes positionnelles ; `FEATURES` étendu ; verdict AUC avec/sans positionnement.

- [ ] **Step 1: Étendre `game_to_row`**

Dans `src/01_data_engineering/build_dataset.py`, à la fin du dict retourné par `game_to_row` (avant l'accolade fermante, après `"frames_in_base_early": ...`), ajouter l'aplatissement du sous-objet `position` :

```python
        **{f"pos_{k}": v for k, v in (g.get("position") or {}).items()},
```

- [ ] **Step 2: Régénérer silver + dataset (0 API)**

```bash
.venv/bin/python src/reextract_silver.py
.venv/bin/python src/01_data_engineering/build_dataset.py
```
Expected : le silver est réécrit avec `position` ; le dataset a 17 nouvelles colonnes `pos_*`. Vérifier :
```bash
.venv/bin/python -c "import pandas as pd; d=pd.read_parquet('data/04_dataset/adc_dataset.parquet'); print([c for c in d.columns if c.startswith('pos_')]); print(len([c for c in d.columns if c.startswith('pos_')]),'colonnes pos')"
```
Expected : 17 colonnes `pos_*`.

- [ ] **Step 3: Étendre `FEATURES` dans `train_ensemble.py`**

Ajouter à la liste `FEATURES` (après `"avg_dragon_prox",`) les 17 features positionnelles :

```python
    # positionnement macro (timeline, 0 CV)
    "pos_frac_own_lane_early", "pos_frac_river_early", "pos_frac_roam_mid",
    "pos_frac_enemy_half", "pos_frac_base",
    "pos_avg_map_depth", "pos_max_map_depth", "pos_frac_overextended",
    "pos_avg_dist_to_ally", "pos_gold_dead_time",
    "pos_wards_placed", "pos_wards_placed_early", "pos_control_wards_placed",
    "pos_wards_killed",
    "pos_frac_deaths_in_fog", "pos_avg_unaccounted_enemies", "pos_overext_x_unaccounted",
```

- [ ] **Step 4: Mesurer le delta d'AUC (verdict positionnement)**

D'abord noter l'AUC SANS positionnement (référence connue : dia_chall ≈ 0.655). Puis avec :
```bash
.venv/bin/python src/02_data_science/train_ensemble.py --target dia_chall 2>/dev/null
```
Expected : sortie AUC par modèle + ensemble. **Comparer l'AUC ensemble à 0.655.** Le delta = valeur ajoutée mesurée du positionnement. Consigner le chiffre dans le commit.

- [ ] **Step 5: Relancer l'analyse EBM-primary**

```bash
.venv/bin/python src/03_data_analyse/shap_analysis.py --target dia_chall 2>/dev/null
```
Expected : ranking/shape-functions incluant les `pos_*`. Inspecter si une feature positionnelle entre dans le top discriminant.

- [ ] **Step 6: Vérifier les tests + commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add src/01_data_engineering/build_dataset.py src/02_data_science/train_ensemble.py
git commit -m "feat(positioning): wire 17 pos_* features into ML pipeline (AUC dia_chall: 0.655 -> <mesuré>)"
```

---

## Notes d'exécution
- L'incrément 2 (câblage coaching `compare.py` sur les 14 `COACHING_SAFE`) fera l'objet d'un plan séparé, conditionné au verdict AUC de la Task 10.
- Si une feature positionnelle s'avère dégrader/ne rien apporter, ce n'est PAS un échec : c'est le verdict "le positionnement macro n'ajoute pas de signal de rang au-delà du laning" — à consigner.
