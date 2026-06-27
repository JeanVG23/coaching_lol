# Composition & matchup context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capter le contexte de situation de botlane (6 champions sur 10) de chaque game et l'utiliser pour conditionner les benchmarks du coaching post-game, sans aucune donnée scrapée.

**Architecture:** Un module isolé `champion_profiles` fusionne Data Dragon (statique, gratuit) + une table de traits curée pour produire un vecteur d'identité par champion, puis dérive 2 axes de contexte coarse (`lane_pattern`, `gank_exposure`). `extract_game` stocke les noms des 6 champions pertinents dans un sous-objet `comp` du silver ; l'agrégation gold ajoute une dimension `by_lane_context` ; `compare.py` conditionne ses benchmarks sur le contexte avec repli loggué. La migration se fait depuis le raw caché, **0 appel API**.

**Tech Stack:** Python 3.14, `.venv` (pytest 8.4, requests 2.32, pandas), JSON/JSONL, Data Dragon.

## Global Constraints

- Lancer tout depuis la racine `/Users/jeanvangysel/code/website/coaching_lol` ; `src/` est sur le path Python (`import riotlib`).
- Interpréteur : `.venv/bin/python` ; tests : `.venv/bin/python -m pytest`.
- **Le repo n'est PAS sous git.** Remplacer chaque étape « Commit » par une étape **Checkpoint** : relancer la suite de tests du module et vérifier qu'elle passe. Aucune commande git.
- **Riot-first** : aucune donnée de win rate / matchup scrapée. Sources autorisées : matchs déjà pullés + Data Dragon + table curée.
- Aucun appel API Riot dans ce plan : toute (ré)extraction lit le **raw caché** (`data/01_raw/`).
- Chemins médaillon définis dans `src/riotlib.py` (`RAW_DIR`/`SILVER_DIR`/`GOLD_DIR`). Nouveau dossier statique : `data/00_static/`.
- Dégradation propre : un champion absent de la table curée → axes `"unknown"`, jamais d'exception. Tout `unknown` rencontré est loggué.
- Tout repli d'un benchmark conditionné sur le global est **loggué** avec sa raison (principe « no silent caps »).
- Le silver stocke les **noms** de champions (`comp`), pas les buckets dérivés — les buckets sont (re)calculés en aval.

---

### Task 1: Module `champion_profiles` — `champion_vector` (fusion DDragon + traits)

**Files:**
- Create: `src/champion_profiles.py`
- Create: `tests/__init__.py` (vide)
- Test: `tests/test_champion_profiles.py`

**Interfaces:**
- Produces:
  - `champion_vector(name: str, traits: dict | None = None, ddragon: dict | None = None) -> dict`
    Retourne `{"name","range_class","tags","power_curve","lane_pattern","playstyle","gank_threat","roam"}`.
    `range_class` ∈ {"ranged","melee","unknown"} (ranged si `attackrange >= 500`).
    Les axes curés absents → `"unknown"`. `traits`/`ddragon` injectables (défaut : chargés via `load_traits()`/`load_ddragon()` — implémentés Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_champion_profiles.py
import champion_profiles as cp


FAKE_DDRAGON = {
    "Caitlyn": {"attackrange": 650, "tags": ["Marksman"]},
    "Leona": {"attackrange": 125, "tags": ["Tank", "Support"]},
}
FAKE_TRAITS = {
    "Caitlyn": {"power_curve": "early", "lane_pattern": "poke"},
    "Leona": {"lane_pattern": "all_in"},
}


def test_vector_merges_ddragon_and_traits():
    v = cp.champion_vector("Caitlyn", traits=FAKE_TRAITS, ddragon=FAKE_DDRAGON)
    assert v["range_class"] == "ranged"
    assert v["tags"] == ["Marksman"]
    assert v["power_curve"] == "early"
    assert v["lane_pattern"] == "poke"


def test_vector_melee_range_class():
    v = cp.champion_vector("Leona", traits=FAKE_TRAITS, ddragon=FAKE_DDRAGON)
    assert v["range_class"] == "melee"
    assert v["lane_pattern"] == "all_in"


def test_vector_unknown_champion_degrades_cleanly():
    v = cp.champion_vector("Nobody", traits=FAKE_TRAITS, ddragon=FAKE_DDRAGON)
    assert v["range_class"] == "unknown"
    assert v["lane_pattern"] == "unknown"
    assert v["power_curve"] == "unknown"
    assert v["tags"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_champion_profiles.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'champion_profiles'`)

> Note : pytest doit avoir `src/` sur le path. Créer `tests/conftest.py` à l'étape suivante.

- [ ] **Step 3: Write minimal implementation**

```python
# tests/conftest.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

```python
# src/champion_profiles.py
"""Identité des champions : Data Dragon (statique) + table de traits curée.

Module isolé, sans appel réseau au runtime (cache disque). Donne un vecteur
d'identité par champion et dérive les axes de contexte de botlane.
"""
from __future__ import annotations

AXES_CURATED = ("power_curve", "lane_pattern", "playstyle", "gank_threat", "roam")
RANGED_MIN = 500  # attackrange >= 500 => ranged


def champion_vector(name: str, traits: dict | None = None,
                    ddragon: dict | None = None) -> dict:
    if traits is None:
        traits = load_traits()
    if ddragon is None:
        ddragon = load_ddragon()
    dd = ddragon.get(name, {})
    tr = traits.get(name, {})
    rng = dd.get("attackrange")
    range_class = "unknown" if rng is None else ("ranged" if rng >= RANGED_MIN else "melee")
    v = {"name": name, "range_class": range_class, "tags": dd.get("tags", [])}
    for axis in AXES_CURATED:
        v[axis] = tr.get(axis, "unknown")
    return v
```

Ajouter des stubs temporaires pour les loaders (remplacés Task 3) :

```python
def load_traits() -> dict:  # remplacé Task 3
    return {}


def load_ddragon() -> dict:  # remplacé Task 3
    return {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_champion_profiles.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest tests/test_champion_profiles.py -v`
Expected: 3 passed. (Pas de git — checkpoint = tests verts.)

---

### Task 2: `derive_context` — les 2 axes coarse (`lane_pattern`, `gank_exposure`)

**Files:**
- Modify: `src/champion_profiles.py`
- Test: `tests/test_derive_context.py`

**Interfaces:**
- Consumes: `champion_vector` (Task 1).
- Produces:
  - `derive_context(comp: dict, traits=None, ddragon=None) -> dict`
    Entrée : `comp` = `{"self_adc","self_support","enemy_adc","enemy_support","self_jungle","enemy_jungle","enemy_mid"}` (valeurs = noms de champions ou `None`).
    Sortie : `{"lane_pattern": <bucket>, "gank_exposure": <bucket>}`.
    `lane_pattern` ∈ {"poke","all_in","scaling","mixed","unknown"} (dérivé du duo ENNEMI = la pression subie).
    `gank_exposure` ∈ {"low","med","high","unknown"}.

**Règles déterministes (documentées dans le code) :**

`lane_pattern` (sur les patterns du duo ennemi `enemy_adc` + `enemy_support`) :
- tous deux `unknown` → `"unknown"`
- un pattern == `"all_in"` → `"all_in"`
- sinon `"poke"` présent → `"poke"`
- sinon tous ∈ {"scaling","sustain"} → `"scaling"`
- sinon → `"mixed"`

`gank_exposure` (score entier) :
- `enemy_jungle.gank_threat` : high=+2, med=+1, low=0, unknown=0
- `enemy_mid.roam` : high=+2, med=+1, low=0, unknown=0
- atténuation `self_jungle.playstyle` : ganking=−1, skirmish=0, farming=+1, unknown=0
- si les 3 entrées sont `unknown` → `"unknown"` ; sinon score ≤1 → `"low"`, 2–3 → `"med"`, ≥4 → `"high"` (clampé à [0,∞)).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_derive_context.py
import champion_profiles as cp

DD = {
    "Cait": {"attackrange": 650}, "Leona": {"attackrange": 125},
    "Zeri": {"attackrange": 500}, "Lux": {"attackrange": 550},
    "Jarvan": {"attackrange": 175}, "Karthus": {"attackrange": 450},
    "Ahri": {"attackrange": 550}, "Cass": {"attackrange": 550},
}
TR = {
    "Cait": {"lane_pattern": "poke"}, "Leona": {"lane_pattern": "all_in"},
    "Zeri": {"lane_pattern": "scaling"}, "Lux": {"lane_pattern": "poke"},
    "Jarvan": {"gank_threat": "high", "playstyle": "ganking"},
    "Karthus": {"gank_threat": "low", "playstyle": "farming"},
    "Ahri": {"roam": "high"}, "Cass": {"roam": "low"},
}


def comp(**kw):
    base = dict(self_adc=None, self_support=None, enemy_adc=None,
                enemy_support=None, self_jungle=None, enemy_jungle=None, enemy_mid=None)
    base.update(kw)
    return base


def test_lane_pattern_all_in_when_enemy_support_engages():
    c = comp(enemy_adc="Cait", enemy_support="Leona")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["lane_pattern"] == "all_in"


def test_lane_pattern_poke():
    c = comp(enemy_adc="Cait", enemy_support="Lux")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["lane_pattern"] == "poke"


def test_lane_pattern_unknown_when_no_traits():
    c = comp(enemy_adc="Ghost", enemy_support="Phantom")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["lane_pattern"] == "unknown"


def test_gank_exposure_high_then_mitigated():
    # jgl ennemi high (+2) + mid roam high (+2) = 4 => high
    c = comp(enemy_jungle="Jarvan", enemy_mid="Ahri")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["gank_exposure"] == "high"
    # ton jgl ganking attenue (-1) => 3 => med
    c2 = comp(enemy_jungle="Jarvan", enemy_mid="Ahri", self_jungle="Jarvan")
    assert cp.derive_context(c2, traits=TR, ddragon=DD)["gank_exposure"] == "med"


def test_gank_exposure_low():
    c = comp(enemy_jungle="Karthus", enemy_mid="Cass")
    assert cp.derive_context(c, traits=TR, ddragon=DD)["gank_exposure"] == "low"


def test_gank_exposure_unknown():
    c = comp()  # tout None
    assert cp.derive_context(c, traits=TR, ddragon=DD)["gank_exposure"] == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_derive_context.py -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'derive_context'`)

- [ ] **Step 3: Write minimal implementation**

Ajouter à `src/champion_profiles.py` :

```python
def _vec(name, traits, ddragon):
    return champion_vector(name, traits, ddragon) if name else {}


def _lane_pattern(enemy_adc_v, enemy_supp_v) -> str:
    pats = [v.get("lane_pattern", "unknown") for v in (enemy_adc_v, enemy_supp_v)]
    pats = [p for p in pats if p and p != "unknown"]
    if not pats:
        return "unknown"
    if "all_in" in pats:
        return "all_in"
    if "poke" in pats:
        return "poke"
    if all(p in ("scaling", "sustain") for p in pats):
        return "scaling"
    return "mixed"


_THREAT = {"high": 2, "med": 1, "low": 0, "unknown": 0}
_ROAM = {"high": 2, "med": 1, "low": 0, "unknown": 0}
_MITIG = {"ganking": -1, "skirmish": 0, "farming": 1, "unknown": 0}


def _gank_exposure(enemy_jgl_v, enemy_mid_v, self_jgl_v) -> str:
    jt = enemy_jgl_v.get("gank_threat", "unknown")
    mr = enemy_mid_v.get("roam", "unknown")
    sp = self_jgl_v.get("playstyle", "unknown")
    if jt == "unknown" and mr == "unknown" and sp == "unknown":
        return "unknown"
    score = max(0, _THREAT.get(jt, 0) + _ROAM.get(mr, 0) + _MITIG.get(sp, 0))
    if score <= 1:
        return "low"
    if score <= 3:
        return "med"
    return "high"


def derive_context(comp: dict, traits=None, ddragon=None) -> dict:
    if traits is None:
        traits = load_traits()
    if ddragon is None:
        ddragon = load_ddragon()
    g = lambda k: _vec(comp.get(k), traits, ddragon)
    return {
        "lane_pattern": _lane_pattern(g("enemy_adc"), g("enemy_support")),
        "gank_exposure": _gank_exposure(g("enemy_jungle"), g("enemy_mid"), g("self_jungle")),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_derive_context.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: tous verts (Task 1 + Task 2).

---

### Task 3: Loaders DDragon/traits + fetch one-shot + seed `champion_traits.json`

**Files:**
- Modify: `src/champion_profiles.py` (remplacer les stubs `load_traits`/`load_ddragon`, ajouter `fetch_ddragon`)
- Create: `data/00_static/champion_traits.json` (seed curé)
- Create: `src/list_unknown_champions.py` (scanner de complétion)
- Test: `tests/test_loaders.py`

**Interfaces:**
- Consumes: `champion_vector`, `derive_context`.
- Produces:
  - `DDRAGON_VERSION: str` (const figée, ex. `"15.13.1"` — ajuster à la version courante au moment de l'exécution).
  - `STATIC_DIR = riotlib-style path -> data/00_static`.
  - `load_ddragon() -> dict` (mappe `championName -> {"attackrange","tags"}` depuis le cache).
  - `load_traits() -> dict`.
  - `fetch_ddragon(version: str | None = None) -> Path` (télécharge `championFull.json`, écrit le cache, idempotent).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loaders.py
import json
import champion_profiles as cp


def test_load_traits_reads_seed():
    traits = cp.load_traits()
    assert isinstance(traits, dict)
    assert "Caitlyn" in traits            # présent dans le seed
    assert traits["Caitlyn"]["lane_pattern"] in ("poke", "all_in", "sustain", "scaling")


def test_loaders_feed_real_vector():
    # Zeri doit être ranged via DDragon caché + scaling via le seed
    v = cp.champion_vector("Zeri")
    assert v["range_class"] in ("ranged", "unknown")   # ranged si DDragon fetché
    assert v["lane_pattern"] == "scaling"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loaders.py -v`
Expected: FAIL (stubs renvoient `{}` → `"Caitlyn" in traits` est faux).

- [ ] **Step 3a: Créer le seed curé**

```json
// data/00_static/champion_traits.json
{
  "_meta": {"note": "Traits curés : axes que Data Dragon n'a pas. Étendre via list_unknown_champions.py.",
            "axes": {"adc_support": ["power_curve", "lane_pattern"],
                     "jungle": ["playstyle", "gank_threat"], "mid": ["roam"]}},
  "Zeri":      {"power_curve": "late",  "lane_pattern": "scaling"},
  "Caitlyn":   {"power_curve": "early", "lane_pattern": "poke"},
  "Draven":    {"power_curve": "early", "lane_pattern": "all_in"},
  "Kalista":   {"power_curve": "early", "lane_pattern": "all_in"},
  "Jhin":      {"power_curve": "mid",   "lane_pattern": "poke"},
  "Ezreal":    {"power_curve": "mid",   "lane_pattern": "poke"},
  "Kaisa":     {"power_curve": "mid",   "lane_pattern": "scaling"},
  "Jinx":      {"power_curve": "late",  "lane_pattern": "scaling"},
  "Aphelios":  {"power_curve": "late",  "lane_pattern": "scaling"},
  "Lucian":    {"power_curve": "early", "lane_pattern": "all_in"},
  "Varus":     {"power_curve": "mid",   "lane_pattern": "poke"},
  "Ashe":      {"power_curve": "mid",   "lane_pattern": "poke"},
  "MissFortune":{"power_curve": "mid",  "lane_pattern": "poke"},
  "Sivir":     {"power_curve": "mid",   "lane_pattern": "sustain"},
  "Twitch":    {"power_curve": "late",  "lane_pattern": "scaling"},
  "Xayah":     {"power_curve": "mid",   "lane_pattern": "scaling"},
  "Vayne":     {"power_curve": "late",  "lane_pattern": "scaling"},
  "Samira":    {"power_curve": "mid",   "lane_pattern": "all_in"},
  "Lux":       {"lane_pattern": "poke"},
  "Leona":     {"lane_pattern": "all_in"},
  "Nautilus":  {"lane_pattern": "all_in"},
  "Thresh":    {"lane_pattern": "all_in"},
  "Pyke":      {"lane_pattern": "all_in"},
  "Blitzcrank":{"lane_pattern": "all_in"},
  "Lulu":      {"lane_pattern": "sustain"},
  "Nami":      {"lane_pattern": "sustain"},
  "Karma":     {"lane_pattern": "poke"},
  "Senna":     {"lane_pattern": "poke"},
  "Brand":     {"lane_pattern": "poke"},
  "Soraka":    {"lane_pattern": "sustain"},
  "Yuumi":     {"lane_pattern": "sustain"},
  "Milio":     {"lane_pattern": "sustain"},
  "Renata":    {"lane_pattern": "all_in"},
  "Rakan":     {"lane_pattern": "all_in"},
  "JarvanIV":  {"playstyle": "ganking",  "gank_threat": "high"},
  "LeeSin":    {"playstyle": "ganking",  "gank_threat": "high"},
  "Elise":     {"playstyle": "ganking",  "gank_threat": "high"},
  "Nidalee":   {"playstyle": "ganking",  "gank_threat": "high"},
  "Vi":        {"playstyle": "ganking",  "gank_threat": "high"},
  "XinZhao":   {"playstyle": "ganking",  "gank_threat": "high"},
  "Hecarim":   {"playstyle": "ganking",  "gank_threat": "high"},
  "Nocturne":  {"playstyle": "ganking",  "gank_threat": "high"},
  "Sejuani":   {"playstyle": "skirmish", "gank_threat": "med"},
  "Viego":     {"playstyle": "skirmish", "gank_threat": "med"},
  "Graves":    {"playstyle": "farming",  "gank_threat": "med"},
  "Khazix":    {"playstyle": "skirmish", "gank_threat": "med"},
  "Kayn":      {"playstyle": "skirmish", "gank_threat": "med"},
  "Belveth":   {"playstyle": "farming",  "gank_threat": "low"},
  "Karthus":   {"playstyle": "farming",  "gank_threat": "low"},
  "MasterYi":  {"playstyle": "farming",  "gank_threat": "low"},
  "Amumu":     {"playstyle": "ganking",  "gank_threat": "med"},
  "Warwick":   {"playstyle": "ganking",  "gank_threat": "high"},
  "Ahri":      {"roam": "high"},
  "Akshan":    {"roam": "high"},
  "TwistedFate":{"roam": "high"},
  "Galio":     {"roam": "high"},
  "Talon":     {"roam": "high"},
  "Zed":       {"roam": "med"},
  "Yasuo":     {"roam": "low"},
  "Yone":      {"roam": "low"},
  "Syndra":    {"roam": "low"},
  "Orianna":   {"roam": "low"},
  "Cassiopeia":{"roam": "low"},
  "Veigar":    {"roam": "low"},
  "Viktor":    {"roam": "low"},
  "Azir":      {"roam": "low"},
  "Vex":       {"roam": "med"},
  "Sylas":     {"roam": "med"},
  "Leblanc":   {"roam": "high"},
  "Malzahar":  {"roam": "low"}
}
```

> Note : les noms de clés doivent matcher l'`id` Data Dragon (PascalCase, casse Riot exacte : `Kaisa`, `JarvanIV`, `Khazix`, `Belveth`, `Leblanc`, `TwistedFate`, `MissFortune`, `MasterYi`, `XinZhao`). Le scanner (Step 3d) liste les manquants après ré-extraction pour corriger toute coquille.

- [ ] **Step 3b: Remplacer les loaders + ajouter fetch_ddragon**

Remplacer les stubs dans `src/champion_profiles.py` :

```python
import json
from functools import lru_cache
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "data" / "00_static"
DDRAGON_VERSION = "15.13.1"  # figée ; refresh = action manuelle (fetch_ddragon)
TRAITS_PATH = STATIC_DIR / "champion_traits.json"


def fetch_ddragon(version: str | None = None) -> Path:
    version = version or DDRAGON_VERSION
    url = (f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/championFull.json")
    dest = STATIC_DIR / "ddragon" / version / "championFull.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_text(resp.text)
    return dest


@lru_cache(maxsize=1)
def load_ddragon() -> dict:
    path = STATIC_DIR / "ddragon" / DDRAGON_VERSION / "championFull.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())["data"]
    out = {}
    for champ in raw.values():
        out[champ["id"]] = {
            "attackrange": champ.get("stats", {}).get("attackrange"),
            "tags": champ.get("tags", []),
        }
    return out


@lru_cache(maxsize=1)
def load_traits() -> dict:
    if not TRAITS_PATH.exists():
        return {}
    data = json.loads(TRAITS_PATH.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}
```

Supprimer les anciens stubs `load_traits`/`load_ddragon`.

- [ ] **Step 3c: Lancer le fetch one-shot et vérifier le cache**

Run:
```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import champion_profiles as cp; p=cp.fetch_ddragon(); print('OK', p); print('champions:', len(cp.load_ddragon()))"
```
Expected: `OK .../data/00_static/ddragon/15.13.1/championFull.json` puis `champions: ` un nombre > 160.

> Si la version 15.13.1 n'existe plus, récupérer la dernière via `https://ddragon.leagueoflegends.com/api/versions.json` (1re entrée) et mettre à jour `DDRAGON_VERSION`.

- [ ] **Step 3d: Créer le scanner de complétion**

```python
# src/list_unknown_champions.py
"""Liste les champions présents dans le silver mais absents de champion_traits.json
(pour compléter la table au fil de l'eau). 0 appel API."""
from __future__ import annotations

import collections

import champion_profiles as cp
import riotlib as rl

ROLES = ("self_adc", "self_support", "enemy_adc", "enemy_support",
         "self_jungle", "enemy_jungle", "enemy_mid")


def main() -> int:
    traits = cp.load_traits()
    seen = collections.Counter()
    for root in (rl.SILVER_DIR / "referentiel", rl.SILVER_DIR / "personal"):
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            for g in rl.read_jsonl(d / "games.jsonl"):
                for name in (g.get("comp") or {}).values():
                    if name and name not in traits:
                        seen[name] += 1
    if not seen:
        print("✓ Tous les champions du silver sont dans la table (ou pas de comp).")
        return 0
    print(f"Champions manquants ({len(seen)}), triés par fréquence :")
    for name, n in seen.most_common():
        print(f"  {name:<18} {n}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 4: Run loader tests**

Run: `.venv/bin/python -m pytest tests/test_loaders.py -v`
Expected: PASS (2 tests ; `range_class == "ranged"` pour Zeri une fois DDragon fetché).

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: tous verts.

---

### Task 4: `extract_game` — sous-objet `comp` (6 champions)

**Files:**
- Modify: `src/riotlib.py` (fonction `extract_game`, ~lignes 204-276)
- Test: `tests/test_extract_comp.py`

**Interfaces:**
- Consumes: structures `match`/`timeline` Riot.
- Produces: chaque record silver gagne une clé `"comp"` :
  `{"self_adc","self_support","enemy_adc","enemy_support","self_jungle","enemy_jungle","enemy_mid"}` (noms de champions ou `None` si rôle absent).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extract_comp.py
import riotlib as rl


def _match():
    # index 0 = moi (BOTTOM, team 100). 10 participants, 2 équipes.
    def p(team, role, champ):
        return {"teamId": team, "teamPosition": role, "championName": champ, "win": True}
    parts = [
        p(100, "BOTTOM", "Zeri"),    # moi
        p(100, "UTILITY", "Lulu"),
        p(100, "JUNGLE", "Graves"),
        p(100, "MIDDLE", "Ahri"),
        p(100, "TOP", "Aatrox"),
        p(200, "BOTTOM", "Caitlyn"),
        p(200, "UTILITY", "Leona"),
        p(200, "JUNGLE", "JarvanIV"),
        p(200, "MIDDLE", "Syndra"),
        p(200, "TOP", "Sett"),
    ]
    return {"metadata": {"matchId": "T1", "participants": [f"puuid{i}" for i in range(10)]},
            "info": {"mapId": 11, "queueId": 420, "gameVersion": "15.13.1.1", "participants": parts}}


def _timeline():
    return {"info": {"frames": []}}


def test_comp_resolves_six_champions():
    g = rl.extract_game(_match(), _timeline(), "puuid0", rank="test")
    comp = g["comp"]
    assert comp["self_adc"] == "Zeri"
    assert comp["self_support"] == "Lulu"
    assert comp["enemy_adc"] == "Caitlyn"
    assert comp["enemy_support"] == "Leona"
    assert comp["self_jungle"] == "Graves"
    assert comp["enemy_jungle"] == "JarvanIV"
    assert comp["enemy_mid"] == "Syndra"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_extract_comp.py -v`
Expected: FAIL (`KeyError: 'comp'`)

- [ ] **Step 3: Write minimal implementation**

Dans `src/riotlib.py`, `extract_game`, juste avant le `return {`, ajouter la résolution par (rôle, équipe) :

```python
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
```

Puis ajouter `"comp": comp,` dans le dict retourné (après `"lane": lane,`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_extract_comp.py -v`
Expected: PASS

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: tous verts.

---

### Task 5: Agrégation — dimension `by_lane_context` dans le gold

**Files:**
- Modify: `src/riotlib.py` (`aggregate`, ~lignes 333-350 ; import de `champion_profiles`)
- Test: `tests/test_aggregate_context.py`

**Interfaces:**
- Consumes: `champion_profiles.derive_context` (Task 2), `_facet` (existant), `comp` (Task 4).
- Produces: `aggregate(...)` gagne une clé `"by_lane_context"` :
  `{"lane_pattern": {bucket: _facet(games), ...}, "gank_exposure": {bucket: _facet(games), ...}}`.
  Chaque sous-facet a la même forme que `overall`/`win`/`loss` (donc `n_games`, `lane`, `deaths_per_game`, etc.).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggregate_context.py
import riotlib as rl

DD = {"Caitlyn": {"attackrange": 650}, "Leona": {"attackrange": 125},
      "Lux": {"attackrange": 550}}
TR = {"Leona": {"lane_pattern": "all_in"}, "Lux": {"lane_pattern": "poke"},
      "Caitlyn": {"lane_pattern": "poke"}}


def _game(enemy_support, gd10):
    return {"match_id": f"m{enemy_support}", "champion": "Zeri", "role": "BOTTOM",
            "win": True, "deaths": [], "lane": {k: None for k in rl.LANE_KEYS} | {"gd10": gd10},
            "comp": {"self_adc": "Zeri", "self_support": None, "enemy_adc": "Caitlyn",
                     "enemy_support": enemy_support, "self_jungle": None,
                     "enemy_jungle": None, "enemy_mid": None}}


def test_by_lane_context_splits_by_pattern(monkeypatch):
    monkeypatch.setattr(rl.cp, "load_ddragon", lambda: DD)
    monkeypatch.setattr(rl.cp, "load_traits", lambda: TR)
    games = [_game("Leona", -300), _game("Leona", -200), _game("Lux", 50)]
    agg = rl.aggregate(games, "adc")
    lp = agg["by_lane_context"]["lane_pattern"]
    assert lp["all_in"]["n_games"] == 2
    assert lp["poke"]["n_games"] == 1
    assert lp["all_in"]["lane"]["gd10"] == -250  # médiane de -300/-200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_aggregate_context.py -v`
Expected: FAIL (`KeyError: 'by_lane_context'` ou `AttributeError: module 'riotlib' has no attribute 'cp'`)

- [ ] **Step 3: Write minimal implementation**

En tête de `src/riotlib.py` (avec les autres imports) :

```python
import champion_profiles as cp
```

Ajouter une fonction helper avant `aggregate` :

```python
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
```

Dans `aggregate`, ajouter la clé au dict retourné :

```python
        "loss": _facet(losses),
        "by_lane_context": _by_lane_context(subset),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_aggregate_context.py -v`
Expected: PASS

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: tous verts.

---

### Task 6: `reextract_silver.py` — raw caché → silver enrichi (0 API)

**Files:**
- Create: `src/reextract_silver.py`
- Test: `tests/test_reextract.py`

**Interfaces:**
- Consumes: `riotlib.extract_game` (Task 4), raw caché, silver existant (pour connaître quels matchs/puuids appartiennent à quel scope).
- Produces: réécrit `02_silver/{referentiel/<rank>,personal/<player>}/games.jsonl` avec le `comp`, sans appel API.

**Principe :** le silver actuel porte déjà `match_id`, `puuid` et `rank`/scope (via le dossier). On rejoue `extract_game` sur le couple (match, timeline) du raw caché pour chaque (match_id, puuid) du silver, et on réécrit. Helper `get_match_timeline` lit le cache ; mais pour garantir 0 API, on lit directement les fichiers raw.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reextract.py
import reextract_silver as rx


def test_reextract_game_adds_comp(tmp_path, monkeypatch):
    # match minimal caché
    def p(team, role, champ):
        return {"teamId": team, "teamPosition": role, "championName": champ, "win": False}
    parts = [p(100, "BOTTOM", "Zeri"), p(100, "UTILITY", "Lulu"), p(100, "JUNGLE", "Graves"),
             p(100, "MIDDLE", "Ahri"), p(100, "TOP", "Aatrox"), p(200, "BOTTOM", "Caitlyn"),
             p(200, "UTILITY", "Leona"), p(200, "JUNGLE", "JarvanIV"),
             p(200, "MIDDLE", "Syndra"), p(200, "TOP", "Sett")]
    match = {"metadata": {"matchId": "EUW1_X", "participants": [f"p{i}" for i in range(10)]},
             "info": {"mapId": 11, "queueId": 420, "gameVersion": "15.13.1.1",
                      "participants": parts}}
    timeline = {"info": {"frames": []}}
    out = rx.reextract_one(match, timeline, "p0", rank="challenger")
    assert out is not None
    assert out["comp"]["enemy_jungle"] == "JarvanIV"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reextract.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'reextract_silver'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/reextract_silver.py
"""Ré-extrait le silver depuis le raw caché — 0 appel API.

À lancer après toute évolution de extract_game (ici : ajout de comp). Lit les
fichiers raw (data/01_raw/<matchId>_match.json + _timeline.json) pour chaque
(match_id, puuid) déjà présent dans le silver, rejoue extract_game, réécrit.
"""
from __future__ import annotations

import json
import sys

import riotlib as rl


def reextract_one(match: dict, timeline: dict, puuid: str, rank):
    return rl.extract_game(match, timeline, puuid, rank=rank)


def _load_raw(match_id: str):
    m = rl.RAW_DIR / f"{match_id}_match.json"
    t = rl.RAW_DIR / f"{match_id}_timeline.json"
    if not m.exists() or not t.exists():
        return None
    return json.loads(m.read_text()), json.loads(t.read_text())


def _reextract_dir(d, rank, scope_kind):
    games = rl.read_jsonl(d / "games.jsonl")
    if not games:
        return 0, 0
    out, miss = [], 0
    for g in games:
        raw = _load_raw(g["match_id"])
        if not raw:
            out.append(g)  # garde l'ancien record si raw absent
            miss += 1
            continue
        new = rl.extract_game(raw[0], raw[1], g["puuid"], rank=g.get("rank"))
        out.append(new if new else g)
    rl.write_jsonl(d / "games.jsonl", out)
    return len(out), miss


def main() -> int:
    total, missing = 0, 0
    for kind, root in (("referentiel", rl.SILVER_DIR / "referentiel"),
                       ("personal", rl.SILVER_DIR / "personal")):
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            n, miss = _reextract_dir(d, d.name, kind)
            total += n
            missing += miss
            print(f"  {kind}/{d.name}: {n} games ré-extraits ({miss} raw manquants)")
    print(f"\n✓ {total} games ré-extraits ({missing} sans raw → record conservé).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_reextract.py -v`
Expected: PASS

- [ ] **Step 5: Run la migration réelle + vérifier le comp dans le silver**

Run:
```bash
.venv/bin/python src/reextract_silver.py
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import riotlib as rl; g=rl.read_jsonl(rl.SILVER_DIR/'personal'/'spadzze'/'games.jsonl'); print('comp present:', 'comp' in g[0]); print(g[0]['comp'])"
```
Expected: `comp present: True` + un dict avec les 6 champions de la game.

- [ ] **Step 6: Checkpoint**

Run: `.venv/bin/python src/list_unknown_champions.py`
Expected: liste (éventuellement vide) des champions à ajouter au seed. **Compléter `champion_traits.json`** pour les champions fréquents listés (validation utilisateur), puis re-lancer le scanner jusqu'à ce que les manquants fréquents soient couverts.

---

### Task 7: `compare.py` — benchmark conditionné sur le contexte (repli N=8 loggué)

**Files:**
- Modify: `src/compare.py` (ajout d'une section + const `MIN_CONTEXT_N`)
- Modify: `src/rebuild_gold.py` (aucun changement de code requis — vérifier qu'il régénère bien `by_lane_context`)
- Test: `tests/test_compare_context.py`

**Interfaces:**
- Consumes: gold avec `by_lane_context` (Task 5).
- Produces:
  - `MIN_CONTEXT_N = 8` (seuil de repli).
  - `context_benchmark(me_agg, ref_agg, axis, outcome) -> dict` :
    `{"bucket","n_me","n_ref","gd10_me","gd10_ref","fallback": bool, "reason": str|None}`.
    Choisit le bucket dominant côté perso ; si `n_ref < MIN_CONTEXT_N` → `fallback=True`,
    `reason` renseigné, valeurs prises sur `overall`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compare_context.py
import compare


def _agg(buckets, overall_gd10):
    # buckets: {name: (n_games, gd10)}
    lp = {name: {"n_games": n, "lane": {"gd10": gd}, "deaths_per_game": 0}
          for name, (n, gd) in buckets.items()}
    return {"overall": {"n_games": 99, "lane": {"gd10": overall_gd10}},
            "by_lane_context": {"lane_pattern": lp, "gank_exposure": {}}}


def test_uses_matching_bucket_when_ref_has_enough():
    me = _agg({"all_in": (6, -510)}, -400)
    ref = _agg({"all_in": (20, -150)}, -100)
    r = compare.context_benchmark(me, ref, "lane_pattern", "overall")
    assert r["bucket"] == "all_in"
    assert r["gd10_me"] == -510 and r["gd10_ref"] == -150
    assert r["fallback"] is False


def test_falls_back_to_global_when_ref_too_thin():
    me = _agg({"all_in": (6, -510)}, -400)
    ref = _agg({"all_in": (3, -150)}, -100)   # 3 < MIN_CONTEXT_N
    r = compare.context_benchmark(me, ref, "lane_pattern", "overall")
    assert r["fallback"] is True
    assert r["gd10_ref"] == -100              # repli sur overall
    assert r["reason"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compare_context.py -v`
Expected: FAIL (`AttributeError: module 'compare' has no attribute 'context_benchmark'`)

- [ ] **Step 3: Write minimal implementation**

Dans `src/compare.py`, ajouter après les constantes :

```python
MIN_CONTEXT_N = 8  # sous ce seuil de games référentiel dans le bucket, on retombe sur global
```

Ajouter la fonction :

```python
def context_benchmark(me_agg, ref_agg, axis, outcome):
    """Compare le bucket de contexte DOMINANT côté perso au même bucket référentiel.

    Repli explicite et loggué sur 'overall' si le référentiel a < MIN_CONTEXT_N games
    dans ce bucket (échantillon trop fin pour un benchmark honnête).
    """
    me_buckets = me_agg.get("by_lane_context", {}).get(axis, {})
    ref_buckets = ref_agg.get("by_lane_context", {}).get(axis, {})
    if not me_buckets:
        return None
    bucket = max(me_buckets, key=lambda b: me_buckets[b].get("n_games", 0))
    n_me = me_buckets[bucket].get("n_games", 0)
    gd10_me = me_buckets[bucket].get("lane", {}).get("gd10")
    ref_b = ref_buckets.get(bucket, {})
    n_ref = ref_b.get("n_games", 0)
    if n_ref < MIN_CONTEXT_N:
        glob = ref_agg.get("overall", {})
        return {"bucket": bucket, "n_me": n_me, "n_ref": n_ref,
                "gd10_me": gd10_me, "gd10_ref": glob.get("lane", {}).get("gd10"),
                "fallback": True,
                "reason": f"réf. {bucket}={n_ref}<{MIN_CONTEXT_N} games → repli global"}
    return {"bucket": bucket, "n_me": n_me, "n_ref": n_ref, "gd10_me": gd10_me,
            "gd10_ref": ref_b.get("lane", {}).get("gd10"), "fallback": False, "reason": None}
```

Brancher l'affichage dans `main()`, juste avant le `# --- verdict ---` :

```python
    print(f"\n  Benchmark conditionné sur le contexte de lane (vs {target}) :")
    for axis in ("lane_pattern", "gank_exposure"):
        r = context_benchmark(me, refs[target], axis, outcome)
        if not r:
            print(f"    {axis}: pas de contexte côté perso (comp manquant ?)")
            continue
        gm = f"{r['gd10_me']:+d}" if r["gd10_me"] is not None else "—"
        gr = f"{r['gd10_ref']:+d}" if r["gd10_ref"] is not None else "—"
        tag = f"  ⚠ {r['reason']}" if r["fallback"] else ""
        print(f"    {axis} = {r['bucket']:<10} gd10 toi {gm} vs {target} {gr} "
              f"(toi n={r['n_me']}, réf n={r['n_ref']}){tag}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_compare_context.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run la chaîne complète de bout en bout (0 API)**

Run:
```bash
.venv/bin/python src/rebuild_gold.py
.venv/bin/python src/compare.py --scope adc --outcome loss
```
Expected: `rebuild_gold` régénère le gold sans erreur ; `compare.py` affiche la nouvelle section « Benchmark conditionné sur le contexte de lane » avec, le cas échéant, le ⚠ de repli loggué.

- [ ] **Step 6: Checkpoint**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: toute la suite verte (Tasks 1-7).

---

## Notes d'intégration

- **Ordre de migration** (récap, 0 API) : `fetch_ddragon` (Task 3c) → `reextract_silver.py` (Task 6) → compléter `champion_traits.json` via `list_unknown_champions.py` → `rebuild_gold.py` → `compare.py`.
- **Dépendance nouvelle** : `riotlib` importe désormais `champion_profiles` (pas de cycle : `champion_profiles` ne dépend pas de `riotlib`).
- **`@lru_cache` sur les loaders** : en test, patcher via `monkeypatch.setattr(rl.cp, "load_ddragon", ...)` (cf. Task 5) ou appeler `load_ddragon.cache_clear()` si besoin.
- **CLAUDE.md** : après implémentation, ajouter `champion_profiles.py`, `reextract_silver.py`, `list_unknown_champions.py`, `data/00_static/` et la dimension `by_lane_context` à la section « Architecture du code ».
