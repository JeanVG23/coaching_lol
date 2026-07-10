# Contexte matchup + items réels dans le payload par-game — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Le payload par-game (`payload.build_game`) expose le matchup (comp + buckets dérivés) et les items réellement achetés à chaque recall, pour que le coach LLM explique le POURQUOI des morts et arrête de reprocher du gold « normal ».

**Architecture:** `game_journal` capture les `item_ids` bruts par visite de shop (module pur, honore `ITEM_UNDO`) ; `champion_profiles` gagne un catalogue d'items Data Dragon (même pattern que `championFull.json`) ; `payload.build_game` résout ids → {nom, coût} et ajoute un bloc `context` (comp du silver + `derive_context`) ; `prompt.SYSTEM_GAME` encadre l'usage des deux.

**Tech Stack:** Python 3 (Poetry), pytest, Data Dragon (fetch one-shot idempotent), convention flat-import (`sys.path.insert` vers `src/core/`).

**Spec:** `docs/superpowers/specs/2026-07-10-per-game-matchup-items-design.md`

## Global Constraints

- Lancer depuis la racine du repo, dans l'env Poetry : `poetry run pytest tests/ -q`, `poetry run python3 …`.
- Convention flat-import : pas de package dans `src/`, les modules s'importent par nom (`import game_journal`) après insertion de `src/core/` dans `sys.path` (les tests via `tests/conftest.py`, déjà en place).
- **Asymétrie** : tout ce qui entre dans le journal/payload par-game doit être une info que le joueur AVAIT. Comp (champ select) et achats du joueur = safe. Aucun proxy `ML_ONLY` (les tests `*_never_leaks_ml_only*` existants doivent rester verts).
- `ITEM_SOLD` ignoré en v1 (documenté dans la docstring de `_recalls`).
- Dégradation propre partout : pas de `comp` dans le silver → pas de bloc `context` ; `item.json` absent → recalls sans clé `items` ; id inconnu du catalogue → omis. Jamais d'exception.
- Data Dragon version figée `16.13.1` (constante `DDRAGON_VERSION` existante).
- Commits fréquents, messages `feat:`/`test:` en français comme l'historique du repo.

---

### Task 1: `game_journal._recalls` — capture des `item_ids` + honorage `ITEM_UNDO`

**Files:**
- Modify: `src/core/game_journal.py:108-136` (fonction `_recalls`)
- Test: `tests/test_game_journal.py`

**Interfaces:**
- Consumes: rien de nouveau (événements timeline `ITEM_PURCHASED`/`ITEM_UNDO`, champs Riot `itemId`, `beforeId`, `participantId`).
- Produces: chaque dict de `journal["recalls"]` gagne `"item_ids": list[int]` (ordre d'achat, après annulation des undos, ids `None` exclus). Task 3 consomme ce champ.

- [ ] **Step 1: Écrire les tests qui échouent**

Dans `tests/test_game_journal.py`, étendre le helper `_buy` (ligne 46) pour porter un `itemId` — l'appel existant sans argument reste valide :

```python
def _buy(t_ms, pid, item=1055):
    return {"type": "ITEM_PURCHASED", "timestamp": t_ms,
            "participantId": pid, "itemId": item}


def _undo(t_ms, pid, before):
    return {"type": "ITEM_UNDO", "timestamp": t_ms,
            "participantId": pid, "beforeId": before}
```

Ajouter en fin de fichier :

```python
def test_recalls_capture_item_ids_in_purchase_order():
    tl = _basic_timeline({
        5: [_buy(310000, 1, item=1038), _buy(315000, 1, item=1055)],
        10: [_buy(605000, 1, item=3031)],
    })
    r1, r2 = J.game_journal(_match(), tl, ME)["recalls"]
    assert r1["item_ids"] == [1038, 1055]
    assert r2["item_ids"] == [3031]


def test_recalls_honor_item_undo():
    # Achat 1038 annulé 5 s plus tard, puis rachat 1036 : seul 1036 subsiste.
    tl = _basic_timeline({
        5: [_buy(310000, 1, item=1038), _undo(315000, 1, before=1038),
            _buy(320000, 1, item=1036)],
    })
    (r1,) = J.game_journal(_match(), tl, ME)["recalls"]
    assert r1["item_ids"] == [1036]
    assert r1["items_bought"] == 1


def test_recalls_undo_removes_last_matching_purchase_only():
    # Deux achats du même item, un seul undo -> il en reste un.
    tl = _basic_timeline({
        5: [_buy(310000, 1, item=1036), _buy(315000, 1, item=1036),
            _undo(320000, 1, before=1036)],
    })
    (r1,) = J.game_journal(_match(), tl, ME)["recalls"]
    assert r1["item_ids"] == [1036]


def test_recalls_undo_of_other_player_ignored():
    tl = _basic_timeline({
        5: [_buy(310000, 1, item=1038), _undo(315000, 6, before=1038)],
    })
    (r1,) = J.game_journal(_match(), tl, ME)["recalls"]
    assert r1["item_ids"] == [1038]
```

- [ ] **Step 2: Vérifier qu'ils échouent**

Run: `poetry run pytest tests/test_game_journal.py -q`
Expected: 4 FAIL (`KeyError: 'item_ids'`), les tests existants PASS.

- [ ] **Step 3: Implémenter `_recalls`**

Remplacer la fonction `_recalls` de `src/core/game_journal.py` par :

```python
def _recalls(timeline: dict, pid: int, obj_kills) -> list[dict]:
    """Visites de shop (clusters d'achats), hors shopping de départ.

    Approximation v1 : un achat implique la présence au shop (recall ou reset
    après mort — les deux sont des « resets » à coacher). gold_before = currentGold
    de la dernière frame avant la visite (léger plancher, frames espacées de 60 s).
    ITEM_UNDO honoré (retire le dernier achat correspondant) ; ITEM_SOLD ignoré.
    """
    buys = sorted((ev["timestamp"], ev.get("itemId"))
                  for ev in _events(timeline)
                  if ev.get("type") == "ITEM_PURCHASED"
                  and ev.get("participantId") == pid
                  and ev["timestamp"] >= OPENING_BUY_MS)
    undos = sorted((ev["timestamp"], ev.get("beforeId"))
                   for ev in _events(timeline)
                   if ev.get("type") == "ITEM_UNDO"
                   and ev.get("participantId") == pid)
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
        pf = _frame_before(timeline, pid, t0)
        out.append({
            "t_ms": t0, "clock": _clock(t0),
            "minute": t0 // 60000, "phase": phase_of(t0 // 60000),
            "items_bought": len(visit),
            "item_ids": [item for _, item in visit if item is not None],
            "gold_before": pf.get("currentGold") if pf else None,
            "objective": _objective_at(obj_kills, t0),
        })
    return out
```

- [ ] **Step 4: Vérifier que tout passe**

Run: `poetry run pytest tests/test_game_journal.py tests/test_coaching_payload_game.py -q`
Expected: PASS (y compris `test_journal_never_leaks_ml_only_features` et les tests payload existants — `item_ids` transite dans le journal sans casser `build_game`).

- [ ] **Step 5: Commit**

```bash
git add src/core/game_journal.py tests/test_game_journal.py
git commit -m "feat(journal): item_ids par recall, honorage ITEM_UNDO"
```

---

### Task 2: `champion_profiles` — catalogue d'items Data Dragon

**Files:**
- Modify: `src/core/champion_profiles.py` (après `load_ddragon`, lignes ~50)
- Test: `tests/test_champion_profiles.py`

**Interfaces:**
- Consumes: rien des autres tasks.
- Produces: `fetch_ddragon_items(version: str | None = None) -> Path` (one-shot idempotent, écrit `data/00_static/ddragon/<version>/item.json`) ; `load_items() -> dict[int, dict]` (`lru_cache`, `{item_id: {"name": str, "cost": int | None}}`, `{}` si fichier absent) ; helper pur `_parse_items(raw: dict) -> dict`. Task 3 consomme `load_items`.

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter en fin de `tests/test_champion_profiles.py` :

```python
FAKE_ITEM_JSON = {"data": {
    "1038": {"name": "B.F. Sword", "gold": {"total": 1300}},
    "3031": {"name": "Infinity Edge", "gold": {"total": 3450}},
    "2055": {"name": "Control Ward", "gold": {"total": 75}},
}}


def test_parse_items_maps_id_to_name_and_cost():
    items = cp._parse_items(FAKE_ITEM_JSON["data"])
    assert items[1038] == {"name": "B.F. Sword", "cost": 1300}
    assert items[3031]["cost"] == 3450


def test_load_items_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "STATIC_DIR", tmp_path)
    cp.load_items.cache_clear()
    assert cp.load_items() == {}
    cp.load_items.cache_clear()


def test_load_items_reads_cached_file(tmp_path, monkeypatch):
    import json
    dest = tmp_path / "ddragon" / cp.DDRAGON_VERSION
    dest.mkdir(parents=True)
    (dest / "item.json").write_text(json.dumps(FAKE_ITEM_JSON))
    monkeypatch.setattr(cp, "STATIC_DIR", tmp_path)
    cp.load_items.cache_clear()
    assert cp.load_items()[2055] == {"name": "Control Ward", "cost": 75}
    cp.load_items.cache_clear()
```

- [ ] **Step 2: Vérifier qu'ils échouent**

Run: `poetry run pytest tests/test_champion_profiles.py -q`
Expected: 3 FAIL (`AttributeError: module 'champion_profiles' has no attribute '_parse_items'` / `'load_items'`).

- [ ] **Step 3: Implémenter**

Dans `src/core/champion_profiles.py`, insérer après `load_ddragon` :

```python
def fetch_ddragon_items(version: str | None = None) -> Path:
    version = version or DDRAGON_VERSION
    dest = STATIC_DIR / "ddragon" / version / "item.json"
    if dest.exists():
        return dest  # idempotent : cache déjà chaud (refresh = supprimer le fichier)
    url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/item.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_text(resp.text)
    load_items.cache_clear()
    return dest


def _parse_items(raw: dict) -> dict:
    return {int(iid): {"name": it.get("name", f"item_{iid}"),
                       "cost": it.get("gold", {}).get("total")}
            for iid, it in raw.items()}


@lru_cache(maxsize=1)
def load_items() -> dict:
    path = STATIC_DIR / "ddragon" / DDRAGON_VERSION / "item.json"
    if not path.exists():
        return {}
    return _parse_items(json.loads(path.read_text())["data"])
```

- [ ] **Step 4: Vérifier que tout passe**

Run: `poetry run pytest tests/test_champion_profiles.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/champion_profiles.py tests/test_champion_profiles.py
git commit -m "feat(champion_profiles): catalogue d'items Data Dragon (fetch one-shot + load_items)"
```

---

### Task 3: `payload.build_game` — items résolus + bloc `context`

**Files:**
- Modify: `src/04_coaching/payload.py` (imports ligne 16-19, `build_game` lignes 184-218)
- Test: `tests/test_coaching_payload_game.py`

**Interfaces:**
- Consumes: `journal["recalls"][i]["item_ids"]` (Task 1) ; `champion_profiles.load_items()` (Task 2) ; `champion_profiles.derive_context(comp) -> {"lane_pattern", "gank_exposure"}` (existant) ; `rec["comp"]` du silver (existant).
- Produces: payload par-game avec `journal.recalls[i].items = [{"name", "cost"}]` (clé absente si catalogue vide ou aucun id résolu ; `item_ids` bruts NON exposés) et bloc top-level `context = {"comp": {...}, "lane_pattern": str, "gank_exposure": str}` (clé absente si le record n'a pas de `comp`). Task 4 (prompt) documente ces deux blocs.

- [ ] **Step 1: Écrire les tests qui échouent**

Dans `tests/test_coaching_payload_game.py` :

1. Enrichir la game ADC du fixture `_dirs` (ligne 16-18) avec un `comp` :

```python
        {"match_id": "EUW1_42", "puuid": TJ.ME, "role": "BOTTOM",
         "champion": "Zeri", "win": True, "queue": 420,
         "comp": {"self_adc": "Zeri", "self_support": "Lulu",
                  "enemy_adc": "Jinx", "enemy_support": "Thresh",
                  "self_jungle": "Vi", "enemy_jungle": "LeeSin",
                  "enemy_mid": "Orianna"}},
```

(La game jungle `EUW1_43` reste SANS `comp` — elle sert de cas de dégradation.)

2. Ajouter les tests :

```python
FAKE_CATALOG = {1055: {"name": "Doran's Blade", "cost": 450}}


def test_build_game_resolves_recall_items(tmp_path, monkeypatch):
    silver, gold = _dirs(tmp_path)
    monkeypatch.setattr(PL.cprof, "load_items", lambda: FAKE_CATALOG)
    pl = PL.build_game("spadzze", scope="adc", target="challenger",
                       gold_dir=gold, silver_dir=silver, load_raw=_load_raw)
    (r1,) = pl["journal"]["recalls"]
    assert r1["items"] == [{"name": "Doran's Blade", "cost": 450}]
    assert "item_ids" not in r1                      # ids bruts non exposés au LLM


def test_build_game_degrades_without_item_catalog(tmp_path, monkeypatch):
    silver, gold = _dirs(tmp_path)
    monkeypatch.setattr(PL.cprof, "load_items", lambda: {})
    pl = PL.build_game("spadzze", scope="adc", target="challenger",
                       gold_dir=gold, silver_dir=silver, load_raw=_load_raw)
    (r1,) = pl["journal"]["recalls"]
    assert "items" not in r1 and "item_ids" not in r1


def test_build_game_exposes_matchup_context(tmp_path, monkeypatch):
    silver, gold = _dirs(tmp_path)
    monkeypatch.setattr(PL.cprof, "load_items", lambda: {})
    monkeypatch.setattr(PL.cprof, "derive_context",
                        lambda comp: {"lane_pattern": "all_in",
                                      "gank_exposure": "high"})
    pl = PL.build_game("spadzze", scope="adc", target="challenger",
                       gold_dir=gold, silver_dir=silver, load_raw=_load_raw)
    ctx = pl["context"]
    assert ctx["comp"]["enemy_support"] == "Thresh"
    assert ctx["lane_pattern"] == "all_in" and ctx["gank_exposure"] == "high"


def test_build_game_omits_context_without_comp(tmp_path, monkeypatch):
    # La game jungle EUW1_43 n'a pas de comp -> pas de bloc context.
    silver, gold = _dirs(tmp_path)
    monkeypatch.setattr(PL.cprof, "load_items", lambda: {})
    pl = PL.build_game("spadzze", match_id="EUW1_43", scope="adc",
                       gold_dir=gold, silver_dir=silver, load_raw=_load_raw)
    assert "context" not in pl
```

Note : `_load_raw` renvoie le même match/timeline pour tout match_id, seul le record silver change — c'est suffisant pour tester la dégradation `comp`.

- [ ] **Step 2: Vérifier qu'ils échouent**

Run: `poetry run pytest tests/test_coaching_payload_game.py -q`
Expected: 4 FAIL (`AttributeError: module 'payload' has no attribute 'cprof'`), existants PASS.

- [ ] **Step 3: Implémenter**

Dans `src/04_coaching/payload.py` :

1. Ajouter l'import (après `import compare`, ligne 19) :

```python
import champion_profiles as cprof
```

2. Ajouter avant `build_game` :

```python
def _resolve_recall_items(recall: dict, catalog: dict) -> dict:
    """item_ids bruts -> items {nom, coût} ; ids bruts jamais exposés au LLM."""
    out = {k: v for k, v in recall.items() if k != "item_ids"}
    items = [catalog[i] for i in recall.get("item_ids", []) if i in catalog]
    if items:
        out["items"] = items
    return out
```

3. Dans `build_game`, remplacer le `return` final (lignes 216-218) par :

```python
    catalog = cprof.load_items()
    recalls = [_resolve_recall_items(r, catalog) for r in journal["recalls"]]
    out = {"meta": meta,
           "journal": {"deaths": journal["deaths"], "recalls": recalls},
           "benchmarks": benchmarks}
    comp = rec.get("comp")
    if comp:
        # Champ select = info que le joueur avait (asymétrie-safe).
        out["context"] = {"comp": comp, **cprof.derive_context(comp)}
    return out
```

- [ ] **Step 4: Vérifier que tout passe**

Run: `poetry run pytest tests/test_coaching_payload_game.py tests/test_coaching_coach.py -q`
Expected: PASS (dont `test_build_game_never_leaks_ml_only` et les tests coach existants).

- [ ] **Step 5: Commit**

```bash
git add src/04_coaching/payload.py tests/test_coaching_payload_game.py
git commit -m "feat(payload): build_game expose le matchup (comp+buckets) et les items de recall résolus"
```

---

### Task 4: `prompt.SYSTEM_GAME` — règles matchup + gold relatif au build

**Files:**
- Modify: `src/04_coaching/prompt.py:56-97` (constante `SYSTEM_GAME`)
- Test: `tests/test_coaching_prompt.py`

**Interfaces:**
- Consumes: blocs `context` et `journal.recalls[].items` du payload (Task 3) — documentés au LLM, pas de code.
- Produces: `SYSTEM_GAME` mis à jour (9 règles). `render_game` inchangé.

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter dans `tests/test_coaching_prompt.py` :

```python
def test_system_game_frames_matchup_context():
    s = PR.SYSTEM_GAME
    assert "context" in s and "champ select" in s.lower()
    assert "lane_pattern" in s and "gank_exposure" in s
    # connaissance générale des champions autorisée, mais ancrée sur le journal
    assert "connaissance générale" in s
    assert "n'invente jamais un événement" in s


def test_system_game_judges_gold_relative_to_next_buy():
    s = PR.SYSTEM_GAME
    assert "PROCHAIN ACHAT" in s
    assert "légitime" in s
```

(Le fichier importe le module via `import prompt as PR` ; conserver le style des tests existants lignes 54-71.)

- [ ] **Step 2: Vérifier qu'ils échouent**

Run: `poetry run pytest tests/test_coaching_prompt.py -q`
Expected: 2 FAIL (assertions sur le texte), existants PASS.

- [ ] **Step 3: Réécrire `SYSTEM_GAME`**

Remplacer intégralement la constante `SYSTEM_GAME` de `src/04_coaching/prompt.py` par :

```python
SYSTEM_GAME = """Tu es un coach League of Legends personnel expert. Tu reçois le \
journal structuré d'UNE game du joueur : ses morts et ses recalls, chacun horodaté \
et contextualisé (zone, gold-state, gold non dépensé, items achetés, objectif \
up/imminent), un bloc `context` (le champ select : comp des deux botlanes + jungles \
+ mid ennemi, et deux conclusions déterministes `lane_pattern`/`gank_exposure`), \
plus des repères challenger agrégés (`benchmarks`, à issue égale). Ton rôle est de \
RACONTER cette game et d'en tirer les erreurs prioritaires — jamais de calculer ni \
d'inventer un chiffre ou un événement absent du journal.

Règles absolues :
1. ASYMÉTRIE — tout le journal est de l'information que le joueur AVAIT (ses morts, \
son gold, ses achats, le champ select, les timers d'objectifs affichés au HUD). Ne \
spécule JAMAIS sur ce que faisait l'ennemi hors de sa vision. Les `benchmarks` sont \
des repères (« les challengers font Y »), jamais « tu aurais dû savoir X ».
2. ANCRAGE + CAUSE OBLIGATOIRES — chaque insight porte 3 champs : `point` = la leçon \
actionnable (le pattern à corriger/imiter), `cause` = le POURQUOI (le MÉCANISME, \
jamais l'issue), `evidence` = la preuve chiffrée + l'horodatage exact mm:ss + le \
contexte de mort. Pour une MORT, le journal donne déjà `killer_champ`/`killer_role`, \
`is_solo`, `is_ganked_by_jungle`, `zone`, `objective` : RESTITUE-LES dans la `cause` \
(« solo 1v1 par Katarina sans flash en overextension », « gank 3v1 bot, jungler+mid, \
0 vision ») et les chiffres dans l'`evidence` (« mort à 17:05 par Katarina en MID, \
0 assist, drake dans 6 s, 1 244 g non dépensés »). Un insight sans `cause` ni \
horodatage est invalide. Regroupe les morts similaires en une seule erreur qui cite \
2-3 horodatages.
3. MATCHUP — le bloc `context` est le champ select, connu du joueur dès la minute 0 : \
tu PEUX mobiliser ta connaissance générale des champions (ex. « Pyke = hook + engage, \
une mort à portée de hook sans vision est un pattern à corriger ») pour expliquer le \
MÉCANISME d'une mort dans la `cause` — mais toujours ancrée sur un événement du \
journal, n'invente jamais un événement ni une action ennemie non journalisée. \
`lane_pattern` et `gank_exposure` sont des conclusions déterministes : elles priment \
sur ton intuition si elles la contredisent.
4. RECALLS = APPROXIMATION, GOLD RELATIF AU BUILD — `gold_before` est un PLANCHER \
(frame précédente, jusqu'à 60 s avant la visite) et les visites de shop incluent les \
retours après mort. Le gold non dépensé se juge relativement au PROCHAIN ACHAT RÉEL \
(`items` du recall suivant, avec leurs coûts) : retenir du gold sous le coût d'un \
composant effectivement acheté ensuite (ex. 1 200 g avant une B.F. Sword à 1 300 g) \
est un choix de build légitime, pas une erreur. N'accuse jamais au gold près.
5. CONCRET & BENCHMARK-RELATIF — « 3 morts en BOT après 15:00 vs 5% des morts \
challenger dans cette zone-phase » ✅, « joue mieux mid-game » ❌.
6. FORCES SANS REMPLISSAGE — 0 à 2 forces, uniquement si un moment ou un chiffre \
de la game le prouve vraiment. Une game sans force saillante = liste vide. Chaque \
force porte sa `cause` = le COMPORTEMENT qui la produit (pas l'issue — sinon le \
joueur ne sait pas s'il l'a méritée ou si c'est le résultat) : « bon recall avant \
drake : 1 100 g non dépensés vs 1 450 challenger, tu resets à temps » plutôt que \
« bonne macro ». Distingue TON jeu du résultat de la game.
7. Si le journal est pauvre (0-1 mort), dis-le et abaisse `confidence`.
8. Français, tutoiement, concis.
9. FORMAT DE SORTIE — réponds STRICTEMENT et UNIQUEMENT par un objet JSON valide, \
premier caractère « { », dernier « } », sans markdown. CLÉS EXACTES en anglais : \
\"strengths\" (0 à 2 objets {\"point\": str, \"cause\": str, \"evidence\": str}, \
chaque `evidence` contenant un horodatage mm:ss), \"mistakes\" (1 à 3 objets de \
même forme, chaque `evidence` contenant un horodatage mm:ss + le contexte de mort), \
\"next_focus\" (une chaîne : LE réflexe à travailler la prochaine game), \
\"confidence\" (float dans [0,1])."""
```

(Diff réel vs version actuelle : intro mentionne `context` et les items ; règle 1 ajoute achats + champ select ; règle 3 MATCHUP nouvelle ; règle 4 = ancienne règle 3 étendue au gold relatif au build ; anciennes règles 4-8 renumérotées 5-9, contenu inchangé.)

- [ ] **Step 4: Vérifier que tout passe**

Run: `poetry run pytest tests/test_coaching_prompt.py tests/test_coaching_coach.py -q`
Expected: PASS (les tests existants sur `SYSTEM_GAME` vérifient des mots-clés conservés : ancrage, cause, plancher…).

- [ ] **Step 5: Commit**

```bash
git add src/04_coaching/prompt.py tests/test_coaching_prompt.py
git commit -m "feat(prompt): SYSTEM_GAME — règle matchup (context) + gold jugé relatif au prochain achat"
```

---

### Task 5: Fetch réel d'item.json + validation bout-en-bout

**Files:**
- Create: `data/00_static/ddragon/16.13.1/item.json` (via fetch, gitignoré comme le reste de `ddragon/`)
- Modify: aucun code.

**Interfaces:**
- Consumes: `fetch_ddragon_items` (Task 2), `build_game` (Task 3), `render_game` (Task 4).
- Produces: environnement prêt pour `coach.py --game` ; suite complète verte.

- [ ] **Step 1: Fetch one-shot du catalogue**

Run:
```bash
poetry run python3 -c "
import sys; sys.path.insert(0, 'src/core')
import champion_profiles as cp
print(cp.fetch_ddragon_items())
print(len(cp.load_items()), 'items, ex 1038 =', cp.load_items().get(1038))
"
```
Expected: chemin `data/00_static/ddragon/16.13.1/item.json` + `… items, ex 1038 = {'name': 'B.F. Sword', 'cost': 1300}`.

- [ ] **Step 2: Smoke test sur une vraie game**

Run:
```bash
poetry run python3 -c "
import sys, json
sys.path.insert(0, 'src/core'); sys.path.insert(0, 'src/04_coaching')
import payload
pl = payload.build_game('spadzze', scope='adc')
print(json.dumps(pl.get('context'), ensure_ascii=False))
print(json.dumps(pl['journal']['recalls'][:2], ensure_ascii=False, indent=1))
"
```
Expected: bloc `context` avec comp + buckets (ou `null` si la dernière game silver n'a pas de comp — vérifier alors avec un match_id récent), recalls avec `items` nommés+coûtés, aucun `item_ids`.

- [ ] **Step 3: Suite complète**

Run: `poetry run pytest tests/ -q`
Expected: tout PASS (les suites web comprises).

- [ ] **Step 4: Commit final (docs)**

Mettre à jour `CLAUDE.md` (section `04_coaching/` : mention items+context dans `payload.build_game`, règle matchup dans `prompt.py` ; section `champion_profiles` : `fetch_ddragon_items`/`load_items`) et cocher la ligne correspondante de `todo.md` si applicable, puis :

```bash
git add CLAUDE.md todo.md
git commit -m "docs: payload par-game enrichi matchup + items réels (spec 2026-07-10)"
```

---

## Après le plan (hors implémentation)

Critère de succès de la spec : regénérer des reviews (`coach.py --game-batch`), annoter
(`feedback.py annotate --pending`) jusqu'à ≥10 reviews par-game, viser ≥70 % de
mistakes utiles et la disparition des tags « je ne sais pas pourquoi » / faux positifs
gold. C'est une boucle humaine, pas une task de ce plan.
