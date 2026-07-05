# Boucle de feedback batch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Débloquer la métrique « ≥70 % de mistakes utiles sur ≥10 reviews par-game annotées » en ajoutant la génération batch (`coach.py --game-batch N`), l'annotation en série (`feedback.py annotate --pending`) et le suivi de la métrique (bloc « Objectif » dans `summary`).

**Architecture:** Trois blocs indépendants sur le code coaching existant (`src/04_coaching/`). Chaque bloc = une fonction pure testable (sélection/jointure/stats) + un câblage CLI fin. Aucun changement de prompt, schéma ou modèle. Spec : `docs/superpowers/specs/2026-07-06-feedback-loop-batch-design.md`.

**Tech Stack:** Python 3 (Poetry), pytest, argparse, Pydantic (schémas existants `Review`/`GameReview`/`Feedback`). Aucune nouvelle dépendance.

## Global Constraints

- Convention flat-import : les modules de `src/04_coaching/` s'importent entre eux directement (`import payload`, `import feedback`) — `tests/conftest.py` ajoute déjà ce dossier au path ; ne PAS créer de package.
- Lancer les tests avec `poetry run pytest ...` depuis la racine du repo.
- Commits : conventional `type(scope): sujet` (minuscule, impératif, ≤72 chars) + trailer exact `Co-Authored-By: Claude <noreply@anthropic.com>` (convention repo — PAS le trailer par défaut du harness). Jamais `git add .`.
- Style code : français dans docstrings/messages, em-dash `—`, commentaires seulement pour les contraintes non évidentes.
- Asymétrie : rien dans ce plan ne touche au contenu des payloads — aucune nouvelle donnée exposée au LLM.
- `--game-batch` est **incompatible avec `--game`** (erreur argparse). Défaut N=10.
- Le bloc « Objectif » se calcule **uniquement sur les mistakes des reviews `kind=game`**, sur les feedbacks NON filtrés (les filtres `--model`/`--tag` de summary ne s'y appliquent pas : c'est la métrique globale du projet).

---

### Task 1: `payload.filter_scope` — extraire le filtre de scope (DRY pour la sélection batch)

Le filtre scope (adc→role BOTTOM, all→tout, sinon champion) vit dans `payload._select_game`. La sélection batch (Task 2) en a besoin aussi → l'extraire en fonction publique.

**Files:**
- Modify: `src/04_coaching/payload.py:162-176` (`_select_game`)
- Test: `tests/test_coaching_payload_game.py`

**Interfaces:**
- Produces: `filter_scope(records: list[dict], scope: str) -> list[dict]` — sous-liste des records du scope, ordre du fichier préservé. `"adc"` → `role == "BOTTOM"` ; `"all"` → copie de tout ; sinon → `champion` égal casse-insensible.

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à la fin de `tests/test_coaching_payload_game.py` :

```python
# --- filter_scope (partagé _select_game / sélection batch) --------------------

def test_filter_scope_by_role_champion_and_all():
    records = [
        {"match_id": "m1", "role": "BOTTOM", "champion": "Zeri"},
        {"match_id": "m2", "role": "MIDDLE", "champion": "Ahri"},
        {"match_id": "m3", "role": "BOTTOM", "champion": "Jinx"},
    ]
    assert [r["match_id"] for r in P.filter_scope(records, "adc")] == ["m1", "m3"]
    assert [r["match_id"] for r in P.filter_scope(records, "zeri")] == ["m1"]
    assert [r["match_id"] for r in P.filter_scope(records, "all")] == ["m1", "m2", "m3"]
```

Vérifier d'abord comment `payload` est importé dans ce fichier de test (`head -20 tests/test_coaching_payload_game.py`) — si c'est `import payload as P` garder `P.`, sinon adapter le préfixe au nom utilisé.

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_coaching_payload_game.py::test_filter_scope_by_role_champion_and_all -v`
Attendu : FAIL avec `AttributeError: module 'payload' has no attribute 'filter_scope'`

- [ ] **Step 3: Implémenter**

Dans `src/04_coaching/payload.py`, juste après `_SCOPE_ROLE = {"adc": "BOTTOM"}` :

```python
def filter_scope(records: list[dict], scope: str) -> list[dict]:
    """Sous-liste des records du scope, ordre du fichier préservé."""
    role = _SCOPE_ROLE.get(scope)
    if role:
        return [r for r in records if r.get("role") == role]
    if scope == "all":
        return list(records)
    return [r for r in records if (r.get("champion") or "").lower() == scope.lower()]
```

Puis remplacer le corps du filtre dans `_select_game` :

```python
def _select_game(records: list[dict], scope: str, match_id: str | None) -> dict:
    if match_id is not None:
        rec = next((r for r in records if r.get("match_id") == match_id), None)
        if rec is None:
            raise FileNotFoundError(f"game {match_id} absente du silver perso")
        return rec
    records = filter_scope(records, scope)
    if not records:
        raise FileNotFoundError(f"aucune game du scope {scope} dans le silver perso")
    return records[-1]           # la plus récente du scope
```

- [ ] **Step 4: Vérifier le vert + non-régression du fichier**

Run: `poetry run pytest tests/test_coaching_payload_game.py -v`
Attendu : PASS partout (les tests `_select_game` existants restent verts).

- [ ] **Step 5: Commit**

```bash
git add src/04_coaching/payload.py tests/test_coaching_payload_game.py
git commit -m "refactor(coaching): extract payload.filter_scope from _select_game

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: `coach.pending_game_matches` — sélection pure des games à reviewer

**Files:**
- Modify: `src/04_coaching/coach.py` (nouvelle fonction, après `generate_game_review`)
- Test: `tests/test_coaching_coach.py`

**Interfaces:**
- Consumes: `payload.filter_scope(records, scope)` (Task 1).
- Produces: `pending_game_matches(records: list[dict], reviews: list[dict], scope: str, n: int) -> list[str]` — match_ids du scope SANS review `kind=game` (dédup quel que soit le modèle), plus récents d'abord (tri `game_ts` décroissant ; records sans `game_ts` — silver perso antérieur au 2026-07-06 — classés après, en ordre d'apparition inversé du fichier), limité à `n`.

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à la fin de `tests/test_coaching_coach.py` :

```python
# --- pending_game_matches (sélection batch, pure) -----------------------------

def test_pending_game_matches_dedups_scopes_and_sorts_recent_first():
    records = [
        {"match_id": "m1", "role": "BOTTOM", "game_ts": 100},
        {"match_id": "m2", "role": "BOTTOM", "game_ts": 300},   # déjà reviewée
        {"match_id": "m3", "role": "MIDDLE", "game_ts": 400},   # hors scope adc
        {"match_id": "m4", "role": "BOTTOM", "game_ts": 200},
    ]
    reviews = [{"kind": "game", "match_id": "m2", "model": "kimi-k2.6"},
               {"outcome_focus": "loss"}]          # agrégée : ne dédupe rien
    got = C.pending_game_matches(records, reviews, "adc", n=10)
    assert got == ["m4", "m1"]                     # ts décroissant, m2/m3 exclues


def test_pending_game_matches_limits_to_n():
    records = [{"match_id": f"m{i}", "role": "BOTTOM", "game_ts": i}
               for i in range(5)]
    got = C.pending_game_matches(records, [], "adc", n=2)
    assert got == ["m4", "m3"]


def test_pending_game_matches_falls_back_to_reversed_file_order():
    # silver perso antérieur au 2026-07-06 : pas de game_ts -> ordre inversé du fichier
    records = [{"match_id": f"m{i}", "role": "BOTTOM"} for i in range(3)]
    got = C.pending_game_matches(records, [], "adc", n=10)
    assert got == ["m2", "m1", "m0"]
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_coaching_coach.py -v -k pending_game_matches`
Attendu : 3 FAIL avec `AttributeError: module 'coach' has no attribute 'pending_game_matches'`

- [ ] **Step 3: Implémenter**

Dans `src/04_coaching/coach.py`, après `generate_game_review` :

```python
def pending_game_matches(records: list[dict], reviews: list[dict],
                         scope: str, n: int) -> list[str]:
    """Match_ids du scope sans review kind=game (dédup quel que soit le modèle),
    plus récents d'abord. Le silver perso antérieur au 2026-07-06 n'a pas de
    game_ts -> ces records (ts traité comme 0) retombent sur l'ordre
    d'apparition inversé du fichier, approximation de l'ordre de collecte."""
    reviewed = {r.get("match_id") for r in reviews if r.get("kind") == "game"}
    indexed = list(enumerate(payload_mod.filter_scope(records, scope)))
    indexed.sort(key=lambda p: (p[1].get("game_ts") or 0, p[0]), reverse=True)
    out: list[str] = []
    for _, rec in indexed:
        mid = rec.get("match_id")
        if mid in reviewed or mid in out:
            continue
        out.append(mid)
        if len(out) >= n:
            break
    return out
```

- [ ] **Step 4: Vérifier le vert**

Run: `poetry run pytest tests/test_coaching_coach.py -v`
Attendu : PASS partout.

- [ ] **Step 5: Commit**

```bash
git add src/04_coaching/coach.py tests/test_coaching_coach.py
git commit -m "feat(coaching): pending_game_matches — sélection pure des games à reviewer

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: `coach.run_batch` + flag CLI `--game-batch N`

**Files:**
- Modify: `src/04_coaching/coach.py` (import `feedback`, fonction `run_batch`, argparse dans `main()`)
- Test: `tests/test_coaching_coach.py`

**Interfaces:**
- Consumes: `pending_game_matches` (Task 2), `payload._personal_records(player, silver_dir)`, `payload.build_game(player, match_id=, scope=, target=, silver_dir=)`, `feedback.list_reviews(player, root)`, `generate_game_review(pl, model)`, `persist(player, model, pl, review, ts, root=)`, `_save_failed(player, ts, raw)`.
- Produces: `run_batch(player: str, scope: str, target: str, model: str, n: int, root=None, silver_dir=None) -> int` — 0 si au moins une review générée ou rien à faire ; 1 si toutes les games tentées ont échoué. CLI : `--game-batch [N]` (const=10), mutuellement exclusif avec `--game`.

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à la fin de `tests/test_coaching_coach.py` (réutilise `_game_payload()` et `_game_review_dict()` définis plus haut dans le fichier) :

```python
# --- run_batch + --game-batch --------------------------------------------------

def _batch_env(tmp_path, n_games=3, reviewed=("EUW1_2",)):
    """Silver perso ADC + reviews.jsonl existant -> (root, silver_dir)."""
    silver = tmp_path / "silver"
    pdir = silver / "personal" / "spadzze"
    pdir.mkdir(parents=True)
    recs = [{"match_id": f"EUW1_{i}", "role": "BOTTOM", "puuid": "p",
             "champion": "Zeri", "game_ts": i} for i in range(n_games)]
    (pdir / "games.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")
    root = tmp_path / "07_coaching"
    out = root / "spadzze"
    out.mkdir(parents=True)
    lines = [{"ts": f"t{m}", "model": "kimi-k2.6", "kind": "game",
              "match_id": m, "scope": "adc", "target": "challenger",
              "payload": {"meta": {}}, "review": _game_review_dict()}
             for m in reviewed]
    (out / "reviews.jsonl").write_text(
        "\n".join(json.dumps(l) for l in lines) + "\n")
    return root, silver


def test_run_batch_generates_dedups_and_continues_on_error(tmp_path, monkeypatch, capsys):
    root, silver = _batch_env(tmp_path)          # EUW1_2 déjà reviewée

    def fake_build_game(player, match_id=None, **kw):
        if match_id == "EUW1_0":
            raise FileNotFoundError(f"raw manquant pour {match_id}")
        pl = _game_payload()
        pl["meta"]["match_id"] = match_id
        return pl

    monkeypatch.setattr(C.payload_mod, "build_game", fake_build_game)
    monkeypatch.setattr(C.llm_client, "generate_json",
                        lambda *a, **k: _game_review_dict())
    rc = C.run_batch("spadzze", "adc", "challenger", "m", 10,
                     root=root, silver_dir=silver)
    assert rc == 0
    lines = [json.loads(l) for l in
             (root / "spadzze" / "reviews.jsonl").read_text().splitlines()]
    new_ids = {l["match_id"] for l in lines if l["ts"] != "tEUW1_2"}
    assert new_ids == {"EUW1_1"}                 # EUW1_0 échouée, EUW1_2 dédupliquée
    out = capsys.readouterr().out
    assert "1 générée" in out and "1 déjà reviewée" in out and "1 échouée" in out


def test_run_batch_returns_1_when_all_attempts_fail(tmp_path, monkeypatch, capsys):
    root, silver = _batch_env(tmp_path, reviewed=())

    def boom(*a, **k):
        raise C.llm_client.LLMError("api down")

    monkeypatch.setattr(C.payload_mod, "build_game",
                        lambda player, match_id=None, **kw: _game_payload())
    monkeypatch.setattr(C.llm_client, "generate_json", boom)
    rc = C.run_batch("spadzze", "adc", "challenger", "m", 2,
                     root=root, silver_dir=silver)
    assert rc == 1
    assert "échouée" in capsys.readouterr().out


def test_run_batch_nothing_to_do(tmp_path, capsys):
    root, silver = _batch_env(tmp_path, n_games=1, reviewed=("EUW1_0",))
    rc = C.run_batch("spadzze", "adc", "challenger", "m", 10,
                     root=root, silver_dir=silver)
    assert rc == 0
    assert "déjà reviewée" in capsys.readouterr().out


def test_main_game_and_game_batch_mutually_exclusive(monkeypatch, capsys):
    monkeypatch.setattr(C.sys, "argv",
                        ["coach.py", "--game", "--game-batch", "5"])
    with pytest.raises(SystemExit) as e:
        C.main()
    assert e.value.code == 2                     # erreur argparse
```

Vérifier d'abord que `llm_client.LLMError("api down")` se construit avec un seul argument positionnel (`grep -n "class LLMError" src/04_coaching/llm_client.py`) ; sinon adapter l'instanciation dans le test.

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_coaching_coach.py -v -k "run_batch or mutually"`
Attendu : 3 FAIL `AttributeError: ... 'run_batch'` + 1 FAIL sur l'exclusivité (argparse accepte aujourd'hui les deux flags, donc `main()` continue et le test échoue autrement — l'important est de le voir ROUGE).

- [ ] **Step 3: Implémenter**

Dans `src/04_coaching/coach.py` :

3a. Ajouter l'import (après `import llm_client`) :

```python
import feedback as feedback_mod
```

3b. Ajouter `run_batch` après `pending_game_matches` :

```python
def run_batch(player: str, scope: str, target: str, model: str, n: int,
              root=None, silver_dir=None) -> int:
    """Génère jusqu'à n reviews par-game sur les games du scope pas encore
    reviewées (kind=game). Continue sur échec d'une game ; bilan final."""
    silver = Path(silver_dir) if silver_dir is not None else rl.SILVER_DIR
    records = payload_mod._personal_records(player, silver)
    reviews = feedback_mod.list_reviews(player, root)
    already = len({r.get("match_id") for r in reviews if r.get("kind") == "game"}
                  & {r.get("match_id") for r in payload_mod.filter_scope(records, scope)})
    pending = pending_game_matches(records, reviews, scope, n)
    done, failed = 0, 0
    seen_ts: set[str] = set()
    for mid in pending:
        ts = datetime.now().isoformat(timespec="seconds")
        if ts in seen_ts:                        # 2 games dans la même seconde
            ts = datetime.now().isoformat()      # (mocks/tests) -> microsecondes
        seen_ts.add(ts)
        try:
            pl = payload_mod.build_game(player, match_id=mid, scope=scope,
                                        target=target, silver_dir=silver)
            review = generate_game_review(pl, model)
        except FileNotFoundError as e:
            print(f"✗ {mid} : {e}", file=sys.stderr)
            failed += 1
            continue
        except llm_client.LLMError as e:
            print(f"✗ {mid} : appel LLM échoué : {e}", file=sys.stderr)
            failed += 1
            continue
        except CoachValidationError as e:
            p = _save_failed(player, ts, e.raw)
            print(f"✗ {mid} : {e} — brut sauvé dans {p}", file=sys.stderr)
            failed += 1
            continue
        persist(player, model, pl, review, ts, root=root)
        m = pl["meta"]
        issue = "victoire" if m.get("win") else "défaite"
        print(f"✓ {mid} : {m.get('champion')} ({issue}) reviewée")
        done += 1
    s = lambda k: "s" if k > 1 else ""
    print(f"\nBilan : {done} générée{s(done)} · {already} déjà reviewée{s(already)} "
          f"· {failed} échouée{s(failed)}")
    return 1 if (pending and done == 0) else 0
```

3c. Dans `main()`, remplacer la déclaration de `--game` par un groupe mutuellement exclusif, et brancher le batch juste après la résolution du modèle :

```python
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--game", nargs="?", const="latest", default=None,
                     metavar="MATCH_ID",
                     help="review par-game : dernière game du scope, ou un match_id")
    grp.add_argument("--game-batch", type=int, nargs="?", const=10, default=None,
                     metavar="N",
                     help="génère les reviews par-game des N dernières games "
                          "du scope pas encore reviewées (défaut 10)")
```

et après le bloc de résolution du modèle (`if args.model is None: ...`) :

```python
    if args.game_batch is not None:
        return run_batch(args.player, args.scope, args.target,
                         args.model, args.game_batch)
```

- [ ] **Step 4: Vérifier le vert**

Run: `poetry run pytest tests/test_coaching_coach.py -v`
Attendu : PASS partout (y compris les tests main/persist existants).

- [ ] **Step 5: Commit**

```bash
git add src/04_coaching/coach.py tests/test_coaching_coach.py
git commit -m "feat(coaching): coach.py --game-batch N — génération batch de reviews par-game

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: `feedback.pending_reviews` + `annotate --pending` (annotation en série)

**Files:**
- Modify: `src/04_coaching/feedback.py` (fonction pure `pending_reviews`, extraction `_annotate_one`, param `pending` d'`annotate`, flag argparse)
- Test: `tests/test_coaching_feedback.py`

**Interfaces:**
- Consumes: `list_reviews`, `load_feedbacks`, `build_feedback`, `persist_feedback`, `_display_items`, `_prompt_useful` (existants).
- Produces: `pending_reviews(reviews: list[dict], fbs: list[schema.Feedback]) -> list[dict]` — reviews sans feedback (jointure `ts`), plus anciennes d'abord. `annotate(player, ts=None, last=False, pending=False, root=None, prompt=None) -> int`. Helper interne `_annotate_one(chosen: dict, player: str, root, prompt) -> int`.

- [ ] **Step 1: Écrire le test de la fonction pure (échec)**

Ajouter à la fin de `tests/test_coaching_feedback.py` :

```python
# --- pending_reviews + annotate --pending --------------------------------------

def test_pending_reviews_joins_by_ts_oldest_first():
    reviews = [{"ts": "2026-07-03T10:00:00"}, {"ts": "2026-07-01T10:00:00"},
               {"ts": "2026-07-02T10:00:00"}]
    fbs = [_fb("2026-07-02T10:00:00", "m", [_it("strength", True)])]
    got = F.pending_reviews(reviews, fbs)
    assert [r["ts"] for r in got] == ["2026-07-01T10:00:00", "2026-07-03T10:00:00"]
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_coaching_feedback.py::test_pending_reviews_joins_by_ts_oldest_first -v`
Attendu : FAIL `AttributeError: module 'feedback' has no attribute 'pending_reviews'`

- [ ] **Step 3: Implémenter la fonction pure**

Dans `src/04_coaching/feedback.py`, avant la section `# --- annotate (flow interactif) + main()` :

```python
def pending_reviews(reviews: list[dict],
                    fbs: list[schema_mod.Feedback]) -> list[dict]:
    """Reviews sans feedback (jointure par ts), plus anciennes d'abord.
    Une review entièrement skippée n'est jamais persistée -> reste pending."""
    done = {f.ts for f in fbs}
    return sorted((r for r in reviews if r.get("ts") not in done),
                  key=lambda r: r.get("ts") or "")
```

Run: `poetry run pytest tests/test_coaching_feedback.py::test_pending_reviews_joins_by_ts_oldest_first -v` → PASS.

- [ ] **Step 4: Écrire les tests du flow `--pending` (échec)**

Ajouter à la suite (réutilise `_review_dict`, `_game_review_record`, `_write_reviews` définis plus haut dans le fichier — la review agrégée a ts `2026-06-30T10:00:00`, la par-game `2026-07-05T10:00:00`) :

```python
def test_annotate_pending_iterates_oldest_first_and_quits(tmp_path):
    _write_reviews(tmp_path, lines=[_game_review_record(), _review_dict()])
    answers = iter([
        "",                                       # Entrée = annoter (l'agrégée, plus ancienne)
        "y", "y", "y", "y", "y", "y", "y", "y", "y",   # ses 9 items
        "q",                                      # quitter avant la par-game
    ])
    rc = F.annotate("spadzze", pending=True, root=tmp_path,
                    prompt=lambda _m: next(answers))
    assert rc == 0
    fbs = F.load_feedbacks("spadzze", root=tmp_path)
    assert len(fbs) == 1 and fbs[0].ts == "2026-06-30T10:00:00"


def test_annotate_pending_skip_leaves_review_pending(tmp_path):
    _write_reviews(tmp_path)                      # 1 review agrégée
    answers = iter(["n"])                         # passer -> rien persisté
    rc = F.annotate("spadzze", pending=True, root=tmp_path,
                    prompt=lambda _m: next(answers))
    assert rc == 0
    assert F.load_feedbacks("spadzze", root=tmp_path) == []


def test_annotate_pending_none_left(tmp_path, capsys):
    _write_reviews(tmp_path)
    fb = _fb("2026-06-30T10:00:00", "kimi-k2.6", [_it("strength", True)])
    F.persist_feedback("spadzze", fb, root=tmp_path)
    rc = F.annotate("spadzze", pending=True, root=tmp_path,
                    prompt=lambda _m: "")
    assert rc == 0
    assert "en attente" in capsys.readouterr().out.lower()
```

- [ ] **Step 5: Vérifier l'échec**

Run: `poetry run pytest tests/test_coaching_feedback.py -v -k annotate_pending`
Attendu : 3 FAIL `TypeError: annotate() got an unexpected keyword argument 'pending'`

- [ ] **Step 6: Implémenter le flow**

Dans `src/04_coaching/feedback.py` :

6a. Extraire le corps d'annotation d'une review (tout ce qui suit la sélection de `chosen` dans `annotate` actuel : de `cls = ...` jusqu'au `return 0` final) en helper — code identique, seulement déplacé :

```python
def _annotate_one(chosen: dict, player: str, root, prompt) -> int:
    """Annote UNE review déjà sélectionnée ; persiste si >= 1 item noté."""
    cls = (schema_mod.GameReview if chosen.get("kind") == "game"
           else schema_mod.Review)
    review = cls.model_validate(chosen["review"])
    items = _display_items(review)
    responses: dict[tuple[str, int], tuple[bool, str | None, str | None]] = {}
    for kind, idx, line in items:
        ans = _prompt_useful(prompt, line)
        if ans is None or ans == "s":
            continue
        useful = (ans == "y")
        tag = None
        note = None
        if not useful:
            menu = "\n".join(f"  {j+1} | {t}" for j, t in enumerate(schema_mod.NEG_TAGS))
            while tag is None:
                ts_in = prompt(f"Pourquoi ? (numéro)\n{menu}\n  tag : ").strip()
                try:
                    tag = schema_mod.NEG_TAGS[int(ts_in) - 1]
                except (ValueError, IndexError):
                    pass
            note = prompt("Note (optionnel, Entrée = skip) : ").strip() or None
        responses[(kind, idx)] = (useful, tag, note)
    if not responses:
        print("Tout skippé — rien à persister.")
        return 0
    rated_at = datetime.now().isoformat(timespec="seconds")
    fb = build_feedback(review, ts=chosen["ts"], player=player,
                        model=chosen["model"], rated_at=rated_at, responses=responses)
    path, overwrote = persist_feedback(player, fb, root)
    n_useful = sum(1 for it in fb.items if it.useful)
    print(f"\n{len(fb.items)} items notés ({n_useful} utiles).")
    if overwrote:
        print(f"réannotation: écrase feedback précédent pour {fb.ts}")
    print(f"✓ feedback persisté dans {path}")
    return 0
```

6b. Réécrire `annotate` pour déléguer à `_annotate_one` et ajouter le mode pending :

```python
def annotate(player: str, ts: str | None = None, last: bool = False,
             pending: bool = False, root=None, prompt=None) -> int:
    if prompt is None:
        prompt = input           # late binding : monkeypatch builtins.input pris en compte
    reviews = list_reviews(player, root)
    if not reviews:
        print("Aucune review pour ce joueur — génère-en via coach.py d'abord.")
        return 0
    if pending:
        pend = pending_reviews(reviews, load_feedbacks(player, root))
        if not pend:
            print("Aucune review en attente d'annotation.")
            return 0
        for i, r in enumerate(pend, 1):
            kind = "game" if r.get("kind") == "game" else "agrégée"
            what = r.get("match_id") or r.get("outcome_focus") or "?"
            ans = prompt(f"\n({i}/{len(pend)}) [{kind}] {r['ts']} | {r['model']} "
                         f"| {what}\n  [Entrée=annoter / n=passer / q=quitter] : "
                         ).strip().lower()
            if ans == "q":
                break
            if ans == "n":
                continue
            _annotate_one(r, player, root, prompt)
        return 0
    if ts is None:
        if last:
            chosen = reviews[-1]
        else:
            print("Reviews disponibles :")
            for i, r in enumerate(reviews, 1):
                what = r.get("outcome_focus") or r.get("match_id") or "?"
                print(f"  {i} | {r['ts']} | {r['model']} | {what}")
            sel = prompt("Choisis une review (numéro) : ").strip()
            try:
                chosen = reviews[int(sel) - 1]
            except (ValueError, IndexError):
                print("✗ sélection invalide")
                return 1
    else:
        chosen = next((r for r in reviews if r.get("ts") == ts), None)
        if chosen is None:
            print(f"✗ ts {ts} introuvable dans reviews.jsonl")
            return 1
    return _annotate_one(chosen, player, root, prompt)
```

6c. Dans `main()`, câbler le flag :

```python
    a.add_argument("--pending", action="store_true",
                   help="annoter en série toutes les reviews sans feedback")
```

et :

```python
    if args.cmd == "annotate":
        return annotate(args.player, ts=args.ts, last=args.last,
                        pending=args.pending)
```

- [ ] **Step 7: Vérifier le vert + non-régression**

Run: `poetry run pytest tests/test_coaching_feedback.py -v`
Attendu : PASS partout (les tests annotate/main existants restent verts — le refactor `_annotate_one` ne change aucun comportement).

- [ ] **Step 8: Commit**

```bash
git add src/04_coaching/feedback.py tests/test_coaching_feedback.py
git commit -m "feat(coaching): feedback.py annotate --pending — annotation en série

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: `objective_stats` + bloc « Objectif » dans `summary`

**Files:**
- Modify: `src/04_coaching/feedback.py` (fonctions `objective_stats`/`render_objective`, câblage `main()` summary)
- Test: `tests/test_coaching_feedback.py`

**Interfaces:**
- Consumes: `load_feedbacks`, `list_reviews` (existants).
- Produces: `objective_stats(fbs: list[schema.Feedback], reviews: list[dict]) -> dict` avec clés `n_game_reviews_annotated: int`, `target_n: int (=10)`, `mistake_useful_rate: float | None`, `target_rate: float (=0.70)` ; `render_objective(obj: dict) -> str`.

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à la fin de `tests/test_coaching_feedback.py` :

```python
# --- objective_stats (métrique >=70 % de mistakes utiles sur >=10 par-game) ----

def test_objective_stats_counts_only_game_review_mistakes():
    reviews = [{"ts": "tg", "kind": "game", "match_id": "m1"},
               {"ts": "ta", "outcome_focus": "loss"}]
    fbs = [_fb("tg", "m", [_it("mistake", True),
                           _it("mistake", False, "trop-vague"),
                           _it("strength", False, "trop-vague")]),  # pas une mistake
           _fb("ta", "m", [_it("mistake", True)])]                  # agrégée : exclue
    s = F.objective_stats(fbs, reviews)
    assert s["n_game_reviews_annotated"] == 1
    assert s["target_n"] == 10
    assert s["mistake_useful_rate"] == pytest.approx(0.5)
    assert s["target_rate"] == 0.70


def test_objective_stats_robust_without_game_annotations():
    s = F.objective_stats([], [])
    assert s["n_game_reviews_annotated"] == 0
    assert s["mistake_useful_rate"] is None


def test_render_objective_formats_line():
    line = F.render_objective({"n_game_reviews_annotated": 3, "target_n": 10,
                               "mistake_useful_rate": 0.5, "target_rate": 0.70})
    assert "Objectif par-game : 3/10 reviews annotées" in line
    assert "50" in line and "70" in line


def test_render_objective_dash_when_no_rate():
    line = F.render_objective({"n_game_reviews_annotated": 0, "target_n": 10,
                               "mistake_useful_rate": None, "target_rate": 0.70})
    assert "0/10" in line and "—" in line


def test_main_summary_shows_objective_block(tmp_path, monkeypatch, capsys):
    root = tmp_path / "07_coaching"
    _write_reviews(root, lines=[_game_review_record()])
    monkeypatch.setattr(F.rl, "DATA", tmp_path)
    answers = iter(["y", "y"])                   # mistake + focus de la par-game
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    assert F.main(["annotate", "--player", "spadzze", "--last"]) == 0
    capsys.readouterr()
    assert F.main(["summary", "--player", "spadzze"]) == 0
    out = capsys.readouterr().out
    assert "Objectif par-game : 1/10 reviews annotées" in out
    assert "100" in out                          # 1/1 mistake utile
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_coaching_feedback.py -v -k objective`
Attendu : 5 FAIL (`AttributeError: ... 'objective_stats'` / `'render_objective'` ; le test main échoue sur l'assertion « Objectif »).

- [ ] **Step 3: Implémenter**

Dans `src/04_coaching/feedback.py` :

3a. Après les constantes de la section summarize (`_NOTES_PER_TAG = 2`), ajouter :

```python
_OBJECTIVE_N = 10          # métrique projet : >=70 % de mistakes utiles
_OBJECTIVE_RATE = 0.70     # sur >=10 reviews par-game annotées
```

3b. Après `summarize`, ajouter :

```python
def objective_stats(fbs: list[schema_mod.Feedback],
                    reviews: list[dict]) -> dict:
    """Métrique de succès par-game. Le feedback ne stocke pas le kind ->
    jointure ts avec reviews.jsonl ; seules les mistakes des reviews kind=game
    comptent (définition de la métrique)."""
    game_ts = {r.get("ts") for r in reviews if r.get("kind") == "game"}
    game_fbs = [f for f in fbs if f.ts in game_ts]
    mistakes = [it for f in game_fbs for it in f.items if it.kind == "mistake"]
    useful = sum(1 for it in mistakes if it.useful)
    return {"n_game_reviews_annotated": len(game_fbs),
            "target_n": _OBJECTIVE_N,
            "mistake_useful_rate": (useful / len(mistakes)) if mistakes else None,
            "target_rate": _OBJECTIVE_RATE}


def render_objective(obj: dict) -> str:
    rate = ("—" if obj["mistake_useful_rate"] is None
            else f"{obj['mistake_useful_rate']:.0%}")
    return (f"Objectif par-game : {obj['n_game_reviews_annotated']}"
            f"/{obj['target_n']} reviews annotées · mistakes utiles {rate} "
            f"(cible ≥{obj['target_rate']:.0%})")
```

3c. Dans `main()`, brancher le bloc summary — la métrique se calcule sur les feedbacks NON filtrés (`--model`/`--tag` ne s'y appliquent pas) :

```python
    if args.cmd == "summary":
        fbs = load_feedbacks(args.player)
        objective = objective_stats(fbs, list_reviews(args.player))
        if args.model:
            fbs = [f for f in fbs if f.model == args.model]
        if args.tag:
            fbs = [f for f in fbs if any(it.tag == args.tag for it in f.items)]
        print(f"FEEDBACK — {args.player}")
        print(render_objective(objective))
        print(render_summary(summarize(fbs)))
        return 0
```

- [ ] **Step 4: Vérifier le vert + suite complète**

Run: `poetry run pytest tests/test_coaching_feedback.py -v` puis `poetry run pytest tests/`
Attendu : PASS partout (~285 tests, 273 avant + les nouveaux).

- [ ] **Step 5: Commit**

```bash
git add src/04_coaching/feedback.py tests/test_coaching_feedback.py
git commit -m "feat(coaching): bloc Objectif dans summary — suivi de la métrique par-game

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Vérification de bout en bout + sync CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (bullet `src/04_coaching/` dans « Architecture du code » ; item 1 de « Prochaines étapes »)

**Interfaces:** aucune — documentation + smoke test.

- [ ] **Step 1: Suite complète**

Run: `poetry run pytest tests/`
Attendu : PASS partout, 0 warning nouveau.

- [ ] **Step 2: Smoke test CLI sans réseau**

```bash
poetry run python3 src/04_coaching/coach.py --help
poetry run python3 src/04_coaching/feedback.py annotate --help
poetry run python3 src/04_coaching/feedback.py summary --player spadzze
```

Attendu : `--game-batch [N]` et `--pending` apparaissent dans les help ; summary affiche la ligne `Objectif par-game : 0/10 ...` (1 seule annotation existante, agrégée). Ne PAS lancer `--game-batch` sans accord de l'utilisateur (10 appels LLM payants).

- [ ] **Step 3: Sync CLAUDE.md (édits chirurgicaux)**

Dans le bullet `src/04_coaching/` d'« Architecture du code » : après la description du chemin par-game (`coach.py --game [latest|MATCH_ID]`), insérer une phrase sur `--game-batch N` (défaut 10, dédup par match_id reviewé, poursuite sur échec, bilan) ; dans la description de `feedback.py`, mentionner `annotate --pending` (itération sur les reviews sans feedback, plus anciennes d'abord) et le bloc `Objectif` de summary (`objective_stats`, mistakes des reviews kind=game uniquement). Dans « Prochaines étapes » item 1, après « ✅ **Compte-rendu par-game** », noter que la boucle batch+pending est en place et que l'étape restante est l'annotation effective des ≥10 reviews (+ approche C — génération auto post-game — explicitement à suivre).

- [ ] **Step 4: Commit final**

```bash
git add CLAUDE.md
git commit -m "docs: sync CLAUDE.md — boucle de feedback batch (game-batch, pending, objectif)

Co-Authored-By: Claude <noreply@anthropic.com>"
```
