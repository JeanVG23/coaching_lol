# Boucle d'évaluation — scoring d'utilité : Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permettre au joueur d'annoter (CLI interactive, par-insight) les reviews persistées et d'agréger le signal « le coach s'améliore-t-il ».

**Architecture:** Un nouveau module `src/04_coaching/feedback.py` (sous-commandes `annotate` + `summary`) consomme les reviews persistées dans `data/07_coaching/<player>/reviews.jsonl` (clé = `ts`) et écrit/lit `feedback.jsonl`. Le schéma Pydantic `Feedback`/`FeedbackItem` vit dans `schema.py` (partagé). Aucun appel réseau.

**Tech Stack:** Python 3.14, Pydantic v2 (`model_validator`, `Literal`), pytest, `argparse` subparsers, stdlib `collections.Counter`. Imports style maison : `sys.path.insert` vers `src/`, modules 04_coaching importés en top-level (`import feedback`, `import schema`).

## Global Constraints

- Python ≥ 3.14 (utilise `Literal[*tuple]` non requis — on dérive le tuple via `typing.get_args`).
- `data/` est gitignoré : `feedback.jsonl` est un output local non versionné.
- Asymétrie (projet) : la boucle doit capter une régression de règle 3 (profondeur prescrite en faute) → tag `profondeur-en-faute` compté dans le récap.
- `tag` obligatoire quand `useful=False` (invariant validé Pydantic).
- Un `ts` apparaît au plus une fois dans `feedback.jsonl` (réannotation = écrase).
- Tests : `tests/conftest.py` ajoute déjà `src/` et `src/04_coaching` au path → `import feedback`, `import schema`, `import riotlib` marchent.
- Aucun appel réseau dans le module ni les tests (lecture/écriture fichiers + stdin seulement).
- Commits français (`feat(coaching): ...`, `test(coaching): ...`), `Co-Authored-By: Claude <noreply@anthropic.com>` en fin de message.

## File Structure

- **Modify** `src/04_coaching/schema.py` — ajouter `TagKind` (Literal), `NEG_TAGS` (tuple dérivé), `FeedbackItem`, `Feedback` + `@model_validator` (tag requis si `useful=False`).
- **Create** `src/04_coaching/feedback.py` — `list_reviews`, `load_review`, `build_feedback`, `persist_feedback`, `summarize`, `annotate` (flow interactif), `render_summary`, `main()` (subparsers `annotate`/`summary`).
- **Create** `tests/test_coaching_feedback.py` — couvre schéma, helpers, summarize, annotate (input monkeypatché).

---

### Task 1: Schéma `Feedback` / `FeedbackItem` dans `schema.py`

**Files:**
- Modify: `src/04_coaching/schema.py`
- Test: `tests/test_coaching_feedback.py`

**Interfaces:**
- Consumes: `pydantic.BaseModel`, `pydantic.model_validator`, `typing.Annotated`, `typing.Literal`, `typing.get_args`.
- Produces:
  - `TagKind = Literal["asymetrie","stat-inventee","profondeur-en-faute","trop-vague","non-actionnable","autre"]`
  - `NEG_TAGS: tuple[str, ...]` (dérivé via `get_args(TagKind)`)
  - `class FeedbackItem(BaseModel)` : `kind: Literal["strength","mistake","habit","focus"]`, `index: int`, `useful: bool`, `tag: TagKind | None = None`, `note: str | None = None` + validator `_tag_required_when_not_useful`.
  - `class Feedback(BaseModel)` : `ts: str`, `player: str`, `rated_at: str`, `model: str`, `overall_useful: bool | None = None`, `items: list[FeedbackItem]`.

- [ ] **Step 1: Écrire les tests (échouent)**

Créer `tests/test_coaching_feedback.py` :
```python
import pytest
from pydantic import ValidationError

import schema as S


def _item(kind="strength", index=0, useful=True, tag=None, note=None):
    return S.FeedbackItem(kind=kind, index=index, useful=useful, tag=tag, note=note)


def test_feedback_item_useful_no_tag_ok():
    it = _item(useful=True, tag=None)
    assert it.useful is True and it.tag is None


def test_feedback_item_not_useful_without_tag_rejected():
    with pytest.raises(ValidationError):
        _item(useful=False, tag=None)


def test_feedback_item_not_useful_with_tag_ok():
    it = _item(useful=False, tag="profondeur-en-faute", note="prof prescrite")
    assert it.tag == "profondeur-en-faute"


def test_feedback_item_bad_tag_rejected():
    with pytest.raises(ValidationError):
        _item(useful=False, tag="inventé")


def test_feedback_roundtrip_and_keys():
    fb = S.Feedback(ts="2026-06-30T10:00:00", player="spadzze",
                   rated_at="2026-06-30T11:00:00", model="kimi-k2.6",
                   items=[_item(useful=True), _item(kind="mistake", index=1,
                              useful=False, tag="asymetrie", note="x")])
    d = fb.model_dump()
    assert d["ts"] == "2026-06-30T10:00:00"
    assert len(d["items"]) == 2
    assert d["items"][1]["tag"] == "asymetrie"
    # re-validation depuis dict brut
    assert S.Feedback.model_validate(d).model == "kimi-k2.6"


def test_neg_tags_matches_literal():
    assert set(S.NEG_TAGS) == {"asymetrie", "stat-inventee", "profondeur-en-faute",
                               "trop-vague", "non-actionnable", "autre"}
```

- [ ] **Step 2: Lancer les tests (échec)**

Run: `.venv/bin/python -m pytest tests/test_coaching_feedback.py -q`
Expected: FAIL (`AttributeError: module 'schema' has no attribute 'FeedbackItem'`).

- [ ] **Step 3: Implémenter dans `schema.py`**

Ajouter à la fin de `src/04_coaching/schema.py` :
```python
from typing import Literal, get_args
from pydantic import model_validator


TagKind = Literal["asymetrie", "stat-inventee", "profondeur-en-faute",
                  "trop-vague", "non-actionnable", "autre"]
NEG_TAGS: tuple[str, ...] = get_args(TagKind)


class FeedbackItem(BaseModel):
    kind: Literal["strength", "mistake", "habit", "focus"]
    index: int                       # position dans sa section (focus = 0)
    useful: bool
    tag: TagKind | None = None        # obligatoire si useful=False (cf. validator)
    note: str | None = None

    @model_validator(mode="after")
    def _tag_required_when_not_useful(self):
        if not self.useful and self.tag is None:
            raise ValueError("tag requis quand useful=False")
        return self


class Feedback(BaseModel):
    ts: str                           # clé = ts de la review annotée
    player: str
    rated_at: str                     # ISO timestamp de l'annotation
    model: str                        # copié de la review (récap par modèle)
    overall_useful: bool | None = None   # non collecté par le flow interactif
    items: list[FeedbackItem]         # items annotés (≤9 ; skips omis)
```
(Les imports `Literal`/`get_args`/`model_validator` peuvent être fusionnés avec l'import existant `Annotated` en haut du fichier — garder le style cohérent.)

- [ ] **Step 4: Lancer les tests (passent)**

Run: `.venv/bin/python -m pytest tests/test_coaching_feedback.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Non-régression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (74 + 6 = 80).

- [ ] **Step 6: Commit**

```bash
git add src/04_coaching/schema.py tests/test_coaching_feedback.py
git commit -m "$(cat <<'EOF'
feat(coaching): schéma Feedback (pydantic) — tag requis si conseil jugé faux

FeedbackItem (kind/index/useful/tag/note) + Feedback (ts/player/rated_at/
model/items). TagKind Literal (asymetrie, stat-inventee, profondeur-en-faute,
trop-vague, non-actionnable, autre) + NEG_TAGS dérivé. Invariant : tag
obligatoire quand useful=False (model_validator).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `feedback.py` — helpers données (list/load/build/persist)

**Files:**
- Create: `src/04_coaching/feedback.py`
- Test: `tests/test_coaching_feedback.py` (ajouts)

**Interfaces:**
- Consumes: `riotlib.DATA` (`rl.DATA / "07_coaching"`), `schema.Feedback`/`FeedbackItem`/`Review`, `json`, `pathlib.Path`, `datetime`.
- Produces:
  - `list_reviews(player: str, root=None) -> list[dict]`
  - `load_review(player: str, ts: str, root=None) -> dict | None`
  - `build_feedback(review: schema.Review, ts: str, player: str, model: str, rated_at: str, responses: dict[tuple[str, int], tuple[bool, str | None, str | None]]) -> schema.Feedback`
  - `persist_feedback(player: str, fb: schema.Feedback, root=None) -> tuple[pathlib.Path, bool]` (bool = a écrasé un ts existant)

- [ ] **Step 1: Ajouter les tests (échouent)**

Ajouter à `tests/test_coaching_feedback.py` (en haut, après les imports existants) :
```python
import json
from pathlib import Path

import feedback as F
import riotlib as rl
```
Puis à la fin :
```python
def _review_dict():
    ins = {"point": "p", "evidence": "e"}
    return {"ts": "2026-06-30T10:00:00", "model": "kimi-k2.6",
            "scope": "adc", "target": "challenger", "outcome_focus": "loss",
            "payload": {"meta": {}},
            "review": {"strengths": [ins, ins, ins], "mistakes": [ins, ins, ins],
                       "habits": ["h1", "h2"], "next_focus": "f", "confidence": 0.6}}


def _write_reviews(tmp_path, player="spadzze", lines=None):
    out = tmp_path / player
    out.mkdir(parents=True, exist_ok=True)
    (out / "reviews.jsonl").write_text("\n".join(json.dumps(l) for l in (lines or [_review_dict()])) + "\n")
    return out / "reviews.jsonl"


def test_list_reviews_empty(tmp_path):
    assert F.list_reviews("nobody", root=tmp_path) == []


def test_list_reviews_reads_lines(tmp_path):
    _write_reviews(tmp_path)
    rs = F.list_reviews("spadzze", root=tmp_path)
    assert len(rs) == 1 and rs[0]["ts"] == "2026-06-30T10:00:00"


def test_load_review_found_and_missing(tmp_path):
    _write_reviews(tmp_path)
    assert F.load_review("spadzze", "2026-06-30T10:00:00", root=tmp_path)["model"] == "kimi-k2.6"
    assert F.load_review("spadzze", "missing", root=tmp_path) is None


def test_build_feedback_skips_unanswered_and_enforces_tag():
    rev = S.Review.model_validate(_review_dict()["review"])
    responses = {
        ("strength", 0): (True, None, None),
        ("mistake", 1): (False, "profondeur-en-faute", "prof prescrite"),
        # les autres items = skip
    }
    fb = F.build_feedback(rev, ts="2026-06-30T10:00:00", player="spadzze",
                          model="kimi-k2.6", rated_at="2026-06-30T11:00:00",
                          responses=responses)
    assert isinstance(fb, S.Feedback)
    assert len(fb.items) == 2                      # skips omis
    assert fb.items[0].kind == "strength" and fb.items[0].useful is True
    assert fb.items[1].tag == "profondeur-en-faute"


def test_persist_feedback_creates_then_overwrites_same_ts(tmp_path):
    fb1 = S.Feedback(ts="t1", player="spadzze", rated_at="a", model="m",
                     items=[S.FeedbackItem(kind="strength", index=0, useful=True)])
    path, overwrote = F.persist_feedback("spadzze", fb1, root=tmp_path)
    assert overwrote is False and path.exists()
    # nouvelle annotation même ts, items différents
    fb2 = S.Feedback(ts="t1", player="spadzze", rated_at="b", model="m",
                     items=[S.FeedbackItem(kind="mistake", index=0, useful=False, tag="asymetrie")])
    path, overwrote = F.persist_feedback("spadzze", fb2, root=tmp_path)
    assert overwrote is True
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1                         # 1 ligne finale, pas 2
    assert lines[0]["rated_at"] == "b" and lines[0]["items"][0]["tag"] == "asymetrie"
```

- [ ] **Step 2: Lancer les tests (échec)**

Run: `.venv/bin/python -m pytest tests/test_coaching_feedback.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'feedback'`).

- [ ] **Step 3: Implémenter `feedback.py` (squelette + helpers)**

Créer `src/04_coaching/feedback.py` :
```python
#!/usr/bin/env python3
"""04_coaching — boucle d'évaluation : annotation + agrégation des reviews.

Sous-commandes :
  annotate  : choisir une review persistée, juger chaque insight (👍/👎 + tag),
              persister dans data/07_coaching/<player>/feedback.jsonl.
  summary   : agréger les annotations (taux utile, par section, top tags, par
              modèle, tendance). Aucun appel réseau.

Usage :
  python3 src/04_coaching/feedback.py annotate --player spadzze [--ts <ts> | --last]
  python3 src/04_coaching/feedback.py summary  --player spadzze [--tag <t>] [--model <m>]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # accès src/
import riotlib as rl

import schema as schema_mod


def _reviews_path(player: str, root=None) -> Path:
    root = Path(root) if root is not None else rl.DATA / "07_coaching"
    return root / player / "reviews.jsonl"


def _feedback_path(player: str, root=None) -> Path:
    root = Path(root) if root is not None else rl.DATA / "07_coaching"
    return root / player / "feedback.jsonl"


def list_reviews(player: str, root=None) -> list[dict]:
    path = _reviews_path(player, root)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_review(player: str, ts: str, root=None) -> dict | None:
    for r in list_reviews(player, root):
        if r.get("ts") == ts:
            return r
    return None


def build_feedback(review: schema_mod.Review, ts: str, player: str, model: str,
                   rated_at: str,
                   responses: dict[tuple[str, int], tuple[bool, str | None, str | None]]
                   ) -> schema_mod.Feedback:
    """Construit un Feedback en n'incluant que les items présents dans responses
    (skip = item omis). Invariant tag-requis validé par FeedbackItem."""
    sections = [("strength", review.strengths), ("mistake", review.mistakes),
                ("habit", review.habits)]
    items = []
    for kind, section in sections:
        for i, _ in enumerate(section):
            key = (kind, i)
            if key not in responses:
                continue
            useful, tag, note = responses[key]
            items.append(schema_mod.FeedbackItem(kind=kind, index=i,
                                                 useful=useful, tag=tag, note=note))
    key = ("focus", 0)
    if key in responses:
        useful, tag, note = responses[key]
        items.append(schema_mod.FeedbackItem(kind="focus", index=0,
                                             useful=useful, tag=tag, note=note))
    return schema_mod.Feedback(ts=ts, player=player, rated_at=rated_at,
                               model=model, items=items)


def persist_feedback(player: str, fb: schema_mod.Feedback,
                     root=None) -> tuple[Path, bool]:
    """Append/écrase : un ts apparaît au plus une fois. Retourne (path, overwrote)."""
    path = _feedback_path(player, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    overwrote = False
    if path.exists():
        for l in path.read_text().splitlines():
            if not l.strip():
                continue
            if json.loads(l).get("ts") == fb.ts:
                overwrote = True
                continue
            kept.append(l)
    kept.append(json.dumps(fb.model_dump(), ensure_ascii=False))
    with path.open("w") as f:
        f.write("\n".join(kept) + "\n")
    return path, overwrote
```

- [ ] **Step 4: Lancer les tests (passent)**

Run: `.venv/bin/python -m pytest tests/test_coaching_feedback.py -q`
Expected: PASS (6 + 6 = 12).

- [ ] **Step 5: Commit**

```bash
git add src/04_coaching/feedback.py tests/test_coaching_feedback.py
git commit -m "$(cat <<'EOF'
feat(coaching): feedback.py helpers — list/load/build/persist (écrase par ts)

list_reviews/load_review (clé ts), build_feedback (skips omis, invariant tag),
persist_feedback (un ts = une ligne, réannotation écrase). Aucun réseau.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `feedback.py` — `summarize` + `render_summary`

**Files:**
- Modify: `src/04_coaching/feedback.py`
- Test: `tests/test_coaching_feedback.py` (ajouts)

**Interfaces:**
- Consumes: `schema.Feedback`, `collections.Counter`.
- Produces:
  - `summarize(fbs: list[schema.Feedback]) -> dict` : clés `n_reviews`, `n_items`, `global_rate`, `by_kind` (dict kind→{n,useful,rate|None}), `top_tags` (list[(tag,count)]), `by_model` (dict model→{n_reviews,n_items,useful,rate|None}), `low_sample` (bool), `trend` (None ou {recent,prior}).
  - `render_summary(stats: dict) -> str` : formatage FR imprimable.
  - `load_feedbacks(player: str, root=None) -> list[schema.Feedback]` : parse `feedback.jsonl`.

- [ ] **Step 1: Ajouter les tests (échouent)**

Ajouter à `tests/test_coaching_feedback.py` :
```python
def _fb(ts, model, items):
    return S.Feedback(ts=ts, player="spadzze", rated_at="r", model=model, items=items)


def _it(kind, useful, tag=None):
    return S.FeedbackItem(kind=kind, index=0, useful=useful, tag=tag)


def test_summarize_empty():
    s = F.summarize([])
    assert s["n_reviews"] == 0 and "global_rate" not in s


def test_summarize_global_and_by_kind():
    fbs = [_fb("t1", "kimi-k2.6", [
            _it("strength", True), _it("mistake", False, "profondeur-en-faute"),
            _it("habit", True)])]
    s = F.summarize(fbs)
    assert s["n_reviews"] == 1 and s["n_items"] == 3
    assert s["global_rate"] == pytest.approx(2 / 3)
    assert s["by_kind"]["strength"]["rate"] == 1.0
    assert s["by_kind"]["mistake"]["rate"] == 0.0
    assert s["by_kind"]["habit"]["rate"] == 1.0


def test_summarize_top_tags():
    fbs = [_fb("t1", "m", [_it("mistake", False, "profondeur-en-faute"),
                           _it("mistake", False, "profondeur-en-faute"),
                           _it("strength", False, "asymetrie")])]
    s = F.summarize(fbs)
    assert s["top_tags"][0] == ("profondeur-en-faute", 2)
    assert s["top_tags"][1] == ("asymetrie", 1)


def test_summarize_by_model():
    fbs = [_fb("t1", "kimi-k2.6", [_it("strength", True), _it("mistake", True)]),
           _fb("t2", "minimax-m3", [_it("mistake", False, "asymetrie")])]
    s = F.summarize(fbs)
    assert s["by_model"]["kimi-k2.6"]["rate"] == 1.0
    assert s["by_model"]["minimax-m3"]["rate"] == 0.0
    assert s["by_model"]["kimi-k2.6"]["n_reviews"] == 1


def test_summarize_low_sample_no_trend():
    fbs = [_fb(f"t{i}", "m", [_it("strength", True)]) for i in range(5)]
    s = F.summarize(fbs)
    assert s["low_sample"] is True and s["trend"] is None


def test_summarize_trend_when_enough():
    # 10 reviews : 5 premières tout faux, 5 dernières tout vrai → tendance haussière
    fbs = ([_fb(f"b{i}", "m", [_it("strength", False, "trop-vague")]) for i in range(5)]
           + [_fb(f"a{i}", "m", [_it("strength", True)]) for i in range(5)])
    s = F.summarize(fbs)
    assert s["low_sample"] is False
    assert s["trend"] is not None
    assert s["trend"]["prior"] == 0.0 and s["trend"]["recent"] == 1.0


def test_render_summary_has_sections():
    fbs = [_fb("t1", "kimi-k2.6", [_it("strength", True),
                                   _it("mistake", False, "profondeur-en-faute")])]
    txt = F.render_summary(F.summarize(fbs))
    assert "Taux d'utilité" in txt and "Top tags" in txt and "Par modèle" in txt
    assert "profondeur-en-faute" in txt


def test_load_feedbacks_roundtrip(tmp_path):
    fb = _fb("t1", "m", [_it("strength", True)])
    F.persist_feedback("spadzze", fb, root=tmp_path)
    loaded = F.load_feedbacks("spadzze", root=tmp_path)
    assert len(loaded) == 1 and loaded[0].ts == "t1"
```

- [ ] **Step 2: Lancer les tests (échec)**

Run: `.venv/bin/python -m pytest tests/test_coaching_feedback.py -q`
Expected: FAIL (`AttributeError: module 'feedback' has no attribute 'summarize'`).

- [ ] **Step 3: Implémenter `summarize` / `render_summary` / `load_feedbacks`**

Ajouter à `src/04_coaching/feedback.py` (après `persist_feedback`) :
```python
from collections import Counter


_KINDS = ("strength", "mistake", "habit", "focus")
_TREND_RECENT = 5
_LOW_SAMPLE = 10


def load_feedbacks(player: str, root=None) -> list[schema_mod.Feedback]:
    path = _feedback_path(player, root)
    if not path.exists():
        return []
    out = []
    for l in path.read_text().splitlines():
        if l.strip():
            out.append(schema_mod.Feedback.model_validate(json.loads(l)))
    return out


def summarize(fbs: list[schema_mod.Feedback]) -> dict:
    if not fbs:
        return {"n_reviews": 0}
    items = [it for fb in fbs for it in fb.items]
    n_items = len(items)
    n_useful = sum(1 for it in items if it.useful)
    by_kind: dict[str, dict] = {}
    for kind in _KINDS:
        ki = [it for it in items if it.kind == kind]
        u = sum(1 for it in ki if it.useful)
        by_kind[kind] = {"n": len(ki), "useful": u,
                         "rate": (u / len(ki)) if ki else None}
    tag_counts = Counter(it.tag for it in items if (not it.useful) and it.tag)
    by_model: dict[str, dict] = {}
    for fb in fbs:
        if not fb.items:
            continue
        m = fb.model
        bm = by_model.setdefault(m, {"n_reviews": 0, "n_items": 0, "useful": 0})
        bm["n_reviews"] += 1
        bm["n_items"] += len(fb.items)
        bm["useful"] += sum(1 for it in fb.items if it.useful)
    for m, bm in by_model.items():
        bm["rate"] = (bm["useful"] / bm["n_items"]) if bm["n_items"] else None

    ordered = sorted(fbs, key=lambda f: f.ts)
    low_sample = len(ordered) < _LOW_SAMPLE
    trend = None
    if not low_sample:
        def _rate(group):
            its = [it for fb in group for it in fb.items]
            return (sum(1 for it in its if it.useful) / len(its)) if its else None
        recent, prior = ordered[-_TREND_RECENT:], ordered[:-_TREND_RECENT]
        trend = {"recent": _rate(recent), "prior": _rate(prior)}

    return {"n_reviews": len(fbs), "n_items": n_items,
            "global_rate": (n_useful / n_items) if n_items else 0.0,
            "by_kind": by_kind, "top_tags": tag_counts.most_common(),
            "by_model": by_model, "low_sample": low_sample, "trend": trend}


def render_summary(stats: dict) -> str:
    if stats.get("n_reviews", 0) == 0:
        return "Aucune annotation — lance `feedback.py annotate` d'abord."
    lines = [f"Taux d'utilité global : {stats['global_rate']:.0%} "
             f"({stats['n_items']} items notés sur {stats['n_reviews']} reviews)",
             "\nPar section :"]
    label = {"strength": "Forces", "mistake": "Erreurs", "habit": "Habitudes",
             "focus": "Focus"}
    for kind in _KINDS:
        s = stats["by_kind"][kind]
        r = "—" if s["rate"] is None else f"{s['rate']:.0%}"
        lines.append(f"  {label[kind]:10} {r}  ({s['useful']}/{s['n']})")
    if stats["top_tags"]:
        lines.append("\nTop tags (conseils jugés faux) :")
        for tag, c in stats["top_tags"]:
            lines.append(f"  {tag:24} ×{c}")
    if stats["by_model"]:
        lines.append("\nPar modèle :")
        for m, bm in sorted(stats["by_model"].items()):
            r = "—" if bm["rate"] is None else f"{bm['rate']:.0%}"
            lines.append(f"  {m:18} {r}  ({bm['n_reviews']} reviews)")
    if stats["low_sample"]:
        lines.append(f"\nTendance : échantillon faible (<{ _LOW_SAMPLE} reviews annotées), "
                     f"pas de tendance calculée.")
    elif stats["trend"]:
        t = stats["trend"]
        rp = lambda v: "—" if v is None else f"{v:.0%}"
        lines.append(f"\nTendance : 5 dernières {rp(t['recent'])} vs précédentes {rp(t['prior'])}")
    return "\n".join(lines)
```

- [ ] **Step 4: Lancer les tests (passent)**

Run: `.venv/bin/python -m pytest tests/test_coaching_feedback.py -q`
Expected: PASS (12 + 8 = 20).

- [ ] **Step 5: Commit**

```bash
git add src/04_coaching/feedback.py tests/test_coaching_feedback.py
git commit -m "$(cat <<'EOF'
feat(coaching): feedback.py summarize — taux, par section, top tags, modèle, tendance

summarize (global_rate, by_kind, top_tags via Counter, by_model, tendance
5-dernières vs précédentes, low_sample <10). render_summary FR. load_feedbacks
parse feedback.jsonl.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `feedback.py` — flow `annotate` interactif + `main()` subcommands

**Files:**
- Modify: `src/04_coaching/feedback.py`
- Test: `tests/test_coaching_feedback.py` (ajouts)

**Interfaces:**
- Consumes: `list_reviews`, `load_review`, `build_feedback`, `persist_feedback`, `schema.Review`, `argparse`, `input` builtin.
- Produces:
  - `annotate(player: str, ts: str | None = None, last: bool = False, root=None, prompt=None) -> int` : flow interactif, retourne code sortie. (`prompt=None` → late binding sur `input`, pour que `monkeypatch.setattr("builtins.input", ...)` soit pris en compte dans les tests ; `main()` ne passe pas prompt.)
  - `main() -> int` : argparse subparsers `annotate`/`summary` (+ `--player`, `--ts`, `--last`, `--tag`, `--model`).

- [ ] **Step 1: Ajouter les tests (échouent)**

Ajouter à `tests/test_coaching_feedback.py`. On monkeypatche `prompt` via un itérateur de réponses pour rendre `annotate` testable sans stdin réel.
```python
def _reviews_file(tmp_path, n=1):
    lines = []
    for i in range(n):
        d = _review_dict()
        d["ts"] = f"2026-06-30T1{i}:00:00"
        lines.append(d)
    _write_reviews(tmp_path, lines=lines)


def test_annotate_interactive_monkeypatched_input(tmp_path, monkeypatch, capsys):
    _reviews_file(tmp_path, n=1)
    # séquence input() : choisir review #1, puis 9 items.
    # item utile: "y" ; item faux: "n" puis tag menu (numéro) puis note.
    answers = iter([
        "1",                       # choix review #1
        "y",                       # strength 0 utile
        "s",                       # strength 1 skip
        "y",                       # strength 2 utile
        "n", "3", "prof prescrite", # mistake 0 faux -> tag menu #3 = profondeur-en-faute, note
        "y",                       # mistake 1 utile
        "y",                       # mistake 2 utile
        "y",                       # habit 0 utile
        "s",                       # habit 1 skip
        "y",                       # focus utile
    ])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    rc = F.annotate("spadzze", root=tmp_path)
    assert rc == 0
    fbs = F.load_feedbacks("spadzze", root=tmp_path)
    assert len(fbs) == 1
    fb = fbs[0]
    assert fb.ts == "2026-06-30T10:00:00"
    assert len(fb.items) == 7            # 9 - 2 skips
    # mistake 0 = faux, tag profondeur-en-faute, note "prof prescrite"
    mk0 = next(it for it in fb.items if it.kind == "mistake" and it.index == 0)
    assert mk0.useful is False and mk0.tag == "profondeur-en-faute"
    assert mk0.note == "prof prescrite"


def test_annotate_no_reviews(tmp_path, capsys):
    rc = F.annotate("nobody", root=tmp_path, prompt=lambda *a, **k: "")
    assert rc == 0
    assert "Aucune review" in capsys.readouterr().out


def test_annotate_ts_not_found(tmp_path, capsys):
    _reviews_file(tmp_path, n=1)
    rc = F.annotate("spadzze", ts="missing", root=tmp_path, prompt=lambda *a, **k: "")
    assert rc == 1
    assert "introuvable" in capsys.readouterr().out.lower()


def test_main_summary_subcommand(tmp_path, monkeypatch, capsys):
    # layout production : rl.DATA / "07_coaching" / player / reviews.jsonl
    _reviews_file(tmp_path / "07_coaching", n=1)
    # annote d'abord
    answers = iter(["1", "y", "y", "y", "y", "y", "y", "y", "y", "y"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    monkeypatch.setattr(F.rl, "DATA", tmp_path)   # rl.DATA/07_coaching = tmp/07_coaching
    assert F.main(["annotate", "--player", "spadzze"]) == 0
    capsys.readouterr()
    assert F.main(["summary", "--player", "spadzze"]) == 0
    out = capsys.readouterr().out
    assert "Taux d'utilité" in out
```

- [ ] **Step 2: Lancer les tests (échec)**

Run: `.venv/bin/python -m pytest tests/test_coaching_feedback.py -q`
Expected: FAIL (`AttributeError: module 'feedback' has no attribute 'annotate'`).

- [ ] **Step 3: Implémenter `annotate` + `main`**

Ajouter à `src/04_coaching/feedback.py` :
```python
import argparse


def _display_items(review: schema_mod.Review) -> list[tuple[str, int, str]]:
    """Retourne [(kind, index, ligne_affichage)] pour les 9 items."""
    out = []
    for i, ins in enumerate(review.strengths):
        out.append(("strength", i, f"Force  {i}: {ins.point}  ({ins.evidence})"))
    for i, ins in enumerate(review.mistakes):
        out.append(("mistake", i, f"Erreur {i}: {ins.point}  ({ins.evidence})"))
    for i, h in enumerate(review.habits):
        out.append(("habit", i, f"Habitude {i}: {h}"))
    out.append(("focus", 0, f"Focus : {review.next_focus}"))
    return out


def _prompt_useful(prompt, label_line) -> str | None:
    """Retourne 'y'/'n'/'s' (skip=None)."""
    while True:
        ans = prompt(f"{label_line}\n  utile ? [y/n/s] : ").strip().lower()
        if ans in ("y", "n", "s", ""):
            return None if ans == "" else ans   # Entrée vide = skip aussi
        # boucle sinon


def annotate(player: str, ts: str | None = None, last: bool = False,
             root=None, prompt=None) -> int:
    if prompt is None:
        prompt = input           # late binding : monkeypatch builtins.input pris en compte
    reviews = list_reviews(player, root)
    if not reviews:
        print("Aucune review pour ce joueur — génère-en via coach.py d'abord.")
        return 0
    if ts is None:
        if last:
            chosen = reviews[-1]
        else:
            print("Reviews disponibles :")
            for i, r in enumerate(reviews, 1):
                print(f"  {i} | {r['ts']} | {r['model']} | {r.get('outcome_focus','?')}")
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
    review = schema_mod.Review.model_validate(chosen["review"])
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="feedback.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("annotate", help="juger les insights d'une review")
    a.add_argument("--player", default="spadzze")
    a.add_argument("--ts", default=None)
    a.add_argument("--last", action="store_true")

    s = sub.add_parser("summary", help="agrège les annotations")
    s.add_argument("--player", default="spadzze")
    s.add_argument("--tag", default=None, help="filtre par tag")
    s.add_argument("--model", default=None, help="filtre par modèle")

    args = ap.parse_args(argv)
    if args.cmd == "annotate":
        return annotate(args.player, ts=args.ts, last=args.last)
    if args.cmd == "summary":
        fbs = load_feedbacks(args.player)
        if args.model:
            fbs = [f for f in fbs if f.model == args.model]
        if args.tag:
            # ne garde que les reviews qui contiennent au moins un item avec ce tag
            fbs = [f for f in fbs if any(it.tag == args.tag for it in f.items)]
        print(f"FEEDBACK — {args.player}")
        print(render_summary(summarize(fbs)))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Lancer les tests (passent)**

Run: `.venv/bin/python -m pytest tests/test_coaching_feedback.py -q`
Expected: PASS (20 + 4 = 24).

- [ ] **Step 5: Non-régression complète**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (74 + 24 = 98).

- [ ] **Step 6: Smoke manuel (optionnel, pas un échec si pas de réseau)**

Run: `.venv/bin/python src/04_coaching/feedback.py summary --player spadzze`
Expected: « Aucune annotation — lance `feedback.py annotate` d'abord. » (exit 0).

- [ ] **Step 7: Commit**

```bash
git add src/04_coaching/feedback.py tests/test_coaching_feedback.py
git commit -m "$(cat <<'EOF'
feat(coaching): feedback.py annotate interactif + CLI subcommands

annotate : sélection review (numéro/--ts/--last), défile 9 items (y/n/s),
tag menu numéroté + note optionnelle sur faux, persiste (écrase par ts),
récap session. main() subparsers annotate/summary (--tag/--model filtres).
prompt injectable pour tests (input monkeypatché).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Doc — `src/04_coaching/README.md` + `CLAUDE.md` + `todo.md`

**Files:**
- Modify: `src/04_coaching/README.md` (section boucle d'évaluation)
- Modify: `CLAUDE.md` (bloc 04_coaching + prochaines étapes)
- Modify: `todo.md` (cocher scoring d'utilité)

- [ ] **Step 1: README — ajouter une section « Boucle d'évaluation »

Ajouter à `src/04_coaching/README.md` (après la section Usage) :
```markdown
## Boucle d'évaluation (`feedback.py`)

Ferme le « ce conseil était-il utile ? » sur les reviews persistées — sans
re-générer. CLI interactive par-insight (9 items), tag fixe sur jugement négatif.

```bash
python3 src/04_coaching/feedback.py annotate --player spadzze   # choisir + juger
python3 src/04_coaching/feedback.py summary  --player spadzze   # agrégation
```

- **annotate** : liste les reviews, défile 3 forces / 3 erreurs / 2 habitudes /
  focus, prompt `y/n/s` (+ tag numéroté + note sur `n`). Persiste dans
  `data/07_coaching/<player>/feedback.jsonl` (1 ligne/review, réannotation écrase).
- **summary** : taux d'utilité global + par section, top tags (conseils faux),
  par modèle, tendance (5 dernières vs précédentes ; low_sample <10).
- **Tags** : `asymetrie`, `stat-inventee`, `profondeur-en-faute`, `trop-vague`,
  `non-actionnable`, `autre` — ciblent les modes d'échec connus du prompt
  (règles 1/2/3). Le top tag est le signal actionnable pour durcir le prompt.
```

- [ ] **Step 2: CLAUDE.md — bloc 04_coaching**

Dans `CLAUDE.md`, ligne du médaillon 04_coaching, ajouter `feedback.py` à la liste
des modules et mentionner la boucle d'éval. Exemple de formulation (adapter au
texte existant) : ajouter après `coach.py (...).` :
« `feedback.py` (CLI `annotate`/`summary` : boucle d'éval par-insight, tag fixe,
persiste `data/07_coaching/<player>/feedback.jsonl`). »

Et dans « Prochaines étapes », passer le scoring d'utilité en ✅ :
« ✅ Boucle d'éval (scoring d'utilité) — `feedback.py` annotate/summary. À suivre :
compte-rendu par-game. »

- [ ] **Step 3: todo.md — cocher**

Dans `todo.md`, section « Court terme », cocher la case « Scoring d'utilité » et
ajouter une ligne de référence : « ✅ Scoring d'utilité — `feedback.py`
annotate/summary (tag fixe, par-insight, tendance). »

- [ ] **Step 4: Commit**

```bash
git add src/04_coaching/README.md CLAUDE.md todo.md
git commit -m "$(cat <<'EOF'
docs(coaching): doc boucle d'évaluation (feedback.py annotate/summary)

README: section boucle d'éval. CLAUDE.md: module feedback.py + prochaine étape
scoring d'utilité cochée. todo.md: case cochée.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```