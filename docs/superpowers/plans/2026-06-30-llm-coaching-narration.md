# Narration LLM du coaching — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformer le diff gold perso↔référentiel en un compte-rendu de coaching typé, généré par Ollama Cloud et persisté.

**Architecture:** Cinq modules sous `src/04_coaching/` : `payload.py` (gold → payload déterministe, pur), `prompt.py` (payload → messages, pur), `schema.py` (Pydantic `Review` + JSON-schema), `llm_client.py` (client Ollama Cloud HTTP), `coach.py` (CLI : payload→prompt→client→validation→affiche+persiste). Approche A : un seul appel LLM, sélection des signaux déterministe côté features, validation Pydantic.

**Tech Stack:** Python, Pydantic v2, `requests` (existant), Ollama Cloud (`https://ollama.com/api/chat`). Spec : `docs/superpowers/specs/2026-06-30-llm-coaching-narration-design.md`.

## Global Constraints

- Lancer depuis la racine : `python3 src/04_coaching/coach.py`. Les modules de `src/04_coaching/` s'importent entre eux (même dossier) ; pour `riotlib`/`positioning`/`compare` ils insèrent `src/` au path via `sys.path.insert(0, Path(__file__).resolve().parent.parent)`.
- Seule dépendance ajoutée : `pydantic>=2`. Le client HTTP utilise `requests` (déjà présent) — **pas** le lib `ollama`.
- **Asymétrie (dur)** : le payload n'expose QUE les 14 features positionnelles `COACHING_SAFE`. `POS_META` (table de salience positionnelle) doit avoir des clés **exactement égales** à `positioning.COACHING_SAFE`. Les 3 proxys `ML_ONLY` n'apparaissent JAMAIS dans le payload ni le prompt.
- **Profondeur (dur)** : `avg_map_depth`, `max_map_depth`, `frac_overextended` sont **toujours `descriptive_only:true`, jamais `notable`** (profondeur ↑ = diamond, pas un défaut).
- Modèle LLM par défaut : `deepseek-v4-pro`, configurable via `--model` / env `OLLAMA_MODEL`.
- Client Ollama Cloud : endpoint `https://ollama.com/api/chat`, header `Authorization: Bearer <OLLAMA_API_KEY>` lu via `riotlib.load_env()`, body `{"model","messages","format":<schema>,"stream":false,"options":{"temperature":0.2}}`, `timeout=180`, retries+backoff sur 429/5xx/timeout, 401 → message « vérifie OLLAMA_API_KEY ».
- `Review` : exactement 3 `strengths`, 3 `mistakes`, 2 `habits`, 1 `next_focus`, `confidence` ∈ [0,1] ; chaque `Insight` porte `point` + `evidence` (preuve chiffrée).
- Persistance : `data/07_coaching/<player>/reviews.jsonl` (append), 1 ligne = `{ts, model, scope, target, outcome_focus, payload, review}`. `data/` reste gitignoré.
- Sortie en **français**, tutoiement.
- Tests : **zéro réseau par défaut** (monkeypatch `requests.post` / `llm_client.generate_json`). Le test d'intégration réel est `skip` si `OLLAMA_API_KEY` absente.
- `tests/conftest.py` met `src/` ET `src/04_coaching/` sur `sys.path`.

---

### Task 1: Schéma de sortie Pydantic + dépendance + conftest

**Files:**
- Modify: `requirements.txt`
- Modify: `tests/conftest.py`
- Create: `src/04_coaching/schema.py`
- Test: `tests/test_coaching_schema.py`

**Interfaces:**
- Produces: `Insight(BaseModel)` (champs `point: str`, `evidence: str`) ; `Review(BaseModel)` (champs `strengths`, `mistakes` : `list[Insight]` longueur 3 ; `habits: list[str]` longueur 2 ; `next_focus: str` ; `confidence: float` ∈ [0,1]) ; `review_json_schema() -> dict`.

- [ ] **Step 1: Ajouter la dépendance et installer**

Modifier `requirements.txt` (ajouter une ligne) :
```
requests>=2.32
zstandard>=0.23
pydantic>=2
```
Run: `.venv/bin/pip install 'pydantic>=2'`
Expected: `Successfully installed pydantic...`

- [ ] **Step 2: Étendre conftest.py pour exposer `src/04_coaching/`**

Remplacer le contenu de `tests/conftest.py` par :
```python
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "04_coaching"))
```

- [ ] **Step 3: Écrire le test (échoue)**

Créer `tests/test_coaching_schema.py` :
```python
import pytest
from pydantic import ValidationError

import schema as S


def _good():
    ins = {"point": "p", "evidence": "e"}
    return {"strengths": [ins, ins, ins], "mistakes": [ins, ins, ins],
            "habits": ["h1", "h2"], "next_focus": "focus", "confidence": 0.7}


def test_review_accepts_valid():
    r = S.Review.model_validate(_good())
    assert len(r.strengths) == 3 and len(r.habits) == 2
    assert r.strengths[0].evidence == "e"


def test_review_rejects_wrong_lengths():
    bad = _good()
    bad["strengths"].append({"point": "x", "evidence": "y"})  # 4 forces
    with pytest.raises(ValidationError):
        S.Review.model_validate(bad)


def test_review_rejects_confidence_out_of_range():
    bad = _good(); bad["confidence"] = 1.5
    with pytest.raises(ValidationError):
        S.Review.model_validate(bad)


def test_json_schema_has_fixed_lengths():
    sch = S.review_json_schema()
    assert sch["properties"]["strengths"]["minItems"] == 3
    assert sch["properties"]["strengths"]["maxItems"] == 3
    assert sch["properties"]["habits"]["maxItems"] == 2
```

Run: `.venv/bin/python -m pytest tests/test_coaching_schema.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'schema'`)

- [ ] **Step 4: Implémenter `schema.py`**

Créer `src/04_coaching/schema.py` :
```python
"""Schéma de sortie typé de la review de coaching (Pydantic v2).

Expose Review (longueurs fixes 3/3/2) et le JSON-schema dérivé pour le
paramètre `format` d'Ollama (structured output). Chaque Insight porte sa
preuve chiffrée — pas de conseil sans stat.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class Insight(BaseModel):
    point: str       # affirmation FR ("tu roams trop peu en mid")
    evidence: str    # preuve chiffrée du payload ("roam mid 50% vs 70% challenger")


class Review(BaseModel):
    strengths: Annotated[list[Insight], Field(min_length=3, max_length=3)]
    mistakes: Annotated[list[Insight], Field(min_length=3, max_length=3)]
    habits: Annotated[list[str], Field(min_length=2, max_length=2)]
    next_focus: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


def review_json_schema() -> dict:
    """JSON-schema passé à Ollama `format`. minItems/maxItems contraignent la génération."""
    return Review.model_json_schema()
```

- [ ] **Step 5: Lancer les tests (passent)**

Run: `.venv/bin/python -m pytest tests/test_coaching_schema.py -q`
Expected: PASS (4 tests)

> Note : Pydantic v2 sérialise `min_length`/`max_length` sur une liste en `minItems`/`maxItems` dans le JSON-schema. Si l'assertion du Step 3 échoue sur le nom de clé, inspecter `review_json_schema()` et aligner le test sur la clé réellement émise (ne pas changer le modèle).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/conftest.py src/04_coaching/schema.py tests/test_coaching_schema.py
git commit -m "feat(coaching): Review Pydantic schema + Ollama JSON-schema"
```

---

### Task 2: Payload déterministe depuis le gold (pur, safe-only)

**Files:**
- Create: `src/04_coaching/payload.py`
- Test: `tests/test_coaching_payload.py`

**Interfaces:**
- Consumes: `riotlib.GOLD_DIR`, `positioning.COACHING_SAFE`, `compare.context_benchmark(me_agg, ref_agg, axis, outcome)`.
- Produces:
  - `POS_META: dict[str, tuple[str, str, float | None, bool]]` (clé → (label, unit, notable_threshold, descriptive_only)), clés == `positioning.COACHING_SAFE`.
  - `_lane_signals(mf: dict, rf: dict) -> list[dict]`
  - `_pos_signals(mf: dict, rf: dict) -> list[dict]`
  - `_zone_phase_signals(mf: dict, rf: dict) -> list[dict]`
  - `_gold_state_signals(mf: dict, rf: dict) -> list[dict]`
  - `build(player: str, scope: str = "adc", target: str = "challenger", outcome: str = "loss", gold_dir=None) -> dict`
  - Chaque signal : `{"group","key","label","you","ref","delta","unit","notable"}` (+ `"descriptive_only": True` sur les features de profondeur). `build` lève `FileNotFoundError` si un gold manque.

- [ ] **Step 1: Écrire les tests (échouent)**

Créer `tests/test_coaching_payload.py` :
```python
import json

import positioning as P
import payload as PL


def test_pos_meta_keys_match_coaching_safe():
    assert set(PL.POS_META) == P.COACHING_SAFE


def test_pos_signals_depth_always_descriptive_never_notable():
    mf = {"positioning": {"max_map_depth": 2728.0, "frac_roam_mid": 0.50}}
    rf = {"positioning": {"max_map_depth": 1633.0, "frac_roam_mid": 0.70}}
    out = {s["key"]: s for s in PL._pos_signals(mf, rf)}
    assert out["max_map_depth"]["descriptive_only"] is True
    assert out["max_map_depth"]["notable"] is False          # malgré delta énorme
    assert out["frac_roam_mid"]["notable"] is True            # |−0.20| ≥ 0.08
    assert "descriptive_only" not in out["frac_roam_mid"]


def test_pos_signals_only_coaching_safe_keys():
    mf = {"positioning": {k: 0.5 for k in P.ALL_FEATURES}}     # inclut ML_ONLY
    rf = {"positioning": {k: 0.5 for k in P.ALL_FEATURES}}
    keys = {s["key"] for s in PL._pos_signals(mf, rf)}
    assert keys <= P.COACHING_SAFE
    assert keys.isdisjoint(P.ML_ONLY)


def test_lane_signals_thresholds():
    mf = {"lane": {"gd10": 100, "csd14": 0}}
    rf = {"lane": {"gd10": -100, "csd14": -5}}
    out = {s["key"]: s for s in PL._lane_signals(mf, rf)}
    assert out["gd10"]["delta"] == 200 and out["gd10"]["notable"] is True   # >150
    assert out["csd14"]["delta"] == 5 and out["csd14"]["notable"] is True   # ≥2 cs


def test_zone_phase_signals_top_overdeaths():
    mf = {"by_zone_phase": {"BOT|mid": 0.29, "MID|late": 0.10}}
    rf = {"by_zone_phase": {"BOT|mid": 0.05, "MID|late": 0.11}}
    out = PL._zone_phase_signals(mf, rf)
    assert out[0]["key"] == "BOT|mid" and out[0]["notable"] is True         # Δ +0.24


def test_build_reads_gold_and_flags_low_sample(tmp_path):
    facet = {"n_games": 3, "deaths_per_game": 6.0,
             "lane": {"gd10": -100, "gd14": 0, "gd20": 0, "csd10": 0, "csd14": -5},
             "positioning": {k: 0.5 for k in P.COACHING_SAFE},
             "death_gold_state": {"ahead": 0.3, "even": 0.2, "behind": 0.5},
             "by_zone_phase": {"BOT|mid": 0.3}}
    agg = {"scope": "adc", "patch": "16.13", "n_games": 3, "winrate": 0.33,
           "overall": facet, "win": facet, "loss": facet, "by_lane_context": {}}
    for who, n in (("personal/spadzze", 3), ("referentiel/challenger", 1000)):
        a = dict(agg); a["n_games"] = n
        d = tmp_path / who / "adc"; d.mkdir(parents=True)
        (d / "aggregate.json").write_text(json.dumps(a))
    pl = PL.build("spadzze", "adc", "challenger", "loss", gold_dir=tmp_path)
    assert pl["meta"]["low_sample"] is True                  # 3 < 30
    assert pl["meta"]["n_games_ref"] == 1000
    # aucune feature ML_ONLY nulle part dans le payload sérialisé
    blob = json.dumps(pl)
    assert all(k not in blob for k in P.ML_ONLY)
    assert any(s["group"] == "positioning" for s in pl["signals"])
```

Run: `.venv/bin/python -m pytest tests/test_coaching_payload.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'payload'`)

- [ ] **Step 2: Implémenter `payload.py`**

Créer `src/04_coaching/payload.py` :
```python
"""Gold (perso + référentiel) -> payload de coaching compact et déterministe.

PUR (sauf lecture des aggregate.json). Les features ont déjà conclu : on
sélectionne ici les signaux saillants (flag `notable`) selon des seuils fixes.
N'expose QUE des métriques asymétrie-safe : positioning ⊂ COACHING_SAFE, et
les features de profondeur sont `descriptive_only` (jamais une erreur).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # accès src/
import riotlib as rl
import positioning
import compare

LANE_SIGNALS = ["gd10", "gd14", "gd20", "csd10", "csd14"]
LANE_LABELS = {"gd10": "gold diff @10", "gd14": "gold diff @14", "gd20": "gold diff @20",
               "csd10": "cs diff @10", "csd14": "cs diff @14"}

# clé -> (label, unit, notable_threshold | None, descriptive_only)
# Profondeur (avg/max_map_depth, frac_overextended) : threshold=None + descriptive_only=True
# -> jamais `notable`, observable mais jamais prescrit.
POS_META = {
    "frac_own_lane_early": ("% lane (early)", "pct", 0.08, False),
    "frac_river_early":    ("% river (early)", "pct", 0.08, False),
    "frac_roam_mid":       ("% roam (mid)", "pct", 0.08, False),
    "frac_enemy_half":     ("% moitié ennemie", "pct", 0.08, False),
    "frac_base":           ("% en base", "pct", 0.08, False),
    "frac_overextended":   ("% over-extended", "pct", None, True),
    "avg_map_depth":       ("profondeur moy.", "u", None, True),
    "max_map_depth":       ("profondeur max", "u", None, True),
    "avg_dist_to_ally":    ("isolement (allié)", "u", 200.0, False),
    "gold_dead_time":      ("temps mort (s)", "s", 20.0, False),
    "wards_placed":        ("wards posées", "ward", 2.0, False),
    "wards_placed_early":  ("wards early", "ward", 1.0, False),
    "control_wards_placed": ("control wards", "ward", 1.0, False),
    "wards_killed":        ("wards détruites", "ward", 2.0, False),
}
# Garde-fou asymétrie : la table doit couvrir EXACTEMENT les features safe, ni plus ni moins.
assert set(POS_META) == positioning.COACHING_SAFE, \
    "POS_META doit refléter exactement positioning.COACHING_SAFE"

LOW_SAMPLE_THRESHOLD = 30


def _lane_signals(mf: dict, rf: dict) -> list[dict]:
    out = []
    lane_me, lane_rf = mf.get("lane", {}), rf.get("lane", {})
    for key in LANE_SIGNALS:
        you, ref = lane_me.get(key), lane_rf.get(key)
        if you is None or ref is None:
            continue
        delta = you - ref
        unit = "cs" if key.startswith("cs") else "g"
        notable = abs(delta) >= 2 if unit == "cs" else abs(delta) > 150
        out.append({"group": "lane", "key": key, "label": LANE_LABELS[key],
                    "you": you, "ref": ref, "delta": delta, "unit": unit,
                    "notable": notable})
    return out


def _pos_signals(mf: dict, rf: dict) -> list[dict]:
    out = []
    pos_me, pos_rf = mf.get("positioning", {}), rf.get("positioning", {})
    for key in sorted(POS_META):
        you, ref = pos_me.get(key), pos_rf.get(key)
        if you is None or ref is None:
            continue
        label, unit, thr, descriptive = POS_META[key]
        delta = round(you - ref, 4)
        notable = (thr is not None) and abs(delta) >= thr
        sig = {"group": "positioning", "key": key, "label": label,
               "you": you, "ref": ref, "delta": delta, "unit": unit,
               "notable": notable}
        if descriptive:
            sig["descriptive_only"] = True
        out.append(sig)
    return out


def _zone_phase_signals(mf: dict, rf: dict, top: int = 5) -> list[dict]:
    me_zp, rf_zp = mf.get("by_zone_phase", {}), rf.get("by_zone_phase", {})
    rows = []
    for key in set(me_zp) | set(rf_zp):
        you, ref = me_zp.get(key, 0.0), rf_zp.get(key, 0.0)
        delta = round(you - ref, 4)
        rows.append({"group": "deaths_zone_phase", "key": key,
                     "label": f"morts {key}", "you": you, "ref": ref,
                     "delta": delta, "unit": "pct", "notable": delta >= 0.08})
    rows.sort(key=lambda s: s["delta"], reverse=True)   # où tu sur-meurs d'abord
    return rows[:top]


def _gold_state_signals(mf: dict, rf: dict) -> list[dict]:
    me_gs, rf_gs = mf.get("death_gold_state", {}), rf.get("death_gold_state", {})
    labels = {"ahead": "morts en avance", "even": "morts à égalité", "behind": "morts en retard"}
    out = []
    for key, label in labels.items():
        you, ref = me_gs.get(key), rf_gs.get(key)
        if you is None or ref is None:
            continue
        delta = round(you - ref, 4)
        out.append({"group": "death_gold_state", "key": key, "label": label,
                    "you": you, "ref": ref, "delta": delta, "unit": "pct",
                    "notable": abs(delta) >= 0.10})
    return out


def _load(gold_dir: Path, *parts) -> dict:
    path = gold_dir.joinpath(*parts) / "aggregate.json"
    if not path.exists():
        raise FileNotFoundError(f"gold manquant : {path}")
    return json.loads(path.read_text())


def build(player: str, scope: str = "adc", target: str = "challenger",
          outcome: str = "loss", gold_dir=None) -> dict:
    gold_dir = Path(gold_dir) if gold_dir is not None else rl.GOLD_DIR
    me = _load(gold_dir, "personal", player, scope)
    ref = _load(gold_dir, "referentiel", target, scope)
    mf, rf = me[outcome], ref[outcome]

    meta = {
        "player": player, "scope": scope, "target": target,
        "outcome_focus": outcome, "patch": me.get("patch", "?"),
        "n_games_me": me["n_games"], "n_games_ref": ref["n_games"],
        "winrate_me": me["winrate"],
        "low_sample": me["n_games"] < LOW_SAMPLE_THRESHOLD,
        "deaths_per_game": {oc: {"you": me[oc]["deaths_per_game"],
                                 "ref": ref[oc]["deaths_per_game"]}
                            for oc in ("overall", "win", "loss")},
    }
    signals = (_lane_signals(mf, rf) + _pos_signals(mf, rf)
               + _zone_phase_signals(mf, rf) + _gold_state_signals(mf, rf))

    context = {}
    for axis in ("lane_pattern", "gank_exposure"):
        cb = compare.context_benchmark(me, ref, axis, outcome)
        if cb:
            context[axis] = cb

    return {"meta": meta, "signals": signals, "context": context}
```

- [ ] **Step 3: Lancer les tests (passent)**

Run: `.venv/bin/python -m pytest tests/test_coaching_payload.py -q`
Expected: PASS (6 tests)

- [ ] **Step 4: Commit**

```bash
git add src/04_coaching/payload.py tests/test_coaching_payload.py
git commit -m "feat(coaching): payload déterministe gold->signaux, safe-only"
```

---

### Task 3: Prompt (payload → messages, pur)

**Files:**
- Create: `src/04_coaching/prompt.py`
- Test: `tests/test_coaching_prompt.py`

**Interfaces:**
- Consumes: payload dict de Task 2, `positioning.ML_ONLY`.
- Produces: `SYSTEM: str` ; `render(payload: dict) -> tuple[str, str]` (system, user).

- [ ] **Step 1: Écrire les tests (échouent)**

Créer `tests/test_coaching_prompt.py` :
```python
import json

import positioning as P
import prompt as PR


def _payload():
    return {"meta": {"player": "spadzze", "scope": "adc", "target": "challenger",
                     "outcome_focus": "loss", "n_games_me": 15, "low_sample": True},
            "signals": [{"group": "positioning", "key": "frac_roam_mid",
                         "you": 0.5, "ref": 0.7, "delta": -0.2, "notable": True}],
            "context": {}}


def test_system_encodes_asymmetry_and_depth_rules():
    s = PR.SYSTEM.lower()
    assert "asym" in s                       # règle d'asymétrie présente
    assert "profondeur" in s                 # nuance profondeur présente
    assert "descriptive_only" in PR.SYSTEM   # le LLM sait ne pas prescrire ces signaux


def test_render_returns_system_and_user_with_payload():
    system, user = PR.render(_payload())
    assert system == PR.SYSTEM
    assert "15 dernières games" in user
    assert "challenger" in user
    assert json.loads(user[user.index("{"):user.rindex("}") + 1])  # le payload JSON est inclus


def test_prompt_never_leaks_ml_only_feature_names():
    system, user = PR.render(_payload())
    for k in P.ML_ONLY:
        assert k not in system and k not in user
```

Run: `.venv/bin/python -m pytest tests/test_coaching_prompt.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'prompt'`)

- [ ] **Step 2: Implémenter `prompt.py`**

Créer `src/04_coaching/prompt.py` :
```python
"""Payload de coaching -> messages (system, user) pour le LLM. PUR.

Le system encode les règles dures (asymétrie, preuve obligatoire, priorité aux
signaux notable, nuance profondeur, benchmark-relatif, FR). La sélection des
signaux est déjà faite dans payload.py : ici on n'impose que le cadre de narration.
"""
from __future__ import annotations

import json

SYSTEM = """Tu es un coach League of Legends personnel expert. Tu reçois un JSON \
de signaux DÉJÀ calculés : le joueur comparé à un benchmark de son rang cible \
(challenger). Ton rôle est de RACONTER et PRIORISER ces signaux, jamais de calculer \
ni d'inventer un chiffre.

Règles absolues :
1. ASYMÉTRIE — ne reproche JAMAIS une décision fondée sur une information que le \
joueur n'avait pas. Les valeurs `ref` sont des repères (« les challengers font Y »), \
jamais « tu aurais dû savoir X ».
2. PREUVE OBLIGATOIRE — chaque point cite la stat correspondante du payload \
(valeur du joueur vs ref). N'invente aucune stat absente du payload.
3. PRIORITÉ — traite d'abord les signaux `notable: true`. Un signal marqué \
`descriptive_only: true` peut être mentionné comme observation neutre, JAMAIS comme \
une erreur à corriger. En particulier la PROFONDEUR de carte élevée n'est PAS un \
défaut (elle corrèle au rang inférieur) : ne prescris jamais « prends plus / moins \
d'espace » à partir d'elle.
4. CONCRET & BENCHMARK-RELATIF — « tu recall à 1450 g vs 1100 g challenger » ✅, \
« meurs moins » ❌.
5. Si `meta.low_sample` vaut true, abaisse `confidence` et signale l'échantillon faible.
6. Français, tutoiement, concis. Respecte strictement le schéma de sortie imposé \
(3 forces, 3 erreurs, 2 habitudes, 1 focus, confidence)."""


def render(payload: dict) -> tuple[str, str]:
    m = payload["meta"]
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    user = (f"Signaux de tes {m['n_games_me']} dernières games "
            f"({m['scope']}, issue={m['outcome_focus']}, vs {m['target']}) :\n\n"
            f"{body}\n\nProduis la review.")
    return SYSTEM, user
```

- [ ] **Step 3: Lancer les tests (passent)**

Run: `.venv/bin/python -m pytest tests/test_coaching_prompt.py -q`
Expected: PASS (3 tests)

- [ ] **Step 4: Commit**

```bash
git add src/04_coaching/prompt.py tests/test_coaching_prompt.py
git commit -m "feat(coaching): prompt système (asymétrie + benchmark-relatif)"
```

---

### Task 4: Client Ollama Cloud

**Files:**
- Create: `src/04_coaching/llm_client.py`
- Test: `tests/test_coaching_llm_client.py`

**Interfaces:**
- Consumes: `riotlib.load_env()`, `requests.post`.
- Produces: `OLLAMA_URL: str` ; `class LLMError(RuntimeError)` ; `generate_json(model: str, system: str, user: str, schema: dict, temperature: float = 0.2, timeout: int = 180) -> dict`.

- [ ] **Step 1: Écrire les tests (échouent)**

Créer `tests/test_coaching_llm_client.py` :
```python
import json

import pytest

import llm_client as LC


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}
    def json(self):
        return self._payload
    def raise_for_status(self):
        pass


def test_generate_json_posts_and_parses(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        captured["headers"] = headers
        content = '{"ok": true}'
        return _Resp(200, {"message": {"content": content}})

    monkeypatch.setattr(LC.rl, "load_env", lambda: {"OLLAMA_API_KEY": "k123"})
    monkeypatch.setattr(LC.requests, "post", fake_post)
    out = LC.generate_json("deepseek-v4-pro", "sys", "usr", {"type": "object"})
    assert out == {"ok": True}
    assert captured["url"] == LC.OLLAMA_URL
    assert captured["headers"]["Authorization"] == "Bearer k123"
    assert captured["body"]["format"] == {"type": "object"}
    assert captured["body"]["stream"] is False
    assert captured["body"]["model"] == "deepseek-v4-pro"


def test_generate_json_missing_key_raises(monkeypatch):
    monkeypatch.setattr(LC.rl, "load_env", lambda: {})
    with pytest.raises(LC.LLMError):
        LC.generate_json("m", "s", "u", {})


def test_generate_json_401_raises_clear(monkeypatch):
    monkeypatch.setattr(LC.rl, "load_env", lambda: {"OLLAMA_API_KEY": "k"})
    monkeypatch.setattr(LC.requests, "post", lambda *a, **k: _Resp(401))
    with pytest.raises(LC.LLMError) as e:
        LC.generate_json("m", "s", "u", {})
    assert "OLLAMA_API_KEY" in str(e.value)
```

Run: `.venv/bin/python -m pytest tests/test_coaching_llm_client.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'llm_client'`)

- [ ] **Step 2: Implémenter `llm_client.py`**

Créer `src/04_coaching/llm_client.py` :
```python
"""Client Ollama Cloud (structured output). Aucune logique métier.

POST https://ollama.com/api/chat avec Authorization: Bearer <OLLAMA_API_KEY>
et `format` = JSON-schema -> renvoie le dict parsé du message. Retries/backoff
sur 429/5xx/timeout, message clair sur clé absente / 401.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # accès src/
import requests
import riotlib as rl

OLLAMA_URL = "https://ollama.com/api/chat"
_MAX_ATTEMPTS = 4


class LLMError(RuntimeError):
    pass


def generate_json(model: str, system: str, user: str, schema: dict,
                  temperature: float = 0.2, timeout: int = 180) -> dict:
    key = rl.load_env().get("OLLAMA_API_KEY")
    if not key:
        raise LLMError("OLLAMA_API_KEY absente du .env")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "format": schema,
        "stream": False,
        "options": {"temperature": temperature},
    }
    headers = {"Authorization": f"Bearer {key}"}
    last = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            r = requests.post(OLLAMA_URL, json=body, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last = e
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 401:
            raise LLMError("401 — vérifie OLLAMA_API_KEY")
        if r.status_code == 429 or r.status_code >= 500:
            last = LLMError(f"HTTP {r.status_code}")
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return json.loads(content)
    raise LLMError(f"échec après {_MAX_ATTEMPTS} tentatives : {last}")
```

- [ ] **Step 3: Lancer les tests (passent)**

Run: `.venv/bin/python -m pytest tests/test_coaching_llm_client.py -q`
Expected: PASS (3 tests)

- [ ] **Step 4: Commit**

```bash
git add src/04_coaching/llm_client.py tests/test_coaching_llm_client.py
git commit -m "feat(coaching): client Ollama Cloud (structured output + retries)"
```

---

### Task 5: Orchestrateur CLI `coach.py` (validation + persistance)

**Files:**
- Create: `src/04_coaching/coach.py`
- Test: `tests/test_coaching_coach.py`

**Interfaces:**
- Consumes: `payload.build`, `prompt.render`, `schema.Review`/`schema.review_json_schema`, `llm_client.generate_json`, `riotlib.DATA`.
- Produces:
  - `DEFAULT_MODEL = "deepseek-v4-pro"`
  - `class CoachValidationError(RuntimeError)` (attribut `.raw`)
  - `generate_review(pl: dict, model: str) -> schema.Review` (1 retry sur ValidationError, sinon lève `CoachValidationError`)
  - `persist(player: str, model: str, pl: dict, review: schema.Review, ts: str, root=None) -> Path`
  - `render_text(review: schema.Review) -> str`
  - `main() -> int` (argparse)

- [ ] **Step 1: Écrire les tests (échouent)**

Créer `tests/test_coaching_coach.py` :
```python
import json

import pytest

import coach as C
import schema as S


def _review_dict():
    ins = {"point": "p", "evidence": "e"}
    return {"strengths": [ins, ins, ins], "mistakes": [ins, ins, ins],
            "habits": ["h1", "h2"], "next_focus": "f", "confidence": 0.5}


def test_generate_review_validates(monkeypatch):
    monkeypatch.setattr(C.llm_client, "generate_json", lambda *a, **k: _review_dict())
    r = C.generate_review({"meta": {"player": "x", "scope": "adc", "target": "challenger",
                                    "outcome_focus": "loss", "n_games_me": 1}}, "m")
    assert isinstance(r, S.Review) and r.confidence == 0.5


def test_generate_review_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def bad(*a, **k):
        calls["n"] += 1
        return {"bogus": True}                       # invalide -> ValidationError

    monkeypatch.setattr(C.llm_client, "generate_json", bad)
    with pytest.raises(C.CoachValidationError):
        C.generate_review({"meta": {"player": "x", "scope": "adc", "target": "challenger",
                                    "outcome_focus": "loss", "n_games_me": 1}}, "m")
    assert calls["n"] == 2                            # 1 essai + 1 retry


def test_persist_appends_jsonl(tmp_path):
    pl = {"meta": {"scope": "adc", "target": "challenger", "outcome_focus": "loss"}}
    review = S.Review.model_validate(_review_dict())
    path = C.persist("spadzze", "deepseek-v4-pro", pl, review,
                     ts="2026-06-30T10:00:00", root=tmp_path)
    line = json.loads(path.read_text().splitlines()[-1])
    assert line["model"] == "deepseek-v4-pro"
    assert line["review"]["confidence"] == 0.5
    assert line["payload"]["meta"]["scope"] == "adc"
    assert line["ts"] == "2026-06-30T10:00:00"


def test_render_text_is_french_and_has_sections():
    txt = C.render_text(S.Review.model_validate(_review_dict()))
    assert "Forces" in txt and "Erreurs" in txt and "Focus" in txt
```

Run: `.venv/bin/python -m pytest tests/test_coaching_coach.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'coach'`)

- [ ] **Step 2: Implémenter `coach.py`**

Créer `src/04_coaching/coach.py` :
```python
#!/usr/bin/env python3
"""04_coaching — compte-rendu de coaching agrégé, narré par Ollama Cloud.

Pipeline : gold (perso + référentiel) -> payload déterministe -> prompt ->
Ollama Cloud (structured output) -> Review validée (Pydantic) -> affichage FR +
persistance data/07_coaching/<player>/reviews.jsonl.

Usage : python3 src/04_coaching/coach.py --player spadzze --scope adc \
        [--outcome loss] [--target challenger] [--model deepseek-v4-pro]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # accès src/
import riotlib as rl
from pydantic import ValidationError

import payload as payload_mod
import prompt as prompt_mod
import schema as schema_mod
import llm_client

DEFAULT_MODEL = "deepseek-v4-pro"


class CoachValidationError(RuntimeError):
    def __init__(self, raw):
        super().__init__("sortie LLM non conforme au schéma après retry")
        self.raw = raw


def generate_review(pl: dict, model: str) -> schema_mod.Review:
    system, user = prompt_mod.render(pl)
    sch = schema_mod.review_json_schema()
    last_raw = None
    for _ in range(2):                       # 1 essai + 1 retry
        last_raw = llm_client.generate_json(model, system, user, sch)
        try:
            return schema_mod.Review.model_validate(last_raw)
        except ValidationError:
            continue
    raise CoachValidationError(last_raw)


def persist(player: str, model: str, pl: dict, review: schema_mod.Review,
            ts: str, root=None) -> Path:
    root = Path(root) if root is not None else rl.DATA / "07_coaching"
    out = root / player
    out.mkdir(parents=True, exist_ok=True)
    record = {"ts": ts, "model": model,
              "scope": pl["meta"]["scope"], "target": pl["meta"]["target"],
              "outcome_focus": pl["meta"]["outcome_focus"],
              "payload": pl, "review": review.model_dump()}
    path = out / "reviews.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def render_text(review: schema_mod.Review) -> str:
    lines = [f"\n  Confiance : {review.confidence:.0%}", "\n  Forces :"]
    lines += [f"    + {i.point}  ({i.evidence})" for i in review.strengths]
    lines.append("\n  Erreurs prioritaires :")
    lines += [f"    - {i.point}  ({i.evidence})" for i in review.mistakes]
    lines.append("\n  Habitudes à corriger :")
    lines += [f"    • {h}" for h in review.habits]
    lines.append(f"\n  Focus prochaine game : {review.next_focus}")
    return "\n".join(lines)


def _save_failed(player: str, ts: str, raw) -> Path:
    out = rl.DATA / "07_coaching" / player / "failed"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{ts.replace(':', '-')}.json"
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default="spadzze")
    ap.add_argument("--scope", default="adc")
    ap.add_argument("--outcome", default="loss", choices=["overall", "win", "loss"])
    ap.add_argument("--target", default="challenger")
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
    args = ap.parse_args()

    ts = datetime.now().isoformat(timespec="seconds")
    try:
        pl = payload_mod.build(args.player, args.scope, args.target, args.outcome)
    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    try:
        review = generate_review(pl, args.model)
    except llm_client.LLMError as e:
        print(f"✗ appel LLM échoué : {e}", file=sys.stderr)
        return 1
    except CoachValidationError as e:
        p = _save_failed(args.player, ts, e.raw)
        print(f"✗ {e} — brut sauvé dans {p}", file=sys.stderr)
        return 1

    path = persist(args.player, args.model, pl, review, ts)
    print(f"COACHING — {args.player} ({args.scope}, issue={args.outcome}, "
          f"vs {args.target}) [modèle {args.model}]")
    print(render_text(review))
    print(f"\n✓ review persistée dans {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Lancer les tests (passent)**

Run: `.venv/bin/python -m pytest tests/test_coaching_coach.py -q`
Expected: PASS (4 tests)

- [ ] **Step 4: Lancer toute la suite (non-régression)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (49 existants + nouveaux coaching, aucun échec)

- [ ] **Step 5: Commit**

```bash
git add src/04_coaching/coach.py tests/test_coaching_coach.py
git commit -m "feat(coaching): orchestrateur CLI coach.py (validation + persistance)"
```

---

### Task 6: Documentation — CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: rien (doc). Décrit `src/04_coaching/`, `data/07_coaching/`, le schéma `Review`.

- [ ] **Step 1: Documenter le module dans l'architecture du code**

Dans `CLAUDE.md`, section « ## Architecture du code », après le bloc `src/03_data_analyse/`, ajouter :
```markdown
  - **`src/04_coaching/`** : narration LLM du coaching (Ollama Cloud, structured output).
    `payload.py` (gold perso+réf → payload déterministe, **safe-only** : positioning ⊂
    COACHING_SAFE, profondeur `descriptive_only`), `prompt.py` (system asymétrie +
    benchmark-relatif, FR), `schema.py` (Pydantic `Review` : 3 forces / 3 erreurs / 2
    habitudes / 1 focus / confidence, **preuve chiffrée par point**), `llm_client.py`
    (client `https://ollama.com/api/chat`, `OLLAMA_API_KEY`, `format`=JSON-schema,
    défaut `deepseek-v4-pro`), `coach.py` (CLI : payload→prompt→client→validation→affiche+
    persiste). Lancer : `python3 src/04_coaching/coach.py --player spadzze --scope adc`.
```

- [ ] **Step 2: Ajouter la couche data 07_coaching**

Dans le bloc arborescence `data/` de `CLAUDE.md`, après la ligne `05_model/`, ajouter :
```markdown
  06_shap/                               # SHAP/EBM outputs (rankings, shape functions)
  07_coaching/<player>/reviews.jsonl     # reviews LLM persistées (payload+review horodatés)
```

- [ ] **Step 3: Mettre à jour le statut + la note evidence**

Dans `CLAUDE.md`, section « ### Prochaines étapes », remplacer la puce 1 (« Brancher Ollama… ») par :
```markdown
1. ✅ **Ollama branché** (Phase 2 narration) — `src/04_coaching/` génère un compte-rendu
   agrégé typé (Ollama Cloud, `deepseek-v4-pro`) depuis le diff perso↔référentiel, persisté
   dans `data/07_coaching/`. À suivre : compte-rendu par-game + boucle d'éval (scoring d'utilité).
```
Et dans la section « ## Pipeline de résumé » / sortie LLM, remplacer la ligne `evidence[]` par :
```markdown
   - chaque `strength`/`mistake` porte sa **preuve chiffrée** (`evidence` par point, fusionné)
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(coaching): documenter src/04_coaching + data/07_coaching"
```

---

## Notes d'implémentation

- **Test d'intégration réel (optionnel, hors plan automatique)** : une fois Task 5 mergée, vérifier un appel cloud réel avec `ollama pull` non requis (cloud) :
  `python3 src/04_coaching/coach.py --player spadzze --scope adc`. Relire la review : conseils concrets, benchmark-relatifs, avec preuve, sans reproche sur info cachée. Si la profondeur est traitée comme un défaut → durcir la règle 3 du prompt.
- **A/B modèles** : rejouer le même payload sur un autre modèle via `--model glm-5.2` / `--model minimax-m3` ; comparer les lignes de `reviews.jsonl`.
