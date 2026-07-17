# Modélisation séquentielle + self-supervised — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mesurer si un transformer sur les séquences d'états par-minute des timelines bat le baseline tabulaire per-game (dia_chall 0.724 / master-GM 0.589), et si un pretraining self-supervised (mask-and-reconstruct) ajoute du signal — branche parallèle de recherche, 0 perturbation du pipeline existant.

**Architecture:** Nouveau dataset `adc_sequence_dataset.npz` (1 ADC/game = séquence [40,20] + mask) construit depuis le raw déjà caché (0 API). Petit transformer encoder écrit à la main en PyTorch (4 couches, d_model=64), tête masked-mean-pool → classification binaire. CV purgé identique au baseline (StratifiedKFold joueurs + purge games miroir). Étape 2 : pretrain mask-and-reconstruct sur toutes les games, fine-tune, delta mesuré. Standardisation per-feature non négociable.

**Tech Stack:** Python 3.13, PyTorch (MPS sur Mac M4), NumPy, pandas, scikit-learn, xgboost (baselines), pytest. Réutilise `riotlib` (`_read_raw`, `_frames_by_minute`, `DATA/RAW_DIR/SILVER_DIR`), `build_dataset` (`adc_puuids`, `build_rank_map`, `_load_raw`, `RANKS`, `RANK_ORD`, `HIGH_ELO`).

## Global Constraints

- **0 appel API** — tout est relu depuis `data/01_raw/*.json.zst` via `riotlib._read_raw`.
- **Standardisation per-feature z-score obligatoire** (mean/std par feature sur le **train** du fold, stats figées, réappliquées à val/test), avant la projection `20→d_model`. La position est normalisée à [0,1] à la construction, puis z-scoreée comme les autres features. Sans ça, un null est un artefact d'optimisation, pas une réponse à la question.
- **CV purgé** : StratifiedKFold sur les joueurs (l'identité du joueur porte le rang) + purge des games miroir (drop de train toute row dont l'ADC adverse est un joueur de val). Même discipline que `train_player_ensemble.py`.
- **Deux tâches de classification co-primaires** : `high_elo` (toutes games référentielles, label = rang ∈ {GM, challenger} vs {diamond, master}) et `dia_chall` (games filtrées rang ∈ {diamond, challenger}, label = challenger). Le verdict se lit sur dia_chall ; master/GM est rapporté mais son null est non interprétable (bruit de label).
- **Perso (rank NaN) exclu** des tâches labellisées (pas de label).
- **Cap T=40 minutes** ; games plus courtes → pad `mask=False` ; games > 40 min → tronquées.
- **Branche parallèle** : ne touche ni le pipeline tabulaire ni le coach web. `torch` est ajouté au groupe `analysis` de `pyproject.toml`.
- **TDD** : chaque tâche écrit d'abord un test qui échoue, puis le code minimal, puis commit.
- **Convention flat-import** : chaque script insère `src/core/` dans `sys.path` avant `import riotlib` (cf. `build_dataset.py:34`).
- Lancer les scripts via `poetry run python3 src/<dossier>/<script>.py`, les tests via `poetry run pytest tests/<fichier>::<test> -v`.

## File Structure

| Fichier | Responsabilité |
|---|---|
| `src/01_data_engineering/build_sequence_dataset.py` | raw → `data/04_dataset/adc_sequence_dataset.npz`. Réutilise `build_dataset` via importlib. Pure extraction, 0 API. |
| `src/02_data_science/sequence_model.py` | Modules PyTorch purs : `SequenceEncoder`, `ClassifierHead`, `SequenceClassifier`, `ReconstructHead`, helper `masked_mean`, `get_device`. |
| `src/02_data_science/sequence_data.py` | Chargement npz, fabrication des labels par tâche, folds purgés (joueurs) + purge miroir, standardisation per-feature. Partagé par train + pretrain (DRY). |
| `src/02_data_science/train_sequence_model.py` | Étape 1 : boucle supervisée sur les 2 tâches, baselines (tabulaire xgb + MLP) sur les mêmes folds, écrit `sequence_metrics.json`. |
| `src/02_data_science/pretrain_sequence_model.py` | Étape 2 : pretrain mask-and-reconstruct, fine-tune, delta `delta_ssl` dans `sequence_metrics.json`, `embed_game`. |
| `tests/test_build_sequence_dataset.py` | Extraction : `frame_state`, `opponent_pid`, `build_sequence` (shapes, mask, valeurs attendues sur un match synthétique). |
| `tests/test_sequence_model.py` | Forward/shapes, masked-mean-pool ignore le pad, reconstruct. |
| `tests/test_sequence_data.py` | Labels par tâche, folds joueur-groupés (pas de chevauchement), purge miroir, standardisation (mean≈0/std≈1 sur train, stats figées sur val). |

> Le spec listait 4 fichiers source ; `sequence_data.py` est ajouté pour DRY (folds + standardisation consommés par `train_sequence_model` et `pretrain_sequence_model`). Responsabilité unique, boundaries claires — cohérent avec les principes du skill.

---

### Task 0: Ajouter PyTorch au projet + vérifier MPS

**Files:**
- Modify: `pyproject.toml` (groupe `[tool.poetry.group.analysis.dependencies]`)

**Interfaces:**
- Produces: `torch` importable dans l'env Poetry ; `torch.backends.mps.is_available() == True` sur Mac M4.

- [ ] **Step 1: Ajouter la dépendance torch**

Modifier `pyproject.toml`, section `[tool.poetry.group.analysis.dependencies]`, ajouter la ligne après `shap` :

```toml
torch = ">=2.6"
```

- [ ] **Step 2: Installer**

Run: `poetry install --with analysis`
Expected: torch installé (déjà présent transitivement, poetry le confirme). Aucune erreur.

- [ ] **Step 3: Vérifier MPS**

Run: `poetry run python3 -c "import torch; print(torch.__version__, 'mps', torch.backends.mps.is_available())"`
Expected: `2.x.x mps True`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "build: ajoute torch au groupe analysis (transformer séquentiel, MPS)"
```

---

### Task 1: `build_sequence_dataset` — extraction d'une frame + résolution opponent (TDD)

**Files:**
- Create: `src/01_data_engineering/build_sequence_dataset.py`
- Test: `tests/test_build_sequence_dataset.py`

**Interfaces:**
- Produces: `frame_state(pf: dict) -> list[float]` (8-dim), `participant_pid(match, puuid) -> int`, `opponent_pid(match, puuid) -> int | None`, constantes `MAP_SIZE`, `MAX_LEN`, `STATE_FIELDS`.

- [ ] **Step 1: Écrire le test qui échoue**

`tests/test_build_sequence_dataset.py` :

```python
"""Tests build_sequence_dataset : extraction de frames + résolution matchup."""
import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
_spec = importlib.util.spec_from_file_location(
    "build_sequence_dataset", _SRC / "01_data_engineering" / "build_sequence_dataset.py")
bsd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsd)


def test_frame_state_returns_8dim_normalized():
    pf = {"position": {"x": 14800, "y": 7400},
          "totalGold": 600, "currentGold": 50, "xp": 100, "level": 2,
          "minionsKilled": 5, "jungleMinionsKilled": 1}
    s = bsd.frame_state(pf)
    assert len(s) == 8
    assert s[0] == 1.0          # x / MAP_SIZE
    assert s[1] == 0.5          # y / MAP_SIZE
    assert s[2] == 600.0 and s[3] == 50.0 and s[4] == 100.0
    assert s[5] == 2.0 and s[6] == 5.0 and s[7] == 1.0


def test_frame_state_missing_fields_zero():
    s = bsd.frame_state({})
    assert s == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _fake_match():
    # 2 équipes × 5 rôles ; puuids[i] ↔ participants[i]
    puuids = [f"p_{r}_{t}" for t in (100, 200)
              for r in ("BOTTOM", "UTILITY", "JUNGLE", "MIDDLE", "TOP")]
    parts = [{"teamId": t, "teamPosition": r, "championName": f"{r}_{t}"}
             for t in (100, 200) for r in ("BOTTOM", "UTILITY", "JUNGLE", "MIDDLE", "TOP")]
    return {"metadata": {"participants": puuids}, "info": {"participants": parts}}


def test_participant_and_opponent_pid():
    m = _fake_match()
    # p_BOTTOM_100 est idx 0 → pid 1
    assert bsd.participant_pid(m, "p_BOTTOM_100") == 1
    # opponent = BOTTOM de l'équipe 200 → idx 5 → pid 6
    assert bsd.opponent_pid(m, "p_BOTTOM_100") == 6
    assert bsd.opponent_pid(m, "p_BOTTOM_200") == 1


def test_opponent_pid_none_if_role_missing():
    m = _fake_match()
    assert bsd.opponent_pid(m, "p_JUNGLE_100") == 3  # jungle opp = idx 7 → pid 8
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_build_sequence_dataset.py -v`
Expected: FAIL — `module 'build_sequence_dataset' has no attribute 'frame_state'` (module vide/non créé).

- [ ] **Step 3: Implémenter le minimal**

`src/01_data_engineering/build_sequence_dataset.py` :

```python
#!/usr/bin/env python3
"""
01_data_engineering — raw -> dataset séquentiel (1 ADC d'une game = 1 séquence [40,20]).

Construit les séquences d'états par-minute DEPUIS LE RAW (0 API), pour le transformer
séquentiel. Réutilise build_dataset (adc_puuids, build_rank_map, _load_raw, RANKS/HIGH_ELO)
via importlib — zéro duplication de la logique métier.

State vector 20-dim/frame = ADC ciblé (8) + ADC adverse (8) + 4 diffs relatives
(gold, cs, xp, level). Aucun event en v1 (les events discrets = étape 2 d'enrichissement,
cf. spec 2026-07-18 §Pièges : un null v1 ne réfute pas la thèse séquence).

Sortie : data/04_dataset/adc_sequence_dataset.npz
Usage : poetry run python3 src/01_data_engineering/build_sequence_dataset.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_CORE = Path(__file__).resolve().parent.parent / "core"
sys.path.insert(0, str(_CORE))                       # import riotlib
import numpy as np
import riotlib as rl

# reutilise build_dataset (vit dans un dossier non-importable) via importlib
_BD = Path(__file__).resolve().parent / "build_dataset.py"
_spec = importlib.util.spec_from_file_location("build_dataset", _BD)
build_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_dataset)

MAP_SIZE = 14800.0          # Summoner's Rift (approx, normalisation position -> [0,1])
MAX_LEN = 40                 # cap minutes ; pad au-delà de la durée réelle
STATE_FIELDS = ["pos_x", "pos_y", "totalGold", "currentGold", "xp",
                "level", "minionsKilled", "jungleMinionsKilled"]


def frame_state(pf: dict) -> list[float]:
    """Vecteur d'état 8-dim depuis un participantFrame. Position normalisée à [0,1]."""
    pos = pf.get("position") or {}
    return [
        float(pos.get("x") or 0) / MAP_SIZE,
        float(pos.get("y") or 0) / MAP_SIZE,
        float(pf.get("totalGold") or 0),
        float(pf.get("currentGold") or 0),
        float(pf.get("xp") or 0),
        float(pf.get("level") or 0),
        float(pf.get("minionsKilled") or 0),
        float(pf.get("jungleMinionsKilled") or 0),
    ]


def participant_pid(match: dict, puuid: str) -> int:
    """puuid -> participantId (1-indexé)."""
    return match["metadata"]["participants"].index(puuid) + 1


def opponent_pid(match: dict, target_puuid: str) -> int | None:
    """participantId de l'adversaire de même rôle (BOTTOM), équipe opposée. None si introuvable."""
    meta = match["metadata"]
    parts = match["info"]["participants"]
    pidx = meta["participants"].index(target_puuid)
    me = parts[pidx]
    my_team = me["teamId"]
    my_role = me.get("teamPosition") or ""
    for i, p in enumerate(parts):
        if p["teamId"] != my_team and (p.get("teamPosition") or "") == my_role and my_role:
            return i + 1
    return None
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `poetry run pytest tests/test_build_sequence_dataset.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/01_data_engineering/build_sequence_dataset.py tests/test_build_sequence_dataset.py
git commit -m "feat(sequence): frame_state + résolution opponent (build_sequence_dataset)"
```

---

### Task 2: `build_sequence` — une game → (seq, mask) (TDD)

**Files:**
- Modify: `src/01_data_engineering/build_sequence_dataset.py`
- Test: `tests/test_build_sequence_dataset.py`

**Interfaces:**
- Consumes: `frame_state`, `participant_pid`, `opponent_pid` (Task 1), `rl._frames_by_minute`.
- Produces: `build_sequence(match, timeline, target_puuid) -> (np.ndarray[40,20] float32, np.ndarray[40] bool) | None`.

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `tests/test_build_sequence_dataset.py` :

```python
def _fake_timeline(n_minutes=3):
    frames = []
    for minute in range(n_minutes):
        pf = {}
        for pid in range(1, 11):
            pf[str(pid)] = {
                "position": {"x": 1000 * pid + minute, "y": 2000 * pid},
                "totalGold": 500 * pid + 100 * minute,
                "currentGold": 50 * pid,
                "xp": 100 * pid * minute if minute else 0,
                "level": 1 + minute,
                "minionsKilled": 5 * minute,
                "jungleMinionsKilled": minute,
            }
        frames.append({"timestamp": minute * 60000, "participantFrames": pf})
    return {"info": {"frames": frames}}


def test_build_sequence_shapes_and_mask():
    m = _fake_match(); t = _fake_timeline(3)
    out = bsd.build_sequence(m, t, "p_BOTTOM_100")
    assert out is not None
    seq, mask = out
    assert seq.shape == (40, 20) and seq.dtype == np.float32
    assert mask.shape == (40,) and mask.dtype == bool
    assert mask.sum() == 3              # 3 minutes valides
    assert mask[0] and mask[1] and mask[2] and not mask[3]


def test_build_sequence_values_minute1():
    m = _fake_match(); t = _fake_timeline(3)
    seq, mask = bsd.build_sequence(m, t, "p_BOTTOM_100")
    # pid1 @min1 : x=1001,y=2000,gold=600,cur=50,xp=100,lvl=2,cs=5,jg=1
    self_state = [1001 / 14800, 2000 / 14800, 600.0, 50.0, 100.0, 2.0, 5.0, 1.0]
    # opp pid6 @min1 : x=6001,y=12000,gold=3100,cur=300,xp=600,lvl=2,cs=5,jg=1
    opp_state = [6001 / 14800, 12000 / 14800, 3100.0, 300.0, 600.0, 2.0, 5.0, 1.0]
    diffs = [600.0 - 3100.0,                 # gold diff
             (5.0 + 1.0) - (5.0 + 1.0),      # cs diff
             100.0 - 600.0,                  # xp diff
             2.0 - 2.0]                      # level diff
    expected = self_state + opp_state + diffs
    np.testing.assert_allclose(seq[1], expected, rtol=1e-5)


def test_build_sequence_none_if_no_opponent_role():
    # match sans adversaire BOTTOM : opponent_pid None -> None
    puuids = ["p_bottom_100"] + [f"p_{r}_{t}" for t in (100, 200)
              for r in ("UTILITY", "JUNGLE", "MIDDLE", "TOP")]
    parts = [{"teamId": 100, "teamPosition": "BOTTOM", "championName": "a"}] + \
            [{"teamId": t, "teamPosition": r, "championName": f"{r}_{t}"}
             for t in (100, 200) for r in ("UTILITY", "JUNGLE", "MIDDLE", "TOP")]
    m = {"metadata": {"participants": puuids}, "info": {"participants": parts}}
    assert bsd.build_sequence(m, _fake_timeline(2), "p_bottom_100") is None
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_build_sequence_dataset.py::test_build_sequence_shapes_and_mask -v`
Expected: FAIL — `module 'build_sequence_dataset' has no attribute 'build_sequence'`.

- [ ] **Step 3: Implémenter**

Ajouter à `src/01_data_engineering/build_sequence_dataset.py` :

```python
def _diffs(self_state: list[float], opp_state: list[float]) -> list[float]:
    """4 diffs relatives : gold, cs, xp, level (signaux de lane)."""
    return [
        self_state[2] - opp_state[2],                              # totalGold
        (self_state[6] + self_state[7]) - (opp_state[6] + opp_state[7]),  # cs
        self_state[4] - opp_state[4],                              # xp
        self_state[5] - opp_state[5],                              # level
    ]


def build_sequence(match: dict, timeline: dict,
                   target_puuid: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Une game -> (seq[40,20] float32, mask[40] bool). None si pas d'opponent ou 0 frame."""
    pid = participant_pid(match, target_puuid)
    opp = opponent_pid(match, target_puuid)
    if opp is None:
        return None
    my_fr = rl._frames_by_minute(timeline, pid)      # {minute_int: participantFrame}
    opp_fr = rl._frames_by_minute(timeline, opp)
    seq = np.zeros((MAX_LEN, 20), dtype=np.float32)
    mask = np.zeros(MAX_LEN, dtype=bool)
    for minute, pf in my_fr.items():
        if minute >= MAX_LEN:
            continue
        self_s = frame_state(pf)
        opp_s = frame_state(opp_fr.get(minute, {}))  # frame adverse manquante -> zeros
        seq[minute] = self_s + opp_s + _diffs(self_s, opp_s)
        mask[minute] = True
    if mask.sum() == 0:
        return None
    return seq, mask
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `poetry run pytest tests/test_build_sequence_dataset.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/01_data_engineering/build_sequence_dataset.py tests/test_build_sequence_dataset.py
git commit -m "feat(sequence): build_sequence (game -> seq[40,20] + mask, matchup)"
```

---

### Task 3: `main()` — build du npz complet

**Files:**
- Modify: `src/01_data_engineering/build_sequence_dataset.py`
- Test: `tests/test_build_sequence_dataset.py` (smoke sur 1 game réelle si raw dispo)

**Interfaces:**
- Consumes: `build_sequence`, `build_dataset.adc_puuids`, `build_dataset.build_rank_map`, `build_dataset._load_raw`, `build_dataset.HIGH_ELO`.
- Produces: `data/04_dataset/adc_sequence_dataset.npz` avec arrays `sequences[N,40,20]`, `mask[N,40]`, `label_highelo[N]`, `rank[N]` (str), `puuid[N]` (str), `match_id[N]` (str), `champion[N]` (str). Fonction `champion_of(match, puuid) -> str`.

- [ ] **Step 1: Écrire le test (champion_of + structure npz sur un fake complet via monkeypatch)**

Ajouter à `tests/test_build_sequence_dataset.py` :

```python
def test_champion_of():
    m = _fake_match()
    assert bsd.champion_of(m, "p_BOTTOM_100") == "BOTTOM_100"
    assert bsd.champion_of(m, "p_BOTTOM_200") == "BOTTOM_200"


def test_main_writes_npz(tmp_path, monkeypatch):
    # redirige DATASET_DIR vers tmp_path ; mock build_rank_map + _load_raw + adc_puuids
    monkeypatch.setattr(bsd, "DATASET_DIR", tmp_path)
    m = _fake_match(); t = _fake_timeline(3)
    monkeypatch.setattr(build_dataset, "build_rank_map",
                        lambda: ({"EUW1_1": "challenger"}, 0))
    monkeypatch.setattr(build_dataset, "_load_raw", lambda mid: (m, t))
    monkeypatch.setattr(build_dataset, "adc_puuids", lambda match: ["p_BOTTOM_100"])
    rc = bsd.main()
    assert rc == 0
    import numpy as np
    d = np.load(tmp_path / "adc_sequence_dataset.npz", allow_pickle=True)
    assert d["sequences"].shape == (1, 40, 20)
    assert d["mask"].sum() == 3
    assert list(d["rank"]) == ["challenger"]
    assert list(d["label_highelo"]) == [1]
    assert list(d["match_id"]) == ["EUW1_1"]
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_build_sequence_dataset.py::test_main_writes_npz -v`
Expected: FAIL — `module 'build_sequence_dataset' has no attribute 'main'` (ou `champion_of`).

- [ ] **Step 3: Implémenter**

Ajouter à `src/01_data_engineering/build_sequence_dataset.py` :

```python
DATASET_DIR = rl.DATA / "04_dataset"


def champion_of(match: dict, puuid: str) -> str:
    pidx = match["metadata"]["participants"].index(puuid)
    return match["info"]["participants"][pidx].get("championName") or "unknown"


def main() -> int:
    rank_of, multi = build_dataset.build_rank_map()
    print(f"  {len(rank_of)} games référentiel distinctes ({multi} multi-rang)")
    seqs, masks, labels, ranks, puuids, mids, champs = [], [], [], [], [], [], []
    raw_miss = 0
    for mid, rank in rank_of.items():
        raw = build_dataset._load_raw(mid)
        if not raw:
            raw_miss += 1
            continue
        match, timeline = raw
        for puuid in build_dataset.adc_puuids(match):
            out = build_sequence(match, timeline, puuid)
            if out is None:
                continue
            seq, mask = out
            seqs.append(seq)
            masks.append(mask)
            ranks.append(rank)
            puuids.append(puuid)
            mids.append(mid)
            champs.append(champion_of(match, puuid))
            labels.append(1 if rank in build_dataset.HIGH_ELO else 0)
    if raw_miss:
        print(f"  ⚠ {raw_miss} games sans raw lisible -> ignorées")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATASET_DIR / "adc_sequence_dataset.npz",
        sequences=np.stack(seqs).astype(np.float32),
        mask=np.stack(masks).astype(bool),
        label_highelo=np.array(labels, dtype=np.int64),
        rank=np.array(ranks, dtype=object),
        puuid=np.array(puuids, dtype=object),
        match_id=np.array(mids, dtype=object),
        champion=np.array(champs, dtype=object),
    )
    print(f"\n✓ {len(seqs)} séquences ADC -> {DATASET_DIR}/adc_sequence_dataset.npz")
    print(f"  high_elo (GM+Chall=1) : {int(np.sum(labels))} / {len(labels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `poetry run pytest tests/test_build_sequence_dataset.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Lancer sur les vraies données (smoke)**

Run: `poetry run python3 src/01_data_engineering/build_sequence_dataset.py`
Expected: affiche `✓ N séquences ADC -> .../adc_sequence_dataset.npz` avec N cohérent (~7000-8000). Vérifier : `poetry run python3 -c "import numpy as np; d=np.load('data/04_dataset/adc_sequence_dataset.npz', allow_pickle=True); print(d['sequences'].shape, d['mask'].sum(1).mean())"` → `(N,40,20)` et durée moyenne ~25-32 min.

- [ ] **Step 6: Commit**

```bash
git add src/01_data_engineering/build_sequence_dataset.py tests/test_build_sequence_dataset.py
git commit -m "feat(sequence): main() build adc_sequence_dataset.npz depuis le raw"
```

---

### Task 4: `SequenceEncoder` — encodeur transformer (TDD)

**Files:**
- Create: `src/02_data_science/sequence_model.py`
- Test: `tests/test_sequence_model.py`

**Interfaces:**
- Produces: `SequenceEncoder(d_in=20, d_model=64, nhead=4, n_layers=4, ff=128, dropout=0.1, max_len=40)`, `forward(x: Tensor[B,T,d_in], mask: Tensor[B,T] bool) -> Tensor[B,T,d_model]`. `get_device()`.

- [ ] **Step 1: Écrire le test qui échoue**

`tests/test_sequence_model.py` :

```python
"""Tests sequence_model : encoder, masked-mean-pool, reconstruct."""
import importlib.util, sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC / "02_data_science"))
_spec = importlib.util.spec_from_file_location(
    "sequence_model", _SRC / "02_data_science" / "sequence_model.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)

import torch

def test_encoder_forward_shapes():
    torch.manual_seed(0)
    enc = sm.SequenceEncoder(d_in=20, d_model=64, max_len=40)
    x = torch.randn(4, 40, 20)
    mask = torch.ones(4, 40, dtype=torch.bool)
    mask[:, 30:] = False                      # 30 minutes valides
    h = enc(x, mask)
    assert h.shape == (4, 40, 64)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_sequence_model.py::test_encoder_forward_shapes -v`
Expected: FAIL — module non trouvé / pas de `SequenceEncoder`.

- [ ] **Step 3: Implémenter**

`src/02_data_science/sequence_model.py` :

```python
#!/usr/bin/env python3
"""
02_data_science — modules PyTorch purs pour le transformer séquentiel (cf. spec
2026-07-18). Pas de HF Transformers : tout est écrit à la main, lisible, pédagogique.

- SequenceEncoder : projection 20->d_model + positional embedding appris + N couches
  TransformerEncoderLayer. src_key_padding_mask ignore les minutes paddées.
- ClassifierHead : masked-mean-pool sur les frames valides -> logit binaire.
  (Tradeoff : le pool est une agrégation -> ablation CLS/attention-pool si null, cf. spec.)
- ReconstructHead : projection d_model -> d_in pour le SSL mask-and-reconstruct.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SequenceEncoder(nn.Module):
    def __init__(self, d_in: int = 20, d_model: int = 64, nhead: int = 4,
                 n_layers: int = 4, ff: int = 128, dropout: float = 0.1,
                 max_len: int = 40):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x [B,T,d_in], mask [B,T] bool (True = frame valide)
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.proj(x) + self.pos(positions)
        h = self.enc(h, src_key_padding_mask=~mask)   # True at pad
        return h                                        # [B,T,d_model]
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `poetry run pytest tests/test_sequence_model.py::test_encoder_forward_shapes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/02_data_science/sequence_model.py tests/test_sequence_model.py
git commit -m "feat(sequence): SequenceEncoder (transformer à la main, masking)"
```

---

### Task 5: `ClassifierHead` + `masked_mean` + `SequenceClassifier` (TDD)

**Files:**
- Modify: `src/02_data_science/sequence_model.py`
- Test: `tests/test_sequence_model.py`

**Interfaces:**
- Produces: `masked_mean(h, mask) -> Tensor[B,d_model]`, `ClassifierHead.forward(h, mask) -> Tensor[B]`, `SequenceClassifier.forward(x, mask) -> Tensor[B]`.

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `tests/test_sequence_model.py` :

```python
def test_masked_mean_ignores_pad():
    torch.manual_seed(0)
    h = torch.randn(2, 40, 64)
    m_full = torch.ones(2, 40, dtype=torch.bool)
    m_pad = m_full.clone(); m_pad[:, 30:] = False       # mêmes 30 frames valides
    # pool sur 40 frames valides == pool sur 30 (les 10 paddées à 0 contribuent 0)
    a = sm.masked_mean(h, m_full)
    b = sm.masked_mean(h, m_pad)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_classifier_head_shape():
    torch.manual_seed(0)
    clf = sm.SequenceClassifier(d_in=20, d_model=64, max_len=40)
    x = torch.randn(8, 40, 20)
    mask = torch.ones(8, 40, dtype=torch.bool); mask[:, 25:] = False
    logits = clf(x, mask)
    assert logits.shape == (8,)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_sequence_model.py::test_masked_mean_ignores_pad -v`
Expected: FAIL — pas de `masked_mean`.

- [ ] **Step 3: Implémenter**

Ajouter à `src/02_data_science/sequence_model.py` :

```python
def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Moyenne sur les frames valides. h [B,T,d], mask [B,T] bool."""
    m = mask.unsqueeze(-1).float()                     # [B,T,1]
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)      # [B,d]


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = masked_mean(h, mask)
        return self.fc(pooled).squeeze(-1)              # [B]


class SequenceClassifier(nn.Module):
    def __init__(self, d_in: int = 20, d_model: int = 64, nhead: int = 4,
                 n_layers: int = 4, ff: int = 128, dropout: float = 0.1, max_len: int = 40):
        super().__init__()
        self.encoder = SequenceEncoder(d_in, d_model, nhead, n_layers, ff, dropout, max_len)
        self.head = ClassifierHead(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, mask), mask)

    def embed(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean(self.encoder(x, mask), mask)  # [B,d_model]
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `poetry run pytest tests/test_sequence_model.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/02_data_science/sequence_model.py tests/test_sequence_model.py
git commit -m "feat(sequence): ClassifierHead + masked_mean + SequenceClassifier"
```

---

### Task 6: `ReconstructHead` (TDD)

**Files:**
- Modify: `src/02_data_science/sequence_model.py`
- Test: `tests/test_sequence_model.py`

**Interfaces:**
- Produces: `ReconstructHead.forward(h: [B,T,d_model]) -> [B,T,d_in]`.

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `tests/test_sequence_model.py` :

```python
def test_reconstruct_head_shape():
    rh = sm.ReconstructHead(d_model=64, d_in=20)
    h = torch.randn(4, 40, 64)
    out = rh(h)
    assert out.shape == (4, 40, 20)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_sequence_model.py::test_reconstruct_head_shape -v`
Expected: FAIL — pas de `ReconstructHead`.

- [ ] **Step 3: Implémenter**

Ajouter à `src/02_data_science/sequence_model.py` :

```python
class ReconstructHead(nn.Module):
    """Tête SSL : reconstruit le state 20-d des frames (utilisée sur les frames masquées)."""
    def __init__(self, d_model: int = 64, d_in: int = 20):
        super().__init__()
        self.fc = nn.Linear(d_model, d_in)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)                               # [B,T,d_in]
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `poetry run pytest tests/test_sequence_model.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/02_data_science/sequence_model.py tests/test_sequence_model.py
git commit -m "feat(sequence): ReconstructHead pour le SSL mask-and-reconstruct"
```

---

### Task 7: `sequence_data` — labels, folds purgés, purge miroir, standardisation (TDD)

**Files:**
- Create: `src/02_data_science/sequence_data.py`
- Test: `tests/test_sequence_data.py`

**Interfaces:**
- Consumes: `adc_sequence_dataset.npz` (Task 3).
- Produces:
  - `load_dataset(path) -> dict` (arrays).
  - `task_subset(data, task) -> (idx: np.ndarray[int], y: np.ndarray[int])` — `task ∈ {"high_elo","dia_chall"}`.
  - `player_folds(puuids, y, n_splits=5, seed=42) -> list[(train_idx, val_idx)]` sur l'espace des rows, joueur-groupé + stratifié.
  - `mirror_purge(train_idx, val_puuids, match_ids, puuids) -> np.ndarray[int]` (train_idx filtré).
  - `standardize_fit(sequences, mask, train_idx) -> (mean[20], std[20])`.
  - `standardize_apply(sequences, mean, std) -> np.ndarray` (même shape).

- [ ] **Step 1: Écrire le test qui échoue**

`tests/test_sequence_data.py` :

```python
"""Tests sequence_data : labels, folds joueur-groupés, purge miroir, standardisation."""
import importlib.util, sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC / "02_data_science"))
import numpy as np
_spec = importlib.util.spec_from_file_location(
    "sequence_data", _SRC / "02_data_science" / "sequence_data.py")
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)


def _data(n=20, seed=0):
    rng = np.random.RandomState(seed)
    return {
        "sequences": rng.randn(n, 40, 20).astype(np.float32),
        "mask": np.ones((n, 40), dtype=bool),
        "label_highelo": rng.randint(0, 2, n),
        "rank": np.array(rng.choice(["diamond", "master", "grandmaster", "challenger"], n),
                         dtype=object),
        "puuid": np.array([f"p{i % 5}" for i in range(n)], dtype=object),
        "match_id": np.array([f"m{i // 2}" for i in range(n)], dtype=object),  # 2 rows/match
        "champion": np.array(["Zeri"] * n, dtype=object),
    }


def test_task_subset_highelo():
    d = _data()
    idx, y = sd.task_subset(d, "high_elo")
    assert len(idx) == len(d["rank"])
    assert set(y.tolist()) <= {0, 1}


def test_task_subset_dia_chall_filters():
    d = _data()
    idx, y = sd.task_subset(d, "dia_chall")
    # ne garde que diamond/challenger
    kept = d["rank"][idx]
    assert set(kept.tolist()) <= {"diamond", "challenger"}
    # label = 1 si challenger
    for i, yy in zip(idx, y):
        assert yy == (1 if d["rank"][i] == "challenger" else 0)


def test_player_folds_no_overlap():
    d = _data(n=20)
    idx, y = sd.task_subset(d, "high_elo")
    folds = sd.player_folds(d["puuid"][idx], y, n_splits=5, seed=42)
    seen = set()
    for tr, va in folds:
        trp = set(d["puuid"][idx][tr]); vap = set(d["puuid"][idx][va])
        assert trp.isdisjoint(vap)          # aucun joueur à la fois train et val
        assert vap.isdisjoint(seen)        # chaque joueur vu une seule fois en val
        seen |= vap


def test_mirror_purge_drops_opponent_of_val():
    # 2 rows par match ; si l'opponent puuid est en val, la row train du même match est purgée
    puuids = np.array(["a", "b", "a", "b"], dtype=object)
    match_ids = np.array(["m0", "m0", "m1", "m1"], dtype=object)
    train_idx = np.array([0, 2])           # row 0 (a,m0), row 2 (a,m1)
    val_puuids = {"b"}                      # b est en val -> row m0 (a) et m1 (a) sont miroir de b
    kept = sd.mirror_purge(train_idx, val_puuids, match_ids, puuids)
    assert len(kept) == 0                   # les 2 rows train partagent leur match avec b(val)


def test_standardize_fit_train_only():
    d = _data(n=20)
    train_idx = np.arange(15)
    mean, std = sd.standardize_fit(d["sequences"], d["mask"], train_idx)
    assert mean.shape == (20,) and std.shape == (20,)
    # sur le train, z-score donne mean≈0 std≈1 (frames valides toutes True ici)
    z = sd.standardize_apply(d["sequences"][train_idx], mean, std)
    assert abs(z.mean(axis=(0, 1))) .max() < 1e-5
    assert abs(z.std(axis=(0, 1)) - 1.0).max() < 1e-5


def test_standardize_uses_train_stats_on_val():
    d = _data(n=20, seed=1)
    train_idx = np.arange(15); val_idx = np.arange(15, 20)
    mean, std = sd.standardize_fit(d["sequences"], d["mask"], train_idx)
    z_val = sd.standardize_apply(d["sequences"][val_idx], mean, std)
    # val n'a PAS mean 0 (stats du train appliquées) -> on vérifie juste la shape + pas de NaN
    assert z_val.shape == (5, 40, 20)
    assert np.isfinite(z_val).all()
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_sequence_data.py -v`
Expected: FAIL — module non trouvé.

- [ ] **Step 3: Implémenter**

`src/02_data_science/sequence_data.py` :

```python
#!/usr/bin/env python3
"""
02_data_science — couche donnée/CV partagée par train_sequence_model et
pretrain_sequence_model (DRY). Chargement npz, labels par tâche, folds joueur-groupés,
purge miroir, standardisation per-feature (z-score sur le train du fold).

⚠ Standardisation non négociable (spec 2026-07-18 décision 3) : totalGold ~15000 dans la
même projection que position [0,1] rend l'entraînement instable ; sans standardisation un
null est un artefact d'optimisation, pas une réponse à la question de recherche.
"""
from __future__ import annotations

import sys
from pathlib import Path

_CORE = Path(__file__).resolve().parent.parent / "core"
sys.path.insert(0, str(_CORE))
import numpy as np
import riotlib as rl

DATASET = rl.DATA / "04_dataset" / "adc_sequence_dataset.npz"
HIGH_ELO = {"grandmaster", "challenger"}
DIA_CHALL = {"diamond", "challenger"}
TASKS = ("high_elo", "dia_chall")


def load_dataset(path: Path = DATASET) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def task_subset(data: dict, task: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (idx rows, y binaire). high_elo : toutes rows label=GM/C. dia_chall : filter
    diamond+challenger, label=challenger."""
    ranks = data["rank"]
    if task == "high_elo":
        # perso (rank None) exclu -> on garde les rows dont rank est défini
        idx = np.array([i for i, r in enumerate(ranks) if r is not None])
        y = np.array([1 if r in HIGH_ELO else 0 for r in ranks[idx]], dtype=np.int64)
    elif task == "dia_chall":
        idx = np.array([i for i, r in enumerate(ranks) if r in DIA_CHALL])
        y = np.array([1 if ranks[i] == "challenger" else 0 for i in idx], dtype=np.int64)
    else:
        raise ValueError(f"task inconnue: {task}")
    return idx, y


def player_folds(puuids: np.ndarray, y: np.ndarray,
                 n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold sur les joueurs (chaque joueur -> 1 fold), expandu en indices de rows.
    Stratification sur le label du joueur (mode de ses rows). Aucun joueur à cheval train/val."""
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    df = pd.DataFrame({"puuid": puuids, "y": y})
    player_label = df.groupby("puuid")["y"].agg(lambda s: s.mode().iloc[0]).reset_index()
    player_label.columns = ["puuid", "plabel"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    player_arr = player_label["puuid"].to_numpy()
    plabel_arr = player_label["plabel"].to_numpy()
    for p_tr, p_va in cv.split(player_arr, plabel_arr):
        tr_p = set(player_arr[p_tr]); va_p = set(player_arr[p_va])
        tr_idx = np.array([i for i, p in enumerate(puuids) if p in tr_p])
        va_idx = np.array([i for i, p in enumerate(puuids) if p in va_p])
        folds.append((tr_idx, va_idx))
    return folds


def mirror_purge(train_idx: np.ndarray, val_puuids: set,
                 match_ids: np.ndarray, puuids: np.ndarray) -> np.ndarray:
    """Drop les rows de train dont l'ADC adverse (autre puuid du même match_id) est un
    joueur de val (fuite par games miroir : les 2 ADC d'une game sont des lignes en miroir).
    O(N) : pré-indexe match_id -> puuids une fois, puis lookup par row de train (vs O(n²)
    naïf qui coûterait ~500M itérations Python sur 5 folds × 2 tâches)."""
    val_puuids = set(val_puuids)
    match_to_puuids: dict[object, list] = {}
    for mid, p in zip(match_ids, puuids):
        match_to_puuids.setdefault(mid, []).append(p)
    keep = []
    for i in train_idx:
        mid = match_ids[i]; me = puuids[i]
        opps = match_to_puuids.get(mid, [])
        if any(o != me and o in val_puuids for o in opps):
            continue
        keep.append(i)
    return np.array(keep, dtype=train_idx.dtype)


def standardize_fit(sequences: np.ndarray, mask: np.ndarray,
                    train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mean/std par feature (20) sur les frames valides des rows de train."""
    X = sequences[train_idx][mask[train_idx]]      # [n_valid, 20]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0                          # garde-fou feature constante
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(sequences: np.ndarray, mean: np.ndarray,
                      std: np.ndarray) -> np.ndarray:
    return ((sequences - mean) / std).astype(np.float32)
```

- [ ] **Step 4: Vérifier que ça passe**

Run: `poetry run pytest tests/test_sequence_data.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/02_data_science/sequence_data.py tests/test_sequence_data.py
git commit -m "feat(sequence): sequence_data (labels, folds joueur-groupés, purge miroir O(N), standardisation)"
```

---

### Task 8: `train_sequence_model` — étape 1 supervisée + baselines + metrics (TDD-light)

**Files:**
- Create: `src/02_data_science/train_sequence_model.py`

**Interfaces:**
- Consumes: `sequence_model` (Task 4-6), `sequence_data` (Task 7), `adc_dataset.parquet` (baselines).
- Produces: `data/05_model/sequence_supervised.pt` (meilleur modèle high_elo), `data/05_model/sequence_metrics.json` (AUC par fold mean±std par tâche, baselines tabulaire+MLP, params, seed, device).

- [ ] **Step 1: Écrire un test d'intégration minimal (boucle tourne sur données factices)**

`tests/test_train_sequence_model.py` :

```python
"""Smoke : train_sequence_model tourne sur un mini-dataset synthétique."""
import importlib.util, sys, json, tempfile
from pathlib import Path
import numpy as np
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC / "02_data_science"))
_spec = importlib.util.spec_from_file_location(
    "train_sequence_model", _SRC / "02_data_science" / "train_sequence_model.py")
trm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trm)


def _mini(n=40, seed=0):
    rng = np.random.RandomState(seed)
    puuids = np.array([f"p{i // 4}" for i in range(n)], dtype=object)   # 4 games/joueur
    ranks = np.array(rng.choice(["diamond", "challenger"], n), dtype=object)
    y = (ranks == "challenger").astype(int)
    # signal : les challenger ont totalGold plus haut à la frame 10 (feature 2)
    seqs = rng.randn(n, 40, 20).astype(np.float32)
    seqs[y == 1, 10, 2] += 5.0
    return {
        "sequences": seqs, "mask": np.ones((n, 40), dtype=bool),
        "rank": ranks, "puuid": puuids,
        "match_id": np.array([f"m{i}" for i in range(n)], dtype=object),
        "champion": np.array(["Zeri"] * n, dtype=object),
    }


def test_train_returns_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(trm, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(trm.sd, "DATASET", tmp_path / "x.npz")
    np.savez(tmp_path / "x.npz", **_mini())
    metrics = trm.run(epochs=5, batch=8, seed=42, device_force="cpu")
    assert "tasks" in metrics and "high_elo" in metrics["tasks"]
    he = metrics["tasks"]["high_elo"]
    assert "auc_mean" in he and "auc_std" in he
    assert "baseline_tabular_auc" in he and "baseline_mlp_auc" in he
    assert (tmp_path / "sequence_supervised.pt").exists()
    assert (tmp_path / "sequence_metrics.json").exists()
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_train_sequence_model.py -v`
Expected: FAIL — module non trouvé.

- [ ] **Step 3: Implémenter**

`src/02_data_science/train_sequence_model.py` :

```python
#!/usr/bin/env python3
"""
02_data_science — étape 1 : transformer séquentiel supervisé (high_elo + dia_chall),
CV purgé (joueurs + miroir), comparé aux baselines tabulaire (xgb) et MLP sur les mêmes
folds. Écrit sequence_metrics.json + sequence_supervised.pt.

Verdict « séquence > agrégat ? » se lit sur dia_chall (master/GM null = bruit de label,
non interprétable — cf. spec 2026-07-18 §Pièges).

Usage : poetry run python3 src/02_data_science/train_sequence_model.py [--epochs N]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

_CORE = Path(__file__).resolve().parent.parent / "core"
sys.path.insert(0, str(_CORE))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # sequence_model, sequence_data
import numpy as np
import pandas as pd
import torch
import riotlib as rl
import sequence_model as sm
import sequence_data as sd

MODEL_DIR = rl.DATA / "05_model"
TABULAR = rl.DATA / "04_dataset" / "adc_dataset.parquet"
SEED = 42


def _baseline_tabular(task, folds, idx, y, data, seed=SEED):
    """Ensemble xgb+rf sur adc_dataset.parquet (features agrégées), MÊME fold partition
    joueur + purge miroir que le transformer. Donne à l'agrégat son meilleur coup (pas un
    xgb sous-tuné qui offrirait une victoire facile au transformer). NB : les 0.724/0.589 de
    CLAUDE.md viennent d'un autre protocole (ensemble 3-modèles, CV per-game non purgée) ->
    notre baseline est le comparatif propre, PAS une reproduction de ces chiffres."""
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(TABULAR)
    feat_cols = [c for c in df.columns if c not in
                 ("match_id", "puuid", "source", "rank", "champion", "win",
                  "patch", "game_ts", "rank_ord", "high_elo")]
    oof_xgb = np.full(len(idx), np.nan)
    oof_rf = np.full(len(idx), np.nan)
    for tr_i, va_i in folds:
        tr_puuids = set(data["puuid"][idx][tr_i]); va_puuids = set(data["puuid"][idx][va_i])
        tr_df = df[df["puuid"].isin(tr_puuids)
                   & ~df["match_id"].isin(set(data["match_id"][idx][va_i]))]
        va_df = df[df["puuid"].isin(va_puuids)]
        if task == "dia_chall":
            tr_df = tr_df[tr_df["rank"].isin(sd.DIA_CHALL)]
            va_df = va_df[va_df["rank"].isin(sd.DIA_CHALL)]
        yt = tr_df["rank"].isin(sd.HIGH_ELO if task == "high_elo" else {"challenger"}).astype(int)
        yv = va_df["rank"].isin(sd.HIGH_ELO if task == "high_elo" else {"challenger"}).astype(int)
        if len(set(yv)) < 2:
            continue
        mx = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                               reg_lambda=1.0, eval_metric="logloss", tree_method="hist",
                               random_state=seed)
        mr = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=4,
                                    max_features="sqrt", n_jobs=-1, random_state=seed)
        Xt = tr_df[feat_cols]; Xv = va_df[feat_cols]
        mx.fit(Xt, yt)
        mr.fit(Xt.fillna(Xt.median(numeric_only=True)), yt)
        p_x = mx.predict_proba(Xv)[:, 1]
        p_r = mr.predict_proba(Xv.fillna(Xv.median(numeric_only=True)))[:, 1]
        va_map = {(r["match_id"], r["puuid"]): (px, pr)
                  for (_, r), px, pr in zip(va_df.iterrows(), p_x, p_r)}
        for j in va_i:
            v = va_map.get((data["match_id"][idx[j]], data["puuid"][idx[j]]))
            if v is not None:
                oof_xgb[j], oof_rf[j] = v
    ens = np.where(np.isnan(oof_xgb) | np.isnan(oof_rf), np.nan, (oof_xgb + oof_rf) / 2.0)
    mask = ~np.isnan(ens)
    return float(roc_auc_score(y[mask], ens[mask])) if mask.sum() and len(set(y[mask])) > 1 else None


def _baseline_mlp(task, folds, idx, y, data, seed=SEED):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(TABULAR)
    feat_cols = [c for c in df.columns if c not in
                 ("match_id", "puuid", "source", "rank", "champion", "win",
                  "patch", "game_ts", "rank_ord", "high_elo")]
    oof = np.full(len(idx), np.nan)
    for tr_i, va_i in folds:
        tr_puuids = set(data["puuid"][idx][tr_i]); va_puuids = set(data["puuid"][idx][va_i])
        tr_df = df[df["puuid"].isin(tr_puuids)
                   & ~df["match_id"].isin(set(data["match_id"][idx][va_i]))]
        va_df = df[df["puuid"].isin(va_puuids)]
        if task == "dia_chall":
            tr_df = tr_df[tr_df["rank"].isin(sd.DIA_CHALL)]
            va_df = va_df[va_df["rank"].isin(sd.DIA_CHALL)]
        yt = tr_df["rank"].isin(sd.HIGH_ELO if task == "high_elo" else {"challenger"}).astype(int)
        yv = va_df["rank"].isin(sd.HIGH_ELO if task == "high_elo" else {"challenger"}).astype(int)
        if len(set(yv)) < 2:
            continue
        m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=80, random_state=seed)
        Xt = tr_df[feat_cols].fillna(0).values; Xv = va_df[feat_cols].fillna(0).values
        m.fit(Xt, yt)
        proba = m.predict_proba(Xv)[:, 1]
        key = lambda i: (data["match_id"][idx[i]], data["puuid"][idx[i]])
        va_map = {(r["match_id"], r["puuid"]): p for (_, r), p in zip(va_df.iterrows(), proba)}
        for j in va_i:
            oof[j] = va_map.get(key(j), np.nan)
    mask = ~np.isnan(oof)
    return float(roc_auc_score(y[mask], oof[mask])) if mask.sum() and len(set(y[mask])) > 1 else None


def _train_one_task(task, data, device, epochs, batch, seed):
    from sklearn.metrics import roc_auc_score
    import copy
    idx, y = sd.task_subset(data, task)
    if len(set(y)) < 2 or len(idx) < 50:
        return ({"auc_mean": None, "auc_std": None, "n_rows": int(len(idx)),
                 "reason": "trop peu de rows ou 1 classe"}, None)
    folds = sd.player_folds(data["puuid"][idx], y, n_splits=5, seed=seed)
    oof = np.full(len(idx), np.nan)
    best_state_global, best_auc_global = None, -1.0
    for fi, (tr_i, va_i) in enumerate(folds):
        val_puuids = set(data["puuid"][idx][va_i])
        tr_purged = sd.mirror_purge(tr_i, val_puuids, data["match_id"][idx], data["puuid"][idx])
        if len(tr_purged) == 0:
            continue
        mean, std = sd.standardize_fit(data["sequences"], data["mask"], idx[tr_purged])
        Xs = sd.standardize_apply(data["sequences"], mean, std)
        Xtr = torch.from_numpy(Xs[idx[tr_purged]]).to(device)
        Mtr = torch.from_numpy(data["mask"][idx[tr_purged]]).to(device)
        ytr = torch.from_numpy(y[tr_purged].astype(np.float32)).to(device)
        Xva = torch.from_numpy(Xs[idx[va_i]]).to(device)
        Mva = torch.from_numpy(data["mask"][idx[va_i]]).to(device)
        model = sm.SequenceClassifier().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_fold_auc, best_state, patience, bad = 0.0, None, 10, 0
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(Xtr))
            for b in range(0, len(perm), batch):
                bi = perm[b:b + batch]
                logits = model(Xtr[bi], Mtr[bi])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ytr[bi])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xva, Mva)).cpu().numpy()
            if len(set(y[va_i])) < 2:
                continue
            auc = roc_auc_score(y[va_i], pv)
            if auc > best_fold_auc:
                best_fold_auc = auc
                bad = 0                                  # patience reset sur amélioration
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        # OOF au MEILLEUR état (restaure best_state, pas le dernier)
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            oof[va_i] = torch.sigmoid(model(Xva, Mva)).cpu().numpy()
        if best_fold_auc > best_auc_global:
            best_auc_global = best_fold_auc
            best_state_global = best_state
    mask = ~np.isnan(oof)
    auc = float(roc_auc_score(y[mask], oof[mask])) if mask.sum() and len(set(y[mask])) > 1 else None
    per_fold = [float(roc_auc_score(y[v], oof[v])) for (_, v) in folds
                if not np.isnan(oof[v]).any() and len(set(y[v])) > 1]
    return ({
        "auc_mean": auc,
        "auc_std": float(np.std(per_fold)) if per_fold else None,
        "n_rows": int(len(idx)), "n_val_folds": len(per_fold),
        "baseline_tabular_auc": _baseline_tabular(task, folds, idx, y, data, seed),
        "baseline_mlp_auc": _baseline_mlp(task, folds, idx, y, data, seed),
        "best_fold_auc": float(best_auc_global),
    }, best_state_global)


def run(epochs=60, batch=64, seed=SEED, device_force=None) -> dict:
    data = sd.load_dataset()
    device = torch.device(device_force) if device_force else sm.get_device()
    print(f"  device={device} | {len(data['sequences'])} séquences")
    metrics = {"tasks": {}, "params": {"epochs": epochs, "batch": batch, "seed": seed,
                "d_model": 64, "n_layers": 4, "nhead": 4, "device": str(device)}}
    saved_state = None
    for task in sd.TASKS:
        print(f"\n=== tâche {task} ===")
        m, state = _train_one_task(task, data, device, epochs, batch, seed)
        metrics["tasks"][task] = m
        print(f"  séquence AUC={m.get('auc_mean')} (±{m.get('auc_std')})  "
              f"tabulaire={m.get('baseline_tabular_auc')}  mlp={m.get('baseline_mlp_auc')}")
        if task == "high_elo" and state is not None:
            saved_state = state                              # meilleur modèle high_elo (best fold)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if saved_state is not None:
        torch.save(saved_state, MODEL_DIR / "sequence_supervised.pt")
    (MODEL_DIR / "sequence_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n✓ {MODEL_DIR}/sequence_metrics.json")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    run(epochs=args.epochs, batch=args.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Vérifier que le smoke passe**

Run: `poetry run pytest tests/test_train_sequence_model.py -v`
Expected: PASS (le mini-dataset a un signal injecté → auc_mean > 0.5 ; baselines tournent).

- [ ] **Step 5: Lancer sur les vraies données**

Run: `poetry run python3 src/02_data_science/train_sequence_model.py --epochs 60`
Expected: affiche AUC séquence / tabulaire / MLP pour high_elo et dia_chall ; écrit `data/05_model/sequence_metrics.json` + `sequence_supervised.pt`. Note : les 0.724 dia_chall / 0.589 master-GM de CLAUDE.md viennent d'un AUTRE protocole (ensemble 3-modèles, CV per-game non purgée) — notre baseline tabulaire (xgb+rf sur folds purgés) N'A PAS à reproduire ces chiffres exacts ; c'est un *prior* (dia_chall attendu ~0.6-0.7, master/GM ~0.55-0.62), pas un oracle. Si la baseline tabulaire sort ~0.5 (aléatoire) sur dia_chall, ALORS seulement debugger le join (match_id, puuid) — sinon c'est juste un modèle/protocole différent.

- [ ] **Step 6: Commit**

```bash
git add src/02_data_science/train_sequence_model.py tests/test_train_sequence_model.py
git commit -m "feat(sequence): étape 1 supervisée + baselines tabulaire/MLP + metrics"
```

---

### Task 9: `pretrain_sequence_model` — étape 2 SSL + delta (TDD-light)

**Files:**
- Create: `src/02_data_science/pretrain_sequence_model.py`

**Interfaces:**
- Consumes: `sequence_model`, `sequence_data`, `sequence_metrics.json` (lit auc_supervised pour le delta).
- Produces: `data/05_model/sequence_encoder_pretrain.pt`, met à jour `sequence_metrics.json` avec `auc_supervised` / `auc_ssl` / `delta_ssl` + `embed_game`.

- [ ] **Step 1: Écrire le smoke test**

`tests/test_pretrain_sequence_model.py` :

```python
"""Smoke : pretrain SSL + finetune tourne sur mini-dataset synthétique."""
import importlib.util, sys
from pathlib import Path
import numpy as np
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC / "02_data_science"))
_spec = importlib.util.spec_from_file_location(
    "pretrain_sequence_model", _SRC / "02_data_science" / "pretrain_sequence_model.py")
ptm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ptm)


def _mini(n=40, seed=0):
    rng = np.random.RandomState(seed)
    puuids = np.array([f"p{i // 4}" for i in range(n)], dtype=object)
    ranks = np.array(rng.choice(["diamond", "challenger"], n), dtype=object)
    y = (ranks == "challenger").astype(int)
    seqs = rng.randn(n, 40, 20).astype(np.float32)
    seqs[y == 1, 10, 2] += 5.0
    return {"sequences": seqs, "mask": np.ones((n, 40), dtype=bool), "rank": ranks,
            "puuid": puuids, "match_id": np.array([f"m{i}" for i in range(n)], dtype=object),
            "champion": np.array(["Zeri"] * n, dtype=object)}


def test_pretrain_returns_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(ptm, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(ptm.sd, "DATASET", tmp_path / "x.npz")
    np.savez(tmp_path / "x.npz", **_mini())
    out = ptm.run(pretrain_epochs=5, finetune_epochs=5, batch=8, seed=42, device_force="cpu")
    assert "auc_supervised" in out and "auc_ssl" in out and "delta_ssl" in out
    assert (tmp_path / "sequence_encoder_pretrain.pt").exists()


def test_embed_game_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(ptm.sd, "DATASET", tmp_path / "x.npz")
    np.savez(tmp_path / "x.npz", **_mini())
    emb = ptm.embed_game(np.random.randn(40, 20).astype(np.float32),
                         np.ones(40, dtype=bool), device_force="cpu")
    assert emb.shape == (64,)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run pytest tests/test_pretrain_sequence_model.py -v`
Expected: FAIL — module non trouvé.

- [ ] **Step 3: Implémenter**

`src/02_data_science/pretrain_sequence_model.py` :

```python
#!/usr/bin/env python3
"""
02_data_science — étape 2 : pretrain self-supervised (mask-and-reconstruct) PURISTE
par-fold : pour chaque fold, on standardise train-only, on pretrain l'encodeur sur le
TRAIN du fold uniquement (joueurs de val jamais vus -> pas de fuite transductive), puis
on finetune le classifieur high_elo. Mêmes stats de standardisation au pretrain et au
finetune -> transfert non saboté. delta_ssl = AUC_ssl - AUC_supervisé (étape 1, même CV),
donc un delta propre (pas d'avantage transductif). NB : moins de données de pretrain par
fold (~train rows seulement) que si on préentraînait sur tout — choix délibéré pour la
propreté du comparatif.

⚠ Le prétexte MSE-reconstruct est FAIBLE sur signaux lisses (gold monotone, position
continue -> quasi-interpolation) : le modèle peut cartonner la reconstruction sans rien
apprendre de pertinent au rang. Un ≈0 delta_ssl n'est PAS un verdict sur le SSL en général,
juste sur ce prétexte (un prétexte prédictif = étape 3, cf. spec 2026-07-18 §Pièges).

Usage : poetry run python3 src/02_data_science/pretrain_sequence_model.py
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

_CORE = Path(__file__).resolve().parent.parent / "core"
sys.path.insert(0, str(_CORE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import riotlib as rl
import sequence_model as sm
import sequence_data as sd

MODEL_DIR = rl.DATA / "05_model"
SEED = 42
MASK_FRAC = 0.15


def _ssl_cv(data, device, pretrain_epochs, finetune_epochs, batch, seed):
    """SSL puriste par-fold : pour chaque fold, (1) standardise train-only, (2) pretrain
    mask-and-reconstruct sur le TRAIN du fold uniquement, (3) finetune classifieur avec
    l'encodeur pré-entraîné. Même échelle d'entrée au pretrain et au finetune (mêmes stats
    train-only) -> transfert non saboté. Aucune fuite : joueurs de val jamais vus au
    pretrain. delta_ssl est donc un signal propre (pas d'avantage transductif)."""
    from sklearn.metrics import roc_auc_score
    idx, y = sd.task_subset(data, "high_elo")
    folds = sd.player_folds(data["puuid"][idx], y, n_splits=5, seed=seed)
    oof = np.full(len(idx), np.nan)
    best_enc, best_auc = None, -1.0
    for tr_i, va_i in folds:
        val_puuids = set(data["puuid"][idx][va_i])
        tr_purged = sd.mirror_purge(tr_i, val_puuids, data["match_id"][idx], data["puuid"][idx])
        if len(tr_purged) == 0:
            continue
        mean, std = sd.standardize_fit(data["sequences"], data["mask"], idx[tr_purged])
        Xs = sd.standardize_apply(data["sequences"], mean, std)
        Xtr = torch.from_numpy(Xs[idx[tr_purged]]).to(device)
        Mtr = torch.from_numpy(data["mask"][idx[tr_purged]]).to(device)
        ytr = torch.from_numpy(y[tr_purged].astype(np.float32)).to(device)
        Xva = torch.from_numpy(Xs[idx[va_i]]).to(device)
        Mva = torch.from_numpy(data["mask"][idx[va_i]]).to(device)
        n = len(Xtr)
        # (1)+(2) pretrain SSL sur le TRAIN du fold
        enc = sm.SequenceEncoder().to(device)
        head = sm.ReconstructHead().to(device)
        opt_p = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                                  lr=3e-4, weight_decay=1e-2)
        for ep in range(pretrain_epochs):
            enc.train(); head.train()
            perm = torch.randperm(n)
            for b0 in range(0, n, batch):
                bi = perm[b0:b0 + batch]
                xb, mb = Xtr[bi], Mtr[bi]
                ssl_mask = (torch.rand_like(mb.float()) < MASK_FRAC) & mb   # 15% des valides masquées
                h = enc(xb, mb & ~ssl_mask)                                 # encode sans les masquées
                pred = head(h)
                loss = ((pred - xb) ** 2 * ssl_mask.unsqueeze(-1)).sum() / \
                       ssl_mask.float().sum().clamp(min=1.0) / 20.0
                opt_p.zero_grad(); loss.backward(); opt_p.step()
        # (3) finetune : encodeur pré-entraîné + tête fraîche
        clf = sm.SequenceClassifier().to(device)
        clf.encoder.load_state_dict(enc.state_dict())
        opt_f = torch.optim.AdamW(clf.parameters(), lr=3e-4, weight_decay=1e-2)
        for ep in range(finetune_epochs):
            clf.train()
            perm = torch.randperm(n)
            for b0 in range(0, n, batch):
                bi = perm[b0:b0 + batch]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    clf(Xtr[bi], Mtr[bi]), ytr[bi])
                opt_f.zero_grad(); loss.backward(); opt_f.step()
        clf.eval()
        with torch.no_grad():
            oof[va_i] = torch.sigmoid(clf(Xva, Mva)).cpu().numpy()
        if len(set(y[va_i])) > 1:
            a = roc_auc_score(y[va_i], oof[va_i])
            if a > best_auc:
                best_auc = a
                best_enc = {k: v.detach().cpu().clone()
                            for k, v in clf.encoder.state_dict().items()}
    mask = ~np.isnan(oof)
    auc = float(roc_auc_score(y[mask], oof[mask])) if mask.sum() and len(set(y[mask])) > 1 else None
    return auc, best_enc


def run(pretrain_epochs=30, finetune_epochs=40, batch=64, seed=SEED, device_force=None) -> dict:
    data = sd.load_dataset()
    device = torch.device(device_force) if device_force else sm.get_device()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    auc_ssl, best_enc = _ssl_cv(data, device, pretrain_epochs, finetune_epochs, batch, seed)
    # lit l'AUC supervisé étape 1 (même protocole CV purgé, sans pretrain)
    metrics_path = MODEL_DIR / "sequence_metrics.json"
    prev = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    auc_sup = prev.get("tasks", {}).get("high_elo", {}).get("auc_mean")
    out = {"auc_supervised": auc_sup, "auc_ssl": auc_ssl,
           "delta_ssl": (auc_ssl - auc_sup) if (auc_ssl is not None and auc_sup is not None) else None,
           "pretext": "mask-and-reconstruct (MSE, 15% mask) — prétexte faible sur signaux lisses ; "
                      "≈0 delta n'est pas un verdict sur le SSL en général (cf. spec §Pièges). "
                      "Pretrain par-fold train-only : pas d'avantage transductif, delta propre.",
           "params": {"pretrain_epochs": pretrain_epochs, "finetune_epochs": finetune_epochs,
                      "seed": seed, "device": str(device)}}
    prev["ssl"] = out
    metrics_path.write_text(json.dumps(prev, indent=2))
    if best_enc is not None:
        torch.save(best_enc, MODEL_DIR / "sequence_encoder_pretrain.pt")
    print(f"  AUC supervisé={auc_sup}  AUC ssl={auc_ssl}  delta={out['delta_ssl']}")
    print(f"✓ {MODEL_DIR}/sequence_encoder_pretrain.pt + delta dans sequence_metrics.json")
    return out


def embed_game(seq: np.ndarray, mask: np.ndarray, device_force=None) -> np.ndarray:
    """Vecteur d'embedding 64-d d'une game (pour inspection : projection 2D colorée rang)."""
    device = torch.device(device_force) if device_force else sm.get_device()
    clf = sm.SequenceClassifier().to(device)
    clf.eval()
    with torch.no_grad():
        x = torch.from_numpy(seq[None].astype(np.float32)).to(device)
        m = torch.from_numpy(mask[None].astype(bool)).to(device)
        return clf.embed(x, m).cpu().numpy()[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    args = ap.parse_args()
    run(pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Vérifier que le smoke passe**

Run: `poetry run pytest tests/test_pretrain_sequence_model.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lancer sur les vraies données**

Run: `poetry run python3 src/02_data_science/pretrain_sequence_model.py`
Expected: affiche `AUC supervisé=X AUC ssl=Y delta=Z` ; met à jour `sequence_metrics.json` (bloc `ssl`) ; écrit `sequence_encoder_pretrain.pt`.

- [ ] **Step 6: Commit**

```bash
git add src/02_data_science/pretrain_sequence_model.py tests/test_pretrain_sequence_model.py
git commit -m "feat(sequence): étape 2 SSL mask-and-reconstruct + finetune + delta"
```

---

### Task 10: Vérification de bout en bout + lecture des résultats

**Files:**
- Modify: `CLAUDE.md` (section « État d'avancement » — ajouter une ligne recherche séquentielle)

- [ ] **Step 1: Tout rejouer bout en bout**

```bash
poetry run pytest tests/test_build_sequence_dataset.py tests/test_sequence_model.py \
  tests/test_sequence_data.py tests/test_train_sequence_model.py \
  tests/test_pretrain_sequence_model.py -v
poetry run python3 src/01_data_engineering/build_sequence_dataset.py
poetry run python3 src/02_data_science/train_sequence_model.py --epochs 60
poetry run python3 src/02_data_science/pretrain_sequence_model.py
```
Expected: tous tests verts ; `sequence_metrics.json` contient `tasks.high_elo`, `tasks.dia_chall` (avec `auc_mean`, `baseline_tabular_auc`, `baseline_mlp_auc`), et `ssl` (avec `auc_supervised`, `auc_ssl`, `delta_ssl`).

- [ ] **Step 2: Lire et interpréter les métriques**

Run: `poetry run python3 -c "import json; m=json.load(open('data/05_model/sequence_metrics.json')); print(json.dumps(m, indent=2))"`

Interprétation (à écrire dans un commentaire de commit / note, pas dans le code) :
- **dia_chall** : si `auc_mean(seq) > baseline_tabular_auc` → la séquence capte un signal que l'agrégat rate (thèse séquence renforcée). Si ≈ → null v1 non concluant (rappeler §Pièges : v1 = frames que l'agrégat résume déjà ; signal fort vit dans les events, étape 2 d'enrichissement).
- **master/GM** : null attendu et non interprétable (bruit de label). Ne pas conclure.
- **ssl.delta_ssl** : pretrain par-fold train-only → delta propre (pas d'avantage
  transductif, pas de fuite val). > 0 → ce prétexte aide vraiment à N=8k (signal réel).
  ≈0 → ce prétexte est faible (interpolation sur signaux lisses), pas un verdict sur le
  SSL en général ; un prétexte prédictif (future-event) reste à tester en étape 3.

- [ ] **Step 3: Mettre à jour CLAUDE.md**

Dans la section « État d'avancement », après le bloc « Régression LP », ajouter :

```markdown
- **Recherche — transformer séquentiel + SSL** 🚧 — 2026-07-18. Branche parallèle (0
  perturbation du pipeline existant). Transformer à la main (4 couches, d_model=64) sur
  les séquences d'états par-minute (20-d : ADC ciblé + adverse + diffs), CV purgé identique
  au baseline tabulaire. Étape 1 supervisée vs ensemble tabulaire + MLP. Étape 2 SSL
  mask-and-reconstruct (delta mesuré). Verdict sur dia_chall ; master/GM null = bruit de
  label non interprétable. Standardisation per-feature non négociable. Spec :
  docs/superpowers/specs/2026-07-18-sequence-transformer-design.md. Métriques :
  data/05_model/sequence_metrics.json.
```

- [ ] **Step 4: Commit final**

```bash
git add CLAUDE.md
git commit -m "docs: état recherche transformer séquentiel + SSL (CLAUDE.md)"
```

---

## Self-Review (post-écriture, post-review critique)

Review externe intégré (7 points) : bugs de validité (#1-3) + bugs de qualité (#4-7) + nit
process. Tous corrigés inline ci-dessus.

**1. Spec coverage :**
- State vector 20-d (8+8+4) — Task 1-2 ✓
- Standardisation per-feature non négociable, IDENTIQUE pretrain+finetune (train-only par fold) — Task 7 (`standardize_fit/apply`) + Task 8 (appliquée par fold) + Task 9 (`_ssl_cv` réutilise les mêmes stats) ✓ (fix #2)
- Cap T=40 + pad mask — Task 2 ✓
- Deux tâches co-primaires high_elo + dia_chall — Task 7 (`task_subset`, `TASKS`) + Task 8 (boucle sur `sd.TASKS`) ✓
- CV purgé joueur-groupé + miroir (O(N)) — Task 7 (`player_folds`, `mirror_purge` indexé) + Task 8/9 (applique) ✓ (fix #6)
- Transformer à la main (pas HF) — Task 4-6 ✓
- masked-mean-pool + ablation CLS notée — Task 5 (commentaire dans `sequence_model.py`) ✓
- Baseline tabulaire = ensemble xgb+rf correctement tuné (pas un xgb sous-tuné), MLP contrôle — Task 8 ✓ (fix #1 + typo #5)
- Assertion de reproduction 0.72/0.59 adoucie en *prior* (pas un oracle) — Task 8 Step 5 ✓ (fix #1)
- SSL mask-and-reconstruct par-fold train-only (pas de fuite val, pas d'avantage transductif) + delta + caveat interpolation + caveat transductif documenté — Task 9 ✓ (fix #3)
- `sequence_supervised.pt` sauve le `best_state` réel (meilleur fold), plus un modèle frais — Task 8 `_train_one_task` retourne best_state + `run` le persiste ✓ (fix #4)
- Early-stop : patience reset sur amélioration + OOF au meilleur état (restaure best_state) — Task 8 ✓ (fix #7)
- `embed_game` bonus — Task 9 ✓
- `sequence_metrics.json` — Task 8 + Task 9 ✓
- Pièges d'interprétation (delta propre vs transductif) — Task 10 Step 2 ✓
- 0 API, branche parallèle, torch ajouté — Task 0 + réutilisation `_read_raw` ✓

**2. Placeholder scan :** aucun reliquat cassé (le code Task 7 est propre du premier coup,
fix nit process). Pas de TBD/TODO. ✓

**3. Type consistency :**
- `frame_state -> list[float]` (Task 1) consommé par `build_sequence` (Task 2) ✓
- `build_sequence -> (np.ndarray[40,20], np.ndarray[40]) | None` (Task 2) ✓
- `SequenceEncoder.forward(x, mask) -> [B,T,d_model]` (Task 4) consommé par `ClassifierHead`/`SequenceClassifier` (Task 5) et `ReconstructHead` (Task 6 via `enc(xb, mb&~ssl_mask)` Task 9) ✓
- `masked_mean(h, mask) -> [B,d_model]` (Task 5) consommé par `embed` (Task 5) et `embed_game` (Task 9) ✓
- `sd.task_subset/task->(idx,y)`, `sd.player_folds->list[(tr,va)]`, `sd.mirror_purge->idx`, `sd.standardize_fit/apply` (Task 7) consommés par Task 8 et Task 9 ✓
- `_train_one_task -> (metrics_dict, best_state_global)` (Task 8) consommé par `run` qui persiste `best_state_global` ✓
- `sm.get_device()`, `sm.SequenceClassifier`, `sm.SequenceEncoder`, `sm.ReconstructHead` (Task 4-6) consommés par Task 8-9 ✓

Tous les noms/signatures alignés entre tâches.