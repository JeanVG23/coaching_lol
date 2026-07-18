# Enrichissement events du transformer séquentiel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Élargir le state vector séquentiel v1 `[40,20]` → v2 `[40,27]` (+7 canaux events binaires COACHING_SAFE par minute), ré-entraîner le transformer supervisé et le SSL sur le même protocole CV purgé, et comparer les AUC à v1 (dia_chall 0.645, high_elo 0.546, delta_ssl −0.0195) pour tester falsifiablement si les events discrets ajoutent du signal que l'agrégat rate.

**Architecture:** Canaux empilés (channel-stacking) — les 7 canaux events rejoignent les 20 continues dans la même frame ; le transformer est inchangé sauf la projection d'entrée `d_in=27` (déjà paramétrée). Standardisation deux-blocs : z-score train-only sur les 20 continues, binaires laissées brutes. Métriques v2 dans `sequence_metrics_v2.json` (record v1 préservé). Spec : `docs/superpowers/specs/2026-07-18-sequence-events-enrichment-design.md`.

**Tech Stack:** Python, PyTorch (transformer à la main), NumPy, sklearn (RF+EBM baseline), pytest. Aucune nouvelle dépendance.

## Global Constraints

- **d_in auto-détecté** depuis `data["sequences"].shape[-1]` — jamais hardcoder 20. v1 (20-d) et v2 (27-d) doivent tous deux tourner via le même code.
- **Standardisation deux-blocs** : 20 cols continues en z-score train-only par fold (non négociable, spec d'origine décision 3) ; 7 cols binaires laissées brutes `[0,1]` via `bin_cols=range(20, d_in) if d_in > 20 else None` passé à `sd.standardize_fit`.
- **COACHING_SAFE uniquement** : info que le joueur avait (sa mort, mort adverse annoncée, timers objectifs = HUD public). Aucun proxy fog, aucune info inférée. Asymétrie du spec d'origine respectée.
- **DRY** : réutiliser `game_journal.OBJECTIVES`, `_objective_kills`, `_events`, `_recalls` (ne pas redéfinir les timers ni la logique recall/death).
- **Comparatif apples-to-apples** : même architecture transformer, même CV purgé (folds joueur-groupés + purge miroir), même seed (42), même budget (supervised `--epochs 30`, SSL pretrain 15 / finetune 30) qu'étape 1.
- **Métriques v2** : `data/05_model/sequence_metrics_v2.json` (fichier distinct via `--metrics-name sequence_metrics_v2.json`). Record v1 `sequence_metrics.json` préservé.
- **Environnement Mac M4** : `--device cpu` (MPS ne supporte pas `aten::_nested_tensor_from_mask_left_aligned` pour `src_key_padding_mask` ; `PYTORCH_ENABLE_MPS_FALLBACK=1` déjà en tête des modules). **Jamais `pytest tests/` complet** — cohabitation torch + xgboost → SIGSEGV libomp. La baseline tabulaire reste **RF + EBM** (xgb exclu), comme en étape 1.
- **Branche parallèle** : `research/sequence-events` depuis `master`. Pas de push. Pipeline tabulaire + coach web non touchés. Aucun overlap avec le pending work non-committé (web/shap/densify/pyproject).

## File Structure

- `src/01_data_engineering/build_sequence_dataset.py` — `_event_channels` (nouveau helper) + `build_sequence` élargi à 27-d (Task 1).
- `src/02_data_science/sequence_data.py` — `standardize_fit`/`standardize_apply` + param `bin_cols` (Task 2).
- `src/02_data_science/train_sequence_model.py` — `d_in` auto + `bin_cols` + `metrics_name` (Task 3).
- `src/02_data_science/pretrain_sequence_model.py` — `d_in` auto + `bin_cols` + `metrics_name` + `embed_game` infère `d_in` + perte SSL `/d_in` (Task 3).
- `tests/test_build_sequence_dataset.py` — nouveaux tests events + MAJ assertions 27-d (Task 1).
- `tests/test_sequence_data.py` — test `bin_cols` (Task 2).
- `tests/test_train_sequence_model.py`, `tests/test_pretrain_sequence_model.py` — `_mini` 27-d (Task 3).
- `tests/test_sequence_model.py` — test `d_in=27` (Task 3).
- `data/05_model/sequence_metrics_v2.json` — produit par Task 4 (run réel).
- `CLAUDE.md` — bloc recherche mis à jour (Task 5).

---

### Task 1: `_event_channels` + `build_sequence` 27-d (TDD)

**Files:**
- Modify: `src/01_data_engineering/build_sequence_dataset.py`
- Test: `tests/test_build_sequence_dataset.py`

**Interfaces:**
- Consumes: `game_journal` (`OBJECTIVES`, `_objective_kills`, `_events`, `_recalls`), `riotlib._frames_by_minute`.
- Produces: `_event_channels(timeline, pid, opp_pid, enemy_jungle_pid) -> np.ndarray[40,7]` ; `build_sequence` retourne désormais `(seq[40,27], mask[40])`.

- [ ] **Step 1: Écrire les tests events (et préparer la MAJ des assertions 27-d)**

Ajouter à `tests/test_build_sequence_dataset.py` (après `_fake_timeline`) :

```python
def _fake_timeline_with_events(n_minutes=15):
    t = _fake_timeline(n_minutes)
    # tous les events placés dans frame[0] — game_journal._events les flatten, peu importe la frame.
    t["info"]["frames"][0]["events"] = [
        # mort gank à 4:42 (minute 4) : killer=jungle(8) + assist=adc(6) -> ganked, pas solo
        {"type": "CHAMPION_KILL", "timestamp": 282000, "victimId": 1, "killerId": 8,
         "assistingParticipantIds": [6], "position": {"x": 1000, "y": 2000}},
        # mort solo à 12:30 (minute 12) : killer=adc(6), 0 assist -> pas ganked, solo
        {"type": "CHAMPION_KILL", "timestamp": 750000, "victimId": 1, "killerId": 6,
         "assistingParticipantIds": [], "position": {"x": 2000, "y": 3000}},
        # mort adverse à 8:15 (minute 8) : victimId=6 (opp)
        {"type": "CHAMPION_KILL", "timestamp": 495000, "victimId": 6, "killerId": 1,
         "assistingParticipantIds": [], "position": {"x": 3000, "y": 4000}},
        # drake tué à 7:00 par équipe 200 -> up avant, down 7-11, respawn à 12:00
        {"type": "ELITE_MONSTER_KILL", "timestamp": 420000, "monsterType": "DRAGON",
         "killerTeamId": 200, "killerId": 8},
        # achat à 2:30 (minute 2), hors opening (<90s) -> recall
        {"type": "ITEM_PURCHASED", "timestamp": 150000, "participantId": 1, "itemId": 1001},
    ]
    return t


def test_event_channels_death_gank_solo():
    t = _fake_timeline_with_events()
    ch = bsd._event_channels(t, pid=1, opp_pid=6, enemy_jungle_pid=8)
    assert ch.shape == (40, 7)
    assert ch[4, 0] == 1.0 and ch[4, 5] == 1.0 and ch[4, 6] == 0.0   # ganked, pas solo
    assert ch[12, 0] == 1.0 and ch[12, 5] == 0.0 and ch[12, 6] == 1.0  # solo, pas ganked


def test_event_channels_opp_death():
    t = _fake_timeline_with_events()
    ch = bsd._event_channels(t, pid=1, opp_pid=6, enemy_jungle_pid=8)
    assert ch[8, 1] == 1.0


def test_event_channels_drake_up_respawn():
    t = _fake_timeline_with_events()
    ch = bsd._event_channels(t, pid=1, opp_pid=6, enemy_jungle_pid=8)
    assert ch[5, 3] == 1.0 and ch[6, 3] == 1.0   # up avant le kill (first spawn 5:00)
    assert ch[9, 3] == 0.0                       # down après kill (respawn 5min)
    assert ch[12, 3] == 1.0                       # respawn à 12:00
    assert ch[0, 4] == 0.0                        # baron pas up en early (first 25:00)


def test_event_channels_recall():
    t = _fake_timeline_with_events()
    ch = bsd._event_channels(t, pid=1, opp_pid=6, enemy_jungle_pid=8)
    assert ch[2, 2] == 1.0                        # achat à 2:30 -> recall minute 2
```

Et **mettre à jour** les assertions existantes pour le 27-d (les 7 canaux sont 0 sur la `_fake_timeline` sans events, car minutes 0-2 < first spawn drake 5:00 et aucun event) :

```python
# test_build_sequence_shapes_and_mask : (40, 20) -> (40, 27)
assert seq.shape == (40, 27) and seq.dtype == np.float32

# test_build_sequence_values_minute1 : expected += 7 zéros
expected = self_state + opp_state + diffs + [0.0] * 7

# test_main_writes_npz : (1, 40, 20) -> (1, 40, 27)
assert d["sequences"].shape == (1, 40, 27)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run python3 -m pytest tests/test_build_sequence_dataset.py -v`
Expected: FAIL — `bsd._event_channels` n'existe pas ; `seq.shape == (40, 27)` échoue (v1 = 20).

- [ ] **Step 3: Implémenter**

Dans `src/01_data_engineering/build_sequence_dataset.py`, ajouter l'import top-level après `import riotlib as rl` (ligne ~25) :

```python
import game_journal as gj      # réutilise OBJECTIVES, _objective_kills, _events, _recalls (DRY)
```

Ajouter les deux helpers avant `build_sequence` :

```python
def _obj_up(obj_kills: dict, name: str, t_ms: int) -> bool:
    """Objectif `name` respawné (up) à t_ms ? Réutilise game_journal.OBJECTIVES (timers)."""
    cfg = gj.OBJECTIVES[name]
    past = [k for k in obj_kills[name] if k <= t_ms]
    next_spawn = past[-1] + cfg["respawn"] if past else cfg["first"]
    return t_ms >= next_spawn


def _event_channels(timeline: dict, pid: int, opp_pid: int,
                    enemy_jungle_pid: int | None) -> np.ndarray:
    """-> [40, 7] float32. Canaux events binaires par minute, COACHING_SAFE (info que le
    joueur avait : sa mort, l'annonce de la mort adverse, timers objectifs = HUD public).
    Ordre : self_death_m, opp_death_m, self_recall_m, drake_up, baron_up, is_ganked,
    is_solo_death. Réutilise game_journal (OBJECTIVES, _objective_kills, _events, _recalls)
    — version allégée : on ne calcule que le bucket minute + gank/solo, PAS gold_state/
    consequences (coût inutile sur 43-95k games × 2 ADC)."""
    ch = np.zeros((MAX_LEN, 7), dtype=np.float32)
    obj_kills = gj._objective_kills(timeline)
    for m in range(MAX_LEN):
        t_end = (m + 1) * 60000 - 1                       # fin de la minute m
        if _obj_up(obj_kills, "DRAGON", t_end):
            ch[m, 3] = 1.0                                # drake_up
        if _obj_up(obj_kills, "BARON_NASHOR", t_end):
            ch[m, 4] = 1.0                                # baron_up
    for ev in gj._events(timeline):
        et = ev.get("timestamp", 0)
        m = et // 60000
        if m >= MAX_LEN:
            continue
        if ev.get("type") == "CHAMPION_KILL":
            vid = ev.get("victimId")
            if vid == pid:
                ch[m, 0] = 1.0                            # self_death_m
                assisters = ev.get("assistingParticipantIds") or []
                involved = ({ev.get("killerId")} | set(assisters)) - {None}
                if enemy_jungle_pid is not None and enemy_jungle_pid in involved:
                    ch[m, 5] = 1.0                        # is_ganked
                if len(assisters) == 0:
                    ch[m, 6] = 1.0                        # is_solo_death
            elif vid == opp_pid:
                ch[m, 1] = 1.0                            # opp_death_m
    for rec in gj._recalls(timeline, pid, obj_kills):     # visites de shop (clusters d'achats)
        m = rec["t_ms"] // 60000
        if m < MAX_LEN:
            ch[m, 2] = 1.0                               # self_recall_m
    return ch
```

Remplacer `build_sequence` par la version 27-d (calcule `enemy_jungle_pid`, concatène les 7 canaux) :

```python
def build_sequence(match: dict, timeline: dict,
                   target_puuid: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Une game -> (seq[40,27] float32, mask[40] bool). None si pas d'opponent ou 0 frame.
    Frame = self(8) + opp(8) + diffs(4) + event_channels(7) = 27-d."""
    pid = participant_pid(match, target_puuid)
    opp = opponent_pid(match, target_puuid)
    if opp is None:
        return None
    parts = match["info"]["participants"]
    pidx = match["metadata"]["participants"].index(target_puuid)
    my_team = parts[pidx]["teamId"]
    enemy_jungle_pid = next((i + 1 for i, p in enumerate(parts)
                             if p["teamId"] != my_team
                             and (p.get("teamPosition") or "") == "JUNGLE"), None)
    my_fr = rl._frames_by_minute(timeline, pid)
    opp_fr = rl._frames_by_minute(timeline, opp)
    ev = _event_channels(timeline, pid, opp, enemy_jungle_pid)
    seq = np.zeros((MAX_LEN, 27), dtype=np.float32)
    mask = np.zeros(MAX_LEN, dtype=bool)
    for minute, pf in my_fr.items():
        if minute >= MAX_LEN:
            continue
        self_s = frame_state(pf)
        opp_s = frame_state(opp_fr.get(minute, {}))   # frame adverse manquante -> zeros
        seq[minute] = self_s + opp_s + _diffs(self_s, opp_s) + list(ev[minute])
        mask[minute] = True
    if mask.sum() == 0:
        return None
    return seq, mask
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `poetry run python3 -m pytest tests/test_build_sequence_dataset.py -v`
Expected: PASS (tous les tests, y compris les nouveaux events + les assertions 27-d).

- [ ] **Step 5: Commit**

```bash
git add src/01_data_engineering/build_sequence_dataset.py tests/test_build_sequence_dataset.py
git commit -m "feat(sequence): canaux events 27-d (deaths/obj/recalls, COACHING_SAFE)"
```

---

### Task 2: `standardize_fit/apply` + param `bin_cols` (TDD)

**Files:**
- Modify: `src/02_data_science/sequence_data.py`
- Test: `tests/test_sequence_data.py`

**Interfaces:**
- Consumes: rien de nouveau.
- Produces: `standardize_fit(..., bin_cols=None)` / `standardize_apply` inchangée_signature ; les cols dans `bin_cols` reçoivent mean=0/std=1 (laissées brutes). Défaut `None` = comportement v1 (toutes cols standardisées) → backward-compatible.

- [ ] **Step 1: Écrire le test**

Ajouter à `tests/test_sequence_data.py` :

```python
def test_standardize_fit_bin_cols_left_raw():
    rng = np.random.RandomState(0)
    d = _data(n=20)
    seq = d["sequences"].copy()
    seq[..., 18] = (rng.rand(20, 40) > 0.5).astype(np.float32)   # simulé binaire
    seq[..., 19] = (rng.rand(20, 40) > 0.5).astype(np.float32)
    train_idx = np.arange(15)
    mean, std = sd.standardize_fit(seq, d["mask"], train_idx, bin_cols=[18, 19])
    assert mean[18] == 0.0 and std[18] == 1.0      # binaires laissés bruts
    assert mean[19] == 0.0 and std[19] == 1.0
    assert mean[0] != 0.0                            # continue standardisée
    z = sd.standardize_apply(seq[train_idx], mean, std)
    np.testing.assert_allclose(z[:, :, 18], seq[train_idx, :, 18])   # inchangé
    np.testing.assert_allclose(z[:, :, 19], seq[train_idx, :, 19])
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run python3 -m pytest tests/test_sequence_data.py::test_standardize_fit_bin_cols_left_raw -v`
Expected: FAIL — `standardize_fit() got an unexpected keyword argument 'bin_cols'`.

- [ ] **Step 3: Implémenter**

Remplacer `standardize_fit` dans `src/02_data_science/sequence_data.py` (`standardize_apply` inchangée) :

```python
def standardize_fit(sequences: np.ndarray, mask: np.ndarray,
                    train_idx: np.ndarray, bin_cols=None) -> tuple[np.ndarray, np.ndarray]:
    """mean/std par feature sur les frames valides des rows de train. bin_cols = indices de
    colonnes à laisser brutes (canaux events binaires v2) : on force mean=0/std=1 -> apply
    est l'identité sur ces cols. Défaut None = toutes cols standardisées (v1, backward-compat)."""
    X = sequences[train_idx][mask[train_idx]]      # [n_valid, F]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0                          # garde-fou feature constante
    if bin_cols is not None:
        bc = list(bin_cols)
        mean[bc] = 0.0
        std[bc] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)
```

- [ ] **Step 4: Vérifier que les tests passent (régression v1 incluse)**

Run: `poetry run python3 -m pytest tests/test_sequence_data.py -v`
Expected: PASS (nouveau test `bin_cols` + les tests v1 existants `test_standardize_fit_train_only` / `test_standardize_uses_train_stats_on_val` inchangés).

- [ ] **Step 5: Commit**

```bash
git add src/02_data_science/sequence_data.py tests/test_sequence_data.py
git commit -m "feat(sequence): standardize_fit bin_cols (binaires laissés bruts, v1 compat)"
```

---

### Task 3: Câbler `d_in` auto + `bin_cols` + `metrics_name` dans train/pretrain (TDD)

**Files:**
- Modify: `src/02_data_science/train_sequence_model.py`, `src/02_data_science/pretrain_sequence_model.py`
- Test: `tests/test_train_sequence_model.py`, `tests/test_pretrain_sequence_model.py`, `tests/test_sequence_model.py`

**Interfaces:**
- Consumes: `sequence_data.standardize_fit(..., bin_cols=...)` (Task 2), `sequences.shape[-1]` (Task 1 produit le 27-d).
- Produces: `run(..., metrics_name="sequence_metrics.json")` dans les deux modules ; `d_in` auto-détecté et propagé à `SequenceClassifier`/`SequenceEncoder`/`ReconstructHead` ; `embed_game` infère `d_in` de l'input ; perte SSL divisée par `d_in` (pas 20).

- [ ] **Step 1: Écrire/mettre à jour les tests**

`tests/test_sequence_model.py` — ajouter :

```python
def test_sequence_classifier_accepts_d_in_27():
    torch.manual_seed(0)
    clf = sm.SequenceClassifier(d_in=27, d_model=64, max_len=40)
    x = torch.randn(4, 40, 27)
    mask = torch.ones(4, 40, dtype=torch.bool)
    assert clf(x, mask).shape == (4,)
    assert clf.embed(x, mask).shape == (4, 64)
```

`tests/test_train_sequence_model.py` — `_mini` passe à 27-d (le signal reste sur la feature continue 2) :

```python
    seqs = rng.randn(n, 40, 27).astype(np.float32)   # 20 -> 27
    seqs[y == 1, 10, 2] += 5.0
```

`tests/test_pretrain_sequence_model.py` — `_mini` passe à 27-d, et `embed_game` reçoit 27-d :

```python
    seqs = rng.randn(n, 40, 27).astype(np.float32)   # 20 -> 27
    seqs[y == 1, 10, 2] += 5.0
```
et dans `test_embed_game_shape` :
```python
    emb = ptm.embed_game(np.random.randn(40, 27).astype(np.float32),   # 20 -> 27
                         np.ones(40, dtype=bool), device_force="cpu")
    assert emb.shape == (64,)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `poetry run python3 -m pytest tests/test_sequence_model.py::test_sequence_classifier_accepts_d_in_27 tests/test_train_sequence_model.py tests/test_pretrain_sequence_model.py -v`
Expected: FAIL — `train`/`pretrain` construisent `SequenceClassifier()` (d_in=20) sur séquences 27-d → mismatch de dimension ; `embed_game` reçoit 27-d mais crée un classifieur 20-d.

- [ ] **Step 3: Implémenter — `train_sequence_model.py`**

Signature `run` + `d_in` auto + `metrics_name` + propagation :

```python
def run(epochs=60, batch=64, seed=SEED, device_force=None,
        metrics_name="sequence_metrics.json") -> dict:
    data = sd.load_dataset(sd.DATASET)
    device = torch.device(device_force) if device_force else sm.get_device()
    d_in = int(data["sequences"].shape[-1])          # auto : 20 (v1) ou 27 (v2)
    print(f"  device={device} | {len(data['sequences'])} séquences | d_in={d_in}")
    metrics = {"tasks": {}, "params": {"epochs": epochs, "batch": batch, "seed": seed,
                "d_model": 64, "n_layers": 4, "nhead": 4, "d_in": d_in, "device": str(device)}}
    saved_state = None
    for task in sd.TASKS:
        print(f"\n=== tâche {task} ===")
        m, state = _train_one_task(task, data, device, epochs, batch, seed, d_in)
        metrics["tasks"][task] = m
        print(f"  séquence AUC={m.get('auc_mean')} (±{m.get('auc_std')})  "
              f"tabulaire={m.get('baseline_tabular_auc')}  mlp={m.get('baseline_mlp_auc')}")
        if task == "high_elo" and state is not None:
            saved_state = state
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if saved_state is not None:
        torch.save(saved_state, MODEL_DIR / "sequence_supervised.pt")
    (MODEL_DIR / metrics_name).write_text(json.dumps(metrics, indent=2))
    print(f"\n✓ {MODEL_DIR}/{metrics_name}")
    return metrics
```

`_train_one_task` — ajouter `d_in`, `bin_cols`, `SequenceClassifier(d_in=d_in)` :

```python
def _train_one_task(task, data, device, epochs, batch, seed, d_in):
    from sklearn.metrics import roc_auc_score
    import copy
    idx, y = sd.task_subset(data, task)
    if len(set(y)) < 2 or len(idx) < 30:
        return ({"auc_mean": None, "auc_std": None, "n_rows": int(len(idx)),
                 "reason": "trop peu de rows ou 1 classe"}, None)
    bin_cols = range(20, d_in) if d_in > 20 else None
    folds = sd.player_folds(data["puuid"][idx], y, n_splits=5, seed=seed)
    oof = np.full(len(idx), np.nan)
    best_state_global, best_auc_global = None, -1.0
    for fi, (tr_i, va_i) in enumerate(folds):
        print(f"    fold {fi+1}/5  train={len(tr_i)} val={len(va_i)}", flush=True)
        val_puuids = set(data["puuid"][idx][va_i])
        tr_purged = sd.mirror_purge(tr_i, val_puuids, data["match_id"][idx], data["puuid"][idx])
        if len(tr_purged) == 0:
            continue
        mean, std = sd.standardize_fit(data["sequences"], data["mask"], idx[tr_purged], bin_cols)
        Xs = sd.standardize_apply(data["sequences"], mean, std)
        Xtr = torch.from_numpy(Xs[idx[tr_purged]]).to(device)
        Mtr = torch.from_numpy(data["mask"][idx[tr_purged]]).to(device)
        ytr = torch.from_numpy(y[tr_purged].astype(np.float32)).to(device)
        Xva = torch.from_numpy(Xs[idx[va_i]]).to(device)
        Mva = torch.from_numpy(data["mask"][idx[va_i]]).to(device)
        model = sm.SequenceClassifier(d_in=d_in).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_fold_auc, best_state, patience, bad = 0.0, None, 10, 0
        for ep in range(epochs):
            if ep % 10 == 0:
                print(f"      ep {ep}/{epochs}", flush=True)
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
                bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            oof[va_i] = torch.sigmoid(model(Xva, Mva)).cpu().numpy()
        if best_fold_auc > best_auc_global:
            best_auc_global = best_fold_auc
            best_state_global = best_state
        print(f"    fold {fi+1} done  best_fold_auc={best_fold_auc:.4f}  (patience@{bad})", flush=True)
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
```

`main` — ajouter `--metrics-name` :

```python
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", type=str, default=None,
                    help="force device (ex: 'cpu') ; défaut = auto (MPS/CUDA/CPU)")
    ap.add_argument("--metrics-name", type=str, default="sequence_metrics.json",
                    help="nom du fichier métriques (v2: sequence_metrics_v2.json)")
    args = ap.parse_args()
    run(epochs=args.epochs, batch=args.batch, device_force=args.device,
        metrics_name=args.metrics_name)
    return 0
```

- [ ] **Step 4: Implémenter — `pretrain_sequence_model.py`**

`_ssl_cv` — ajouter `d_in`, `bin_cols`, `SequenceEncoder/ReconstructHead/SequenceClassifier(d_in=d_in)`, perte `/d_in` :

```python
def _ssl_cv(data, device, pretrain_epochs, finetune_epochs, batch, seed, d_in):
    from sklearn.metrics import roc_auc_score
    idx, y = sd.task_subset(data, "high_elo")
    bin_cols = range(20, d_in) if d_in > 20 else None
    folds = sd.player_folds(data["puuid"][idx], y, n_splits=5, seed=seed)
    oof = np.full(len(idx), np.nan)
    best_enc, best_auc = None, -1.0
    for tr_i, va_i in folds:
        val_puuids = set(data["puuid"][idx][va_i])
        tr_purged = sd.mirror_purge(tr_i, val_puuids, data["match_id"][idx], data["puuid"][idx])
        if len(tr_purged) == 0:
            continue
        mean, std = sd.standardize_fit(data["sequences"], data["mask"], idx[tr_purged], bin_cols)
        Xs = sd.standardize_apply(data["sequences"], mean, std)
        Xtr = torch.from_numpy(Xs[idx[tr_purged]]).to(device)
        Mtr = torch.from_numpy(data["mask"][idx[tr_purged]]).to(device)
        ytr = torch.from_numpy(y[tr_purged].astype(np.float32)).to(device)
        Xva = torch.from_numpy(Xs[idx[va_i]]).to(device)
        Mva = torch.from_numpy(data["mask"][idx[va_i]]).to(device)
        n = len(Xtr)
        enc = sm.SequenceEncoder(d_in=d_in).to(device)
        head = sm.ReconstructHead(d_in=d_in).to(device)
        opt_p = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                                  lr=3e-4, weight_decay=1e-2)
        for ep in range(pretrain_epochs):
            enc.train(); head.train()
            perm = torch.randperm(n)
            for b0 in range(0, n, batch):
                bi = perm[b0:b0 + batch]
                xb, mb = Xtr[bi], Mtr[bi]
                ssl_mask = (torch.rand_like(mb.float()) < MASK_FRAC) & mb
                h = enc(xb, mb & ~ssl_mask)
                pred = head(h)
                loss = ((pred - xb) ** 2 * ssl_mask.unsqueeze(-1)).sum() / \
                       ssl_mask.float().sum().clamp(min=1.0) / d_in
                opt_p.zero_grad(); loss.backward(); opt_p.step()
        clf = sm.SequenceClassifier(d_in=d_in).to(device)
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
```

`run` — `d_in` auto + `metrics_name` :

```python
def run(pretrain_epochs=30, finetune_epochs=40, batch=64, seed=SEED, device_force=None,
        metrics_name="sequence_metrics.json") -> dict:
    data = sd.load_dataset(sd.DATASET)
    device = torch.device(device_force) if device_force else sm.get_device()
    d_in = int(data["sequences"].shape[-1])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    auc_ssl, best_enc = _ssl_cv(data, device, pretrain_epochs, finetune_epochs, batch, seed, d_in)
    metrics_path = MODEL_DIR / metrics_name
    prev = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    auc_sup = prev.get("tasks", {}).get("high_elo", {}).get("auc_mean")
    out = {"auc_supervised": auc_sup, "auc_ssl": auc_ssl,
           "delta_ssl": (auc_ssl - auc_sup) if (auc_ssl is not None and auc_sup is not None) else None,
           "pretext": "mask-and-reconstruct (MSE, 15% mask) — prétexte faible sur signaux lisses ; "
                      "≈0 delta n'est pas un verdict sur le SSL en général (cf. spec §Pièges). "
                      "Pretrain par-fold train-only : pas d'avantage transductif, delta propre. "
                      "v2 : canaux events binaires inclus dans la reconstruction (signal non-lisse).",
           "params": {"pretrain_epochs": pretrain_epochs, "finetune_epochs": finetune_epochs,
                      "seed": seed, "device": str(device), "d_in": d_in}}
    prev["ssl"] = out
    metrics_path.write_text(json.dumps(prev, indent=2))
    if best_enc is not None:
        torch.save(best_enc, MODEL_DIR / "sequence_encoder_pretrain.pt")
    print(f"  AUC supervisé={auc_sup}  AUC ssl={auc_ssl}  delta={out['delta_ssl']}")
    print(f"✓ {MODEL_DIR}/sequence_encoder_pretrain.pt + delta dans {metrics_name}")
    return out
```

`embed_game` — infère `d_in` de l'input :

```python
def embed_game(seq: np.ndarray, mask: np.ndarray, device_force=None) -> np.ndarray:
    """Vecteur d'embedding 64-d d'une game (pour inspection : projection 2D colorée rang)."""
    device = torch.device(device_force) if device_force else sm.get_device()
    clf = sm.SequenceClassifier(d_in=seq.shape[-1]).to(device)
    clf.eval()
    with torch.no_grad():
        x = torch.from_numpy(seq[None].astype(np.float32)).to(device)
        m = torch.from_numpy(mask[None].astype(bool)).to(device)
        return clf.embed(x, m).cpu().numpy()[0]
```

`main` — ajouter `--metrics-name` :

```python
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--device", type=str, default=None,
                    help="force device (ex: 'cpu') ; défaut = auto (MPS/CUDA/CPU)")
    ap.add_argument("--metrics-name", type=str, default="sequence_metrics.json",
                    help="nom du fichier métriques (v2: sequence_metrics_v2.json)")
    args = ap.parse_args()
    run(pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
        device_force=args.device, metrics_name=args.metrics_name)
    return 0
```

- [ ] **Step 5: Vérifier que les tests passent**

Run (convention Mac — pas de `pytest tests/` global, torch séparé des baselines xgb) :
```bash
poetry run python3 -m pytest tests/test_sequence_model.py tests/test_train_sequence_model.py tests/test_pretrain_sequence_model.py -v
```
Expected: PASS (test_sequence_model 5/5 dont `d_in=27` ; train smoke 1/1 ; pretrain smoke 2/2).

Puis la suite non-torch (régression) :
```bash
poetry run python3 -m pytest tests/test_build_sequence_dataset.py tests/test_sequence_data.py -v
```
Expected: PASS (Task 1 + Task 2 verts).

- [ ] **Step 6: Commit**

```bash
git add src/02_data_science/train_sequence_model.py src/02_data_science/pretrain_sequence_model.py \
        tests/test_train_sequence_model.py tests/test_pretrain_sequence_model.py tests/test_sequence_model.py
git commit -m "feat(sequence): d_in auto + bin_cols + metrics_name (train/pretrain v2-ready)"
```

---

### Task 4: Regen dataset + ré-entraînement supervisé + SSL + verdict

**Files:**
- Run: `src/01_data_engineering/build_sequence_dataset.py`, `src/02_data_science/train_sequence_model.py`, `src/02_data_science/pretrain_sequence_model.py`
- Produces: `data/04_dataset/adc_sequence_dataset.npz` (27-d, overwrite), `data/05_model/sequence_metrics_v2.json`.

- [ ] **Step 1: Regen le dataset 27-d**

Run: `poetry run python3 src/01_data_engineering/build_sequence_dataset.py`
Expected: `~7 873 séquences ADC -> adc_sequence_dataset.npz` (même volume que v1 ; `sequences` shape `[N, 40, 27]`). Vérifier : `poetry run python3 -c "import numpy as np; d=np.load('data/04_dataset/adc_sequence_dataset.npz', allow_pickle=True); print(d['sequences'].shape)"` → `(N, 40, 27)`.

- [ ] **Step 2: Ré-entraîner le transformer supervisé (v2)**

Run: `poetry run python3 src/02_data_science/train_sequence_model.py --device cpu --epochs 30 --metrics-name sequence_metrics_v2.json`
Expected: affiche AUC séquence / tabulaire / MLP pour `high_elo` et `dia_chall` ; écrit `data/05_model/sequence_metrics_v2.json` + `sequence_supervised.pt`. Durée ~1h12 sur M4 CPU (cf. étape 1). Si >2.5h, réduire `--epochs 20` (early-stop à ep 10 attendu → impact nul).

- [ ] **Step 3: Ré-entraîner le SSL (v2)**

Run: `poetry run python3 src/02_data_science/pretrain_sequence_model.py --device cpu --pretrain-epochs 15 --finetune-epochs 30 --metrics-name sequence_metrics_v2.json`
Expected: affiche `AUC supervisé=X AUC ssl=Y delta=Z` ; merge le bloc `ssl` dans `sequence_metrics_v2.json` ; écrit `sequence_encoder_pretrain.pt`. Durée ~3h wall sur M4 CPU (cf. étape 1). Lancer via `nohup ... & disown` si risque de sleep laptop.

- [ ] **Step 4: Lire et interpréter les métriques v2**

Run: `poetry run python3 -c "import json; m=json.load(open('data/05_model/sequence_metrics_v2.json')); print(json.dumps(m, indent=2))"`

Interprétation (à reporter dans le commit Task 5 / CLAUDE.md) :
- **dia_chall v2** : `auc_mean` vs v1 0.645. > 0.645 → thèse renforcée (events ajoutent du signal que l'agrégat rate). ≈ 0.645 → canaux events à résolution frame insuffisants (piste : predictive SSL étape 3, ou event-tokens).
- **high_elo v2** : ≈ 0.546 attendu (bruit de label, non interprétable — cf. §Pièges). Ne pas conclure.
- **delta_ssl v2** : vs v1 −0.0195. > 0 → le prétexte non-lisse (events binaires enfin reconstruits) décolle. ≈ 0 → mask-and-reconstruct reste faible même avec events (prétexte prédictif = étape 3).
- Comparer aussi `baseline_tabular_auc` v2 vs v1 0.633 (la baseline tabulaire lit `adc_dataset.parquet` inchangé → doit rester ~0.633 ; si elle bouge, c'est un signal que le join a changé — debugger).

- [ ] **Step 5: Commit (métriques gitignorées — pas de commit de data/)**

Les métriques sont gitignorées (`data/`). Pas de commit de fichiers data. Si on veut tracer le verdict, le reporter dans le commit Task 5 (CLAUDE.md).

---

### Task 5: Mettre à jour CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (bloc « Recherche — transformer séquentiel + SSL »)

- [ ] **Step 1: Mettre à jour le bloc recherche**

Dans la section « État d'avancement », dans le bloc « Recherche — transformer séquentiel + SSL », ajouter après le paragraphe existant un sous-paragraphe **étape 2 livrée** avec : date 2026-07-18, état (`✅`), le passage v1→v2 (state vector 20→27, +7 canaux events binaires COACHING_SAFE : self/opp death, recall, drake/baron up, ganked, solo death), les métriques v2 réelles (`dia_chall` v2 AUC vs 0.645, `high_elo` v2 vs 0.546, `delta_ssl` v2 vs −0.0195, `baseline_tabular` v2 vs 0.633), le verdict (thèse renforcée / null / SSL décolle), et un lien vers le spec `docs/superpowers/specs/2026-07-18-sequence-events-enrichment-design.md` + métriques `data/05_model/sequence_metrics_v2.json`. Garder le texte concis (cf. densité des blocs voisins).

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: état étape 2 enrichissement events du transformer (CLAUDE.md)"
```

---

## Self-Review (post-écriture)

**1. Spec coverage :**
- 7 canaux events COACHING_SAFE (deaths/obj/recalls + gank/solo) — Task 1 ✓
- Standardisation deux-blocs (continues z-score, binaires brutes) — Task 2 ✓
- d_in auto-détecté (v1/v2 résilients) — Task 3 ✓
- Comparatif apples-to-apples (même arch/CV/seed/epochs) — Task 3+4 (mêmes params) ✓
- Métriques v2 fichier séparé (record v1 préservé) — Task 3 (`metrics_name`) + Task 4 ✓
- SSL perte /d_in (pas 20) — Task 3 ✓
- embed_game infère d_in — Task 3 ✓
- Reuse game_journal (DRY) — Task 1 ✓
- Mac env (--device cpu, pas de pytest tests/ global, RF+EBM) — Global Constraints + Task 3 Step 5 ✓
- Regen + retrain + verdict — Task 4 ✓
- CLAUDE.md mis à jour — Task 5 ✓
- Branche parallèle, pas de push, pas de perturbation — Global Constraints ✓

**2. Placeholder scan :** aucun TBD/TODO. Tout le code des Steps est complet. ✓

**3. Type consistency :**
- `_event_channels -> np.ndarray[40,7]` consommé par `build_sequence` (`list(ev[minute])` → 7 floats) ✓
- `standardize_fit(..., bin_cols=range(20,d_in)|None) -> (mean[F], std[F])` consommé par train/pretrain ✓
- `run(..., metrics_name=str)` dans train et pretrain, consommé par `main(--metrics-name)` ✓
- `_train_one_task(..., d_in)` / `_ssl_cv(..., d_in)` — `d_in` calculé dans `run` (`data["sequences"].shape[-1]`) ✓
- `SequenceClassifier/Encoder/ReconstructHead(d_in=d_in)` — `d_in` param déjà existant sur les 3 classes ✓
- `embed_game(seq[40,d_in])` → `SequenceClassifier(d_in=seq.shape[-1])` ✓

Tous les noms/signatures alignés entre tâches.