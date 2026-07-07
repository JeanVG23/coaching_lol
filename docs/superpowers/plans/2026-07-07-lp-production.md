# LP Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Industrialiser la régression LP (Master/GM/Challenger) validée par le POC : fetch LP → dataset per-player sans balance-cap → ensemble xgb+rf+ebm tuné en purged CV → LP estimé affiché sur le web en complément du placement binaire existant.

**Architecture:** 5 briques : `src/collection/fetch_apex_lp.py` (3 appels API → `data/04_dataset/apex_lp.json`), `src/01_data_engineering/build_player_lp_dataset.py` (0 API → `adc_player_lp_dataset.parquet`), `src/02_data_science/train_player_lp.py` (random search en purged CV, purge précalculée par fold → `{xgb,rf,ebm}_player_lp.pkl` + métriques), `web/backend/ml_rank.py` (hybride : `predicted_lp` ajouté si rang apex), frontend (affichage « ~N LP estimés »). Spec : `docs/superpowers/specs/2026-07-07-lp-production-design.md`.

**Tech Stack:** Python (Poetry), pandas, numpy, scipy, scikit-learn, XGBoost, interpret (EBM), shap, pytest. Convention flat-import du repo (`sys.path.insert` puis `import module`, pas de package).

## Global Constraints

- **Interdiction d'importer depuis `poc/`** dans tout code sous `src/` ou `web/`. `poc/` reste intact (référence historique).
- **Pipeline binaire intact** : `build_player_dataset.py`, `train_player_ensemble.py`, `calibrate_player_rank.py`, `calibrate_rank.py`, tous les `.pkl` existants, `player_metrics.json`, `player_rank_calibration.json` ne sont ni modifiés ni réécrits. Seule exception : `train_player_lp.py` IMPORTE (sans les modifier) `purged_train_features` et `dispersion_share_analysis` de `train_player_ensemble.py`.
- **Le placement 4 rangs de `ml_rank.predict_rank()` est inchangé** : mêmes clés de retour existantes (`predicted_rank`, `proba`, `n_games_used`), le champ `predicted_lp` est purement additif.
- **Noms d'artefacts exacts** : `data/04_dataset/apex_lp.json`, `data/04_dataset/adc_player_lp_dataset.parquet` (+ `.csv`, + `adc_player_lp_dataset.meta.json`), `data/05_model/xgb_player_lp.pkl`, `rf_player_lp.pkl`, `ebm_player_lp.pkl`, `player_lp_features.json`, `player_lp_metrics.json`.
- **Constantes** : `MIN_PLAYER_GAMES = 15`, `APEX_TIERS = {"master", "grandmaster", "challenger"}` (diamond exclu), graine `SEED = 42` partout, `POC_BASELINE_SPEARMAN_POOLED = 0.5028`.
- **Purged CV obligatoire** (StratifiedKFold 5 folds stratifié sur le tier, shuffle, seed 42) ; les features de train purgées sont **précalculées une fois par fold** et réutilisées pour toutes les configs du random search (la purge ne dépend pas des hyperparamètres).
- **Jamais de NaN silencieux dans les métriques** : Spearman via `_safe_spearman` (None si <3 points ou entrée constante).
- Lancer les commandes depuis la racine du repo avec `poetry run` (ex. `poetry run pytest tests/ -q`).
- `data/` est gitignoré : aucun artefact de `data/` n'est commité.

---

### Task 1: `src/collection/fetch_apex_lp.py` — fetch LP prod

Promotion du script POC `poc/script/fetch_apex_lp.py` vers `src/collection/`, avec 3 changements : sortie dans `data/04_dataset/apex_lp.json`, enveloppe `{"fetched_at": ISO-UTC, "players": {...}}`, warning si `master` renvoie exactement 10 000 entrées (cap API suspecté au POC). Le POC n'est PAS modifié ni supprimé.

**Files:**
- Create: `src/collection/fetch_apex_lp.py`
- Test: `tests/test_fetch_apex_lp.py`

**Interfaces:**
- Consomme : `riotlib` (`load_env`, `PLATFORM_TO_REGIONAL`, `RiotClient.apex_league(tier)`, `rl.DATA`).
- Produit : `data/04_dataset/apex_lp.json` au format `{"fetched_at": str ISO-8601 UTC, "players": {puuid: {"tier": str, "leaguePoints": int}}}` — consommé par Task 3. Fonctions pures : `build_lp_lookup(entries_by_tier: dict[str, list[dict]]) -> dict[str, dict]`, `build_payload(entries_by_tier: dict, fetched_at: str) -> dict`.

- [ ] **Step 1: Écrire les tests (échouants)**

Créer `tests/test_fetch_apex_lp.py` :

```python
"""Tests des fonctions pures du fetch LP prod (src/collection/fetch_apex_lp.py).
Les appels API réels sont vérifiés à l'exécution (Task 6 du plan)."""
import fetch_apex_lp  # src/collection est sur sys.path via tests/conftest.py


def test_build_lp_lookup_merges_tiers_and_skips_missing_puuid():
    entries_by_tier = {
        "challenger": [{"puuid": "a", "leaguePoints": 800}],
        "master": [{"puuid": "b", "leaguePoints": 50},
                   {"puuid": None, "leaguePoints": 10}],
    }
    lookup = fetch_apex_lp.build_lp_lookup(entries_by_tier)
    assert lookup == {
        "a": {"tier": "challenger", "leaguePoints": 800},
        "b": {"tier": "master", "leaguePoints": 50},
    }


def test_build_payload_wraps_players_with_fetched_at():
    entries_by_tier = {"master": [{"puuid": "b", "leaguePoints": 50}]}
    payload = fetch_apex_lp.build_payload(entries_by_tier, "2026-07-07T12:00:00+00:00")
    assert payload["fetched_at"] == "2026-07-07T12:00:00+00:00"
    assert payload["players"] == {"b": {"tier": "master", "leaguePoints": 50}}
```

- [ ] **Step 2: Vérifier qu'ils échouent**

Run: `poetry run pytest tests/test_fetch_apex_lp.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'fetch_apex_lp'`)

- [ ] **Step 3: Implémenter le script**

Créer `src/collection/fetch_apex_lp.py` :

```python
#!/usr/bin/env python3
"""
collection — fetch en masse du LP courant des joueurs apex (Master/GM/Challenger).

`apex_league(tier)` (riotlib) retourne la liste COMPLÈTE d'un tier apex (puuid +
leaguePoints pour chaque joueur classé) en 1 seul appel — récupérer le LP de tous
les joueurs apex ne coûte que 3 appels API. À relancer juste avant chaque
entraînement du modèle LP (le label LP dérive avec le temps : drift borné par la
fraîcheur du dataset, limite connue actée en spec —
docs/superpowers/specs/2026-07-07-lp-production-design.md). Promu du POC
poc/script/fetch_apex_lp.py (qui reste intact, référence historique).

Sortie : data/04_dataset/apex_lp.json =
  {"fetched_at": ISO-8601 UTC, "players": {puuid: {"tier": str, "leaguePoints": int}}}
Usage : poetry run python3 src/collection/fetch_apex_lp.py --region euw1
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import riotlib as rl

OUTPUT = rl.DATA / "04_dataset" / "apex_lp.json"
TIERS = ("challenger", "grandmaster", "master")
# Au POC, apex_league("master") a renvoyé pile 10 000 entrées — possible cap de
# l'API (non confirmé). On loggue un warning si ça se reproduit, sans bloquer :
# ~89 % des qualifiés master matchaient quand même (churn normal de tier).
SUSPECT_MASTER_COUNT = 10_000


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def build_lp_lookup(entries_by_tier: dict[str, list[dict]]) -> dict[str, dict]:
    """entries_by_tier : {"challenger": [...], "grandmaster": [...], "master": [...]}
    (format brut apex_league, une entrée par joueur classé). Retourne
    {puuid: {"tier": str, "leaguePoints": int}}. Entrées sans puuid ignorées."""
    lookup: dict[str, dict] = {}
    for tier, entries in entries_by_tier.items():
        for e in entries:
            puuid = e.get("puuid")
            if not puuid:
                continue
            lookup[puuid] = {"tier": tier, "leaguePoints": int(e.get("leaguePoints", 0))}
    return lookup


def build_payload(entries_by_tier: dict[str, list[dict]], fetched_at: str) -> dict:
    """Enveloppe du fichier de sortie : fetched_at trace la fraîcheur du label LP
    (reporté jusque dans player_lp_metrics.json)."""
    return {"fetched_at": fetched_at, "players": build_lp_lookup(entries_by_tier)}


def main() -> int:
    env = rl.load_env()
    api_key = env.get("RIOT_API_ID")
    platform = (arg("--region") or env.get("RIOT_REGION", "")).lower()
    if not api_key or not platform:
        print("✗ RIOT_API_ID + RIOT_REGION requis (--region euw1).", file=sys.stderr)
        return 1
    regional = rl.PLATFORM_TO_REGIONAL.get(platform)
    if not regional:
        print(f"✗ Région inconnue: {platform!r}", file=sys.stderr)
        return 1

    client = rl.RiotClient(api_key, regional, platform)

    entries_by_tier = {}
    for tier in TIERS:
        entries = client.apex_league(tier)
        entries_by_tier[tier] = entries
        print(f"  {tier}: {len(entries)} entrées", file=sys.stderr)
    if len(entries_by_tier.get("master", [])) == SUSPECT_MASTER_COUNT:
        print(f"  ⚠ master = {SUSPECT_MASTER_COUNT} pile — possible cap de l'API, "
              "une partie des masters peut manquer du lookup.", file=sys.stderr)

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = build_payload(entries_by_tier, fetched_at)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(f"\n✓ {len(payload['players'])} joueurs (LP courant, fetched_at={fetched_at}) "
          f"écrits dans {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `poetry run pytest tests/test_fetch_apex_lp.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/collection/fetch_apex_lp.py tests/test_fetch_apex_lp.py
git commit -m "feat(lp): fetch_apex_lp promu en prod — LP apex horodaté dans data/04_dataset"
```

---

### Task 2: `src/02_data_science/lp_metrics.py` — métriques Spearman prod

Copie prod du module POC `poc/script/lp_metrics.py` (le code prod ne doit pas importer depuis `poc/` — contrainte globale). Contenu identique au POC, docstring adaptée.

**Files:**
- Create: `src/02_data_science/lp_metrics.py`
- Test: `tests/test_lp_metrics.py`

**Interfaces:**
- Produit : `_safe_spearman(a, b) -> float | None` et `spearman_report(df: pd.DataFrame) -> dict` (df à colonnes `rank`, `y_true`, `y_pred` ; retour `{"spearman_pooled", "spearman_by_tier": {tier: {"spearman", "n"}}, "rmse_pooled", "n_players_total"}`). Consommés par Task 4.

- [ ] **Step 1: Écrire les tests (échouants)**

Créer `tests/test_lp_metrics.py` (mêmes cas que le POC, pointés sur le module prod ; `src/02_data_science` n'est pas dans `tests/conftest.py`, on insère le chemin ici) :

```python
"""Tests du module prod lp_metrics (src/02_data_science/), copie assumée du module
POC (le code prod n'importe pas depuis poc/, contrainte de la spec LP prod)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "02_data_science"))
import lp_metrics


def test_spearman_report_pooled_and_by_tier():
    df = pd.DataFrame({
        "rank": ["master"] * 4 + ["challenger"] * 4,
        "y_true": [10, 20, 30, 40, 500, 600, 700, 800],
        "y_pred": [12, 18, 33, 38, 510, 590, 710, 790],
    })
    report = lp_metrics.spearman_report(df)
    assert report["spearman_pooled"] > 0.9
    assert report["spearman_by_tier"]["master"]["spearman"] == 1.0
    assert report["spearman_by_tier"]["master"]["n"] == 4
    assert report["n_players_total"] == 8


def test_spearman_report_small_tier_and_degenerate_input_give_none():
    df = pd.DataFrame({
        "rank": ["master", "master", "challenger"],
        "y_true": [10, 20, 500],
        "y_pred": [15, 15, 510],
    })
    report = lp_metrics.spearman_report(df)
    assert report["spearman_by_tier"]["challenger"]["spearman"] is None  # n=1 < MIN_TIER_N
    assert report["spearman_by_tier"]["master"]["spearman"] is None      # y_pred constant sur le tier
    assert report["spearman_pooled"] is not None


def test_spearman_report_rmse():
    df = pd.DataFrame({"rank": ["master", "master"], "y_true": [0, 10], "y_pred": [0, 0]})
    report = lp_metrics.spearman_report(df)
    assert report["rmse_pooled"] == pytest.approx(7.07, abs=0.01)


def test_safe_spearman_none_on_constant_or_short():
    assert lp_metrics._safe_spearman([1, 1, 1], [1, 2, 3]) is None
    assert lp_metrics._safe_spearman([1, 2], [1, 2]) is None
    assert lp_metrics._safe_spearman([1, 2, 3], [10, 20, 30]) == 1.0
```

- [ ] **Step 2: Vérifier qu'ils échouent**

Run: `poetry run pytest tests/test_lp_metrics.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'lp_metrics'` — attention : ce nom de module existe aussi dans `poc/script/`, mais seul `src/02_data_science` est inséré par ce test ; `tests/test_poc_lp_regression.py` insère `poc/script` de son côté. Les deux fichiers de tests ne doivent PAS tourner dans le même process avec des attentes divergentes — le contenu étant identique, ce n'est pas un problème).

- [ ] **Step 3: Implémenter le module**

Créer `src/02_data_science/lp_metrics.py` :

```python
#!/usr/bin/env python3
"""
02_data_science — métriques de la régression LP : Spearman pooled + par tier, RMSE.

Le Spearman pooled seul peut juste redécouvrir la frontière de tier connue (un
Challenger a par définition un LP plus haut qu'un Master) : le Spearman PAR TIER
(calculé séparément à l'intérieur de master, GM, challenger) est le vrai test —
il isole si le modèle discrimine une granularité de skill au-delà du tier.
Copie prod du module POC poc/script/lp_metrics.py (le code prod n'importe pas
depuis poc/). Cf. docs/superpowers/specs/2026-07-07-lp-production-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_TIER_N = 3  # sous ce seuil, spearman non significatif -> None plutôt qu'une valeur trompeuse


def _safe_spearman(a, b) -> float | None:
    """Spearman, ou None s'il est indéfini : moins de MIN_TIER_N points, ou a/b
    constant (nunique < 2). Jamais de NaN silencieux."""
    a, b = pd.Series(a), pd.Series(b)
    if len(a) < MIN_TIER_N or a.nunique() < 2 or b.nunique() < 2:
        return None
    return round(float(spearmanr(a, b)[0]), 4)


def spearman_report(df: pd.DataFrame) -> dict:
    """df : colonnes rank (str), y_true (float), y_pred (float). Une ligne par
    joueur. Retourne spearman pooled + par tier (None si <MIN_TIER_N lignes ou
    y_true/y_pred constant sur le tier) + rmse pooled."""
    pooled_rho = _safe_spearman(df["y_true"], df["y_pred"])

    by_tier: dict[str, dict] = {}
    for tier, g in df.groupby("rank"):
        rho = _safe_spearman(g["y_true"], g["y_pred"])
        by_tier[str(tier)] = {"spearman": rho, "n": int(len(g))}

    rmse = float(np.sqrt(np.mean((df["y_true"] - df["y_pred"]) ** 2)))

    return {
        "spearman_pooled": pooled_rho,
        "spearman_by_tier": by_tier,
        "rmse_pooled": round(rmse, 2),
        "n_players_total": int(len(df)),
    }
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `poetry run pytest tests/test_lp_metrics.py tests/test_poc_lp_regression.py -v`
Expected: tous PASS (le test POC existant doit rester vert — poc/ intact)

- [ ] **Step 5: Commit**

```bash
git add src/02_data_science/lp_metrics.py tests/test_lp_metrics.py
git commit -m "feat(lp): lp_metrics prod — spearman pooled/by-tier + rmse, garde anti-NaN"
```

---

### Task 3: `src/01_data_engineering/build_player_lp_dataset.py` — dataset LP sans cap

Dataset per-player pour la régression LP : pool qualifié ≥15 games ADC référentiel, rang au mode ∈ {master, grandmaster, challenger}, joint au LP courant. **Sans balance-cap** (régression). `build_player_dataset.py` (binaire) n'est pas touché.

**Files:**
- Create: `src/01_data_engineering/build_player_lp_dataset.py`
- Test: `tests/test_build_player_lp_dataset.py`

**Interfaces:**
- Consomme : `data/04_dataset/adc_dataset.parquet` (per-game), `data/04_dataset/apex_lp.json` (format Task 1 : `{"fetched_at", "players"}`), `ml_features` (`FEATURES`, `resolve_rank`, `aggregate_player_features`).
- Produit : `data/04_dataset/adc_player_lp_dataset.parquet` (+ `.csv`) — colonnes `puuid`, `rank`, `lp` (int), agrégats `{feature}__{stat}`, `win_rate`, `n_games` ; et `data/04_dataset/adc_player_lp_dataset.meta.json` = `{"fetched_at": str, "n_dropped_no_lp": int, "n_by_tier": {tier: int}}`. Fonction pure : `build_lp_player_rows(ref, lp_players, min_games=15, features=None) -> tuple[pd.DataFrame, int]`.

- [ ] **Step 1: Écrire les tests (échouants)**

Créer `tests/test_build_player_lp_dataset.py` :

```python
"""Tests du dataset per-player LP (sans balance-cap, apex tiers seulement)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "01_data_engineering"))
import build_player_lp_dataset as bld


def _games(puuid, rank, n, csm10):
    return pd.DataFrame({
        "puuid": [puuid] * n,
        "rank": [rank] * n,
        "win": [1] * n,
        "csm10": csm10,
    })


LP = {"p1": {"tier": "master", "leaguePoints": 120},
      "p4": {"tier": "master", "leaguePoints": 40}}


def test_filters_min_games_and_excludes_diamond():
    ref = pd.concat([
        _games("p1", "master", 3, [4.0, 6.0, 8.0]),   # qualifie
        _games("p2", "master", 1, [5.0]),              # exclu : trop peu de games
        _games("p3", "diamond", 3, [4.0, 6.0, 8.0]),   # exclu : diamond hors échelle LP
    ], ignore_index=True)
    out, n_dropped = bld.build_lp_player_rows(ref, LP, min_games=2, features=["csm10"])
    assert set(out["puuid"]) == {"p1"}
    assert out.iloc[0]["lp"] == 120
    assert out.iloc[0]["csm10__mean"] == pytest.approx(6.0)
    assert n_dropped == 0


def test_drops_and_counts_players_without_current_lp():
    ref = pd.concat([
        _games("p1", "master", 2, [4.0, 6.0]),
        _games("p9", "master", 2, [5.0, 7.0]),         # qualifié mais absent du lookup LP
    ], ignore_index=True)
    out, n_dropped = bld.build_lp_player_rows(ref, LP, min_games=2, features=["csm10"])
    assert set(out["puuid"]) == {"p1"}
    assert n_dropped == 1


def test_rank_resolved_by_mode_across_all_games():
    ref = pd.concat([
        _games("p4", "master", 2, [4.0, 6.0]),
        _games("p4", "diamond", 1, [5.0]),
    ], ignore_index=True)
    out, _ = bld.build_lp_player_rows(ref, LP, min_games=2, features=["csm10"])
    assert list(out["rank"]) == ["master"]  # mode : 2 master > 1 diamond


def test_no_balance_cap():
    # 3 masters vs 1 challenger : une régression garde tout le monde (pas d'undersampling)
    lp = {p: {"tier": "master", "leaguePoints": 10} for p in ("a", "b", "c", "d")}
    ref = pd.concat([
        _games("a", "master", 2, [4.0, 6.0]),
        _games("b", "master", 2, [4.0, 6.0]),
        _games("c", "master", 2, [4.0, 6.0]),
        _games("d", "challenger", 2, [4.0, 6.0]),
    ], ignore_index=True)
    out, _ = bld.build_lp_player_rows(ref, lp, min_games=2, features=["csm10"])
    assert len(out) == 4
```

- [ ] **Step 2: Vérifier qu'ils échouent**

Run: `poetry run pytest tests/test_build_player_lp_dataset.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implémenter le script**

Créer `src/01_data_engineering/build_player_lp_dataset.py` :

```python
#!/usr/bin/env python3
"""
01_data_engineering — dataset per-player pour la régression LP (Master/GM/Chall).

Comme build_player_dataset.py (binaire) mais SANS balance-cap : une régression n'a
pas de classes à équilibrer, donc on garde TOUS les joueurs qualifiés (le cap du
binaire jette ~40 % des masters qualifiés — c'est précisément le pool que le modèle
LP récupère, cf. spec). Restreint aux tiers apex (diamond exclu : divisions I-IV
avec reset, LP non comparable à l'échelle continue master→challenger).

Label : LP courant depuis data/04_dataset/apex_lp.json (fetch_apex_lp.py, à relancer
avant ce script pour un label frais). Joueurs qualifiés absents du lookup = tier
changé depuis la collecte → droppés et comptés (n_dropped_no_lp).

0 appel API. Sorties : data/04_dataset/adc_player_lp_dataset.parquet (+ .csv)
et adc_player_lp_dataset.meta.json (fetched_at du label, n_dropped_no_lp, n_by_tier).
Usage : poetry run python3 src/01_data_engineering/build_player_lp_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import pandas as pd
import riotlib as rl
import ml_features as mf

DATASET_DIR = rl.DATA / "04_dataset"
LP_PATH = DATASET_DIR / "apex_lp.json"
MIN_PLAYER_GAMES = 15
APEX_TIERS = {"master", "grandmaster", "challenger"}


def build_lp_player_rows(ref: pd.DataFrame, lp_players: dict,
                         min_games: int = MIN_PLAYER_GAMES,
                         features: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """ref : rows per-game référentiel (colonnes puuid, rank, win, + features).
    lp_players : {puuid: {"tier", "leaguePoints"}} (clé "players" d'apex_lp.json).
    Retourne (1 ligne par joueur qualifié apex AVEC LP courant — colonnes puuid,
    rank, lp, agrégats + win_rate + n_games ; nombre de qualifiés droppés faute de
    LP courant). PAS de balance-cap. Rang résolu au mode sur tout l'historique
    (même sémantique que le binaire, cf. ml_features.resolve_rank)."""
    features = mf.FEATURES if features is None else features
    rows, n_dropped = [], 0
    for puuid, g in ref.groupby("puuid"):
        if len(g) < min_games:
            continue
        rank = mf.resolve_rank(g)
        if rank not in APEX_TIERS:
            continue
        entry = lp_players.get(puuid)
        if entry is None:
            n_dropped += 1
            continue
        rec = {"puuid": puuid, "rank": rank, "lp": int(entry["leaguePoints"])}
        rec.update(mf.aggregate_player_features(g, features))
        rows.append(rec)
    return pd.DataFrame(rows), n_dropped


def main() -> int:
    if not LP_PATH.exists():
        print(f"✗ {LP_PATH} introuvable — lancer "
              "src/collection/fetch_apex_lp.py d'abord.", file=sys.stderr)
        return 1
    lp_file = json.loads(LP_PATH.read_text())

    df = pd.read_parquet(DATASET_DIR / "adc_dataset.parquet")
    ref = df[df["source"] == "referentiel"].copy()
    print(f"  {len(ref)} games référentiel | {ref['puuid'].nunique()} joueurs uniques")
    print(f"  label LP : {len(lp_file['players'])} joueurs apex, "
          f"fetched_at={lp_file['fetched_at']}")

    out, n_dropped = build_lp_player_rows(ref, lp_file["players"])
    print(f"  >= {MIN_PLAYER_GAMES} games apex avec LP courant : {len(out)} joueurs "
          f"({n_dropped} qualifiés droppés, tier changé depuis la collecte)")
    if out.empty:
        print("  ⚠ aucun joueur ne qualifie -> rien à écrire")
        return 1
    n_by_tier = {k: int(v) for k, v in out["rank"].value_counts().items()}
    print(f"  répartition tiers : {n_by_tier}")

    out.to_parquet(DATASET_DIR / "adc_player_lp_dataset.parquet", index=False)
    out.to_csv(DATASET_DIR / "adc_player_lp_dataset.csv", index=False)
    (DATASET_DIR / "adc_player_lp_dataset.meta.json").write_text(json.dumps({
        "fetched_at": lp_file["fetched_at"],
        "n_dropped_no_lp": n_dropped,
        "n_by_tier": n_by_tier,
    }, indent=2))
    print(f"\n✓ Dataset LP per-player écrit dans {DATASET_DIR}/adc_player_lp_dataset.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `poetry run pytest tests/test_build_player_lp_dataset.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/01_data_engineering/build_player_lp_dataset.py tests/test_build_player_lp_dataset.py
git commit -m "feat(lp): dataset per-player LP sans balance-cap (apex tiers, label LP courant)"
```

---

### Task 4: `src/02_data_science/train_player_lp.py` — ensemble tuné en purged CV

Cœur du chantier : random search (graine fixe) par modèle (xgb 40 configs / rf 20 / ebm 8) en purged CV 5 folds, **purge précalculée une fois par fold**, sélection au Spearman pooled OOF, ensemble = moyenne des 3 meilleurs, refit final + SHAP + métriques.

**Files:**
- Create: `src/02_data_science/train_player_lp.py`
- Test: `tests/test_train_player_lp.py`

**Interfaces:**
- Consomme : `adc_player_lp_dataset.parquet` + `.meta.json` (Task 3), `adc_dataset.parquet` (purge), `lp_metrics` (Task 2 : `_safe_spearman`, `spearman_report`), `train_player_ensemble.purged_train_features(ref, train_puuids, val_puuids, features=...) -> (DataFrame, list)` et `train_player_ensemble.dispersion_share_analysis(per_feature: dict) -> dict` (existants, non modifiés), `ml_features` (`FEATURES`, `player_feature_names`).
- Produit : `data/05_model/{xgb,rf,ebm}_player_lp.pkl`, `player_lp_features.json` (liste ordonnée des colonnes), `player_lp_metrics.json`. Fonctions pures : `sample_configs(grid: dict, n: int, seed: int = 42) -> list[dict]`, `search_best(name, spec, folds, y_true, n_rows) -> dict` (clés `config`, `spearman`, `oof`).

- [ ] **Step 1: Écrire les tests (échouants)**

Créer `tests/test_train_player_lp.py` :

```python
"""Tests des fonctions pures du train LP (sample_configs, search_best). La CV
complète et le SHAP sont vérifiés par exécution réelle (Task 6 du plan)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "02_data_science"))
import train_player_lp as tlp


def test_sample_configs_deterministic_and_within_grid():
    grid = {"a": [1, 2, 3], "b": [10, 20]}
    c1 = tlp.sample_configs(grid, n=4, seed=42)
    c2 = tlp.sample_configs(grid, n=4, seed=42)
    assert c1 == c2                      # déterministe à graine fixe
    assert len(c1) == 4
    assert len({tuple(sorted(c.items())) for c in c1}) == 4   # sans doublon
    for c in c1:
        assert c["a"] in grid["a"] and c["b"] in grid["b"]


def test_sample_configs_returns_full_product_when_small():
    grid = {"a": [1, 2], "b": [10]}
    configs = tlp.sample_configs(grid, n=50, seed=42)
    assert len(configs) == 2             # produit cartésien < n -> tout


def test_search_best_picks_highest_spearman(monkeypatch):
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fake_oof = {
        (("depth", 1),): np.array([5.0, 4.0, 3.0, 2.0, 1.0]),  # spearman -1
        (("depth", 2),): np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # spearman +1
    }
    monkeypatch.setattr(
        tlp, "oof_predictions",
        lambda name, config, folds, n_rows: fake_oof[tuple(sorted(config.items()))])
    spec = {"n_configs": 2, "grid": {"depth": [1, 2]}}
    best = tlp.search_best("xgb", spec, folds=[], y_true=y_true, n_rows=5)
    assert best["config"] == {"depth": 2}
    assert best["spearman"] == 1.0


def test_search_best_survives_degenerate_predictions(monkeypatch):
    # une config qui prédit une constante (spearman indéfini -> None) ne doit ni
    # crasher ni gagner face à une config avec un vrai spearman
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fake_oof = {
        (("depth", 1),): np.zeros(5),                            # constant -> None
        (("depth", 2),): np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    monkeypatch.setattr(
        tlp, "oof_predictions",
        lambda name, config, folds, n_rows: fake_oof[tuple(sorted(config.items()))])
    spec = {"n_configs": 2, "grid": {"depth": [1, 2]}}
    best = tlp.search_best("xgb", spec, folds=[], y_true=y_true, n_rows=5)
    assert best["config"] == {"depth": 2}
```

- [ ] **Step 2: Vérifier qu'ils échouent**

Run: `poetry run pytest tests/test_train_player_lp.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implémenter le script**

Créer `src/02_data_science/train_player_lp.py` :

```python
#!/usr/bin/env python3
"""
02_data_science — régression LP per-player (Master/GM/Challenger), ensemble tuné.

Prédit le LP courant d'un joueur apex depuis ses features agrégées
(mean/std/p10/p50/p90 + win_rate, cf. ml_features). Suite prod du POC
poc/script/train_lp_regression.py (spearman pooled 0.5028 avec un XGBRegressor
unique non tuné — baseline rappelée dans les métriques pour mesurer le gain).

Optimisation : random search à graine fixe par modèle (xgb ~40 configs, rf ~20,
ebm ~8) en PURGED CV 5 folds (StratifiedKFold sur le TIER — y est continu, on
stratifie la catégorie pour équilibrer master/GM/chall par fold). La purge
(agrégats de train recalculés sans les matchs partagés avec la val, cf.
train_player_ensemble.purged_train_features : ~37 % des games opposent 2 ADC du
dataset, features en miroir) ne dépend QUE du découpage en folds, pas des
hyperparamètres → précalculée UNE FOIS par fold et réutilisée pour toutes les
configs (sinon chaque config repayerait le recalcul des agrégats, ~×70 le coût).
Critère de sélection : Spearman pooled OOF (le within-tier reste la métrique de
REPORTING décisive mais trop bruitée par tier pour piloter une recherche — GM
n'a que ~78 joueurs).

N'écrase AUCUN artefact du pipeline binaire : marqueur "player_lp" partout.
Durée attendue : quelques minutes (xgb/rf) + ~10-30 min pour les 8 configs EBM.

Sorties : data/05_model/{xgb,rf,ebm}_player_lp.pkl, player_lp_features.json,
player_lp_metrics.json
Usage : poetry run python3 src/02_data_science/train_player_lp.py
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # lp_metrics, train_player_ensemble
import numpy as np
import pandas as pd
import riotlib as rl
import ml_features as mf
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from interpret.glassbox import ExplainableBoostingRegressor

from lp_metrics import _safe_spearman, spearman_report
from train_player_ensemble import purged_train_features, dispersion_share_analysis

DATASET = rl.DATA / "04_dataset" / "adc_player_lp_dataset.parquet"
DATASET_META = rl.DATA / "04_dataset" / "adc_player_lp_dataset.meta.json"
DATASET_PER_GAME = rl.DATA / "04_dataset" / "adc_dataset.parquet"  # pour la purge
MODEL_DIR = rl.DATA / "05_model"
SEED = 42
POC_BASELINE_SPEARMAN_POOLED = 0.5028

GRIDS = {
    "xgb": {
        "n_configs": 40,
        "grid": {
            "max_depth": [2, 3, 4],
            "n_estimators": [200, 300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "min_child_weight": [3, 5, 10],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_lambda": [0.5, 1.0, 3.0],
        },
    },
    "rf": {
        "n_configs": 20,
        "grid": {
            "n_estimators": [300, 500],
            "max_depth": [None, 8, 12],
            "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt", 0.3, 0.5],
        },
    },
    "ebm": {
        # EBM est lent à fitter (~minutes/config) : budget volontairement réduit
        "n_configs": 8,
        "grid": {
            "max_bins": [128, 256],
            "interactions": [0, 10, 20],
            "learning_rate": [0.01, 0.02],
        },
    },
}


def sample_configs(grid: dict, n: int, seed: int = SEED) -> list[dict]:
    """n combinaisons distinctes tirées uniformément du produit cartésien de grid
    (toutes si le produit est <= n). Déterministe à graine fixe (clés triées)."""
    keys = sorted(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    if len(combos) > n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(combos), size=n, replace=False)
        combos = [combos[i] for i in idx]
    return [dict(zip(keys, c)) for c in combos]


def make_model(name: str, config: dict):
    if name == "xgb":
        return xgb.XGBRegressor(tree_method="hist", random_state=SEED, **config)
    if name == "rf":
        return RandomForestRegressor(n_jobs=-1, random_state=SEED, **config)
    if name == "ebm":
        return ExplainableBoostingRegressor(random_state=SEED, **config)
    raise ValueError(f"modèle inconnu: {name!r}")


def prepare_folds(df: pd.DataFrame, ref: pd.DataFrame,
                  features: list[str]) -> list[tuple]:
    """Précalcule par fold : (X_train purgé, y_train, X_val, val_idx). Les agrégats
    purgés ne dépendent pas des hyperparamètres — calculés UNE fois, réutilisés
    pour toutes les configs du random search."""
    X = df.reindex(columns=features)
    y_of = dict(zip(df["puuid"], df["lp"].astype(float)))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for train_idx, val_idx in cv.split(X, df["rank"]):
        train_puuids = df["puuid"].iloc[train_idx].tolist()
        val_puuids = set(df["puuid"].iloc[val_idx])
        Xtr, dropped = purged_train_features(ref, train_puuids, val_puuids,
                                             features=mf.FEATURES)
        if dropped:
            print(f"    (purge : {len(dropped)} joueurs de train droppés sur ce fold)")
        y_train = Xtr["puuid"].map(y_of).astype(float)
        folds.append((Xtr.reindex(columns=features), y_train, X.iloc[val_idx], val_idx))
    return folds


def oof_predictions(name: str, config: dict, folds: list[tuple],
                    n_rows: int) -> np.ndarray:
    oof = np.zeros(n_rows)
    for X_train, y_train, X_val, val_idx in folds:
        model = make_model(name, config)
        model.fit(X_train, y_train)
        oof[val_idx] = model.predict(X_val)
    return oof


def search_best(name: str, spec: dict, folds: list[tuple],
                y_true: np.ndarray, n_rows: int) -> dict:
    """Random search : retourne {"config", "spearman", "oof"} de la meilleure config
    au Spearman pooled OOF. Une config au spearman indéfini (prédiction constante)
    est traitée comme -1 (ne gagne jamais face à un vrai score)."""
    best = None
    for config in sample_configs(spec["grid"], spec["n_configs"]):
        oof = oof_predictions(name, config, folds, n_rows)
        rho = _safe_spearman(y_true, oof)
        rho = -1.0 if rho is None else rho
        if best is None or rho > best["spearman"]:
            best = {"config": config, "spearman": rho, "oof": oof}
    return best


def shap_top20(model, X: pd.DataFrame) -> dict:
    """SHAP TreeExplainer sur le XGB final : top-20 features + part de dispersion
    (std/p10/p90 vs mean/p50) — vérifie si l'hypothèse constance tient sur la
    cible LP comme elle tenait sur le binaire (~58-62 % du signal)."""
    vals = np.abs(shap.TreeExplainer(model).shap_values(X))
    per_feat = dict(zip(X.columns, vals.mean(axis=0).tolist()))
    disp = dispersion_share_analysis(per_feat)
    top20 = sorted(per_feat.items(), key=lambda kv: kv[1], reverse=True)[:20]
    return {
        "shap_share_by_stat": disp["share_by_stat"],
        "shap_dispersion_share": disp["dispersion_share_of_signal"],
        "top20_shap": [{"feature": k, "mean_abs_shap": round(v, 5)} for k, v in top20],
    }


def main() -> int:
    df = pd.read_parquet(DATASET)
    meta = json.loads(DATASET_META.read_text()) if DATASET_META.exists() else {}
    ref = pd.read_parquet(DATASET_PER_GAME)
    ref = ref[(ref["source"] == "referentiel")
              & ref["puuid"].isin(set(df["puuid"]))].copy()
    features = mf.player_feature_names(mf.FEATURES)
    X = df.reindex(columns=features)
    y = df["lp"].astype(float).values
    print(f"  {len(df)} joueurs | tiers : {df['rank'].value_counts().to_dict()} | "
          f"{len(ref)} games per-game pour la purge | "
          f"label fetched_at={meta.get('fetched_at', '?')}")

    print("\n  Précalcul des folds purgés (1 fois, réutilisés par toutes les configs)…")
    folds = prepare_folds(df, ref, features)

    best, per_model = {}, {}
    for name, spec in GRIDS.items():
        print(f"\n  Random search {name} ({spec['n_configs']} configs max)…")
        best[name] = search_best(name, spec, folds, y, len(df))
        per_model[name] = {"spearman_pooled": round(best[name]["spearman"], 4),
                           "best_config": {k: (v if v is None or isinstance(v, (int, float, str))
                                               else str(v))
                                           for k, v in best[name]["config"].items()}}
        print(f"    -> spearman={best[name]['spearman']:.4f}  "
              f"config={best[name]['config']}")

    ens_oof = np.mean([best[n]["oof"] for n in GRIDS], axis=0)
    report = spearman_report(pd.DataFrame({
        "rank": df["rank"].values, "y_true": y, "y_pred": ens_oof}))
    print(f"\n  Ensemble OOF (purgé) : spearman pooled = {report['spearman_pooled']}  "
          f"(baseline POC {POC_BASELINE_SPEARMAN_POOLED})  rmse={report['rmse_pooled']}")
    for tier, r in report["spearman_by_tier"].items():
        print(f"    {tier:<12} spearman={r['spearman']}  n={r['n']}")

    print("\n  Refit final sur 100 % du dataset…")
    final_models = {name: make_model(name, best[name]["config"]) for name in GRIDS}
    for model in final_models.values():
        model.fit(X, df["lp"].astype(float))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in final_models.items():
        with open(MODEL_DIR / f"{name}_player_lp.pkl", "wb") as f:
            pickle.dump(model, f)
    (MODEL_DIR / "player_lp_features.json").write_text(json.dumps(features, indent=2))

    shap_block = shap_top20(final_models["xgb"], X.fillna(X.median()))
    print(f"  dispersion (std/p10/p90) = {shap_block['shap_dispersion_share']:.1%} "
          "du signal SHAP (xgb)")

    (MODEL_DIR / "player_lp_metrics.json").write_text(json.dumps({
        **report,
        "per_model_cv": per_model,
        "n_players_by_tier": {k: int(v) for k, v in df["rank"].value_counts().items()},
        "n_dropped_no_lp": meta.get("n_dropped_no_lp"),
        "lp_fetched_at": meta.get("fetched_at"),
        "poc_baseline_spearman_pooled": POC_BASELINE_SPEARMAN_POOLED,
        "features": features,
        "shap": shap_block,
    }, indent=2))

    print(f"\n✓ Modèles LP écrits dans {MODEL_DIR}/ (marqueur 'player_lp')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `poetry run pytest tests/test_train_player_lp.py -v`
Expected: 4 PASS

- [ ] **Step 5: Vérifier la non-régression du reste**

Run: `poetry run pytest tests/ -q`
Expected: tout PASS (299+ tests)

- [ ] **Step 6: Commit**

```bash
git add src/02_data_science/train_player_lp.py tests/test_train_player_lp.py
git commit -m "feat(lp): train_player_lp — ensemble xgb/rf/ebm regressors, random search en purged CV"
```

---

### Task 5: Serving web hybride — `ml_rank.py` + frontend

Ajoute `predicted_lp` au retour de `predict_rank()` quand le rang placé est apex ET que les artefacts LP existent (dégradation propre sinon). Frontend : affiche « · ~N LP estimés » dans le bandeau du rang ML.

**Files:**
- Modify: `web/backend/ml_rank.py` (fin de `predict_rank`, ~ligne 89, + nouveaux helpers)
- Modify: `web/frontend/index.html` (bandeau rang ML, lignes 110-116)
- Test: `tests/test_ml_rank_lp.py`

**Interfaces:**
- Consomme : `data/05_model/{xgb,rf,ebm}_player_lp.pkl` + `player_lp_features.json` (Task 4), l'agrégat `agg` déjà calculé dans `predict_rank`.
- Produit : `predict_rank()` retourne en plus `"predicted_lp": int` (≥0) quand applicable. Helpers : `_load_lp_bundle() -> tuple[list, list[str]] | None` (lru_cache ; None si artefacts absents), `predict_lp(agg: dict) -> int | None`, `attach_lp(result: dict, agg: dict) -> dict`. Le router `web/backend/routers/predicted_rank.py` n'a PAS besoin de changer (il renvoie le dict tel quel).

- [ ] **Step 1: Écrire les tests (échouants)**

Créer `tests/test_ml_rank_lp.py` :

```python
"""Tests du chemin LP hybride de web/backend/ml_rank.py (helpers purs, modèles
mockés — le placement binaire existant n'est pas re-testé ici)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web" / "backend"))
import ml_rank


class _FakeReg:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value]


_AGG = {"csm10__mean": 6.0}
_FEATURES = ["csm10__mean"]


def test_predict_lp_none_when_models_missing(monkeypatch):
    monkeypatch.setattr(ml_rank, "_load_lp_bundle", lambda: None)
    assert ml_rank.predict_lp(_AGG) is None


def test_predict_lp_averages_ensemble_and_rounds(monkeypatch):
    bundle = ([_FakeReg(100.0), _FakeReg(200.0), _FakeReg(310.0)], _FEATURES)
    monkeypatch.setattr(ml_rank, "_load_lp_bundle", lambda: bundle)
    assert ml_rank.predict_lp(_AGG) == 203   # mean(100, 200, 310) = 203.33 -> round


def test_predict_lp_clamped_at_zero(monkeypatch):
    bundle = ([_FakeReg(-50.0)], _FEATURES)
    monkeypatch.setattr(ml_rank, "_load_lp_bundle", lambda: bundle)
    assert ml_rank.predict_lp(_AGG) == 0


def test_attach_lp_only_for_apex_ranks(monkeypatch):
    bundle = ([_FakeReg(250.0)], _FEATURES)
    monkeypatch.setattr(ml_rank, "_load_lp_bundle", lambda: bundle)
    apex = ml_rank.attach_lp({"predicted_rank": "master"}, _AGG)
    assert apex["predicted_lp"] == 250
    diamond = ml_rank.attach_lp({"predicted_rank": "diamond"}, _AGG)
    assert "predicted_lp" not in diamond


def test_attach_lp_graceful_without_models(monkeypatch):
    monkeypatch.setattr(ml_rank, "_load_lp_bundle", lambda: None)
    result = ml_rank.attach_lp({"predicted_rank": "challenger"}, _AGG)
    assert "predicted_lp" not in result
    assert result["predicted_rank"] == "challenger"   # rien d'autre ne change
```

- [ ] **Step 2: Vérifier qu'ils échouent**

Run: `poetry run pytest tests/test_ml_rank_lp.py -v`
Expected: FAIL (`AttributeError: module 'ml_rank' has no attribute '_load_lp_bundle'`)

- [ ] **Step 3: Modifier `web/backend/ml_rank.py`**

Trois ajouts. (a) Après la fonction `_load_calibration()` (~ligne 67), insérer :

```python
APEX_RANKS = {"master", "grandmaster", "challenger"}


@functools.lru_cache(maxsize=1)
def _load_lp_bundle() -> tuple[list, list[str]] | None:
    """Regressors LP + ordre des features, ou None si les artefacts sont absents
    (modèle LP pas encore entraîné sur cette machine) — dégradation propre : le
    placement binaire suffit, pas de crash ni de log bruyant."""
    try:
        models = []
        for name in ("xgb", "rf", "ebm"):
            with open(MODEL_DIR / f"{name}_player_lp.pkl", "rb") as f:
                models.append(pickle.load(f))
        features = json.loads((MODEL_DIR / "player_lp_features.json").read_text())
        return models, features
    except FileNotFoundError:
        return None


def predict_lp(agg: dict) -> int | None:
    """LP estimé (moyenne de l'ensemble, arrondi, borné >= 0) depuis l'agrégat de
    features per-player déjà calculé par predict_rank. None si modèles absents.
    N'a de sens que pour un joueur placé apex (échelle LP continue master->chall,
    diamond hors échelle — divisions avec reset)."""
    bundle = _load_lp_bundle()
    if bundle is None:
        return None
    models, features = bundle
    X = pd.DataFrame([agg]).reindex(columns=features).astype(float)
    preds = [float(m.predict(X)[0]) for m in models]
    return max(0, round(sum(preds) / len(preds)))


def attach_lp(result: dict, agg: dict) -> dict:
    """Ajoute predicted_lp au retour de predict_rank quand le rang placé est apex
    ET que le modèle LP est disponible. Purement additif : ne touche à rien d'autre."""
    if result.get("predicted_rank") in APEX_RANKS:
        lp = predict_lp(agg)
        if lp is not None:
            result["predicted_lp"] = lp
    return result
```

(b) Dans `predict_rank`, remplacer le `return` final (lignes 90-94) :

```python
    return {
        "predicted_rank": closest["rank"],
        "proba": round(player_proba, 4),
        "n_games_used": len(adc_games),
    }
```

par :

```python
    return attach_lp({
        "predicted_rank": closest["rank"],
        "proba": round(player_proba, 4),
        "n_games_used": len(adc_games),
    }, agg)
```

(c) Compléter la docstring de module (fin du bloc existant, après la ligne sur MIN_ADC_GAMES) avec :

```python
Hybride LP (2026-07-07) : si le rang placé est apex (master/GM/chall) et que les
regressors LP ({xgb,rf,ebm}_player_lp.pkl, cf. train_player_lp.py) sont présents,
le retour porte en plus "predicted_lp" (LP estimé sur l'échelle continue
master->challenger). Diamond n'en a jamais (divisions avec reset, hors échelle).
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `poetry run pytest tests/test_ml_rank_lp.py -v`
Expected: 5 PASS

- [ ] **Step 5: Modifier le frontend**

Dans `web/frontend/index.html`, le bloc du bandeau rang ML (lignes 110-116) :

```html
      <template x-if="predictedRank && predictedRank.predicted_rank">
        <div class="meta-strip row wrap">
          <span>🤖 Rang ML estimé (<span x-text="predictedRank.n_games_used"></span> dernières games ADC) :</span>
          <span class="badge" x-text="rankTierLabel(predictedRank.predicted_rank)"></span>
          <span class="faint" x-text="'confiance ' + Math.round(predictedRank.proba * 100) + '%'"></span>
        </div>
      </template>
```

devient (une ligne ajoutée entre le badge et la confiance) :

```html
      <template x-if="predictedRank && predictedRank.predicted_rank">
        <div class="meta-strip row wrap">
          <span>🤖 Rang ML estimé (<span x-text="predictedRank.n_games_used"></span> dernières games ADC) :</span>
          <span class="badge" x-text="rankTierLabel(predictedRank.predicted_rank)"></span>
          <span class="faint" x-show="predictedRank.predicted_lp != null"
                x-text="'· ~' + predictedRank.predicted_lp + ' LP estimés'"></span>
          <span class="faint" x-text="'confiance ' + Math.round(predictedRank.proba * 100) + '%'"></span>
        </div>
      </template>
```

(`!= null` couvre `undefined` ET `null` en JS — champ absent → span caché, affichage actuel inchangé.)

- [ ] **Step 6: Non-régression complète**

Run: `poetry run pytest tests/ -q`
Expected: tout PASS

- [ ] **Step 7: Commit**

```bash
git add web/backend/ml_rank.py web/frontend/index.html tests/test_ml_rank_lp.py
git commit -m "feat(lp): serving hybride — predicted_lp pour les rangs apex, affichage web"
```

---

### Task 6: Exécution réelle du pipeline + documentation

Pas de nouveau code : exécution bout-en-bout, lecture des résultats, mise à jour de CLAUDE.md.

**Files:**
- Modify: `CLAUDE.md` (section « Rang ML per-player » de l'État d'avancement + description de `src/02_data_science/` dans l'architecture)

**Interfaces:**
- Consomme : tout ce qui précède.
- Produit : artefacts data (non commités) + verdict chiffré + doc à jour.

- [ ] **Step 1: Fetch LP**

Run: `poetry run python3 src/collection/fetch_apex_lp.py --region euw1`
Expected: `challenger: ~300 entrées`, `grandmaster: ~700-1000`, `master: plusieurs milliers` (warning si pile 10 000), `✓ ~11000 joueurs … écrits dans data/04_dataset/apex_lp.json`

- [ ] **Step 2: Build dataset LP**

Run: `poetry run python3 src/01_data_engineering/build_player_lp_dataset.py`
Expected: `~1150 joueurs` retenus (ordre de grandeur POC : 1278 qualifiés − ~130 sans LP courant), répartition ≈ {master: ~700, challenger: ~370, grandmaster: ~80}, meta.json écrit.

- [ ] **Step 3: Train (long — EBM ~10-30 min)**

Run: `poetry run python3 src/02_data_science/train_player_lp.py`
Expected: spearman pooled OOF de l'ensemble **≥ 0.5028** (baseline POC — si en dessous, STOP : investiguer avant de servir, ne pas passer à la suite), spearman by_tier affichés, 3 `.pkl` + 2 `.json` écrits dans `data/05_model/`.

- [ ] **Step 4: Vérifier le serving**

Run: `poetry run python3 -c "
import sys; sys.path.insert(0, 'web/backend'); sys.path.insert(0, 'src/core')
import ml_rank
b = ml_rank._load_lp_bundle()
assert b is not None, 'bundle LP introuvable'
print('modèles LP chargés :', len(b[0]), '| features :', len(b[1]))
print('predict_lp(agg vide) =', ml_rank.predict_lp({}))
"`
Expected: `modèles LP chargés : 3 | features : 212` et un LP entier ≥ 0 (prédiction sur agrégat vide = NaN partout, les 3 modèles gèrent le NaN à l'inférence ; si RF/EBM refusent le NaN à l'inférence, le vérifier plutôt avec un joueur réel via l'API web et noter le comportement dans le rapport).

- [ ] **Step 5: Mettre à jour CLAUDE.md**

Dans la section « État d'avancement », après le bloc « **Rang ML per-player (constance)** ✅ … », ajouter :

```markdown
- **Régression LP (hybride, apex tiers)** ✅ — 2026-07-07. Suite prod du POC LP
  (signal within-tier confirmé : master 0.38, GM 0.55, chall 0.60). Pipeline :
  `fetch_apex_lp.py` (3 appels API, LP courant horodaté) →
  `build_player_lp_dataset.py` (per-player SANS balance-cap, apex seulement,
  diamond exclu — LP non comparable) → `train_player_lp.py` (ensemble
  xgb/rf/ebm REGRESSORS, random search graine fixe en purged CV — purge
  précalculée par fold —, sélection au Spearman pooled OOF, SHAP, baseline POC
  0.5028 rappelée dans `player_lp_metrics.json`). Serving hybride :
  `ml_rank.predict_rank` ajoute `predicted_lp` (moyenne ensemble, ≥0) quand le
  rang placé est apex et que les `.pkl` LP existent (dégradation propre sinon) ;
  le placement binaire 4 rangs est inchangé. Drift temporel du label LP (fetch
  au train vs games jusqu'à ~13 j) = limite connue actée. Spec :
  `docs/superpowers/specs/2026-07-07-lp-production-design.md`.
```

Et dans la description de `src/02_data_science/` (architecture du code), après la phrase sur `calibrate_player_rank.py`, ajouter :

```markdown
    `train_player_lp.py` — régression LP per-player (apex tiers, cf. bloc
    « Régression LP » de l'État d'avancement) ; `lp_metrics.py` — Spearman
    pooled/by-tier + RMSE (garde anti-NaN `_safe_spearman`).
```

- [ ] **Step 6: Reporter les chiffres réels dans CLAUDE.md**

Compléter le bloc ajouté au Step 5 avec les métriques réellement obtenues au Step 3 (spearman pooled + by_tier + n_players), à la place d'aucun chiffre inventé — copier depuis `data/05_model/player_lp_metrics.json`.

- [ ] **Step 7: Non-régression finale + commit**

```bash
poetry run pytest tests/ -q
git add CLAUDE.md
git commit -m "docs: pipeline LP prod exécuté — métriques réelles dans CLAUDE.md"
```

---

## Self-Review (faite à l'écriture du plan)

- **Couverture spec** : fetch (Task 1), dataset sans cap (Task 3), ensemble+tuning+purge précalculée+SHAP+métriques (Task 4), hybride web+frontend (Task 5), exécution+seuil baseline+doc (Task 6), lp_metrics prod sans import poc (Task 2). Le spec mentionnait « tests/test_ml_rank.py (existant, étendu) » — ce fichier n'existe PAS ; le plan crée `tests/test_ml_rank_lp.py` à la place (correction d'une erreur factuelle du spec, pas un écart de scope).
- **Placeholders** : aucun ; chaque step code contient le code complet.
- **Cohérence de types** : `build_lp_player_rows -> (DataFrame, int)` (Task 3) ; `apex_lp.json = {"fetched_at", "players"}` produit en Task 1 et consommé en Task 3 ; `search_best -> {"config", "spearman", "oof"}` (Task 4) ; `_load_lp_bundle -> (models, features) | None` (Task 5, mocké dans les tests avec un tuple).
