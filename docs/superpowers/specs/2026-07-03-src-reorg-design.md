# Spec — Réorganisation de src/

Date : 2026-07-03
Statut : approuvé

## Objectif

`src/` a accumulé 14 scripts au niveau racine (collecte API, maintenance médaillon,
libs partagées, reporting, un vieux spike) mélangés sans structure, à côté de 4 dossiers
numérotés déjà bien organisés (`01_data_engineering` → `04_coaching`, pipeline ML récent).
Regrouper le vrac par rôle pour que la structure du dossier reflète l'architecture réelle,
sans toucher au pipeline ML déjà propre.

## Périmètre

- Seulement `src/` + les points d'intégration qui importent dedans (`web/backend/main.py`,
  `tests/conftest.py`, `tests/web/conftest.py`) + la doc (`CLAUDE.md`).
- `01_data_engineering/`, `02_data_science/`, `03_data_analyse/`, `04_coaching/` gardent
  leurs noms et emplacements — seul le chemin qu'ils importent pour les libs partagées change.
- `poc/`, `scrap_diamond*.log` (racine du repo, non trackés) : hors périmètre.
- Aucun changement de comportement/logique — uniquement des déplacements de fichiers et
  des ajustements de chemins d'import.

## Structure cible

```
src/
  core/                        # libs partagées, importées par (presque) tout le reste
    riotlib.py
    positioning.py
    champion_profiles.py
  collection/                  # scripts qui APPELLENT l'API Riot / Live Client
    build_referential.py
    aggregate_games.py
    live_capture.py            # reste 0-dépendance stdlib, juste rangé
  pipeline_ops/                # maintenance médaillon, 0 appel API
    reextract_silver.py
    rebuild_gold.py
    compress_raw.py
    archive_patch.py
    list_unknown_champions.py
  reporting/                   # livrable heuristique pré-ML (Phase 1)
    compare.py
  experiments/                 # spikes / prototypes historiques
    phase1_pull.py
  01_data_engineering/          # inchangé
    build_dataset.py
  02_data_science/               # inchangé + audit_leakage.py ajouté (même famille
    train_ensemble.py            # que train_ensemble/calibrate_rank : diagnostic ML)
    calibrate_rank.py
    audit_leakage.py
  03_data_analyse/              # inchangé
    plot_custom_shap.py
    plotter.py
    shap_analysis.py
  04_coaching/                  # inchangé
    coach.py
    feedback.py
    llm_client.py
    payload.py
    prompt.py
    schema.py
```

Aucun `__init__.py` créé — cohérent avec l'état actuel (`src/` n'a aucun package Python,
tout repose sur `sys.path` + imports plats).

## Mécanique des imports

Convention déjà établie dans le repo, réutilisée telle quelle (pas de nouveau style
introduit) : chaque script qui a besoin de `riotlib`/`positioning`/`champion_profiles`
insère leur dossier dans `sys.path` puis fait un `import riotlib` plat.

- **Fichiers déplacés dans `collection/`, `pipeline_ops/`, `reporting/`, `experiments/`** :
  ajout de la ligne (identique au pattern déjà présent dans
  `01_data_engineering/build_dataset.py` et `04_coaching/payload.py`) :
  ```python
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
  ```
- **Fichiers déjà dans `01_data_engineering/`, `02_data_science/`, `03_data_analyse/`,
  `04_coaching/`** qui font `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` :
  le chemin pointait vers `src/` (où vivait `riotlib.py`) → devient
  `.../parent.parent / "core"`.
- **`riotlib.py`** importe déjà `champion_profiles` en plat (`import champion_profiles as cp`)
  et `positioning` en lazy-import (évite un cycle) — les deux restent dans le même dossier
  `core/`, donc ces imports internes ne changent pas.
- **`web/backend/main.py`** : ajout d'un 4e insert, même schéma que l'existant `COACH` :
  ```python
  CORE = SRC / "core"
  if str(CORE) not in sys.path:
      sys.path.insert(0, str(CORE))
  ```
- **`tests/conftest.py`** : ajout de `sys.path.insert(0, str(_SRC / "core"))`.
- **`tests/web/conftest.py`** : inchangé (dépend déjà de `tests/conftest.py` pour
  `src`/`04_coaching`, et de son propre insert pour `web/backend`).

## Fichiers sans changement d'import

- `live_capture.py` : zéro dépendance vers `riotlib`/`core` (conçu pour être copiable
  seul sur une machine sans le reste du repo) — déplacé dans `collection/` sans aucune
  modification de son contenu.

## Documentation

- `CLAUDE.md`, section « Architecture du code » : mise à jour des chemins de chaque
  script déplacé (la description de chaque script reste la même, seul son emplacement
  change dans l'arborescence documentée).
- Aucune autre doc (`web/README.md`, autres specs) ne référence de chemin `src/` cassé
  par ce déplacement (vérifié : ils citent des noms de scripts, pas des chemins complets).

## Vérification

1. `.venv/bin/python -m pytest tests/` doit passer à l'identique (aucune logique changée,
   seulement des chemins) — sert de garde-fou principal.
2. `grep -rn "^import riotlib\|^import positioning\|^import champion_profiles\|from riotlib\|from positioning\|from champion_profiles"` sur tout le repo (hors `.venv`) pour
   confirmer qu'aucun import n'a été oublié après déplacement.
3. Lancer manuellement le backend web (`uvicorn main:app --app-dir web/backend`) et
   vérifier `/api/health` répond, pour confirmer que `main.py` résout bien les nouveaux
   chemins.
4. `git mv` pour chaque déplacement (préserve l'historique de chaque fichier).

## Hors scope

- Renumérotation de `01_data_engineering` → `04_coaching` dans le même schéma que le
  reste (décision explicite : trop de churn pour un gain cosmétique sur du code déjà
  bien organisé).
- Restructuration de `web/backend/` ou `tests/` (non demandée, déjà raisonnablement
  organisés).
- Packaging propre (`pyproject.toml` avec `src/` layout, `__init__.py`) — hors périmètre,
  le style d'import plat existant est conservé tel quel.
