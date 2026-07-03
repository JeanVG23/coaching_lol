# Migration .venv/requirements.txt → Poetry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer `requirements.txt` + `.venv` géré à la main par Poetry (dépendances verrouillées, groupes prod/dev/analyse séparés), en conservant la convention flat-import de `src/` (pas de package Python installable) et le venv en local (`.venv/`) pour ne pas casser les habitudes d'éditeur.

**Architecture:** `pyproject.toml` en mode **non-package** (`package-mode = false` — le repo reste un ensemble de scripts, pas une lib distribuable), avec 3 groupes de dépendances : `main` (prod, image Docker), `dev` (pytest), `analysis` (scipy/statsmodels/matplotlib/seaborn/interpret/shap — utilisés uniquement par `src/03_data_analyse/`, jamais en prod). Un `poetry.toml` local force le venv **dans le projet** (`.venv/`) pour préserver `.venv/bin/python` tel quel.

**Tech Stack:** Poetry 2.3.2 (déjà installé : `/opt/homebrew/bin/poetry`), Python 3.13/3.14 (Docker pin 3.13-slim, dev local testé sous 3.14.6).

## Global Constraints

- Ne pas transformer `src/` en package Python installable — convention flat-import (`sys.path.insert` vers `src/core/`) documentée dans `CLAUDE.md`, à préserver telle quelle.
- `requires-python` doit couvrir à la fois le pin Docker (`python:3.13-slim`) et le poste de dev local (3.14.6) : `>=3.13,<3.15`.
- Aucune régression : la suite `tests/` (pytest) doit passer identiquement avant/après, et l'image Docker doit toujours servir `/api/health`.
- Ne pas re-migrer les paquets fantômes de l'ancien `.venv` (`dash`, `dash_cytoscape`, `plotly`, `jupyter*`, `notebook*`, `category_encoders`, `ipykernel` — installés au fil de l'eau mais **non importés nulle part** dans `src/`, `tests/`, `web/`) : ce sont des résidus d'exploration, pas des dépendances réelles. Vérifié par grep exhaustif sur les imports.
- Ne pas toucher aux fichiers historiques (`docs/superpowers/plans/*.md` antérieurs, `.superpowers/sdd/*.md`) qui mentionnent encore `.venv`/`requirements.txt` — ce sont des comptes-rendus datés, pas de la doc vivante.

---

### Task 1: Créer `pyproject.toml` + `poetry.toml`

**Files:**
- Create: `pyproject.toml`
- Create: `poetry.toml`

**Interfaces:**
- Consumes: rien (fichiers de config racine).
- Produces: déclaration des 3 groupes de dépendances (`main` implicite via `[project.dependencies]`, `dev`, `analysis`) que les tâches suivantes verrouillent et installent.

- [ ] **Step 1: Écrire `pyproject.toml`**

```toml
[project]
name = "coaching-lol"
version = "0.1.0"
description = "Coach IA personnalisé pour League of Legends (positionnement > stats brutes)"
requires-python = ">=3.13,<3.15"
dependencies = [
    "requests>=2.32",
    "zstandard>=0.23",
    "pydantic>=2",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "httpx>=0.27",
    "numpy>=2.0",
    "pandas>=2.0",
    "scikit-learn>=1.5",
    "xgboost>=2.1",
]

[tool.poetry]
package-mode = false

[tool.poetry.group.dev.dependencies]
pytest = ">=8.4"
pytest-mock = ">=3.15"

[tool.poetry.group.analysis.dependencies]
scipy = ">=1.16"
statsmodels = ">=0.14"
matplotlib = ">=3.10"
seaborn = ">=0.13"
interpret = ">=0.7"
shap = ">=0.52"
```

Notes :
- `dependencies` (table `[project]`) = l'ancien `requirements.txt` à l'identique (mêmes bornes `>=`), c'est le groupe `main` installé en prod (Docker).
- `dev` et `analysis` sont **nouveaux** : ces paquets tournaient dans l'ancien `.venv` sans jamais être déclarés nulle part (`pytest`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `interpret`, `shap` sont bien importés dans `tests/` et `src/03_data_analyse/` respectivement — vérifié par grep). Les versions `>=` reprennent celles actuellement installées comme plancher.
- Pas de section `[build-system]` : inutile en mode non-package (vérifié : `poetry check`/`poetry lock`/`poetry install` fonctionnent sans).

- [ ] **Step 2: Écrire `poetry.toml`**

```toml
[virtualenvs]
in-project = true
```

Sans ce fichier, Poetry crée le venv hors du repo (`~/Library/Caches/pypoetry/virtualenvs/`). Avec, `poetry install` crée `.venv/` à la racine du projet — déjà dans `.gitignore`, préserve `.venv/bin/python` tel qu'utilisé aujourd'hui par l'éditeur/les scripts.

- [ ] **Step 3: Vérifier la syntaxe**

Run: `poetry check`
Expected: `All set!`

---

### Task 2: Verrouiller et installer l'environnement complet

**Files:**
- Create: `poetry.lock` (généré, à committer)

**Interfaces:**
- Consumes: `pyproject.toml`, `poetry.toml` (Task 1).
- Produces: `.venv/` peuplé (main + dev + analysis), `poetry.lock` figé — base pour toutes les commandes `poetry run ...` des tâches suivantes.

- [ ] **Step 1: Générer le lock file**

Run: `poetry lock`
Expected: se termine par `Writing lock file` sans erreur de résolution.

- [ ] **Step 2: Installer tous les groupes (dev machine)**

Run: `poetry install`
Expected: crée `.venv/` à la racine (`ls -d .venv` doit réussir), installe main+dev+analysis (pas de `--only`/`--without` par défaut).

- [ ] **Step 3: Vérifier que le nouvel environnement fait tourner la suite de tests existante**

Run: `poetry run pytest tests/ -q`
Expected: même résultat (pass/fail count) qu'avec l'ancien `.venv/bin/python -m pytest tests/` — aucune régression d'import (`scipy`, `shap`, `interpret`, `matplotlib`, `seaborn` doivent tous résoudre pour les tests qui les exercent, ex. `tests/test_positioning.py`, `tests/web/test_api.py`).

- [ ] **Step 4: Vérifier l'installation prod-only (celle utilisée par Docker au Task 4)**

Run: `poetry install --only main --no-root --dry-run`
Expected: liste uniquement les paquets du groupe `main` (requests, zstandard, pydantic, fastapi, uvicorn, httpx, numpy, pandas, scikit-learn, xgboost + leurs transitives) — ni pytest, ni scipy/shap/matplotlib ne doivent apparaître.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml poetry.toml poetry.lock
git commit -m "build: introduce Poetry (pyproject.toml + lock), package-mode=false"
```

---

### Task 3: Migrer le Dockerfile de `pip -r requirements.txt` vers Poetry

**Files:**
- Modify: `Dockerfile`

**Interfaces:**
- Consumes: `pyproject.toml`/`poetry.lock` (Task 2), groupe `main` uniquement.
- Produces: image Docker installant les mêmes paquets qu'avant (main only), sans `requirements.txt`.

- [ ] **Step 1: Remplacer le bloc d'installation des dépendances**

Contenu actuel de `Dockerfile` :

```dockerfile
# coaching_lol — image backend FastAPI (sert aussi le frontend statique).
FROM python:3.13-slim

WORKDIR /app

# Déps d'abord (cache Docker).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'app + modules existants (src/). data/ est exclu (cf. .dockerignore)
# — la donnée vivra sur un volume persistant Fly, pas dans l'image.
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# uvicorn sert l'API + les fichiers statiques montés par main.py.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--app-dir", "/app/web/backend"]
```

Nouveau contenu :

```dockerfile
# coaching_lol — image backend FastAPI (sert aussi le frontend statique).
FROM python:3.13-slim

WORKDIR /app

# Déps d'abord (cache Docker) : poetry.lock ne bouge pas à chaque commit de code.
COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry==2.3.2 \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

# Code de l'app + modules existants (src/). data/ est exclu (cf. .dockerignore)
# — la donnée vivra sur un volume persistant Fly, pas dans l'image.
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# uvicorn sert l'API + les fichiers statiques montés par main.py.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--app-dir", "/app/web/backend"]
```

`virtualenvs.create false` fait installer Poetry directement dans le Python système du conteneur (pas de venv imbriqué) — comportement identique à l'actuel `pip install -r requirements.txt`, donc `CMD` n'a pas besoin de `poetry run`.

- [ ] **Step 2: Builder l'image et vérifier qu'elle démarre**

Run: `docker build -t coaching-lol-poetry-test .`
Expected: build réussi, dernière étape `poetry install --only main` n'installe que le groupe main (pas de pytest/scipy dans les logs).

Run: `docker run --rm -p 8000:8000 -e PYTHONUNBUFFERED=1 coaching-lol-poetry-test &`
puis `curl -sf http://127.0.0.1:8000/api/health`
Expected: réponse HTTP 200 (même sonde que documentée dans `web/README.md`). Arrêter ensuite le conteneur (`docker stop` sur le container id, ou `kill %1`).

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "build: switch Docker image install from pip/requirements.txt to poetry"
```

---

### Task 4: Mettre à jour la documentation vivante (`web/README.md`, `CLAUDE.md`)

**Files:**
- Modify: `web/README.md`
- Modify: `CLAUDE.md:218`, `CLAUDE.md:338`

**Interfaces:**
- Consumes: rien (documentation).
- Produces: instructions cohérentes avec le nouveau flux Poetry pour tout futur lecteur/agent.

- [ ] **Step 1: `web/README.md` — remplacer les commandes de lancement local**

Ancien :

```
.venv/bin/pip install -r requirements.txt        # fastapi + uvicorn la première fois
.venv/bin/python -m uvicorn main:app --app-dir web/backend --reload
```

Nouveau :

```
poetry install                                    # crée .venv/ et installe les deps (1re fois)
poetry run uvicorn main:app --app-dir web/backend --reload
```

- [ ] **Step 2: `CLAUDE.md` — mettre à jour la convention de lancement des scripts (ligne 218)**

Ancien :

```
racine : `python3 src/<dossier>/<script>.py` — chaque script insère lui-même
```

Nouveau :

```
racine, dans l'environnement Poetry (`poetry shell`, ou préfixer chaque commande par
`poetry run`) : `python3 src/<dossier>/<script>.py` — chaque script insère lui-même
```

- [ ] **Step 3: `CLAUDE.md` — mettre à jour la commande pytest (ligne 338)**

Ancien : `` Lancer : `.venv/bin/python -m pytest tests/`. ``
Nouveau : `` Lancer : `poetry run pytest tests/`. ``

- [ ] **Step 4: Vérifier qu'il ne reste aucune référence vivante à `.venv/bin` ou `requirements.txt` hors historique**

Run: `grep -rn "\.venv/bin\|requirements\.txt" CLAUDE.md web/README.md Dockerfile`
Expected: aucune occurrence.

- [ ] **Step 5: Commit**

```bash
git add web/README.md CLAUDE.md
git commit -m "docs: update local-run and test commands for Poetry"
```

---

### Task 5: Nettoyer l'ancien setup (`requirements.txt`, ancien `.venv`)

**Files:**
- Delete: `requirements.txt`
- Delete (local, non versionné): ancien répertoire `.venv/`

**Interfaces:**
- Consumes: Tasks 2–4 doivent être vertes (Poetry installe tout, Docker build passe, docs à jour) avant de supprimer l'ancien chemin.
- Produces: plus aucune ambiguïté sur la source de vérité des dépendances.

- [ ] **Step 1: Supprimer `requirements.txt` du repo**

```bash
git rm requirements.txt
```

- [ ] **Step 2: Vérifier qu'aucun outil vivant ne le référence plus**

Run: `grep -rln "requirements.txt" --include="*.md" --include="Dockerfile" --include="*.py" . | grep -v "^./docs/superpowers/plans/" | grep -v "^./.superpowers/"`
Expected: aucune sortie (seuls les documents historiques exclus par le filtre peuvent encore le mentionner).

- [ ] **Step 3: Commit la suppression**

```bash
git commit -m "chore: remove requirements.txt, superseded by pyproject.toml/poetry.lock"
```

- [ ] **Step 4: Supprimer l'ancien `.venv` local (non tracké par git, action manuelle post-migration)**

```bash
rm -rf .venv
poetry install
```

Expected: `poetry install` recrée un `.venv/` propre à partir du seul `poetry.lock` (donc sans les résidus `dash`/`plotly`/`jupyter*`/`category_encoders` de l'ancien environnement).

---

### Task 6: Vérification finale end-to-end

**Files:** aucun (validation uniquement).

**Interfaces:**
- Consumes: l'ensemble des tâches précédentes.
- Produces: confiance que la migration n'a rien cassé côté tests, CLI et web.

- [ ] **Step 1: Suite de tests complète**

Run: `poetry run pytest tests/ -q`
Expected: tous les tests passent (même compte qu'à l'état initial, cf. Task 2 Step 3).

- [ ] **Step 2: Un script CLI représentatif**

Run: `poetry run python3 src/reporting/compare.py --scope adc --outcome overall`
Expected: sortie identique à l'exécution pré-migration (même format de rapport) — confirme que la convention flat-import (`sys.path.insert` vers `src/core/`) fonctionne sous l'interpréteur du venv Poetry.

- [ ] **Step 3: App web en local**

Run: `poetry run uvicorn main:app --app-dir web/backend --reload &` puis `curl -sf http://127.0.0.1:8000/api/health`
Expected: `✅` / 200 OK, comme documenté dans `web/README.md`. Arrêter le process ensuite.

- [ ] **Step 4: Commit final (si des ajustements ont eu lieu pendant la vérification)**

```bash
git status
```

Expected: working tree propre (tout déjà committé dans les tâches précédentes) — sinon committer les derniers ajustements découverts pendant la vérification.
