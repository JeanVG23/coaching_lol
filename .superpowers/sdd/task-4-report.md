# Task 4: Documentation Update for Poetry Migration

## Status
✅ **DONE**

## Changes Made

### Edit 1: `web/README.md` (lines 12–13)

**Before:**
```bash
.venv/bin/pip install -r requirements.txt        # fastapi + uvicorn la première fois
.venv/bin/python -m uvicorn main:app --app-dir web/backend --reload
```

**After:**
```bash
poetry install                                    # crée .venv/ et installe les deps (1re fois)
poetry run uvicorn main:app --app-dir web/backend --reload
```

### Edit 2: `CLAUDE.md` (line 218)

**Before:**
```
numérotés `01_data_engineering` → `04_coaching` (pipeline ML, inchangés). Lancer depuis
la racine : `python3 src/<dossier>/<script>.py` — chaque script insère lui-même
```

**After:**
```
numérotés `01_data_engineering` → `04_coaching` (pipeline ML, inchangés). Lancer depuis
la racine, dans l'environnement Poetry (`poetry shell`, ou préfixer chaque commande par
`poetry run`) : `python3 src/<dossier>/<script>.py` — chaque script insère lui-même
```

### Edit 3: `CLAUDE.md` (line 338)

**Before:**
```
- **Tests** : `tests/` (pytest), couvrent la dérivation déterministe + l'extraction comp +
  l'agrégation contextuelle. Lancer : `.venv/bin/python -m pytest tests/`.
```

**After:**
```
- **Tests** : `tests/` (pytest), couvrent la dérivation déterministe + l'extraction comp +
  l'agrégation contextuelle. Lancer : `poetry run pytest tests/`.
```

## Verification

**Command:** `grep -rn "\.venv/bin\|requirements\.txt" CLAUDE.md web/README.md Dockerfile`

**Result:** No matches (exit code 1) ✅

All references to `.venv/bin` and `requirements.txt` have been removed from documentation.

## Commit

- **SHA:** `a171bf9`
- **Message:** `docs: update local-run and test commands for Poetry`
- **Files committed:** `web/README.md`, `CLAUDE.md`

## Self-Review

✅ All three edits match the brief exactly (word-for-word)
✅ No approximations or fuzzy matches — exact replacements
✅ Grep verification confirms no stale references remain
✅ Commit message is correct
✅ Only the two required files were staged and committed
