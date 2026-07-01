# web/ — interface web de coaching_lol

Un seul process FastAPI (sur Fly.io) sert à la fois :
- **l'API** sous `/api/*` (`web/backend/main.py`),
- **le frontend statique** (`index.html`, CSS, JS) sous `/` et `/static` (`web/frontend/`).

Les clés (Riot, Ollama) restent côté serveur — jamais dans le navigateur.

## Lancer en local

```bash
.venv/bin/pip install -r requirements.txt        # fastapi + uvicorn la première fois
.venv/bin/python -m uvicorn main:app --app-dir web/backend --reload
```

Ouvre http://127.0.0.1:8000 — la page doit afficher « ✅ Backend en ligne ».
Sonde API : http://127.0.0.1:8000/api/health

## Déployer sur Fly.io

Prérequis : `flyctl` installé et authentifié (`fly auth login`).

```bash
fly apps create coaching-lol       # nom libre mais unique ; ajuste fly.toml en conséquence
fly deploy                          # build le Dockerfile + pousse l'image
fly open                            # ouvre l'URL publique
```

> La donnée (`data/`) n'est PAS dans l'image (cf. `.dockerignore`). Elle vivra sur un
> volume persistant Fly, monté au moment où on câblera le coaching réel.

## Structure

```
web/
  backend/main.py    # FastAPI : /api/* + sert index.html et /static
  frontend/
    index.html       # page (affiche le statut backend pour l'instant)
    style.css
    app.js           # fetch /api/health
fly.toml             # config Fly.io
Dockerfile           # image unique (back + statique)
.dockerignore        # exclut data/ (lourd) et l'env local
```

## À venir (endpoints prévus)

- `POST /api/accounts` — gérer plusieurs comptes de coaching
- `POST /api/fetch` — récupérer les dernières games (Riot API, job async)
- `GET  /api/jobs/{id}` — suivi des jobs longs (fetch + coaching)
- `POST /api/coach` — générer le coaching (LLM)
- `POST /api/feedback` — annoter les conseils (boucle d'éval)