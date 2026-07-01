"""coaching_lol — backend FastAPI.

Un seul process Fly.io sert à la fois :
  - l'API sous /api/* (futurs endpoints : accounts, fetch, coach, feedback),
  - le frontend statique (index.html + assets) sous / et /static.

Les clés (Riot, Ollama) restent serveur, jamais exposées au navigateur.

Lancement local :
    uvicorn main:app --app-dir web/backend --reload
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Rend les modules existants (src/) importables pour les futurs endpoints
# (riotlib, payload, coach, feedback...). Non importés ici tant qu'inutile.
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="coaching_lol", version="0.1.0")


@app.get("/api/health")
def health() -> dict:
    """Sonde de vie : prouve que le backend répond."""
    return {
        "status": "ok",
        "service": "coaching-lol",
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/")
def index() -> FileResponse:
    """Sert la page statique à la racine."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Assets statiques (CSS/JS) — montés APRÈS les routes API pour ne pas les masquer.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")