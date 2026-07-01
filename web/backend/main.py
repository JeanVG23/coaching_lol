"""coaching_lol — backend FastAPI. Un seul process Fly.io sert l'API (/api/*)
et le frontend statique."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
COACH = SRC / "04_coaching"
if str(COACH) not in sys.path:
    sys.path.insert(0, str(COACH))
BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

from routers import accounts, games, jobs, reviews, feedback, shap, rank  # noqa: E402

app = FastAPI(title="coaching_lol", version="0.1.0")

app.include_router(accounts.router)
app.include_router(games.router)
app.include_router(reviews.router)
app.include_router(jobs.router)
app.include_router(feedback.router)
app.include_router(shap.router)
app.include_router(rank.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "coaching-lol",
            "server_time": datetime.now(timezone.utc).isoformat()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/c/{slug}")
@app.get("/readme")
def spa(slug: str | None = None) -> FileResponse:
    """Catch-all SPA : sert le même index.html, le routeur Alpine lit l'URL."""
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")