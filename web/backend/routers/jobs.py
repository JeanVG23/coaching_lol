"""Jobs async : POST /api/fetch, POST /api/coach, GET /api/jobs/{id}."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import pipeline
import settings
from jobs import JobStore, submit_job

router = APIRouter()

# Store partagé pour l'instance.
_store = JobStore()


class FetchReq(BaseModel):
    slug: str
    n: int = 20


class CoachReq(BaseModel):
    slug: str
    scope: str = "adc"
    outcome: str = "loss"
    target: str = "challenger"
    model: str | None = None


@router.post("/api/fetch")
def fetch(req: FetchReq):
    account = settings.account_for(req.slug)
    if not account:
        raise HTTPException(404, "compte inconnu")
    job = submit_job(_store, lambda on_progress=None: pipeline.fetch_games(
        account, n=req.n, on_progress=on_progress), type="fetch", slug=req.slug)
    return {"job_id": job["id"]}


@router.post("/api/coach")
def coach(req: CoachReq):
    if not settings.account_for(req.slug):
        raise HTTPException(404, "compte inconnu")
    job = submit_job(_store, lambda on_progress=None: pipeline.run_coach(
        req.slug, scope=req.scope, outcome=req.outcome, target=req.target,
        model=req.model, on_progress=on_progress), type="coach", slug=req.slug)
    return {"job_id": job["id"]}


@router.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = _store.get(job_id)
    if not j:
        raise HTTPException(404, "job inconnu")
    return j