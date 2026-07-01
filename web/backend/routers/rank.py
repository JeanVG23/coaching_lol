from fastapi import APIRouter

import readers

router = APIRouter()

_EMPTY_RANK = {"tier": None, "division": None, "league_points": None,
               "wins": None, "losses": None, "fetched_at": None}


@router.get("/api/c/{slug}/rank")
def rank(slug: str):
    return readers.read_rank(slug) or _EMPTY_RANK
