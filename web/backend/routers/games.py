from fastapi import APIRouter, HTTPException

import readers

router = APIRouter()


@router.get("/api/c/{slug}/games")
def games(slug: str, page: int = 1, size: int = 20):
    # Validation route-layer (le reader ne valide pas délibérément).
    if page < 1 or not (1 <= size <= 200):
        raise HTTPException(422, "page>=1 et size in [1,200]")
    return readers.read_games(slug, page=page, size=size)