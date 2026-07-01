from fastapi import APIRouter

import ml_rank
import readers

router = APIRouter()

_EMPTY = {"predicted_rank": None, "proba": None, "n_games_used": 0}


@router.get("/api/c/{slug}/predicted-rank")
def predicted_rank(slug: str):
    games = readers.read_games(slug, page=1, size=20)["items"]
    return ml_rank.predict_rank(games) or _EMPTY
