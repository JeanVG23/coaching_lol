from fastapi import APIRouter

import readers

router = APIRouter()


@router.get("/api/c/{slug}/reviews")
def reviews(slug: str):
    return readers.read_reviews(slug)


@router.get("/api/c/{slug}/feedback")
def feedback_list(slug: str):
    return readers.read_feedback(slug)