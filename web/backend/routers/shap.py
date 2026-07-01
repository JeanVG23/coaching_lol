from fastapi import APIRouter

import readers

router = APIRouter()


@router.get("/api/c/{slug}/shap")
def shap(slug: str):
    return readers.read_shap(slug)