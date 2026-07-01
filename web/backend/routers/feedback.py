from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import feedback
import settings

router = APIRouter()


class FeedbackReq(BaseModel):
    slug: str
    ts: str
    responses: dict[str, dict]  # "kind,index" -> {useful, tag?, note?}


@router.post("/api/feedback")
def post_feedback(req: FeedbackReq):
    if not settings.account_for(req.slug):
        raise HTTPException(404, "compte inconnu")
    review_dict = feedback.load_review(req.slug, req.ts)
    if not review_dict:
        raise HTTPException(404, "review introuvable")
    review = feedback.schema_mod.Review.model_validate(review_dict["review"])
    responses = {}
    for k, v in req.responses.items():
        kind, idx = k.split(",")
        responses[(kind, int(idx))] = (v["useful"], v.get("tag"), v.get("note"))
    from datetime import datetime
    fb = feedback.build_feedback(review, req.ts, req.slug, review_dict["model"],
                                  datetime.now().isoformat(timespec="seconds"), responses)
    feedback.persist_feedback(req.slug, fb)
    return {"ok": True}