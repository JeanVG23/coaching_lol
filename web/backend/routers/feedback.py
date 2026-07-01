from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

import feedback
import settings

router = APIRouter()


class FeedbackReq(BaseModel):
    slug: str
    ts: str
    responses: dict[str, dict]  # "kind,index" -> {useful, tag?, note?}


class FeedbackResponse(BaseModel):
    useful: bool
    tag: feedback.schema_mod.TagKind | None = None
    note: str | None = None


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
        try:
            kind, idx = k.split(",")
            idx = int(idx)
        except ValueError:
            raise HTTPException(422, f"clé de réponse invalide : {k!r} (attendu 'kind,index')")
        try:
            r = FeedbackResponse.model_validate(v)
        except ValidationError as e:
            raise HTTPException(422, f"réponse invalide pour {k!r} : {e.errors()}")
        responses[(kind, idx)] = (r.useful, r.tag, r.note)
    rated_at = datetime.now().isoformat(timespec="seconds")
    try:
        fb = feedback.build_feedback(review, req.ts, req.slug, review_dict["model"],
                                      rated_at, responses)
    except ValidationError as e:
        raise HTTPException(422, f"feedback invalide (tag requis si useful=False) : {e.errors()}")
    feedback.persist_feedback(req.slug, fb)
    return {"ok": True}