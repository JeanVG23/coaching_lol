"""Schéma de sortie typé de la review de coaching (Pydantic v2).

Expose Review (longueurs fixes 3/3/2) et le JSON-schema dérivé pour le
paramètre `format` d'Ollama (structured output). Chaque Insight porte sa
preuve chiffrée — pas de conseil sans stat.
"""
from __future__ import annotations

from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field, model_validator


class Insight(BaseModel):
    point: str       # affirmation FR ("tu roams trop peu en mid")
    evidence: str    # preuve chiffrée du payload ("roam mid 50% vs 70% challenger")


class Review(BaseModel):
    strengths: Annotated[list[Insight], Field(min_length=3, max_length=3)]
    mistakes: Annotated[list[Insight], Field(min_length=3, max_length=3)]
    habits: Annotated[list[str], Field(min_length=2, max_length=2)]
    next_focus: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


def review_json_schema() -> dict:
    """JSON-schema passé à Ollama `format`. minItems/maxItems contraignent la génération."""
    return Review.model_json_schema()


# --- Boucle d'évaluation : annotation des reviews persistées -----------------

TagKind = Literal["asymetrie", "stat-inventee", "profondeur-en-faute",
                 "trop-vague", "non-actionnable", "autre"]
NEG_TAGS: tuple[str, ...] = get_args(TagKind)   # affiché en menu (restera synchronisé)


class FeedbackItem(BaseModel):
    kind: Literal["strength", "mistake", "habit", "focus"]
    index: int                       # position dans sa section (focus = 0)
    useful: bool
    tag: TagKind | None = None        # obligatoire si useful=False (cf. validator)
    note: str | None = None

    @model_validator(mode="after")
    def _tag_required_when_not_useful(self):
        if not self.useful and self.tag is None:
            raise ValueError("tag requis quand useful=False")
        return self


class Feedback(BaseModel):
    ts: str                           # clé = ts de la review annotée
    player: str
    rated_at: str                     # ISO timestamp de l'annotation
    model: str                        # copié de la review (récap par modèle)
    overall_useful: bool | None = None   # non collecté par le flow interactif
    items: list[FeedbackItem]         # items annotés (≤9 ; skips omis)
