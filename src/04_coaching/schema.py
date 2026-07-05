"""Schéma de sortie typé de la review de coaching (Pydantic v2).

Expose Review (1-3 forces / 3 erreurs / 2 habitudes) et le JSON-schema dérivé pour le
paramètre `format` d'Ollama (structured output). Chaque Insight porte sa
preuve chiffrée — pas de conseil sans stat.
"""
from __future__ import annotations

import re
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator


class Insight(BaseModel):
    point: str       # affirmation FR ("tu roams trop peu en mid")
    evidence: str    # preuve chiffrée du payload ("roam mid 50% vs 70% challenger")


class Review(BaseModel):
    # 1 à 3 forces : forcer exactement 3 poussait le LLM à remplir avec du filler
    # quand le profil n'a que 1-2 forces réellement saillantes (feedback "trop-vague").
    strengths: Annotated[list[Insight], Field(min_length=1, max_length=3)]
    mistakes: Annotated[list[Insight], Field(min_length=3, max_length=3)]
    habits: Annotated[list[str], Field(min_length=2, max_length=2)]
    next_focus: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


def review_json_schema() -> dict:
    """JSON-schema passé à Ollama `format`. minItems/maxItems contraignent la génération."""
    return Review.model_json_schema()


# --- Review par-game : chaque erreur ancrée sur un moment précis --------------

_CLOCK_RE = re.compile(r"\d+:\d\d")


class AnchoredInsight(Insight):
    """Insight dont la preuve cite un horodatage mm:ss du journal (contrainte
    de schéma : une erreur par-game sans moment cité est invalide par construction)."""

    @field_validator("evidence")
    @classmethod
    def _evidence_has_clock(cls, v: str) -> str:
        if not _CLOCK_RE.search(v):
            raise ValueError("evidence sans horodatage mm:ss")
        return v


class GameReview(BaseModel):
    """Review d'UNE game. Pas de section habits : une habitude est un pattern
    multi-games, indétectable sur une game isolée (source de vague assurée)."""
    strengths: Annotated[list[Insight], Field(max_length=2)]
    mistakes: Annotated[list[AnchoredInsight], Field(min_length=1, max_length=3)]
    next_focus: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


def game_review_json_schema() -> dict:
    return GameReview.model_json_schema()


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
