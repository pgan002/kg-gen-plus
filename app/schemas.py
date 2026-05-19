from __future__ import annotations

from pydantic import BaseModel

from kg_gen.models import EntityType, OntologyPredicate, TypedEntity


class HeartBeatResponse(BaseModel):
    is_alive: bool = True


class OntologyConversionInput(BaseModel):
    classes: list[EntityType]
    predicates: list[OntologyPredicate]


class GenerateSingleInput(BaseModel):
    id: str
    text: str
    terms: list[TypedEntity]
