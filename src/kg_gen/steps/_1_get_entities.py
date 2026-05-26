from typing import Optional
import dspy

from kg_gen.models import (
    TextEntities,
    TypedEntity,
    TypedEntities,
    EntityType,
    Entity,
)


async def get_entities(
    input_data: str,
    temperature: float = 0.0,
    types: list[EntityType] | str | None = None,
    context: str | None = None,
) -> list[Entity]:
    extract = dspy.Predict(TextEntities, temperature=temperature)
    if types is None:
        result = await extract.acall(source_text=input_data, context=context)
    else:
        result = await extract.acall(
            source_text=input_data, context=context, types_to_extract=types
        )
    return result.entities


async def type_terms(
    input_data: str,
    terms: list[str],
    types: Optional[list[EntityType] | str] = None,
    temperature: float = 0.0,
    context: Optional[str] = None,
    provenance_ids: Optional[list[str]] = None,
) -> list[TypedEntity]:
    predict_type = dspy.Predict(TypedEntities, temperature=temperature)
    result = await predict_type.acall(
        entities=terms, types=types, source_text=input_data, context=context
    )
    for entity in result.typed_entities:
        entity.provenance_ids = provenance_ids or []
    return result.typed_entities
