from typing import Optional
import dspy

from kg_gen.steps._2_get_relations import validate_ontology_conformance
from kg_gen.models import (
    TextEntities,
    ExtractTypedEntities,
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
    enforce_type_conformance: bool = False,
) -> list[Entity]:
    if enforce_type_conformance and types and isinstance(types, list):

        def reward_fn(args, pred: dspy.Prediction) -> float:
            typed_entities = [
                e
                if isinstance(e, TypedEntity)
                else TypedEntity(surface_form=e.surface_form, uri=e.uri)
                if not isinstance(e, dict)
                else TypedEntity(**e)
                for e in (pred.entities or [])
            ]
            conforms, _ = validate_ontology_conformance(
                typed_entities=typed_entities,
                enforce_type_conformance=True,
                allowed_types=types,
            )
            return conforms

        extract = dspy.Predict(TextEntities, temperature=temperature)
        refine = dspy.Refine(module=extract, N=2, reward_fn=reward_fn, threshold=1.0)
        result = await dspy.asyncify(refine)(
            source_text=input_data, context=context, types_to_extract=types
        )
    else:
        extract = dspy.Predict(TextEntities, temperature=temperature)
        result = await extract.acall(
            source_text=input_data, context=context, types_to_extract=types
        )
    return result.entities


async def extract_entities(
    input_data: str,
    temperature: float = 0.0,
    types: list[EntityType] | str | None = None,
    context: str | None = None,
    enforce_type_conformance: bool = False,
    provenance_ids: Optional[list[str]] = None,
) -> list[TypedEntity]:
    if enforce_type_conformance and types and isinstance(types, list):

        def reward_fn(args, pred: dspy.Prediction) -> float:
            typed_entities = [
                e if isinstance(e, TypedEntity) else TypedEntity(**e)
                for e in (pred.typed_entities or [])
            ]
            conforms, _ = validate_ontology_conformance(
                typed_entities=typed_entities,
                enforce_type_conformance=True,
                allowed_types=types,
            )
            return conforms

        extract = dspy.Predict(ExtractTypedEntities, temperature=temperature)
        refine = dspy.Refine(module=extract, N=2, reward_fn=reward_fn, threshold=1.0)
        result = await dspy.asyncify(refine)(
            source_text=input_data, context=context, types_to_extract=types
        )
    else:
        extract = dspy.Predict(ExtractTypedEntities, temperature=temperature)
        result = await extract.acall(
            source_text=input_data, context=context, types_to_extract=types
        )
    for entity in result.typed_entities:
        entity.provenance_ids = provenance_ids or []
    return result.typed_entities


async def type_terms(
    input_data: str,
    terms: list[str],
    types: Optional[list[EntityType] | str] = None,
    temperature: float = 0.0,
    context: Optional[str] = None,
    provenance_ids: Optional[list[str]] = None,
    enforce_type_conformance: bool = False,
) -> list[TypedEntity]:
    if enforce_type_conformance and types and isinstance(types, list):

        def reward_fn(args, pred: dspy.Prediction) -> float:
            typed_entities = [
                e if isinstance(e, TypedEntity) else TypedEntity(**e)
                for e in (pred.typed_entities or [])
            ]
            conforms, _ = validate_ontology_conformance(
                typed_entities=typed_entities,
                enforce_type_conformance=True,
                allowed_types=types,
            )
            return conforms

        predict_type = dspy.Predict(TypedEntities, temperature=temperature)
        refine = dspy.Refine(
            module=predict_type, N=2, reward_fn=reward_fn, threshold=1.0
        )
        result = await dspy.asyncify(refine)(
            entities=terms, types=types, source_text=input_data, context=context
        )
    else:
        predict_type = dspy.Predict(TypedEntities, temperature=temperature)
        result = await predict_type.acall(
            entities=terms, types=types, source_text=input_data, context=context
        )
    for entity in result.typed_entities:
        entity.provenance_ids = provenance_ids or []
    return result.typed_entities
