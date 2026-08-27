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


async def _refine_or_surface_error(refine, module, /, **call_kwargs):
    """Await a ``dspy.Refine``, and if it gives up, re-raise the real cause.

    ``dspy.Refine`` catches whatever the wrapped module raises, logs it as
    "Attempt failed with rollout id N", and returns ``None`` once every attempt
    has failed. Callers then touch an attribute on that ``None`` and the operator
    sees ``'NoneType' object has no attribute 'entities'`` -- which says nothing
    about the authentication failure, unsupported parameter or rate limit that
    actually stopped the run. Since this path only runs when the work has already
    failed, calling the bare module once more costs nothing and lets the genuine
    exception propagate.
    """
    result = await dspy.asyncify(refine)(**call_kwargs)
    if result is not None:
        return result
    # Every refine attempt failed. Re-run unguarded so the cause is visible.
    return await module.acall(**call_kwargs)


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
                for e in ((pred.entities if pred is not None else None) or [])
            ]
            conforms, _ = validate_ontology_conformance(
                typed_entities=typed_entities,
                enforce_type_conformance=True,
                allowed_types=types,
            )
            return conforms

        extract = dspy.Predict(TextEntities, temperature=temperature)
        refine = dspy.Refine(module=extract, N=2, reward_fn=reward_fn, threshold=1.0)
        result = await _refine_or_surface_error(
            refine,
            extract,
            source_text=input_data,
            context=context,
            types_to_extract=types,
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
                for e in ((pred.typed_entities if pred is not None else None) or [])
            ]
            conforms, _ = validate_ontology_conformance(
                typed_entities=typed_entities,
                enforce_type_conformance=True,
                allowed_types=types,
            )
            return conforms

        extract = dspy.Predict(ExtractTypedEntities, temperature=temperature)
        refine = dspy.Refine(module=extract, N=2, reward_fn=reward_fn, threshold=1.0)
        result = await _refine_or_surface_error(
            refine,
            extract,
            source_text=input_data,
            context=context,
            types_to_extract=types,
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
                for e in ((pred.typed_entities if pred is not None else None) or [])
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
        result = await _refine_or_surface_error(
            refine,
            predict_type,
            entities=terms,
            types=types,
            source_text=input_data,
            context=context,
        )
    else:
        predict_type = dspy.Predict(TypedEntities, temperature=temperature)
        result = await predict_type.acall(
            entities=terms, types=types, source_text=input_data, context=context
        )
    for entity in result.typed_entities:
        entity.provenance_ids = provenance_ids or []
    return result.typed_entities
