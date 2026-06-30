from rdflib import Graph as RDFGraph, URIRef, RDFS, XSD
from typing import List, Tuple, Optional, Literal, Type
from pathlib import Path
import json
import dspy
import litellm
from pydantic import BaseModel, create_model, ValidationError

from kg_gen.models import (
    TypedEntity,
    ExtractTextRelations,
    Relation,
    OntologyPredicate,
    EntityType,
)


def parse_relations_response(
    raw_json: str,
    entities: List[str],
    response_model: Optional[Type[BaseModel]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Parse a relations JSON response with graceful fallback.

    First attempts strict Pydantic validation. If that fails (e.g., due to
    EntityLiteral validation), falls back to raw JSON parsing and filters
    out items with invalid subject/object.

    Args:
        raw_json: The raw JSON string from the LLM response
        entities: List of valid entity strings
        response_model: Optional Pydantic model for strict validation

    Returns:
        List of (subject, predicate, object) tuples with valid entities
    """
    entities_set = set(entities)

    # Try strict Pydantic validation first if model provided
    if response_model is not None:
        try:
            parsed = response_model.model_validate_json(raw_json)
            return [(r.subject, r.predicate, r.object) for r in parsed.relations]
        except ValidationError:
            pass  # Fall through to JSON parsing

    # Fallback: parse as raw JSON and filter
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return []

    # Handle both {"relations": [...]} and direct list formats
    items = data.get("relations", data) if isinstance(data, dict) else data

    if not isinstance(items, list):
        return []

    relations = []
    for item in items:
        if not isinstance(item, dict):
            continue

        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")

        # Skip if missing required fields
        if not all([subject, predicate, obj]):
            continue

        # Skip if subject or object not in valid entities
        if subject not in entities_set or obj not in entities_set:
            continue

        relations.append((subject, predicate, obj))

    return relations


def _load_relations_prompt() -> str:
    """Load the relations prompt template from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "relations.txt"
    return prompt_path.read_text()


def _create_relations_model(entities: List[str]):
    """Dynamically create Pydantic models with entity literals for subject/object."""
    # Create a Literal type from the entities list
    EntityLiteral = Literal[tuple(entities)]  # type: ignore

    # Create RelationItem with constrained subject/object
    RelationItem = create_model(
        "RelationItem",
        subject=(EntityLiteral, ...),
        predicate=(str, ...),
        object=(EntityLiteral, ...),
    )

    # Create RelationsResponse containing list of RelationItem
    RelationsResponse = create_model(
        "RelationsResponse",
        relations=(List[RelationItem], ...),
    )

    return RelationItem, RelationsResponse


def _get_relations_litellm(
    input_data: str,
    entities: List[str],
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
) -> List[Tuple[str, str, str]]:
    prompt_template = _load_relations_prompt()
    entities_str = "\n".join(f"- {e}" for e in entities)
    user_prompt = f"""
Here is the list of entities that were previously extracted from the source text:

<entities>
{entities_str}
</entities>

Here is the source text to analyze:

<text>
{input_data}
</text>
    """

    # Create dynamic model with entity constraints
    _, RelationsResponse = _create_relations_model(entities)

    # Build schema with additionalProperties: false (required by OpenAI)
    schema = RelationsResponse.model_json_schema()
    schema["additionalProperties"] = False
    # Also need to set additionalProperties on nested objects
    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            if def_schema.get("type") == "object":
                def_schema["additionalProperties"] = False

    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "relations_response",
                "schema": schema,
                "strict": True,
            }
        },
    }

    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.responses(**kwargs)
    raw_json = response.output[-1].content[0].text
    return parse_relations_response(raw_json, entities, RelationsResponse)


def extraction_sig(
    RelationModel: Type[BaseModel], is_conversation: bool, context: str = ""
) -> Type[dspy.Signature]:
    if not is_conversation:

        class ExtractTextRelations(dspy.Signature):
            __doc__ = f"""Extract subject-predicate-object triples from the source text.
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text. {context}"""

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[RelationModel] = dspy.OutputField(
                desc="List of subject-predicate-object tuples. Be thorough."
            )

        return ExtractTextRelations
    else:

        class ExtractConversationRelations(dspy.Signature):
            __doc__ = f"""Extract subject-predicate-object triples from the conversation, including:
      1. Relations between concepts discussed
      2. Relations between speakers and concepts (e.g. user asks about X)
      3. Relations between speakers (e.g. assistant responds to user)
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text. {context}"""

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[RelationModel] = dspy.OutputField(
                desc="List of subject-predicate-object tuples where subject and object are exact matches to items in entities list. Be thorough"
            )

        return ExtractConversationRelations


def fallback_extraction_sig(
    entities, is_conversation, context: str = ""
) -> dspy.Signature:
    """This fallback extraction does not strictly type the subject and object strings."""

    entities_str = "\n- ".join(entities)

    class Relation(BaseModel):
        # TODO: should use literal's here instead.
        __doc__ = f"""Knowledge graph subject-predicate-object tuple. Subject and object entities must be one of: {entities_str}"""

        subject: str = dspy.InputField(desc="Subject entity", examples=["Kevin"])
        predicate: str = dspy.InputField(desc="Predicate", examples=["is brother of"])
        object: str = dspy.InputField(desc="Object entity", examples=["Vicky"])

    return Relation, extraction_sig(Relation, is_conversation, context)


def _filter_entities(entities: List[str]) -> List[str]:
    """Filter out entities that contain backslashes."""
    return [e for e in entities if '"' not in e]  # not received by oai api


def validate_ontology_conformance(
    typed_entities: list[TypedEntity],
    relations: Optional[list[Relation]] = None,
    predicate_domain_range: Optional[list[OntologyPredicate]] = None,
    enforce_domain_conformance: bool = True,
    enforce_range_conformance: bool = True,
    enforce_predicate_conformance: bool = False,
    enforce_type_conformance: bool = False,
    allowed_types: Optional[list[EntityType]] = None,
) -> tuple[float, str]:
    if not predicate_domain_range and not enforce_type_conformance:
        return 1.0, ""

    allowed_labels = {t.label for t in allowed_types} if allowed_types else set()
    errors = []
    successes = 0
    total_checks = 0

    if enforce_type_conformance and allowed_labels:
        for ent in typed_entities:
            total_checks += 1
            if ent.type and ent.type.label not in allowed_labels:
                errors.append(
                    f"Entity {ent.surface_form}: type {ent.type.label} not in allowed types"
                )
            else:
                successes += 1

    if not relations:
        if total_checks == 0:
            return 1.0, ""
        return successes / total_checks, "; ".join(errors)

    entity_map = {te.surface_form: te for te in typed_entities}
    predicate_map = (
        {p.label: p for p in predicate_domain_range} if predicate_domain_range else {}
    )

    for rel in relations:
        subj_ent = entity_map.get(rel.subject.surface_form)
        obj_ent = entity_map.get(rel.object.surface_form)
        pred_obj = predicate_map.get(rel.predicate.surface_form)

        if enforce_predicate_conformance:
            total_checks += 1
            if not pred_obj:
                errors.append(
                    f"Relation {rel.subject} -[{rel.predicate}]-> {rel.object}: Predicate {rel.predicate.surface_form} not in ontology"
                )
            else:
                successes += 1

        if not pred_obj:
            continue

        if (
            enforce_domain_conformance
            and subj_ent
            and subj_ent.type
            and pred_obj.domain
        ):
            total_checks += 1
            if subj_ent.type not in pred_obj.domain:
                errors.append(
                    f"Relation {rel.subject} -[{rel.predicate}]-> {rel.object}: Subject type {subj_ent.type.label} not in domain {[d.label for d in pred_obj.domain]}"
                )
            else:
                successes += 1

        if enforce_range_conformance and obj_ent and obj_ent.type and pred_obj.range:
            total_checks += 1
            if obj_ent.type not in pred_obj.range:
                errors.append(
                    f"Relation {rel.subject} -[{rel.predicate}]-> {rel.object}: Object type {obj_ent.type.label} not in range {[r.label for r in pred_obj.range]}"
                )
            else:
                successes += 1

    if total_checks == 0:
        return 1.0, ""
    return successes / total_checks, "; ".join(errors)


def get_all_superclasses(g: RDFGraph, type_uri: URIRef) -> set[URIRef]:
    superclasses = set()
    q = [type_uri]
    while q:
        curr = q.pop(0)
        if curr not in superclasses:
            superclasses.add(curr)
            scs = g.objects(curr, RDFS.subClassOf)
            q.extend(scs)
    return superclasses


async def get_relations_typed(
    input_text: str,
    typed_entities: list[TypedEntity],
    ontology: Optional[RDFGraph] = None,
    predicate_domain_range: Optional[list[OntologyPredicate]] = None,
    context: str = "",
    temperature: float = 0.0,
    n_retries=2,
    provenance_ids: Optional[List[str]] = None,
    enforce_domain_conformance: bool = True,
    enforce_range_conformance: bool = True,
    enforce_predicate_conformance: bool = False,
    enforce_type_conformance: bool = False,
    allowed_types: Optional[list[EntityType]] = None,
) -> list[Relation]:
    assert n_retries > 0, "n_retries must be greater than 0"

    entity_types_with_superclasses = set()
    if ontology is not None:
        label_to_uri = {
            str(label): uri
            for uri, _, label in ontology.triples((None, RDFS.label, None))
        }
        for e in typed_entities:
            if e.type and e.type.label in label_to_uri:
                type_uri = label_to_uri[e.type.label]
                entity_types_with_superclasses.update(
                    get_all_superclasses(ontology, type_uri)
                )

    if predicate_domain_range:
        filtered_predicates = []
        for p in predicate_domain_range:
            domain_uris = {URIRef(d.uri) for d in p.domain if d.uri}
            range_uris = {URIRef(r.uri) for r in p.range if r.uri}
            # Include if domain is compatible or not specified
            domain_match = (
                not domain_uris
                or not entity_types_with_superclasses.isdisjoint(domain_uris)
                or not enforce_domain_conformance
            )
            # Include if range is a simple datatype or is compatible
            range_is_simple = any(str(r).startswith(str(XSD)) for r in range_uris)
            range_match = (
                not range_uris
                or range_is_simple
                or not entity_types_with_superclasses.isdisjoint(range_uris)
                or not enforce_range_conformance
            )

            if domain_match and range_match:
                filtered_predicates.append(p)
    else:
        filtered_predicates = None

    if enforce_predicate_conformance and not filtered_predicates:
        return []

    def reward_fn(args, pred: dspy.Prediction) -> float:
        conforms, _ = validate_ontology_conformance(
            typed_entities=typed_entities,
            relations=pred.relations,
            predicate_domain_range=filtered_predicates,
            enforce_domain_conformance=enforce_domain_conformance,
            enforce_range_conformance=enforce_range_conformance,
            enforce_predicate_conformance=enforce_predicate_conformance,
            enforce_type_conformance=enforce_type_conformance,
            allowed_types=allowed_types,
        )
        return conforms

    extract_module = dspy.Predict(ExtractTextRelations)
    refine = dspy.Refine(
        module=extract_module, N=n_retries, reward_fn=reward_fn, threshold=1.0
    )

    result = await dspy.asyncify(refine)(
        source_text=input_text,
        typed_entities=typed_entities,
        predicate_domain_range=filtered_predicates,
        context=context,
        config={"temperature": temperature},
    )
    if result is None:
        raise Exception(
            f"Refine failed to produce a result after {n_retries} attempts."
        )

    for relation in result.relations:
        relation.provenance_ids = provenance_ids or []
    return result.relations
