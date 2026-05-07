import json

import dspy
from pydantic import BaseModel, Field, computed_field
from typing import Optional, List, Set

from typing_extensions import TypeVar


class InputData(BaseModel):
    text: str
    id: str


class Entity(BaseModel):
    surface_form: str = Field(..., description="The surface form of the entity")
    uri: Optional[str] = Field(
        default=None, description="Optional identifier or URI for the entity"
    )

    def __hash__(self):
        return hash((self.surface_form, self.uri))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return NotImplemented
        return self.surface_form == other.surface_form and self.uri == other.uri

    def __str__(self):
        return self.surface_form

    def __repr__(self):
        return f"Entity(surface_form='{self.surface_form}', uri='{self.uri}')"


class EntityType(BaseModel):
    label: str = Field(..., description="Label or name of the entity type")
    uri: Optional[str] = Field(
        default=None, description="Identifier or URI for the entity type"
    )

    def __hash__(self):
        return hash((self.label, self.uri))


class TypedEntity(Entity):
    """Structured response for entity typing."""

    type: Optional[EntityType] = Field(
        None,
        description="Type of the entity, e.g., 'Person', 'Location', 'Organization', etc, and the URI of the type. "
        "If no type is suitable, leave empty or put None.",
    )
    provenance_ids: List[str] = Field(
        default_factory=list,
        description="Identifier of the source of the entity typing",
    )

    def __hash__(self):
        return hash(
            (self.surface_form, self.uri, self.type, tuple(self.provenance_ids))
        )

    def __eq__(self, other):
        if not isinstance(other, TypedEntity):
            return NotImplemented
        return (
            self.surface_form == other.surface_form
            and self.uri == other.uri
            and self.type == other.type
            and self.provenance_ids == other.provenance_ids
        )

    def __repr__(self):
        return f"TypedEntity(surface_form='{self.surface_form}', uri='{self.uri}', type={self.type!r})"

    @property
    def type_entity(self) -> Entity | None:
        type_entity = (
            Entity(surface_form=self.type.label, uri=self.type.uri)
            if self.type
            else None
        )
        return type_entity

    @property
    def class_assertion(
        self, assertion_predicate: Entity = Entity(surface_form="is a", uri="rdf:type")
    ) -> Optional["Relation"]:
        if self.type_entity is not None:
            assertion = Relation(
                subject=self,
                predicate=assertion_predicate,
                object=self.type_entity,
                provenance_ids=self.provenance_ids,
            )
        else:
            assertion = None
        return assertion


EntityOrSubclass = TypeVar("EntityOrSubclass", bound="Entity")


class TextEntities(dspy.Signature):
    """Extract key entities from the source text. Extracted entities are subjects or objects.
    This is for an extraction task, please be THOROUGH and accurate to the reference text."""

    source_text: str = dspy.InputField()
    context: Optional[str] = dspy.InputField(
        default=None,
        desc="Optional context. If provided, consider it when extracting entities.",
    )
    types_to_extract: Optional[list[EntityType] | str] = dspy.InputField(
        default_factory=list,
        desc="List of entity types or string describing the types to extract. If empty, all entity types are extracted.",
    )
    entities: list[Entity] = dspy.OutputField(desc="THOROUGH list of key entities")


class ConversationEntities(dspy.Signature):
    """Extract key entities from the conversation Extracted entities are subjects or objects.
    Consider both explicit entities and participants in the conversation.
    This is for an extraction task, please be THOROUGH and accurate."""

    source_text: str = dspy.InputField()
    entities: list[Entity] = dspy.OutputField(desc="THOROUGH list of key entities")


class EntitiesResponse(BaseModel):
    """Structured response for entity extraction."""

    entities: List[Entity]


class TypedEntities(dspy.Signature):
    """
    Find the mentions of the given entities in the source_text and predict their type/class.
    If types are provided, only use these types as target types/classes. Otherwise, if no types are provided,
    predict your own types.
    """

    entities: list[str] = dspy.InputField()
    context: Optional[str] = dspy.InputField(
        default=None,
        desc="Optional context. If provided, consider it when extracting entities.",
    )
    types: list[EntityType] | str | None = dspy.InputField(
        default=None,
        desc="List of entity types or string describing the types to extract. If empty, all entity types are extracted.",
    )
    source_text: str = dspy.InputField()
    typed_entities: list[TypedEntity] = dspy.OutputField(
        desc="List of all entities and their types/classes. Every type should come from a list of types, if provided."
    )


class Relation(BaseModel):
    """Knowledge graph subject-predicate-object tuple."""

    subject: Entity = dspy.InputField(desc="Subject entity")
    predicate: Entity = dspy.InputField(desc="Predicate")
    object: Entity = dspy.InputField(desc="Object entity")
    provenance_ids: List[str] = Field(
        default_factory=list, description="Identifier of the source of the triple"
    )

    def __hash__(self):
        return (
            self.subject.__hash__()
            ^ self.predicate.__hash__()
            ^ self.object.__hash__()
            ^ hash(tuple(self.provenance_ids))
        )


class OntologyPredicate(EntityType):
    description: Optional[str] = None
    domain: Set[EntityType] = Field(default_factory=set)
    range: Set[EntityType] = Field(default_factory=set)


class Ontology(BaseModel):
    classes: Set[EntityType] = Field(default_factory=set)
    predicates: Set[OntologyPredicate] = Field(default_factory=set)


class OntologyPredicateExtraction(BaseModel):
    predicate_label: str = dspy.InputField(
        desc="Predicate label", examples=["is_brother_of"]
    )
    predicate_definition: Optional[str] = dspy.InputField(
        default=None, desc="Predicate definition"
    )
    predicate_domain: list[str] = dspy.InputField(
        desc="Predicate domain. Any subject/head of this predicate should have a type from this list.",
        examples=[["Person"]],
    )
    predicate_range: list[str] = dspy.InputField(
        desc="Predicate range. Any object/tail of this predicate should have a type from this list.",
        examples=[["Person"]],
    )


class ExtractTextRelations(dspy.Signature):
    """
    Extract subject-predicate-object triples from the source text.
    Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
    This is for an extraction task, please be thorough, accurate, and faithful to the reference text.
    """

    source_text: str = dspy.InputField()
    typed_entities: list[TypedEntity] = dspy.InputField()
    predicate_domain_range: Optional[list[OntologyPredicateExtraction]] = (
        dspy.InputField(
            default=None,
            desc="List of predicate specification with domains and ranges. If not provided, use any predicates.",
        )
    )
    context: Optional[str] = dspy.InputField(
        default=None,
        desc="Optional context. If provided, consider it when extracting relations.",
    )
    relations: list[Relation] = dspy.OutputField(
        desc="List of subject-predicate-object tuples. Be thorough."
    )


class Graph(BaseModel):
    typed_entities: Set[TypedEntity] = Field(
        ..., description="All entities including additional ones from response"
    )
    relations_wo_class_assertions: List[Relation] = Field(
        ..., description="List of (subject, predicate, object) triples"
    )
    entity_clusters: Optional[dict[str, Set[TypedEntity | Entity]]] = None
    edge_clusters: Optional[dict[str, Set[str]]] = None

    entity_metadata: dict[TypedEntity, Set[str]] | None = None

    output_class_assertions: bool = True

    @computed_field
    @property
    def entities(self) -> set[Entity]:
        out = {
            Entity(surface_form=te.surface_form, uri=te.uri)
            for te in self.typed_entities
        }
        out |= {te.type_entity for te in self.typed_entities if te.type_entity}
        return out

    @computed_field
    @property
    def relations(self) -> list[Relation]:
        if self.output_class_assertions:
            out = self.relations_wo_class_assertions + [
                te.class_assertion for te in self.typed_entities if te.type_entity
            ]
            return out
        else:
            return self.relations_wo_class_assertions

    @computed_field
    @property
    def edges(self) -> set[Entity]:
        out = {relation.predicate for relation in self.relations}
        return out

    # @staticmethod
    # def from_file(file_path: str) -> "Graph":
    #     """
    #     Load the graph from a file.
    #     Fix graph entities and edges for missing ones defined in relations.
    #     """
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #         # Convert entity strings to TypedEntity objects during loading
    #         if "typed_entities" in data and isinstance(data["typed_entities"], list):
    #             data["typed_entities"] = [
    #                 TypedEntity(**e) if isinstance(e, dict) else TypedEntity(surface_form=e)
    #                 for e in data["typed_entities"]
    #             ]
    #         if "relations_wo_class_assertions" in data and isinstance(data["relations_wo_class_assertions"], list):
    #             new_relations = []
    #             for r in data["relations_wo_class_assertions"]:
    #                 if isinstance(r, dict):
    #                     subject = TypedEntity(**r["subject"]) if isinstance(r["subject"], dict) else TypedEntity(surface_form=r["subject"])
    #                     predicate = Entity(**r["predicate"]) if isinstance(r["predicate"], dict) else Entity(surface_form=r["predicate"])
    #                     object_ = TypedEntity(**r["object"]) if isinstance(r["object"], dict) else TypedEntity(surface_form=r["object"])
    #                     new_relations.append(Relation(subject=subject, predicate=predicate, object=object_))
    #             data["relations_wo_class_assertions"] = new_relations
    #
    #         if "entity_metadata" in data and isinstance(data["entity_metadata"], dict):
    #             new_metadata = {}
    #             for k_str, v in data["entity_metadata"].items():
    #                 try:
    #                     k_dict = json.loads(k_str)
    #                     key_obj = TypedEntity(**k_dict)
    #                     new_metadata[key_obj] = set(v)
    #                 except (json.JSONDecodeError, TypeError):
    #                     new_metadata[TypedEntity(surface_form=k_str)] = set(v)
    #             data["entity_metadata"].pop(k_str)
    #             data["entity_metadata"][key_obj] = v
    #
    #         graph = Graph.model_validate(data)
    #
    #     # Fix graph entities and edges
    #     for relation in graph.relations:
    #         if relation.subject not in graph.entities:
    #             graph.entities.add(relation.subject)
    #         if relation.predicate not in graph.edges:
    #             graph.edges.add(relation.predicate)
    #         if relation.object not in graph.entities:
    #             graph.entities.add(relation.object)
    #
    #     return graph

    def to_file(self, file_path: str):
        """
        Save the graph to a file.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

    def stats(self, name: Optional[str] = None):
        """
        Print the stats of the graph.
        """
        return f"{name or 'Graph'} with:\n\t{len(self.entities)} entities\n\t{len(self.edges)} edges\n\t{len(self.relations)} relations"


class LMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class StepStats(BaseModel):
    lm_usage: Optional[LMUsage] = LMUsage()
    execution_time: Optional[float] = 0

    def __add__(self, other: "StepStats") -> "StepStats":
        new_lm_usage = LMUsage(
            prompt_tokens=self.lm_usage.prompt_tokens + other.lm_usage.prompt_tokens,
            completion_tokens=self.lm_usage.completion_tokens
            + other.lm_usage.completion_tokens,
            total_tokens=self.lm_usage.total_tokens + other.lm_usage.total_tokens,
        )
        new_stats = StepStats(
            lm_usage=new_lm_usage,
            execution_time=self.execution_time + other.execution_time,
        )
        return new_stats


class KGGenStats(BaseModel):
    get_entities: Optional[StepStats] = StepStats()
    type_terms: Optional[StepStats] = StepStats()
    get_relations_typed: Optional[StepStats] = StepStats()
    deduplicate: Optional[StepStats] = None

    @computed_field
    @property
    def overall_usage(self) -> StepStats:
        overall_stats = StepStats(lm_usage=LMUsage(), execution_time=0.0)
        for step_stats in [
            self.get_entities,
            self.type_terms,
            self.get_relations_typed,
            self.deduplicate,
        ]:
            if step_stats:
                overall_stats.lm_usage.prompt_tokens += (
                    step_stats.lm_usage.prompt_tokens
                )
                overall_stats.lm_usage.completion_tokens += (
                    step_stats.lm_usage.completion_tokens
                )
                overall_stats.lm_usage.total_tokens += step_stats.lm_usage.total_tokens
                overall_stats.execution_time += step_stats.execution_time
        return overall_stats

    def __add__(self, other: "KGGenStats") -> "KGGenStats":
        if self.deduplicate:
            if other.deduplicate:
                new_deduplicate = self.deduplicate + other.deduplicate
            else:
                new_deduplicate = self.deduplicate
        elif other.deduplicate:
            new_deduplicate = other.deduplicate
        else:
            new_deduplicate = None
        new_stats = KGGenStats(
            get_entities=self.get_entities + other.get_entities,
            type_terms=self.type_terms + other.type_terms,
            get_relations_typed=self.get_relations_typed + other.get_relations_typed,
            deduplicate=new_deduplicate,
        )
        return new_stats
