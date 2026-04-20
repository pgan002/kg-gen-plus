import json

import dspy
from pydantic import BaseModel, Field
from typing import Optional, List


class TextEntities(dspy.Signature):
    """Extract key entities from the source text. Extracted entities are subjects or objects.
    This is for an extraction task, please be THOROUGH and accurate to the reference text."""

    source_text: str = dspy.InputField()
    context: Optional[str] = dspy.InputField(
        default=None,
        desc="Optional context. If provided, consider it when extracting entities.",
    )
    types_to_extract: list[str] = dspy.InputField(
        default=["Person", "Location", "Organization"],
        desc="List of entity types to extract. If empty, all entity types are extracted.",
    )
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class ConversationEntities(dspy.Signature):
    """Extract key entities from the conversation Extracted entities are subjects or objects.
    Consider both explicit entities and participants in the conversation.
    This is for an extraction task, please be THOROUGH and accurate."""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class EntitiesResponse(BaseModel):
    """Structured response for entity extraction."""

    entities: List[str]


class TypedEntity(BaseModel):
    """Structured response for entity typing."""

    entity: str
    type: str | None = Field(
        None,
        description="Type of the entity, e.g., 'Person', 'Location', 'Organization', etc. If no type is suitable, leave empty or put None.",
    )


class TypedEntities(dspy.Signature):
    """
    Find the mentions of the given entities in the source_text and predict their type/class.
    If types are provided, only use these types as target types/classes. Otherwise, if no types are provided,
    predict your own types.
    """

    entities: List[str] = dspy.InputField()
    context: Optional[str] = dspy.InputField(
        default=None,
        desc="Optional context. If provided, consider it when extracting entities.",
    )
    types: Optional[list[str]] = dspy.InputField(
        default=["Person", "Location", "Organization"],
        desc="List of entity types to extract. If empty, all entity types are extracted.",
    )
    source_text: str = dspy.InputField()
    typed_entities: list[TypedEntity] = dspy.OutputField(
        desc="List of all entities and their types/classes. Every type should come from a list of types, if provided."
    )


class Relation(BaseModel):
    """Knowledge graph subject-predicate-object tuple."""

    subject: str = dspy.InputField(desc="Subject entity", examples=["Kevin"])
    predicate: str = dspy.InputField(desc="Predicate", examples=["is brother of"])
    object: str = dspy.InputField(desc="Object entity", examples=["Vicky"])

    def __hash__(self):
        return (
            self.subject.__hash__() ^ self.predicate.__hash__() ^ self.object.__hash__()
        )


class OntologyPredicate(BaseModel):
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
    predicate_domain_range: Optional[list[OntologyPredicate]] = dspy.InputField(
        default=None,
        desc="List of predicate specification with domains and ranges. If not provided, use any predicates.",
    )
    context: Optional[str] = dspy.InputField(
        default=None,
        desc="Optional context. If provided, consider it when extracting relations.",
    )
    relations: list[Relation] = dspy.OutputField(
        desc="List of subject-predicate-object tuples. Be thorough. For each entity also add a relation to its type"
    )


class Graph(BaseModel):
    entities: set[str] = Field(
        ..., description="All entities including additional ones from response"
    )
    edges: set[str] = Field(..., description="All edges")
    relations: list[Relation] = Field(
        ..., description="List of (subject, predicate, object) triples"
    )
    entity_clusters: Optional[dict[str, set[str]]] = None
    edge_clusters: Optional[dict[str, set[str]]] = None

    entity_metadata: dict[str, set[str]] | None = None

    @staticmethod
    def from_file(file_path: str) -> "Graph":
        """
        Load the graph from a file.
        Fix graph entities and edges for missing ones defined in relations.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            graph = Graph.model_validate(data)

        # Fix graph entities and edges
        for relation in graph.relations:
            if relation.subject not in graph.entities:
                graph.entities.add(relation.subject)
            if relation.predicate not in graph.edges:
                graph.edges.add(relation.predicate)
            if relation.object not in graph.entities:
                graph.entities.add(relation.object)

        return graph

    def to_file(self, file_path: str):
        """
        Save the graph to a file.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    def stats(self, name: Optional[str] = None):
        """
        Print the stats of the graph.
        """
        print(
            f"{name or 'Graph'} with:\n\t{len(self.entities)} entities\n\t{len(self.edges)} edges\n\t{len(self.relations)} relations"
        )
