import json
import math
import re

import dspy
from pydantic import BaseModel, Field, computed_field
from typing import Optional, List, Set

from typing_extensions import TypeVar


def _guess_xsd_type(value: str) -> str:
    """Guess the XSD datatype of a string value."""
    if not value:
        return "xsd:string"

    # Try integer (reject underscore grouping, which int() would otherwise accept)
    if "_" not in value:
        try:
            int(value)
            return "xsd:integer"
        except ValueError:
            pass

        # Try float/decimal, but exclude non-finite values ("nan", "inf", "-inf"),
        # which float() parses successfully but are not valid xsd:decimal literals.
        try:
            parsed = float(value)
            if math.isfinite(parsed):
                return "xsd:decimal"
        except ValueError:
            pass

    # Try boolean
    if value.lower() in ("true", "false"):
        return "xsd:boolean"

    # Try date (ISO 8601 basic check: YYYY-MM-DD)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return "xsd:date"

    # Try dateTime (YYYY-MM-DDTHH:MM:SS)
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value):
        return "xsd:dateTime"

    return "xsd:string"


_XSD_NAMESPACE = "http://www.w3.org/2001/XMLSchema#"


def _is_xsd_uri(uri: Optional[str]) -> bool:
    """True if `uri` refers to an XSD datatype (full URI or 'xsd:' shorthand)."""
    return bool(uri) and (uri.startswith("xsd:") or uri.startswith(_XSD_NAMESPACE))


def _normalize_xsd_datatype(uri: str) -> str:
    """Return the 'xsd:<type>' shorthand for an XSD datatype URI."""
    if uri.startswith(_XSD_NAMESPACE):
        return "xsd:" + uri[len(_XSD_NAMESPACE) :]
    return uri


def _predicate_datatype(predicate: "OntologyPredicate") -> Optional[str]:
    """Return the XSD datatype implied by a datatype predicate, else None.

    A predicate is treated as a datatype property if it is declared as one, or
    if every URI in its range is an XSD datatype. When the range names exactly
    one XSD datatype, that datatype is returned so it can be used instead of
    guessing from the literal value.
    """
    xsd_ranges = [r.uri for r in predicate.range if _is_xsd_uri(r.uri)]
    is_datatype = predicate.property_type == "owl:DatatypeProperty" or (
        bool(predicate.range) and len(xsd_ranges) == len(predicate.range)
    )
    if not is_datatype:
        return None
    if len(xsd_ranges) == 1:
        return _normalize_xsd_datatype(xsd_ranges[0])
    return ""


class Entity(BaseModel):
    surface_form: str = Field(..., description="The surface form of the entity")
    uri: Optional[str] = Field(
        default=None, description="Optional identifier or URI for the entity"
    )
    description: Optional[str] = Field(
        default=None, description="A short description of the entity"
    )

    # Identity is based on surface_form only. URIs are excluded because LLMs
    # generate them inconsistently; including them would over-count otherwise
    # identical entities. Reconciliation to an ontology still happens later via
    # surface form / label (see Graph.to_knowledge_graph).
    def __hash__(self):
        return hash(self.surface_form)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return NotImplemented
        return self.surface_form == other.surface_form

    def __str__(self):
        return self.surface_form

    def __repr__(self):
        return f"Entity(surface_form='{self.surface_form}', uri='{self.uri}')"


class EntityType(BaseModel):
    label: str = Field(..., description="Label or name of the entity type")
    uri: Optional[str] = Field(
        default=None, description="Identifier or URI for the entity type"
    )
    description: Optional[str] = Field(
        default=None, description="A short description of the entity type"
    )

    # Identity is based on label only; URIs are excluded (see Entity).
    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if not isinstance(other, EntityType):
            return NotImplemented
        return self.label == other.label


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

    # Identity is based on (surface_form, type); URIs are excluded (see Entity).
    # Type is kept so that homonyms with distinct types (e.g. "Mercury" the
    # planet vs. the element) are not collapsed.
    def __hash__(self):
        return hash((self.surface_form, self.type))

    def __eq__(self, other):
        if not isinstance(other, TypedEntity):
            return NotImplemented
        return self.surface_form == other.surface_form and self.type == other.type

    def __repr__(self):
        return f"TypedEntity(surface_form='{self.surface_form}', uri='{self.uri}', type={self.type!r}, provenance={self.provenance_ids})"

    @property
    def type_entity(self) -> Entity | None:
        if self.type is not None:
            return Entity(surface_form=self.type.label, uri=self.type.uri)
        return None

    @property
    def class_assertion(
        self, assertion_predicate: Entity = Entity(surface_form="is a", uri="rdf:type")
    ) -> Optional["Relation"]:
        if self.type_entity is not None:
            return Relation(
                subject=self,
                predicate=assertion_predicate,
                object=self.type_entity,
                provenance_ids=self.provenance_ids,
            )
        return None


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


class ExtractTypedEntities(dspy.Signature):
    """Extract key entities from the source text and predict their type/class. Extracted entities are subjects or objects.
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
    typed_entities: list[TypedEntity] = dspy.OutputField(
        desc="THOROUGH list of key entities and their types/classes. Every type should come from a list of types, if provided."
    )


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
            self.subject.__hash__() ^ self.predicate.__hash__() ^ self.object.__hash__()
        )


class OntologyPredicate(BaseModel):
    domain: Set[EntityType] = Field(
        default_factory=set,
        description="Predicate domain. Any subject/head of this predicate should have a type from this list.",
        examples=[[{"label": "Person", "uri": "http://www.wikidata.org/entity/Q5"}]],
    )
    range: Set[EntityType] = Field(
        default_factory=set,
        description="Predicate range. Any object/tail of this predicate should have a type from this list.",
        examples=[[{"label": "Person", "uri": "http://schema.org/Person"}]],
    )
    label: str = Field(..., description="Predicate label", examples=["is_brother_of"])
    uri: Optional[str] = Field(
        default=None, description="Identifier or URI of the predicate"
    )
    description: Optional[str] = Field(
        default=None, description="A short description of the predicate"
    )
    property_type: Optional[str] = Field(
        default="owl:ObjectProperty",
        description="Type of the property, e.g., 'owl:ObjectProperty' or 'owl:DatatypeProperty'.",
    )

    # Identity is based on label only; URIs are excluded (see Entity).
    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if not isinstance(other, OntologyPredicate):
            return NotImplemented
        return self.label == other.label


class Ontology(BaseModel):
    classes: Set[EntityType] = Field(default_factory=set)
    predicates: Set[OntologyPredicate] = Field(default_factory=set)


class ExtractTextRelations(dspy.Signature):
    """
    Extract subject-predicate-object triples from the source text.

    Rules:
    1. Only use entities provided in `entities_with_types`.
    2. Only use predicates provided in `allowed_predicates`.
    3. Ensure domain/range conformance:
       - The subject's type must match the predicate's domain.
       - The object's type must match the predicate's range.
    4. Be thorough, accurate, and faithful to the source text.
    """

    source_text: str = dspy.InputField()
    entities_with_types: str = dspy.InputField(
        desc="List of available entities and their types in 'Entity (Type)' format."
    )
    allowed_predicates: str = dspy.InputField(
        desc="List of allowed predicates and their domain/range constraints in 'Predicate: [Domain] -> [Range]' format."
    )
    context: Optional[str] = dspy.InputField(
        default=None,
        desc="Optional context for relation extraction.",
    )
    relations: list[Relation] = dspy.OutputField(
        desc="List of extracted subject-predicate-object triples."
    )


class LMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class StepStats(BaseModel):
    lm_usage: LMUsage = Field(default_factory=LMUsage)
    execution_time: float = 0.0

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
    extract_entities: StepStats = Field(default_factory=StepStats)
    get_entities: StepStats = Field(default_factory=StepStats)
    type_terms: StepStats = Field(default_factory=StepStats)
    get_relations_typed: StepStats = Field(default_factory=StepStats)
    deduplicate: Optional[StepStats] = None
    class_usage: dict[str, int] = Field(default_factory=dict)
    predicate_usage: dict[str, int] = Field(default_factory=dict)

    @computed_field
    @property
    def overall_usage(self) -> StepStats:
        overall_stats = StepStats(lm_usage=LMUsage(), execution_time=0.0)
        for step_stats in [
            self.extract_entities,
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
        new_class_usage = self.class_usage.copy()
        for k, v in other.class_usage.items():
            new_class_usage[k] = new_class_usage.get(k, 0) + v

        new_predicate_usage = self.predicate_usage.copy()
        for k, v in other.predicate_usage.items():
            new_predicate_usage[k] = new_predicate_usage.get(k, 0) + v

        new_stats = KGGenStats(
            extract_entities=self.extract_entities + other.extract_entities,
            get_entities=self.get_entities + other.get_entities,
            type_terms=self.type_terms + other.type_terms,
            get_relations_typed=self.get_relations_typed + other.get_relations_typed,
            deduplicate=new_deduplicate,
            class_usage=new_class_usage,
            predicate_usage=new_predicate_usage,
        )
        return new_stats


class InputData(BaseModel):
    text: str
    id: str
    terms: Optional[list[TypedEntity | str]] = Field(default_factory=list)


class EntityTypeReference(BaseModel):
    label: Optional[str] = None
    uri: Optional[str] = None


class OutputEntity(BaseModel):
    surface_form: str
    uri: Optional[str] = None
    description: Optional[str] = None
    type: Optional[EntityTypeReference] = None
    provenance_ids: List[str] = Field(default_factory=list)


class PredicateReference(BaseModel):
    label: Optional[str] = None
    uri: Optional[str] = None


class OutputRelation(BaseModel):
    subject_id: str
    predicate: PredicateReference
    object_id: Optional[str] = None
    object_value: Optional[str] = None
    object_datatype: Optional[str] = None
    is_literal: bool
    provenance_ids: List[str] = Field(default_factory=list)


class ClassExtension(BaseModel):
    label: str
    uri: str
    description: Optional[str] = None
    superclasses: List[str] = Field(default_factory=list)


class PredicateExtension(BaseModel):
    label: str
    uri: str
    description: Optional[str] = None
    domain: List[EntityTypeReference] = Field(default_factory=list)
    range: List[EntityTypeReference] = Field(default_factory=list)


class OntologyExtensions(BaseModel):
    classes: List[ClassExtension] = Field(default_factory=list)
    predicates: List[PredicateExtension] = Field(default_factory=list)


class Clusters(BaseModel):
    entities: Optional[dict[str, list[OutputEntity]]] = None
    edges: Optional[dict[str, list[Entity]]] = None


class KnowledgeGraph(BaseModel):
    entities: dict[str, OutputEntity]
    relations: list[OutputRelation]
    ontology_extensions: Optional[OntologyExtensions] = None
    clusters: Optional[Clusters] = None
    stats: dict


class Graph(BaseModel):
    typed_entities: set[TypedEntity] = Field(
        ..., description="All entities including additional ones from response"
    )
    relations_wo_class_assertions: list[Relation] = Field(
        ..., description="List of (subject, predicate, object) triples"
    )
    entity_clusters: Optional[dict[str, list[TypedEntity | Entity]]] = None
    edge_clusters: Optional[dict[str, list[Entity]]] = None

    entity_metadata: dict[TypedEntity, set[str]] | None = None

    output_class_assertions: bool = True

    @computed_field
    @property
    def entities(self) -> set[Entity]:
        out = {
            Entity(surface_form=te.surface_form, uri=te.uri)
            for te in self.typed_entities
        }
        out |= {
            te.type_entity for te in self.typed_entities if te.type_entity is not None
        }
        return out

    @computed_field
    @property
    def relations(self) -> list[Relation]:
        if self.output_class_assertions:
            class_assertions = [
                te.class_assertion
                for te in self.typed_entities
                if te.class_assertion is not None
            ]
            out = self.relations_wo_class_assertions + class_assertions
            return out
        else:
            return self.relations_wo_class_assertions

    @computed_field
    @property
    def edges(self) -> set[Entity]:
        out = {relation.predicate for relation in self.relations}
        return out

    def to_file(self, file_path: str):
        """
        Save the graph to a file.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

    def to_knowledge_graph(
        self, stats: KGGenStats, ontology: Optional[Ontology] = None
    ) -> "KnowledgeGraph":
        entities_dict = {}
        surface_form_to_id = {}

        # Prepare ontology mappings for matching
        uri_to_class = {}
        label_to_class = {}
        uri_to_pred = {}
        label_to_pred = {}

        if ontology:
            for c in ontology.classes:
                if c.uri:
                    uri_to_class[c.uri] = c
                label_to_class[c.label] = c
            for p in ontology.predicates:
                if p.uri:
                    uri_to_pred[p.uri] = p
                label_to_pred[p.label] = p

        # 1. Map TypedEntities to OutputEntities
        for i, te in enumerate(
            sorted(list(self.typed_entities), key=lambda x: x.surface_form)
        ):
            entity_id = f"e{i}"
            surface_form_to_id[te.surface_form] = entity_id

            type_ref = None
            if te.type:
                type_uri = te.type.uri
                type_label = te.type.label

                if ontology:
                    matched_class = None
                    if te.type.uri:
                        matched_class = uri_to_class.get(te.type.uri)
                    if not matched_class:
                        matched_class = label_to_class.get(te.type.label)

                    if matched_class:
                        type_uri = matched_class.uri
                        type_label = matched_class.label

                type_ref = EntityTypeReference(uri=type_uri, label=type_label)

            entities_dict[entity_id] = OutputEntity(
                surface_form=te.surface_form,
                uri=te.uri,
                description=te.description,
                type=type_ref,
                provenance_ids=te.provenance_ids,
            )

        # 2. Map relations
        output_relations = []
        for rel in self.relations_wo_class_assertions:
            subj_id = surface_form_to_id.get(rel.subject.surface_form)
            obj_id = surface_form_to_id.get(rel.object.surface_form)

            pred_label = rel.predicate.surface_form
            pred_uri = rel.predicate.uri

            matched_pred = None
            if ontology:
                if rel.predicate.uri:
                    matched_pred = uri_to_pred.get(rel.predicate.uri)
                if not matched_pred:
                    matched_pred = label_to_pred.get(rel.predicate.surface_form)

                if matched_pred:
                    pred_label = matched_pred.label
                    pred_uri = matched_pred.uri

            if subj_id:
                # Prefer the ontology's own knowledge: a datatype predicate always
                # yields a literal object (with the datatype from its range when
                # available). Otherwise fall back to "literal iff the object was
                # not matched to an extracted entity".
                range_datatype = (
                    _predicate_datatype(matched_pred) if matched_pred else None
                )
                is_literal = range_datatype is not None or obj_id is None

                if is_literal:
                    obj_value = rel.object.surface_form
                    obj_datatype = range_datatype or _guess_xsd_type(obj_value)
                    final_obj_id = None
                else:
                    obj_value = None
                    obj_datatype = None
                    final_obj_id = obj_id

                output_relations.append(
                    OutputRelation(
                        subject_id=subj_id,
                        predicate=PredicateReference(label=pred_label, uri=pred_uri),
                        object_id=final_obj_id,
                        is_literal=is_literal,
                        object_value=obj_value,
                        object_datatype=obj_datatype,
                        provenance_ids=rel.provenance_ids,
                    )
                )

        # 3. Handle ontology extensions
        classes_ext = []
        predicates_ext = []

        if ontology:
            known_class_uris = {c.uri for c in ontology.classes if c.uri}
            known_class_labels = {c.label for c in ontology.classes}
            known_predicate_uris = {p.uri for p in ontology.predicates if p.uri}
            known_predicate_labels = {p.label for p in ontology.predicates}

            # Use sets to keep track of what we already added to extensions in this call
            added_ext_class_uris = set()
            added_ext_class_labels = set()
            added_ext_pred_uris = set()
            added_ext_pred_labels = set()

            for te in self.typed_entities:
                if te.type:
                    is_known = False
                    if te.type.uri and te.type.uri in known_class_uris:
                        is_known = True
                    elif te.type.label in known_class_labels:
                        is_known = True

                    if not is_known:
                        if te.type.uri:
                            if te.type.uri not in added_ext_class_uris:
                                classes_ext.append(
                                    ClassExtension(
                                        label=te.type.label,
                                        uri=te.type.uri,
                                        description=te.type.description,
                                    )
                                )
                                added_ext_class_uris.add(te.type.uri)
                        else:
                            if te.type.label not in added_ext_class_labels:
                                # ClassExtension REQUIRES uri (str)
                                # If it's an extension and has no URI, we might need to skip or generate
                                # Currently we only add it if it has a URI, as per previous logic
                                pass

            for rel in self.relations_wo_class_assertions:
                is_known = False
                if rel.predicate.uri and rel.predicate.uri in known_predicate_uris:
                    is_known = True
                elif rel.predicate.surface_form in known_predicate_labels:
                    is_known = True

                if not is_known:
                    if rel.predicate.uri:
                        if rel.predicate.uri not in added_ext_pred_uris:
                            predicates_ext.append(
                                PredicateExtension(
                                    label=rel.predicate.surface_form,
                                    uri=rel.predicate.uri,
                                    description=rel.predicate.description,
                                )
                            )
                            added_ext_pred_uris.add(rel.predicate.uri)
                    else:
                        if rel.predicate.surface_form not in added_ext_pred_labels:
                            # PredicateExtension REQUIRES uri (str)
                            pass

        extensions = OntologyExtensions(classes=classes_ext, predicates=predicates_ext)

        # 4. Prepare stats
        stats_dict = stats.model_dump()
        # Ensure overall_usage is included as it's a computed field
        stats_dict["overall_usage"] = stats.overall_usage.model_dump()

        # 5. Handle clusters
        out_entity_clusters = {}
        if self.entity_clusters:
            for canonical_form, cluster in self.entity_clusters.items():
                entity_id = surface_form_to_id.get(canonical_form)
                if entity_id:
                    cluster_members = []
                    for member in cluster:
                        if isinstance(member, TypedEntity):
                            cluster_members.append(
                                OutputEntity(
                                    surface_form=member.surface_form,
                                    uri=member.uri,
                                    description=member.description,
                                    type=EntityTypeReference(uri=member.type.uri)
                                    if member.type
                                    else None,
                                    provenance_ids=member.provenance_ids,
                                )
                            )
                        else:
                            cluster_members.append(
                                OutputEntity(
                                    surface_form=member.surface_form,
                                    uri=member.uri,
                                    description=member.description,
                                )
                            )
                    out_entity_clusters[entity_id] = cluster_members

        out_edge_clusters = {}
        if self.edge_clusters:
            for canonical_form, cluster in self.edge_clusters.items():
                out_edge_clusters[canonical_form] = cluster

        clusters = None
        if out_entity_clusters or out_edge_clusters:
            clusters = Clusters(
                entities=out_entity_clusters if out_entity_clusters else None,
                edges=out_edge_clusters if out_edge_clusters else None,
            )

        return KnowledgeGraph(
            entities=entities_dict,
            relations=output_relations,
            ontology_extensions=extensions
            if (extensions.classes or extensions.predicates)
            else None,
            clusters=clusters,
            stats=stats_dict,
        )

    def stats(self, name: Optional[str] = None):
        """
        Print the stats of the graph.
        """
        return f"{name or 'Graph'} with:\n\t{len(self.entities)} entities\n\t{len(self.edges)} edges\n\t{len(self.relations)} relations"
