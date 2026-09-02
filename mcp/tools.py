"""Deterministic tool bodies for the kg-gen MCP server.

Every function here is a thin, side-effect-free wrapper over logic that already
exists in ``app`` / ``kg_gen`` — there is no LLM call anywhere in this module.
The functions are kept free of any MCP/FastMCP dependency so they can be unit
tested directly (see ``mcp/tests``); ``mcp/server.py`` registers them as tools.

The server is stateless: any tool that needs the ontology takes the Turtle text
directly, so the caller simply passes the ontology again on each call rather
than the server holding parsed state between calls.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

from app import blobs, settings
from app.utils import (
    parse_ontology_from_string,
    serialize_ontology_to_ttl,
)
from kg_gen.models import (
    Entity,
    EntityType,
    Graph,
    KGGenStats,
    KnowledgeGraph,
    Ontology,
    OntologyPredicate,
    Relation,
    TypedEntity,
)
from kg_gen.steps._2_get_relations import (
    filter_predicates_by_entity_types,
    validate_ontology_conformance,
)
from kg_gen.utils.deduplicate import DeduplicateList


class ConformanceReport(BaseModel):
    """Result of checking entities/relations against an ontology."""

    score: float
    conformant: bool
    errors: list[str]


class SchemaValidationResult(BaseModel):
    """Result of validating a graph payload against the output schema."""

    valid: bool
    errors: list[str]


class GraphWriteResult(BaseModel):
    """Returned by ``serialize_graph``/``apply_clusters`` instead of the full
    ``KnowledgeGraph`` when the caller passes ``output_file``: the graph was
    written to disk rather than returned inline, so a large result never has
    to pass back through the caller's own context/output tokens.
    """

    num_entities: int
    num_relations: int
    output_file: str


class ClusterProposalWriteResult(BaseModel):
    """Returned by ``suggest_clusters`` instead of the full proposal when the
    caller passes ``output_file``.

    A corpus-scale proposal is the one large result with nowhere else to go: it
    is the agent's to review, so it cannot be summarised away, but at a few
    thousand clusters it does not belong in a single context either. Written to
    disk it can be reviewed in slices, by as many agents as it takes.
    """

    num_entity_clusters: int
    num_clustered_entities: int
    output_file: str


_ListItemT = TypeVar("_ListItemT", bound=BaseModel)


def _read_reference(value: str) -> bytes:
    """Read a ``blob:<id>`` handle or a filesystem path, whichever this is."""
    if blobs.is_handle(value):
        try:
            return blobs.load(value)
        except KeyError:
            raise ValueError(
                f"Unknown or expired blob handle {value!r}. Blobs live for "
                f"{settings.BLOB_TTL_SECONDS}s; re-upload via POST /api/blobs."
            )
    with open(value, "rb") as f:
        return f.read()


def resolve_ontology_ttl(ontology_ttl: Optional[str]) -> Optional[str]:
    """The ontology Turtle, given the text itself, a path, or a blob handle.

    The Turtle rides along on every call that needs it -- ~4,700 tokens for a
    real 18-class ontology, on a tool call about a 90-token document -- so
    passing it once as a blob and referring to the handle thereafter is the
    single largest saving available to a caller.

    Disambiguated conservatively: a blob handle announces itself, and a path is
    a single line that exists on disk. Anything else is treated as the Turtle
    itself, so an ontology that happens to be one line long and to name an
    existing file is the only ambiguous case, and reading that file is the more
    useful reading of it.
    """
    if ontology_ttl is None:
        return None
    if blobs.is_handle(ontology_ttl):
        return _read_reference(ontology_ttl).decode("utf-8")
    if "\n" not in ontology_ttl.strip() and os.path.exists(ontology_ttl):
        return _read_reference(ontology_ttl).decode("utf-8")
    return ontology_ttl


def _resolve_list(
    value: list[_ListItemT] | str, model: type[_ListItemT]
) -> list[_ListItemT]:
    """Resolve a tool argument that is either the actual list already, or (as
    a plain string) a **path to a JSON file** containing that list.

    This lets a caller with a large ``typed_entities``/``relations``/cluster
    list pass a file path instead of retyping the whole thing as literal
    tool-call output -- on a real large corpus that retyping can exceed the
    calling model's own output-token limit (confirmed in practice: an
    apply_clusters call carrying ~300 relations inline hit exactly this wall).
    A normal call with an actual JSON array still works exactly as before;
    this is purely additive. A ``blob:<id>`` handle from ``POST /api/blobs``
    works too, and is the form to use when the server shares no filesystem with
    the caller. See SKILL.md's "Large inputs" section.
    """
    if isinstance(value, str):
        raw = json.loads(_read_reference(value))
        return [model.model_validate(item) for item in raw]
    return value


def _maybe_write_output(
    graph: KnowledgeGraph, output_file: Optional[str]
) -> "KnowledgeGraph | GraphWriteResult":
    """Return ``graph`` unchanged if ``output_file`` is falsy (the default,
    fully backward compatible). Otherwise write it to disk as JSON and return
    a small ``GraphWriteResult`` summary instead -- so a large final graph
    doesn't have to pass back through the caller's own context.
    """
    if not output_file:
        return graph
    with open(output_file, "w") as f:
        f.write(graph.model_dump_json(indent=2))
    return GraphWriteResult(
        num_entities=len(graph.entities),
        num_relations=len(graph.relations),
        output_file=output_file,
    )


def parse_ontology(ontology_ttl: str) -> Ontology:
    """Parse a Turtle ontology into structured classes and predicates."""
    onto, _ = parse_ontology_from_string(resolve_ontology_ttl(ontology_ttl))
    return onto


def list_target_types(ontology_ttl: str) -> list[EntityType]:
    """Return the entity types (classes) the agent should extract, sorted by label."""
    onto, _ = parse_ontology_from_string(resolve_ontology_ttl(ontology_ttl))
    return sorted(onto.classes, key=lambda t: t.label)


def suggest_predicates(
    ontology_ttl: str,
    found_entity_types: list[EntityType],
    enforce_domain_conformance: bool = True,
    enforce_range_conformance: bool = True,
) -> list[OntologyPredicate]:
    """Given the entity types actually found in the text, return only the
    ontology predicates whose domain/range are compatible with those types.

    Compatibility accounts for the class hierarchy: a predicate declared on a
    superclass is suggested for an entity of a subclass. Predicates with a
    literal (XSD) range are always kept.

    The domain and range classes come back as label and URI only. Their
    descriptions are what `list_target_types` (or the ontology file) already
    delivers once, whereas here they were repeated inside every predicate that
    shares a domain: on an 18-class/17-predicate ontology that was 12,894
    characters of duplication, or ~5,800 tokens of a ~6,300-token result.
    """
    onto, g = parse_ontology_from_string(resolve_ontology_ttl(ontology_ttl))
    predicates = filter_predicates_by_entity_types(
        entity_types=found_entity_types,
        predicate_domain_range=list(onto.predicates),
        ontology=g,
        enforce_domain_conformance=enforce_domain_conformance,
        enforce_range_conformance=enforce_range_conformance,
    )
    return [_without_class_descriptions(p) for p in predicates]


class PredicateSuggestionGroups(BaseModel):
    """Compatible predicates for many groups of found entity types at once.

    ``legend`` describes each predicate exactly once. ``predicate_sets`` holds
    the *distinct* answers, and ``group_predicate_set`` maps each caller group
    key to the index of its set.
    """

    legend: list[OntologyPredicate]
    predicate_sets: list[list[str]]
    group_predicate_set: dict[str, int]


def _resolve_type_groups(
    value: "dict[str, list[EntityType | str]] | str",
) -> dict[str, list[EntityType]]:
    """Resolve the groups argument: the mapping itself, or a path/blob handle
    to a JSON file holding it.

    Each group's types may be given as ``EntityType`` objects or, more
    compactly, as bare label strings -- the labels are all the filtering needs
    once they match the ontology.
    """
    raw = json.loads(_read_reference(value)) if isinstance(value, str) else value
    if not isinstance(raw, dict):
        raise ValueError(
            "found_entity_types_by_group must be a mapping of group key -> "
            f"list of entity types, got {type(raw).__name__}."
        )
    groups: dict[str, list[EntityType]] = {}
    for key, types in raw.items():
        groups[str(key)] = [
            EntityType(label=t) if isinstance(t, str) else EntityType.model_validate(t)
            for t in types
        ]
    return groups


def suggest_predicates_batch(
    ontology_ttl: str,
    found_entity_types_by_group: "dict[str, list[EntityType | str]] | str",
    enforce_domain_conformance: bool = True,
    enforce_range_conformance: bool = True,
) -> PredicateSuggestionGroups:
    """``suggest_predicates`` for many groups of found entity types in one call.

    Narrowing predicates per *chunk* is what actually constrains the choice;
    narrowing per document *set* does not. Measured on the MuSiQue ontology
    (18 classes, 17 predicates) over a 200-chunk slice: a single chunk yields a
    median of 2 entity types and so 8 candidate predicates, while the union of
    types over the whole slice yields 17 of 18 classes and therefore 17 of 17
    predicates -- no narrowing at all. Per-chunk calls recover that, but cost a
    call and a full description payload per chunk.

    Two collapses make one call enough. Groups sharing a type signature share an
    answer, and distinct signatures still collapse onto far fewer distinct
    answers: those 200 chunks hold 77 distinct type signatures but only 18
    distinct predicate sets. So this returns each predicate's description once
    in ``legend``, the distinct sets in ``predicate_sets``, and an index per
    group -- instead of repeating ~17 descriptions per chunk.

    Group keys are the caller's own labels for the groups, typically chunk ids.
    Pass ``found_entity_types_by_group`` as a path or ``blob:<id>`` handle to
    keep the mapping out of the caller's context (see ``_resolve_list``).
    """
    groups = _resolve_type_groups(found_entity_types_by_group)
    onto, g = parse_ontology_from_string(resolve_ontology_ttl(ontology_ttl))
    predicates = list(onto.predicates)

    answer_by_signature: dict[frozenset[str], list[str]] = {}
    described: dict[str, OntologyPredicate] = {}
    predicate_sets: list[list[str]] = []
    index_by_set: dict[tuple[str, ...], int] = {}
    group_predicate_set: dict[str, int] = {}

    for key, types in groups.items():
        signature = frozenset(t.uri or t.label for t in types)
        labels = answer_by_signature.get(signature)
        if labels is None:
            matched = filter_predicates_by_entity_types(
                entity_types=types,
                predicate_domain_range=predicates,
                ontology=g,
                enforce_domain_conformance=enforce_domain_conformance,
                enforce_range_conformance=enforce_range_conformance,
            )
            labels = [p.label for p in matched]
            answer_by_signature[signature] = labels
            for p in matched:
                described.setdefault(p.label, _without_class_descriptions(p))
        fingerprint = tuple(labels)
        if fingerprint not in index_by_set:
            index_by_set[fingerprint] = len(predicate_sets)
            predicate_sets.append(list(labels))
        group_predicate_set[key] = index_by_set[fingerprint]

    return PredicateSuggestionGroups(
        legend=[described[label] for label in sorted(described)],
        predicate_sets=predicate_sets,
        group_predicate_set=group_predicate_set,
    )


def _without_class_descriptions(predicate: OntologyPredicate) -> OntologyPredicate:
    """The predicate with its domain/range classes reduced to label and URI."""
    trim = lambda types: {  # noqa: E731
        EntityType(label=t.label, uri=t.uri) for t in types
    }
    return predicate.model_copy(
        update={"domain": trim(predicate.domain), "range": trim(predicate.range)}
    )


def validate_conformance(
    typed_entities: list[TypedEntity] | str,
    relations: list[Relation] | str,
    ontology_ttl: Optional[str] = None,
    enforce_domain_conformance: bool = True,
    enforce_range_conformance: bool = True,
    enforce_predicate_conformance: bool = False,
    enforce_type_conformance: bool = False,
) -> ConformanceReport:
    """Score how well the extracted entities/relations conform to the ontology.

    Returns a score in ``[0, 1]`` (1.0 = fully conformant) and a list of
    human-readable violations the agent can use to correct its extraction.

    ``typed_entities``/``relations`` each accept either the list directly, or
    a path to a JSON file containing that list (see ``_resolve_list``) --
    useful once either list is large.
    """
    typed_entities = _resolve_list(typed_entities, TypedEntity)
    relations = _resolve_list(relations, Relation)

    predicate_domain_range = None
    allowed_types = None
    # Keep the RDF graph, not just the extracted classes/predicates: domain and
    # range conformance needs it to honour rdfs:subClassOf.
    rdf_ontology = None
    if ontology_ttl:
        onto, rdf_ontology = parse_ontology_from_string(
            resolve_ontology_ttl(ontology_ttl)
        )
        predicate_domain_range = list(onto.predicates)
        allowed_types = list(onto.classes)

    score, errors_str = validate_ontology_conformance(
        typed_entities=typed_entities,
        relations=relations,
        predicate_domain_range=predicate_domain_range,
        enforce_domain_conformance=enforce_domain_conformance,
        enforce_range_conformance=enforce_range_conformance,
        enforce_predicate_conformance=enforce_predicate_conformance,
        enforce_type_conformance=enforce_type_conformance,
        allowed_types=allowed_types,
        ontology=rdf_ontology,
    )
    errors = errors_str.split("; ") if errors_str else []
    return ConformanceReport(score=score, conformant=score >= 1.0, errors=errors)


def validate_graph_schema(graph: dict) -> SchemaValidationResult:
    """Structurally validate a graph payload against the ``KnowledgeGraph`` schema."""
    try:
        KnowledgeGraph.model_validate(graph)
        return SchemaValidationResult(valid=True, errors=[])
    except ValidationError as exc:
        errors = [
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        ]
        return SchemaValidationResult(valid=False, errors=errors)


def serialize_graph(
    typed_entities: list[TypedEntity] | str,
    relations: list[Relation] | str,
    ontology_ttl: Optional[str] = None,
    output_file: Optional[str] = None,
) -> "KnowledgeGraph | GraphWriteResult":
    """Build the canonical output graph from extracted entities and relations.

    Handles URI reconciliation against the ontology, literal-vs-object detection,
    XSD datatype guessing, and ontology-extension detection.

    ``typed_entities``/``relations`` each accept either the list directly, or
    a path to a JSON file containing that list -- useful once either list is
    large. If ``output_file`` is given, the resulting graph is written there
    instead of being returned inline, and this returns a small
    ``GraphWriteResult`` summary instead of the full ``KnowledgeGraph``.
    """
    typed_entities = _resolve_list(typed_entities, TypedEntity)
    relations = _resolve_list(relations, Relation)

    onto = None
    rdf_ontology = None
    if ontology_ttl:
        onto, rdf_ontology = parse_ontology_from_string(
            resolve_ontology_ttl(ontology_ttl)
        )

    graph = Graph(
        typed_entities=set(typed_entities),
        relations_wo_class_assertions=relations,
    )
    return _maybe_write_output(
        graph.to_knowledge_graph(KGGenStats(), onto, rdf_ontology), output_file
    )


def convert_ontology(
    classes: list[EntityType], predicates: list[OntologyPredicate]
) -> str:
    """Serialize classes and predicates (e.g. discovered extensions) to Turtle."""
    return serialize_ontology_to_ttl(classes, predicates)


class EntityCluster(BaseModel):
    """A candidate group of near-duplicate entities, proposed by ``suggest_clusters``.

    Nothing is merged yet — this is a proposal for the agent to review. Drop
    members that don't actually belong, split the cluster if it lumps together
    distinct things, or swap ``representative`` for a different (or new)
    canonical surface form. Pass the reviewed result to ``apply_clusters``.
    """

    members: list[TypedEntity]
    representative: TypedEntity


class EdgeCluster(BaseModel):
    """Same shape as ``EntityCluster``, but for predicates/edges.

    Not proposed by ``suggest_clusters`` (predicates already come from the
    ontology's controlled vocabulary, so they don't need semantic
    deduplication) — this exists for callers who want to hand ``apply_clusters``
    manually authored edge merges.
    """

    members: list[Entity]
    representative: Entity


class ClusterProposal(BaseModel):
    entity_clusters: list[EntityCluster]
    edge_clusters: list[EdgeCluster]


def _cluster(items, threshold: float, model) -> list[tuple]:
    """Run embedding clustering and return (representative, members) pairs for
    every cluster with more than one member. Does not merge anything."""
    if not items:
        return []

    dedup = DeduplicateList(threshold)
    dedup.deduplicate(items, model=model)

    grouped = defaultdict(list)
    for item in items:
        singular = dedup.original_map[item.surface_form]
        canonical_surface_form = dedup.items_map[singular]
        grouped[canonical_surface_form].append(item)

    return [
        (dedup.surface_form2entity_map[rep_surface_form], members)
        for rep_surface_form, members in grouped.items()
        if len(members) > 1
    ]


def suggest_clusters(
    typed_entities: list[TypedEntity] | str,
    entity_similarity_threshold: float = 0.8,
    retrieval_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
    output_file: Optional[str] = None,
) -> "ClusterProposal | ClusterProposalWriteResult":
    """Propose candidate duplicate clusters for entities using local embeddings.
    This only proposes — nothing is merged.

    Predicates are not clustered here: they come from the ontology's
    controlled vocabulary (see ``suggest_predicates``), so they are already
    canonical and don't need semantic deduplication.

    The agent, not the embedding model, has the final say: review each
    cluster (drop members that don't belong, split it, or change the
    representative), then call ``apply_clusters`` with the reviewed result to
    actually update the graph. Clusters with no duplicates are omitted.

    Parameters
    ----------
    typed_entities:
        Either the list directly, or a path to a JSON file containing that
        list -- useful once it's large (e.g. clustering across many
        documents at once; see SKILL.md's "Large inputs" section). The
        returned proposal only includes clusters with more than one member,
        so it's typically much smaller than the input regardless.
    entity_similarity_threshold:
        Cosine-similarity threshold above which two entities are proposed as a
        cluster. Higher = stricter (fewer candidates); lower = more aggressive.
        Default 0.8.
    retrieval_model:
        Sentence-transformers model used to embed surface forms. ``None`` falls
        back to the deduplication library's built-in default encoder.
    output_file:
        Write the proposal here as JSON and return a small summary instead of
        the proposal itself. Use this at corpus scale: reviewing a few thousand
        clusters inline costs more context than any one reviewer has, whereas a
        file can be read in slices by several reviewers in parallel.
    """
    typed_entities = _resolve_list(typed_entities, TypedEntity)

    model = None
    if retrieval_model:
        from kg_gen.kg_gen import _get_shared_sentence_transformer

        model = _get_shared_sentence_transformer(retrieval_model)

    entity_clusters = [
        EntityCluster(representative=rep, members=members)
        for rep, members in _cluster(typed_entities, entity_similarity_threshold, model)
    ]
    proposal = ClusterProposal(entity_clusters=entity_clusters, edge_clusters=[])
    if not output_file:
        return proposal
    with open(output_file, "w") as f:
        f.write(proposal.model_dump_json(indent=2))
    return ClusterProposalWriteResult(
        num_entity_clusters=len(entity_clusters),
        num_clustered_entities=sum(len(c.members) for c in entity_clusters),
        output_file=output_file,
    )


def apply_clusters(
    typed_entities: list[TypedEntity] | str,
    relations: list[Relation] | str,
    entity_clusters: list[EntityCluster] | str,
    edge_clusters: list[EdgeCluster] | str,
    ontology_ttl: Optional[str] = None,
    output_file: Optional[str] = None,
) -> "KnowledgeGraph | GraphWriteResult":
    """Merge entities/edges per the (agent-reviewed) clusters and rebuild the
    canonical graph. This is the deterministic half of deduplication — it does
    not compute similarity itself, it just applies decisions already made
    (typically the output of ``suggest_clusters`` after the agent has checked
    it).

    Every member of a cluster is merged into that cluster's ``representative``;
    anything not named in a cluster passes through unchanged. Provenance from
    every merged entity/relation is aggregated onto the result. Pass an empty
    list for either cluster kind to skip that merge entirely.

    Every list argument accepts either the list directly, or a path to a JSON
    file containing that list. This matters most here: merging across many
    documents means ``typed_entities``/``relations`` can be hundreds of items,
    and retyping them as literal tool-call output can exceed the calling
    model's own output-token limit (confirmed in practice on a real ~300-
    relation merge). If ``output_file`` is given, the merged graph is written
    there instead of being returned inline, returning a small
    ``GraphWriteResult`` summary instead of the full ``KnowledgeGraph`` — the
    result can be just as large as the input. See SKILL.md's "Large inputs"
    section for guidance on when to use file paths instead of inline data.
    """
    typed_entities = _resolve_list(typed_entities, TypedEntity)
    relations = _resolve_list(relations, Relation)
    entity_clusters = _resolve_list(entity_clusters, EntityCluster)
    edge_clusters = _resolve_list(edge_clusters, EdgeCluster)

    entity_map = {
        member.surface_form: cluster.representative
        for cluster in entity_clusters
        for member in cluster.members
    }
    edge_map = {
        member.surface_form: cluster.representative
        for cluster in edge_clusters
        for member in cluster.members
    }

    def canon_entity(entity: Entity) -> Entity:
        return entity_map.get(entity.surface_form, entity)

    def canon_edge(edge: Entity) -> Entity:
        return edge_map.get(edge.surface_form, edge)

    canonical_entities: dict[str, TypedEntity] = {}
    entity_provenance: dict[str, list[str]] = defaultdict(list)
    for entity in typed_entities:
        canonical = canon_entity(entity)
        canonical_entities[canonical.surface_form] = canonical
        entity_provenance[canonical.surface_form].extend(entity.provenance_ids)
    for surface_form, canonical in canonical_entities.items():
        canonical.provenance_ids = entity_provenance[surface_form]

    merged_relations: dict[tuple[str, str, str], Relation] = {}
    for relation in relations:
        new_subject = canon_entity(relation.subject)
        new_predicate = canon_edge(relation.predicate)
        new_object = canon_entity(relation.object)
        key = (
            new_subject.surface_form,
            new_predicate.surface_form,
            new_object.surface_form,
        )
        if key in merged_relations:
            merged_relations[key].provenance_ids.extend(relation.provenance_ids)
        else:
            merged_relations[key] = Relation(
                subject=new_subject,
                predicate=new_predicate,
                object=new_object,
                provenance_ids=list(relation.provenance_ids),
            )

    onto = None
    rdf_ontology = None
    if ontology_ttl:
        onto, rdf_ontology = parse_ontology_from_string(
            resolve_ontology_ttl(ontology_ttl)
        )

    graph = Graph(
        typed_entities=set(canonical_entities.values()),
        relations_wo_class_assertions=list(merged_relations.values()),
        entity_clusters={
            c.representative.surface_form: c.members
            for c in entity_clusters
            if len(c.members) > 1
        },
        edge_clusters={
            c.representative.surface_form: c.members
            for c in edge_clusters
            if len(c.members) > 1
        },
    )
    return _maybe_write_output(
        graph.to_knowledge_graph(KGGenStats(), onto, rdf_ontology), output_file
    )
