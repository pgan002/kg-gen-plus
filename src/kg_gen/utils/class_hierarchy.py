"""Reading ``rdfs:subClassOf`` out of an ontology.

Shared by the two places that need to know whether one class subsumes another:
domain/range conformance in ``kg_gen.steps._2_get_relations``, and the decision
in ``kg_gen.utils.deduplicate`` about whether two entities with the same
normalized surface form are actually the same thing.
"""

from __future__ import annotations

from typing import Optional

from rdflib import Graph as RDFGraph, RDFS, URIRef

from kg_gen.models import EntityType
from kg_gen.utils.label_matching import normalize_label, resolve_uri_by_label


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


def class_closure_uris(
    ontology: Optional[RDFGraph], entity_type: Optional[EntityType]
) -> set[URIRef]:
    """URIs of ``entity_type`` plus all its ancestors via ``rdfs:subClassOf``.

    Returns an empty set when there is no ontology graph or the type cannot be
    resolved in it; callers must read that as "hierarchy unknown" rather than
    "matches nothing".

    Resolution prefers ``entity_type.uri`` and falls back to matching
    ``rdfs:label``. The URI is the more reliable key: extracted types often carry
    a uri with a null label, and a label lookup spans every labelled subject in
    the graph, so a predicate sharing a class's label could shadow it.
    """
    if entity_type is None:
        return set()

    # With no ontology graph there is no hierarchy to walk, but the class is
    # still a member of its own closure. Returning an empty set here instead
    # made domain/range filtering vacuous: `filter_predicates_by_entity_types`
    # intersects this closure with each predicate's domain and range, so an
    # empty closure discards every predicate that declares either. Callers that
    # pass `predicate_domain_range` without `ontology` -- the documented way to
    # constrain predicates without an RDF graph -- therefore got *no* usable
    # predicates, which then either silently degraded to "any predicate
    # allowed" or, with `enforce_predicate_conformance`, returned no relations
    # at all without so much as a model call.
    if ontology is None:
        return {URIRef(entity_type.uri)} if entity_type.uri else set()

    if entity_type.uri:
        candidate = URIRef(entity_type.uri)
        if (candidate, None, None) in ontology:
            return get_all_superclasses(ontology, candidate)

    # Compare as normalized strings so a language-tagged label, a different
    # case, or a separator difference still resolves; SKOS labels count too.
    uri = resolve_uri_by_label(ontology, entity_type.label)
    if uri is not None:
        return get_all_superclasses(ontology, uri)

    return set()


def resolve_class_uri(
    ontology: Optional[RDFGraph], entity_type: Optional[EntityType]
) -> Optional[URIRef]:
    """The ontology URI ``entity_type`` denotes, by its uri or any of its labels."""
    if ontology is None or entity_type is None:
        return None
    if entity_type.uri:
        candidate = URIRef(entity_type.uri)
        if (candidate, None, None) in ontology:
            return candidate
    return resolve_uri_by_label(ontology, entity_type.label)


def types_compatible(
    ontology: Optional[RDFGraph],
    first: Optional[EntityType],
    second: Optional[EntityType],
) -> bool:
    """Whether two entity types could describe the same thing.

    True when either side is untyped (an untyped mention contradicts nothing),
    when both denote the same class, or when one class subsumes the other -- so
    ``Air Force`` typed ``GovernmentOrganization`` and ``Air Force`` typed
    ``Organization`` are the same entity, the second simply less specific.

    False for two *different* classes with no subsumption between them, which is
    the case this exists for: ``Alan`` the Person and ``Alans`` the Organization
    normalize to the same surface form but are not the same entity. Siblings are
    refused even though some are one entity the model typed inconsistently --
    keeping two nodes states the disagreement rather than hiding it in a merge.

    Without an ontology to consult there is no way to tell a subclass from a
    sibling, so this falls back to the types naming the same class, compared the
    same forgiving way labels are matched everywhere else.
    """
    if first is None or second is None:
        return True

    first_uri = resolve_class_uri(ontology, first)
    second_uri = resolve_class_uri(ontology, second)
    if first_uri is not None and second_uri is not None:
        if first_uri == second_uri:
            return True
        return second_uri in get_all_superclasses(
            ontology, first_uri
        ) or first_uri in get_all_superclasses(ontology, second_uri)

    return normalize_label(first.label or first.uri) == normalize_label(
        second.label or second.uri
    )


def most_specific_type(
    ontology: Optional[RDFGraph],
    types: "list[Optional[EntityType]]",
) -> Optional[EntityType]:
    """The most specific of a set of mutually compatible types.

    The deepest class wins, measured by the size of its ancestor closure, so
    merging ``Organization`` into ``GovernmentOrganization`` keeps the subclass.
    Ties and unknown hierarchies fall back to the first typed entry, which the
    caller orders deterministically.
    """
    typed = [t for t in types if t is not None]
    if not typed:
        return None
    return max(typed, key=lambda t: len(class_closure_uris(ontology, t)))
