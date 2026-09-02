"""Deduplication must not merge entities whose types contradict each other.

The normalized surface form alone over-merges. Measured on the 16,131-document
MuSiQue graph, grouping by string only made 4,506 merges, of which 334 joined
entities whose ontology types are siblings -- with a systematic failure mode:
an individual collapsing into the group named after them (`Alan`/`Alans`,
`Aggie`/`Aggies`). Consulting `rdfs:subClassOf` keeps those apart and still
makes 4,172 of the merges.
"""

import collections

import pytest

from app.utils import parse_ontology_from_string
from kg_gen.models import Entity, EntityType, TypedEntity
from kg_gen.utils.deduplicate import DeduplicateList

ONTOLOGY_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <http://schema.org/> .
@prefix : <http://example.org/> .

schema:Person a owl:Class ; rdfs:label "Person" .
schema:Organization a owl:Class ; rdfs:label "Organization" .
:GovernmentOrganization a owl:Class ; rdfs:subClassOf schema:Organization ;
    rdfs:label "Government Organization" .
:SocialOrganization a owl:Class ; rdfs:subClassOf schema:Organization ;
    rdfs:label "Social Organization" .
"""

PERSON = EntityType(label="Person", uri="http://schema.org/Person")
ORG = EntityType(label="Organization", uri="http://schema.org/Organization")
GOV_ORG = EntityType(
    label="Government Organization", uri="http://example.org/GovernmentOrganization"
)
SOCIAL_ORG = EntityType(
    label="Social Organization", uri="http://example.org/SocialOrganization"
)


@pytest.fixture
def ontology():
    _, graph = parse_ontology_from_string(ONTOLOGY_TTL)
    return graph


def group(items, ontology):
    """The surface forms that would be merged together, as a set of frozensets."""
    dedup = DeduplicateList()
    buckets = dedup._type_buckets(items, ontology)
    groups = collections.defaultdict(set)
    for item in items:
        singular = dedup.singularize(dedup.normalize(item.surface_form)).casefold()
        groups[dedup._bucket_key(singular, item, buckets)].add(item.surface_form)
    return {frozenset(v) for v in groups.values()}


def test_a_subclass_merges_with_its_superclass(ontology):
    items = [
        TypedEntity(surface_form="Air Force", type=GOV_ORG),
        TypedEntity(surface_form="air force", type=ORG),
    ]

    # The same entity, one mention typed less specifically.
    assert group(items, ontology) == {frozenset({"Air Force", "air force"})}


def test_siblings_do_not_merge(ontology):
    items = [
        TypedEntity(surface_form="Abbasid", type=GOV_ORG),
        TypedEntity(surface_form="Abbasids", type=SOCIAL_ORG),
    ]

    assert group(items, ontology) == {frozenset({"Abbasid"}), frozenset({"Abbasids"})}


def test_a_person_does_not_merge_into_the_group_named_after_them(ontology):
    items = [
        TypedEntity(surface_form="Alan", type=PERSON),
        TypedEntity(surface_form="Alans", type=ORG),
    ]

    assert group(items, ontology) == {frozenset({"Alan"}), frozenset({"Alans"})}


def test_the_same_type_still_merges_plurals(ontology):
    items = [
        TypedEntity(surface_form="Cats", type=PERSON),
        TypedEntity(surface_form="Cat", type=PERSON),
    ]

    assert group(items, ontology) == {frozenset({"Cats", "Cat"})}


def test_an_untyped_entity_merges_with_a_typed_one(ontology):
    items = [
        TypedEntity(surface_form="Cats", type=None),
        TypedEntity(surface_form="Cat", type=PERSON),
    ]

    # An untyped mention contradicts nothing.
    assert group(items, ontology) == {frozenset({"Cats", "Cat"})}


def test_predicates_have_no_type_and_still_merge(ontology):
    # Edges are plain Entities; bucketing must not break them.
    items = [Entity(surface_form="publishers"), Entity(surface_form="publisher")]

    assert group(items, ontology) == {frozenset({"publishers", "publisher"})}


def test_keys_are_unchanged_when_no_types_conflict(ontology):
    """Bucket 0 keeps the bare normalized form, so untouched corpora are stable."""
    items = [
        TypedEntity(surface_form="Cats", type=PERSON),
        TypedEntity(surface_form="Cat", type=PERSON),
    ]
    dedup = DeduplicateList()

    buckets = dedup._type_buckets(items, ontology)

    assert {dedup._bucket_key("cat", item, buckets) for item in items} == {"cat"}


def test_grouping_does_not_depend_on_item_order(ontology):
    items = [
        TypedEntity(surface_form="Alan", type=PERSON),
        TypedEntity(surface_form="Alans", type=ORG),
        TypedEntity(surface_form="alan", type=PERSON),
    ]

    assert group(items, ontology) == group(list(reversed(items)), ontology)


def test_without_an_ontology_the_type_label_decides(ontology):
    same = [
        TypedEntity(surface_form="Cats", type=EntityType(label="animal")),
        TypedEntity(surface_form="Cat", type=EntityType(label="Animal")),
    ]
    different = [
        TypedEntity(surface_form="Alan", type=EntityType(label="Person")),
        TypedEntity(surface_form="Alans", type=EntityType(label="Organization")),
    ]

    # Case and separators are matched forgivingly, as labels are everywhere else.
    assert group(same, None) == {frozenset({"Cats", "Cat"})}
    assert group(different, None) == {frozenset({"Alan"}), frozenset({"Alans"})}


def test_end_to_end_through_run_semhash_deduplication(ontology):
    from kg_gen.models import Graph, Relation
    from kg_gen.utils.deduplicate import run_semhash_deduplication

    alan = TypedEntity(surface_form="Alan", type=PERSON)
    alans = TypedEntity(surface_form="Alans", type=ORG)
    graph = Graph(
        typed_entities={alan, alans},
        relations_wo_class_assertions=[
            Relation(
                subject=alan, predicate=Entity(surface_form="member of"), object=alans
            )
        ],
    )

    merged = run_semhash_deduplication(
        graph, use_embeddings=False, deduplicate_edges=False, ontology=ontology
    )

    surface_forms = {e.surface_form for e in merged.typed_entities}
    assert surface_forms == {"Alan", "Alans"}
    # ... and the relation still connects the two rather than becoming a self-loop.
    relation = merged.relations_wo_class_assertions[0]
    assert relation.subject.surface_form != relation.object.surface_form


def test_one_surface_form_with_two_types_still_collapses(ontology):
    """A known limitation, pinned so it is explicit rather than a surprise.

    Deduplication is keyed by surface form throughout (`original_map`,
    `surface_form2entity_map`), so two entities that share a surface form get one
    bucket between them however their types disagree. The corpus really contains
    this: "African American" typed both Person and Place. Fixing it means keying
    deduplication by entity, not by surface form.
    """
    items = [
        TypedEntity(surface_form="African American", type=PERSON),
        TypedEntity(surface_form="African American", type=ORG),
    ]

    assert group(items, ontology) == {frozenset({"African American"})}
