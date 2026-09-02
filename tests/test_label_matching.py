"""Matching model labels back to ontology terms.

Every case here is taken from the 16,131-document MuSiQue run: those labels are
what the model actually returned, and each one was silently lost.
"""

import pytest

from app.utils import parse_ontology_from_string
from kg_gen.models import (
    Entity,
    Graph,
    KGGenStats,
    Ontology,
    OntologyPredicate,
    EntityType,
    Relation,
    TypedEntity,
)
from kg_gen.utils.label_matching import (
    build_alias_index,
    find_normalized_label_collisions,
    match_label,
    normalize_label,
    resolve_uri_by_label,
)

ONTOLOGY_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <http://schema.org/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

schema:Person a owl:Class ; rdfs:label "Person" .
schema:Place a owl:Class ; rdfs:label "Place" .
schema:Province a owl:Class ; rdfs:label "Province" ;
    skos:altLabel "State Division" , "State" .
schema:birthPlace a owl:ObjectProperty ;
    rdfs:domain schema:Person ; rdfs:range schema:Place ;
    rdfs:label "birth place" .
"""


@pytest.mark.parametrize(
    "label,expected",
    [
        ("birth place", "birth place"),
        ("birth_date", "birth date"),
        ("birthPlace", "birth place"),
        ("Birth Place", "birth place"),
        ("  birth   place ", "birth place"),
        ("ex-spouse", "ex spouse"),
        ("CEO", "ceo"),  # an acronym must not be split letter by letter
        ("HTTPServer", "http server"),
        (None, ""),
    ],
)
def test_normalize_label(label, expected):
    assert normalize_label(label) == expected


def test_normalization_does_not_guess():
    """A misspelling stays a miss -- approximate matching is not done here."""
    assert normalize_label("preceeded by") != normalize_label("preceded by")


def test_country_and_county_stay_distinct():
    """Why there is no fuzzy matching: these two are 0.92 similar and both real."""
    index = build_alias_index(
        [
            EntityType(label="Country", uri="http://schema.org/Country"),
            EntityType(label="County", uri="http://example.org/County"),
        ]
    )
    assert match_label(index, "country").label == "Country"
    assert match_label(index, "county").label == "County"


def test_alias_index_accepts_skos_labels():
    onto, graph = parse_ontology_from_string(ONTOLOGY_TTL)
    index = build_alias_index(onto.classes, graph)

    # "State" is only reachable through skos:altLabel on Province.
    assert match_label(index, "State").label == "Province"
    assert match_label(index, "state division").label == "Province"
    # Without the RDF graph the SKOS labels are simply unavailable.
    assert match_label(build_alias_index(onto.classes), "State") is None


def test_primary_label_is_not_shadowed_by_another_terms_alias():
    aliased = EntityType(label="Aliased", uri="http://example.org/Aliased")
    primary = EntityType(label="Person", uri="http://schema.org/Person")
    ttl = """
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
    <http://example.org/Aliased> a owl:Class ; rdfs:label "Aliased" ;
        skos:altLabel "Person" .
    """
    _, graph = parse_ontology_from_string(ttl)

    # Order of items must not decide the winner: the real label always does.
    for items in ([aliased, primary], [primary, aliased]):
        index = build_alias_index(items, graph)
        assert match_label(index, "Person").uri == "http://schema.org/Person"


def test_resolve_uri_by_label_uses_skos_and_normalization():
    _, graph = parse_ontology_from_string(ONTOLOGY_TTL)

    assert str(resolve_uri_by_label(graph, "birth_place")) == (
        "http://schema.org/birthPlace"
    )
    assert str(resolve_uri_by_label(graph, "State")) == "http://schema.org/Province"
    assert resolve_uri_by_label(graph, "no such term") is None
    assert resolve_uri_by_label(None, "Person") is None


def test_collisions_are_reported_not_resolved_silently():
    ttl = """
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    <http://example.org/a> a owl:Class ; rdfs:label "Sports League" .
    <http://example.org/b> a owl:Class ; rdfs:label "sports_league" .
    """
    _, graph = parse_ontology_from_string(ttl)

    collisions = find_normalized_label_collisions(graph)

    assert len(collisions) == 1
    form, uris = collisions[0]
    assert form == "sports league"
    assert sorted(uris) == ["http://example.org/a", "http://example.org/b"]


def test_skos_labels_reach_the_model_via_the_description():
    onto, _ = parse_ontology_from_string(ONTOLOGY_TTL)

    province = next(c for c in onto.classes if c.label == "Province")

    # The prompt shows the description, so this is how the model learns that
    # "Province" is the class it should use for a state.
    assert "Also known as: State Division, State." in province.description


def _graph_with_predicate(label):
    person = TypedEntity(
        surface_form="Erik Hort",
        type={"label": "Person", "uri": "http://schema.org/Person"},
    )
    place = TypedEntity(
        surface_form="Montebello",
        type={"label": "Place", "uri": "http://schema.org/Place"},
    )
    return Graph(
        typed_entities={person, place},
        relations_wo_class_assertions=[
            Relation(subject=person, predicate=Entity(surface_form=label), object=place)
        ],
    )


def test_snake_case_predicate_resolves_to_the_ontology_uri():
    onto, rdf = parse_ontology_from_string(ONTOLOGY_TTL)

    kg = _graph_with_predicate("birth_place").to_knowledge_graph(
        KGGenStats(), onto, rdf
    )

    relation = kg.relations[0]
    assert relation.predicate.uri == "http://schema.org/birthPlace"
    assert relation.predicate.label == "birth place"
    # ... and it is therefore not reported as an invention at all.
    assert kg.ontology_extensions is None


def test_unknown_predicate_without_a_uri_is_still_reported():
    onto, rdf = parse_ontology_from_string(ONTOLOGY_TTL)

    kg = _graph_with_predicate("preceeded by").to_knowledge_graph(
        KGGenStats(), onto, rdf
    )

    assert kg.relations[0].predicate.uri is None
    reported = kg.ontology_extensions.predicates
    assert [p.label for p in reported] == ["preceeded by"]
    assert reported[0].uri is None


def test_unknown_class_without_a_uri_is_still_reported():
    ontology = Ontology(
        classes={EntityType(label="Person", uri="http://schema.org/Person")},
        predicates=set(),
    )
    graph = Graph(
        typed_entities={TypedEntity(surface_form="Ohio", type={"label": "State"})},
        relations_wo_class_assertions=[],
    )

    kg = graph.to_knowledge_graph(KGGenStats(), ontology)

    reported = kg.ontology_extensions.classes
    assert [c.label for c in reported] == ["State"]
    assert reported[0].uri is None


def test_each_novelty_is_reported_once():
    ontology = Ontology(classes={EntityType(label="Person")}, predicates=set())
    graph = Graph(
        typed_entities={
            TypedEntity(surface_form="Ohio", type={"label": "State"}),
            TypedEntity(surface_form="Utah", type={"label": "State"}),
        },
        relations_wo_class_assertions=[],
    )

    kg = graph.to_knowledge_graph(KGGenStats(), ontology)

    assert [c.label for c in kg.ontology_extensions.classes] == ["State"]


def test_conformance_accepts_a_normalized_predicate_label():
    from kg_gen.steps._2_get_relations import validate_ontology_conformance

    person = TypedEntity(
        surface_form="Erik Hort",
        type={"label": "Person", "uri": "http://schema.org/Person"},
    )
    place = TypedEntity(
        surface_form="Montebello",
        type={"label": "Place", "uri": "http://schema.org/Place"},
    )
    predicate = OntologyPredicate(
        label="birth place",
        uri="http://schema.org/birthPlace",
        domain={EntityType(label="Person", uri="http://schema.org/Person")},
        range={EntityType(label="Place", uri="http://schema.org/Place")},
    )
    relation = Relation(
        subject=person, predicate=Entity(surface_form="birth_place"), object=place
    )

    score, errors = validate_ontology_conformance(
        typed_entities=[person, place],
        relations=[relation],
        predicate_domain_range=[predicate],
        enforce_predicate_conformance=True,
    )

    assert score == 1.0, errors
