"""Unit tests for the deterministic MCP tool bodies (mcp/tools.py).

These exercise everything except embedding-based ``suggest_clusters`` (which
loads a sentence-transformers model and is therefore not run offline here).
``apply_clusters`` has no embedding dependency (it merges clusters it is
handed, it does not compute them), so it is tested directly. ``tools`` is
importable because ``mcp/`` is on ``pythonpath`` (see pyproject pytest config)."""

import tools
from kg_gen.models import Entity, EntityType, Relation, TypedEntity
from tools import EdgeCluster, EntityCluster

# A tiny ontology: Person and Organization classes, one object property
# (works_for: Person -> Organization) and one datatype property
# (birth_date: Person -> xsd:date).
ONTOLOGY_TTL = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <http://example.org/> .

ex:Person a owl:Class ; rdfs:label "Person" ; rdfs:comment "A human being" .
ex:Organization a owl:Class ; rdfs:label "Organization" .

ex:works_for a owl:ObjectProperty ;
    rdfs:label "works_for" ;
    rdfs:domain ex:Person ;
    rdfs:range ex:Organization .

ex:birth_date a owl:DatatypeProperty ;
    rdfs:label "birth_date" ;
    rdfs:domain ex:Person ;
    rdfs:range xsd:date .
"""

PERSON = EntityType(label="Person", uri="http://example.org/Person")
ORG = EntityType(label="Organization", uri="http://example.org/Organization")


def test_parse_ontology():
    onto = tools.parse_ontology(ONTOLOGY_TTL)
    labels = {c.label for c in onto.classes}
    assert labels == {"Person", "Organization"}
    assert {p.label for p in onto.predicates} == {"works_for", "birth_date"}


def test_list_target_types_sorted():
    types = tools.list_target_types(ONTOLOGY_TTL)
    assert [t.label for t in types] == ["Organization", "Person"]


def test_suggest_predicates_filters_by_domain_and_range():
    # Both endpoints' types present: works_for (Person->Organization) is kept,
    # and birth_date (Person->xsd:date, literal range) is always kept.
    suggested = tools.suggest_predicates(ONTOLOGY_TTL, [PERSON, ORG])
    assert {p.label for p in suggested} == {"works_for", "birth_date"}

    # Only Person present: works_for's range (Organization) is absent, so it is
    # dropped; birth_date stays because its range is a literal (xsd:date).
    suggested_person = tools.suggest_predicates(ONTOLOGY_TTL, [PERSON])
    assert {p.label for p in suggested_person} == {"birth_date"}

    # Only Organization present: neither predicate's domain (Person) matches.
    suggested_org = tools.suggest_predicates(ONTOLOGY_TTL, [ORG])
    assert {p.label for p in suggested_org} == set()


def test_validate_conformance_reports_domain_violation():
    # "Acme" typed as Organization used as the subject of works_for violates the
    # predicate's domain (Person).
    person = TypedEntity(surface_form="Ada", type=PERSON)
    org = TypedEntity(surface_form="Acme", type=ORG)
    good = Relation(
        subject=Entity(surface_form="Ada"),
        predicate=Entity(surface_form="works_for"),
        object=Entity(surface_form="Acme"),
    )
    report = tools.validate_conformance([person, org], [good], ONTOLOGY_TTL)
    assert report.conformant and report.score == 1.0

    bad = Relation(
        subject=Entity(surface_form="Acme"),
        predicate=Entity(surface_form="works_for"),
        object=Entity(surface_form="Ada"),
    )
    report_bad = tools.validate_conformance([person, org], [bad], ONTOLOGY_TTL)
    assert not report_bad.conformant
    assert report_bad.errors


def test_serialize_graph_produces_output_entities():
    person = TypedEntity(surface_form="Ada", type=PERSON)
    org = TypedEntity(surface_form="Acme", type=ORG)
    rel = Relation(
        subject=Entity(surface_form="Ada"),
        predicate=Entity(surface_form="works_for"),
        object=Entity(surface_form="Acme"),
    )
    kg = tools.serialize_graph([person, org], [rel], ONTOLOGY_TTL)
    surface_forms = {e.surface_form for e in kg.entities.values()}
    assert {"Ada", "Acme"}.issubset(surface_forms)
    assert len(kg.relations) == 1


def test_convert_ontology_roundtrip():
    onto = tools.parse_ontology(ONTOLOGY_TTL)
    ttl = tools.convert_ontology(list(onto.classes), list(onto.predicates))
    reparsed = tools.parse_ontology(ttl)
    assert {c.label for c in reparsed.classes} == {"Person", "Organization"}
    assert {p.label for p in reparsed.predicates} == {"works_for", "birth_date"}


def test_validate_graph_schema():
    person = TypedEntity(surface_form="Ada", type=PERSON)
    kg = tools.serialize_graph([person], [], ONTOLOGY_TTL)
    ok = tools.validate_graph_schema(kg.model_dump())
    assert ok.valid and ok.errors == []

    bad = tools.validate_graph_schema({"entities": "not-a-dict", "relations": []})
    assert not bad.valid
    assert bad.errors


def test_apply_clusters_merges_entities_edges_and_provenance():
    ada = TypedEntity(surface_form="Ada", type=PERSON, provenance_ids=["d1"])
    lovelace = TypedEntity(
        surface_form="A. Lovelace", type=PERSON, provenance_ids=["d2"]
    )
    acme = TypedEntity(surface_form="Acme", type=ORG)

    works_for = Entity(surface_form="works for")
    employed_by = Entity(surface_form="employed by")

    rel1 = Relation(
        subject=ada, predicate=works_for, object=acme, provenance_ids=["d1"]
    )
    rel2 = Relation(
        subject=lovelace, predicate=employed_by, object=acme, provenance_ids=["d2"]
    )

    entity_clusters = [EntityCluster(members=[ada, lovelace], representative=ada)]
    edge_clusters = [
        EdgeCluster(members=[works_for, employed_by], representative=works_for)
    ]

    kg = tools.apply_clusters(
        typed_entities=[ada, lovelace, acme],
        relations=[rel1, rel2],
        entity_clusters=entity_clusters,
        edge_clusters=edge_clusters,
        ontology_ttl=ONTOLOGY_TTL,
    )

    surface_forms = {e.surface_form for e in kg.entities.values()}
    assert surface_forms == {"Ada", "Acme"}
    assert "A. Lovelace" not in surface_forms

    # Both original relations collapse into one, with merged provenance.
    assert len(kg.relations) == 1
    merged = kg.relations[0]
    assert sorted(merged.provenance_ids) == ["d1", "d2"]

    # Provenance from both merged mentions of "Ada" is aggregated.
    ada_out = next(e for e in kg.entities.values() if e.surface_form == "Ada")
    assert sorted(ada_out.provenance_ids) == ["d1", "d2"]


def test_apply_clusters_no_clusters_is_a_passthrough():
    ada = TypedEntity(surface_form="Ada", type=PERSON)
    rel = Relation(
        subject=ada,
        predicate=Entity(surface_form="works for"),
        object=TypedEntity(surface_form="Acme", type=ORG),
    )

    kg = tools.apply_clusters(
        typed_entities=[ada],
        relations=[rel],
        entity_clusters=[],
        edge_clusters=[],
        ontology_ttl=ONTOLOGY_TTL,
    )

    assert {e.surface_form for e in kg.entities.values()} >= {"Ada"}
    assert len(kg.relations) == 1
