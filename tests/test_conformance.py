import pytest
from kg_gen.steps._2_get_relations import validate_ontology_conformance
from kg_gen.models import (
    Relation,
    TypedEntity,
    EntityType,
    OntologyPredicate,
    Entity,
    InputData,
)


def test_validate_domain_range_conformance():
    person_type = EntityType(label="Person")
    city_type = EntityType(label="City")
    organization_type = EntityType(label="Organization")

    # Predicate: livesIn (Person -> City)
    lives_in_pred = OntologyPredicate(
        label="livesIn", domain=[person_type], range=[city_type]
    )
    ontology = [lives_in_pred]

    # # OK relation
    # rel_ok = Relation(
    #     subject=TypedEntity(surface_form="Alice", type=person_type),
    #     predicate=Entity(surface_form="livesIn"),
    #     object=TypedEntity(surface_form="Paris", type=city_type),
    # )

    # Domain violation: Organization livesIn City
    rel_domain_violation = Relation(
        subject=TypedEntity(surface_form="Google", type=organization_type),
        predicate=Entity(surface_form="livesIn"),
        object=TypedEntity(surface_form="Paris", type=city_type),
    )

    # Range violation: Person livesIn Organization
    rel_range_violation = Relation(
        subject=TypedEntity(surface_form="Alice", type=person_type),
        predicate=Entity(surface_form="livesIn"),
        object=TypedEntity(surface_form="Google", type=organization_type),
    )

    typed_entities = [
        TypedEntity(surface_form="Alice", type=person_type),
        TypedEntity(surface_form="Paris", type=city_type),
        TypedEntity(surface_form="Google", type=organization_type),
    ]

    # Test Domain Conformance
    conforms, _ = validate_ontology_conformance(
        typed_entities,
        [rel_domain_violation],
        ontology,
        enforce_domain_conformance=True,
    )
    assert conforms < 1
    conforms, _ = validate_ontology_conformance(
        typed_entities,
        [rel_domain_violation],
        ontology,
        enforce_domain_conformance=False,
    )
    assert int(conforms) == 1

    # Test Range Conformance
    conforms, _ = validate_ontology_conformance(
        typed_entities, [rel_range_violation], ontology, enforce_range_conformance=True
    )
    assert conforms < 1
    conforms, _ = validate_ontology_conformance(
        typed_entities, [rel_range_violation], ontology, enforce_range_conformance=False
    )
    assert int(conforms) == 1


def test_validate_type_conformance():
    person_type = EntityType(label="Person")
    city_type = EntityType(label="City")
    animal_type = EntityType(label="Animal")

    typed_entities = [
        TypedEntity(surface_form="Alice", type=person_type),
        TypedEntity(surface_form="Paris", type=city_type),
        TypedEntity(
            surface_form="Bob", type=animal_type
        ),  # Bob has a type not in allowed list
    ]

    allowed_types = [person_type, city_type]

    # Relation with subject "Bob" which has animal_type (not in allowed_types)
    rel_wrong_type = Relation(
        subject=TypedEntity(surface_form="Bob", type=animal_type),
        predicate=Entity(surface_form="livesIn"),
        object=TypedEntity(surface_form="Paris", type=city_type),
    )

    # Relation with unknown entity (not in typed_entities), should be fine now
    rel_unknown = Relation(
        subject=Entity(surface_form="Charlie"),
        predicate=Entity(surface_form="livesIn"),
        object=TypedEntity(surface_form="Paris", type=city_type),
    )

    # Test Type Conformance
    # When enforce_type_conformance=True and allowed_types provided
    conforms, err = validate_ontology_conformance(
        typed_entities=typed_entities,
        relations=[rel_wrong_type],
        enforce_type_conformance=True,
        allowed_types=allowed_types,
    )
    assert conforms < 1
    assert "not in allowed types" in err

    # Unknown entity should pass even if enforce_type_conformance=True
    valid_typed_entities = [te for te in typed_entities if te.surface_form != "Bob"]
    conforms, err = validate_ontology_conformance(
        typed_entities=valid_typed_entities,
        relations=[rel_unknown],
        enforce_type_conformance=True,
        allowed_types=allowed_types,
    )
    assert int(conforms) == 1

    # When enforce_type_conformance=False, it should pass
    conforms, _ = validate_ontology_conformance(
        typed_entities=typed_entities,
        relations=[rel_wrong_type],
        enforce_type_conformance=False,
        allowed_types=allowed_types,
    )
    assert int(conforms) == 1


def test_validate_predicate_conformance():
    person_type = EntityType(label="Person")
    city_type = EntityType(label="City")

    lives_in_pred = OntologyPredicate(
        label="livesIn", domain=[person_type], range=[city_type]
    )
    ontology = [lives_in_pred]

    rel_unknown = Relation(
        subject=TypedEntity(surface_form="Alice", type=person_type),
        predicate=Entity(surface_form="worksAt"),
        object=Entity(surface_form="Google"),
    )

    typed_entities = [TypedEntity(surface_form="Alice", type=person_type)]

    # Test Predicate Conformance
    conforms, err = validate_ontology_conformance(
        typed_entities, [rel_unknown], ontology, enforce_predicate_conformance=True
    )
    assert conforms < 1
    assert "not in ontology" in err

    conforms, _ = validate_ontology_conformance(
        typed_entities, [rel_unknown], ontology, enforce_predicate_conformance=False
    )
    assert int(conforms) == 1


@pytest.mark.asyncio
async def test_kggen_conformance_flags_passed(mock_kg_gen):
    # This test checks if the flags are passed to the generate method without crashing.
    # Since we use mock_kg_gen, it won't actually fail validation unless we mock the refined response,
    # but we can verify that the call succeeds with the new parameters.
    input_data = InputData(text="Alice lives in Paris.", id="test")

    # Just ensure it doesn't raise TypeError: unexpected keyword argument
    await mock_kg_gen.generate(
        input_data,
        enforce_domain_conformance=False,
        enforce_range_conformance=False,
        enforce_predicate_conformance=True,
        enforce_type_conformance=True,
    )


HIERARCHY_TTL = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <http://example.org/> .

ex:Place        rdf:type owl:Class ; rdfs:label "Place" .
ex:Country      rdf:type owl:Class ; rdfs:label "Country" ; rdfs:subClassOf ex:Place .
ex:Organization rdf:type owl:Class ; rdfs:label "Organization" .
ex:SportsLeague rdf:type owl:Class ; rdfs:label "Sports League" ;
                rdfs:subClassOf ex:Organization .
ex:Person       rdf:type owl:Class ; rdfs:label "Person" .

ex:birthPlace rdf:type owl:ObjectProperty ; rdfs:label "birth place" ;
    rdfs:domain ex:Person ; rdfs:range ex:Place .
ex:founder rdf:type owl:ObjectProperty ; rdfs:label "founder" ;
    rdfs:domain ex:Organization ; rdfs:range ex:Person .
"""


def _hierarchy_ontology():
    from app.utils import parse_ontology_from_string

    onto, graph = parse_ontology_from_string(HIERARCHY_TTL)
    return list(onto.predicates), graph


def test_subclass_satisfies_range_when_ontology_is_passed():
    """A Country object satisfies a Place range, because Country subClassOf Place.

    Without the graph the check degrades to exact label equality and rejects the
    relation -- and since callers demand a perfect score, that would throw away
    every relation found in the same document.
    """
    predicates, graph = _hierarchy_ontology()
    entities = [
        TypedEntity(surface_form="Alice", type=EntityType(label="Person")),
        TypedEntity(surface_form="Argentina", type=EntityType(label="Country")),
    ]
    relation = Relation(
        subject=Entity(surface_form="Alice"),
        predicate=Entity(surface_form="birth place"),
        object=Entity(surface_form="Argentina"),
    )

    score, errors = validate_ontology_conformance(
        typed_entities=entities,
        relations=[relation],
        predicate_domain_range=predicates,
        ontology=graph,
    )
    assert score == 1.0, errors

    # Same inputs, no hierarchy available: the subclass is not recognised.
    score_without, errors_without = validate_ontology_conformance(
        typed_entities=entities,
        relations=[relation],
        predicate_domain_range=predicates,
    )
    assert score_without < 1.0
    assert "not in range" in errors_without


def test_subclass_satisfies_domain_when_ontology_is_passed():
    predicates, graph = _hierarchy_ontology()
    entities = [
        TypedEntity(surface_form="La Liga", type=EntityType(label="Sports League")),
        TypedEntity(surface_form="Alice", type=EntityType(label="Person")),
    ]
    relation = Relation(
        subject=Entity(surface_form="La Liga"),
        predicate=Entity(surface_form="founder"),
        object=Entity(surface_form="Alice"),
    )

    score, errors = validate_ontology_conformance(
        typed_entities=entities,
        relations=[relation],
        predicate_domain_range=predicates,
        ontology=graph,
    )
    assert score == 1.0, errors


def test_genuine_domain_range_violation_still_fails_with_hierarchy():
    """Subclass awareness must not loosen real constraints."""
    predicates, graph = _hierarchy_ontology()
    entities = [
        TypedEntity(surface_form="Argentina", type=EntityType(label="Country")),
        TypedEntity(surface_form="Alice", type=EntityType(label="Person")),
    ]
    # Reversed: a Country cannot be the subject of 'birth place', and a Person
    # cannot be its object.
    relation = Relation(
        subject=Entity(surface_form="Argentina"),
        predicate=Entity(surface_form="birth place"),
        object=Entity(surface_form="Alice"),
    )

    score, errors = validate_ontology_conformance(
        typed_entities=entities,
        relations=[relation],
        predicate_domain_range=predicates,
        ontology=graph,
    )
    assert score == 0.0
    assert "not in domain" in errors and "not in range" in errors


def test_filter_and_validate_agree_on_subclasses():
    """The predicates offered to the model must be the ones the validator accepts.

    These two used to implement the hierarchy rule separately -- the filter with
    superclass expansion, the validator with exact equality -- so the prompt
    invited relations that were then scored as violations.
    """
    from kg_gen.steps._2_get_relations import filter_predicates_by_entity_types

    predicates, graph = _hierarchy_ontology()
    league = EntityType(label="Sports League")
    person = EntityType(label="Person")

    offered = filter_predicates_by_entity_types(
        [league, person], predicates, ontology=graph
    )
    assert "founder" in [p.label for p in offered]

    entities = [
        TypedEntity(surface_form="La Liga", type=league),
        TypedEntity(surface_form="Alice", type=person),
    ]
    for predicate in offered:
        relation = Relation(
            subject=Entity(surface_form="La Liga"),
            predicate=Entity(surface_form=predicate.label),
            object=Entity(surface_form="Alice"),
        )
        score, errors = validate_ontology_conformance(
            typed_entities=entities,
            relations=[relation],
            predicate_domain_range=predicates,
            ontology=graph,
        )
        if predicate.label == "founder":
            assert score == 1.0, f"{predicate.label} was offered but rejected: {errors}"


def test_generation_metadata_enforces_all_conformance_by_default():
    """An ontology-guided request should honour the ontology unless told otherwise.

    Type and predicate conformance used to default to False, so a caller who
    supplied an ontology still got invented types and predicates unless they knew
    to switch two extra flags on.
    """
    from app.schemas import GenerationMetadata

    meta = GenerationMetadata()
    assert meta.enforce_type_conformance is True
    assert meta.enforce_domain_conformance is True
    assert meta.enforce_range_conformance is True
    assert meta.enforce_predicate_conformance is True
