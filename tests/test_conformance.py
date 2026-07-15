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
