"""Tests for ``suggest_predicates_batch`` (mcp/tools.py).

The point of the batch tool is to make *per-chunk* predicate narrowing
affordable. Narrowing per chunk is what constrains the choice; narrowing over a
whole document set does not, because the union of entity types over enough text
is the whole ontology. These tests pin the two properties that make one call
enough: groups that share a type signature share an answer, and each predicate
is described exactly once no matter how many groups it serves.
"""

import json

import pytest
import tools
from kg_gen.models import EntityType

# Person/Organization with a Company subclass, so the class hierarchy actually
# matters: a predicate declared on Organization must be suggested for a Company.
ONTOLOGY_TTL = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <http://example.org/> .

ex:Person a owl:Class ; rdfs:label "Person" ; rdfs:comment "A human being" .
ex:Organization a owl:Class ; rdfs:label "Organization" ;
    rdfs:comment "A structured group of people" .
ex:Company a owl:Class ; rdfs:label "Company" ; rdfs:subClassOf ex:Organization ;
    rdfs:comment "A for-profit organization" .
ex:Place a owl:Class ; rdfs:label "Place" ; rdfs:comment "A location" .

ex:works_for a owl:ObjectProperty ;
    rdfs:label "works_for" ; rdfs:comment "Employment" ;
    rdfs:domain ex:Person ; rdfs:range ex:Organization .

ex:birth_place a owl:ObjectProperty ;
    rdfs:label "birth_place" ; rdfs:comment "Where someone was born" ;
    rdfs:domain ex:Person ; rdfs:range ex:Place .

ex:birth_date a owl:DatatypeProperty ;
    rdfs:label "birth_date" ; rdfs:comment "When someone was born" ;
    rdfs:domain ex:Person ; rdfs:range xsd:date .

ex:founded_date a owl:DatatypeProperty ;
    rdfs:label "founded_date" ; rdfs:comment "When an organization began" ;
    rdfs:domain ex:Organization ; rdfs:range xsd:gYear .
"""


def _labels_for(batch, key):
    """The predicate labels the batch result assigns to one group."""
    return set(batch.predicate_sets[batch.group_predicate_set[key]])


def test_each_group_matches_the_singular_tool():
    """The batch answer per group is exactly what one call would have returned.

    This is the property that lets the batch tool replace per-chunk calls
    outright rather than approximate them.
    """
    groups = {
        "a": ["Person", "Organization"],
        "b": ["Person", "Place"],
        "c": ["Company"],
        "d": ["Place"],
    }
    batch = tools.suggest_predicates_batch(ONTOLOGY_TTL, groups)
    for key, types in groups.items():
        singular = tools.suggest_predicates(
            ONTOLOGY_TTL, [EntityType(label=t) for t in types]
        )
        assert _labels_for(batch, key) == {p.label for p in singular}, key


def test_narrowing_actually_narrows_and_respects_the_hierarchy():
    groups = {"person_org": ["Person", "Organization"], "place_only": ["Place"]}
    batch = tools.suggest_predicates_batch(ONTOLOGY_TTL, groups)
    # A Person+Organization chunk can carry employment and both date properties,
    # but not birth_place -- no Place was found in it.
    assert _labels_for(batch, "person_org") == {
        "works_for",
        "birth_date",
        "founded_date",
    }
    # A chunk with only Places supports nothing: every predicate here needs a
    # Person or Organization subject.
    assert _labels_for(batch, "place_only") == set()


def test_subclass_inherits_superclass_predicates():
    batch = tools.suggest_predicates_batch(ONTOLOGY_TTL, {"co": ["Company"]})
    # founded_date is declared on Organization; Company is a subclass.
    assert "founded_date" in _labels_for(batch, "co")


def test_groups_sharing_a_signature_share_one_set():
    batch = tools.suggest_predicates_batch(
        ONTOLOGY_TTL,
        {
            "c1": ["Person", "Organization"],
            "c2": ["Organization", "Person"],  # same signature, different order
            "c3": ["Place"],
        },
    )
    assert batch.group_predicate_set["c1"] == batch.group_predicate_set["c2"]
    assert batch.group_predicate_set["c3"] != batch.group_predicate_set["c1"]


def test_distinct_signatures_yielding_the_same_answer_collapse():
    """Different type sets often imply the same predicates; store that set once.

    On real data this is the larger of the two collapses: a 200-chunk MuSiQue
    slice holds 77 distinct type signatures but only 18 distinct predicate sets.
    """
    batch = tools.suggest_predicates_batch(
        ONTOLOGY_TTL,
        {
            "org": ["Organization"],
            "company": ["Company"],  # different signature...
        },
    )
    # ...but the same answer, since Company inherits Organization's predicates
    # and neither group contains a Person to satisfy works_for's domain.
    assert _labels_for(batch, "org") == _labels_for(batch, "company")
    assert batch.group_predicate_set["org"] == batch.group_predicate_set["company"]
    assert len(batch.predicate_sets) == 1


def test_legend_describes_each_predicate_once_and_only_if_used():
    batch = tools.suggest_predicates_batch(
        ONTOLOGY_TTL,
        {f"c{i}": ["Person", "Organization"] for i in range(20)},
    )
    labels = [p.label for p in batch.legend]
    assert labels == sorted(labels)
    assert len(labels) == len(set(labels)) == 3, "described once each, 20 groups"
    # birth_place serves no group here, so it is not described at all.
    assert "birth_place" not in labels
    assert all(p.description for p in batch.legend), "descriptions are kept"


def test_legend_omits_class_descriptions_like_the_singular_tool():
    batch = tools.suggest_predicates_batch(ONTOLOGY_TTL, {"c": ["Person", "Place"]})
    classes = [t for p in batch.legend for t in list(p.domain) + list(p.range)]
    assert classes, "the fixture predicates all declare a domain"
    assert all(t.description is None for t in classes)
    assert all(t.label for t in classes)


def test_bare_labels_and_entity_type_dicts_agree():
    as_labels = tools.suggest_predicates_batch(ONTOLOGY_TTL, {"c": ["Person"]})
    as_dicts = tools.suggest_predicates_batch(
        ONTOLOGY_TTL,
        {"c": [{"label": "Person", "uri": "http://example.org/Person"}]},
    )
    assert _labels_for(as_labels, "c") == _labels_for(as_dicts, "c")


def test_groups_can_be_passed_as_a_file_path(tmp_path):
    """The mapping is per-chunk, so on a real slice it is the bulky argument;
    passing it by reference keeps it out of the caller's context."""
    groups = {"c1": ["Person", "Organization"], "c2": ["Place"]}
    path = tmp_path / "groups.json"
    path.write_text(json.dumps(groups))
    from_file = tools.suggest_predicates_batch(ONTOLOGY_TTL, str(path))
    inline = tools.suggest_predicates_batch(ONTOLOGY_TTL, groups)
    # Compared as JSON: domain/range are sets of models, so model_dump() leaves
    # unhashable dicts inside a set and blows up.
    assert from_file.model_dump_json() == inline.model_dump_json()


def test_empty_groups_returns_nothing_rather_than_failing():
    batch = tools.suggest_predicates_batch(ONTOLOGY_TTL, {})
    assert batch.legend == []
    assert batch.predicate_sets == []
    assert batch.group_predicate_set == {}


def test_a_group_with_no_types_gets_an_empty_set():
    batch = tools.suggest_predicates_batch(ONTOLOGY_TTL, {"untyped": []})
    assert _labels_for(batch, "untyped") == set()


def test_non_mapping_is_rejected_with_a_useful_message():
    path_free_list = json.dumps([["Person"], ["Place"]])
    with pytest.raises(ValueError, match="mapping of group key"):
        tools.suggest_predicates_batch(ONTOLOGY_TTL, json.loads(path_free_list))
