from __future__ import annotations

import pytest

from kg_gen.ontology.term_roles import (
    classify_class,
    classify_property,
    filter_ontology_term_roles,
)


def test_exact_class_property_overlap_prefers_property_role():
    classes, properties, excluded = filter_ontology_term_roles(
        {"CQ1": {"animal", "eat"}},
        {"CQ1": {"eat"}},
    )

    assert classes == {"CQ1": {"animal"}}
    assert properties == {"CQ1": {"eat"}}
    assert excluded["CQ1"]["class:eat"] == "also_observed_as_relation_predicate"


def test_induced_alias_is_property_evidence_without_being_exported():
    classes, properties, excluded = filter_ontology_term_roles(
        {"CQ1": {"can read", "software"}},
        {"CQ1": {"has specified data input"}},
        additional_property_evidence={"can read"},
    )

    assert classes["CQ1"] == {"software"}
    assert properties["CQ1"] == {"has specified data input"}
    assert excluded["CQ1"]["class:can read"] == (
        "also_observed_as_relation_predicate"
    )


def test_domain_wide_property_evidence_removes_class_in_another_cq():
    classes, properties, excluded = filter_ontology_term_roles(
        {"CQ1": {"eat"}, "CQ2": {"animal"}},
        {"CQ2": {"eat"}},
    )

    assert classes["CQ1"] == set()
    assert properties["CQ2"] == {"eat"}
    assert excluded["CQ1"]["class:eat"] == "also_observed_as_relation_predicate"


@pytest.mark.parametrize("term", ["is", "is a", "is an", "has type", "type of"])
def test_copular_support_relations_are_not_properties(term):
    assert classify_property(term).role == "ignore"


@pytest.mark.parametrize(
    "term",
    [
        "2026-10-02",
        "3.14",
        "hour",
        "money",
        "release date",
        "firmware version",
        "kill count",
        "fabrication number",
        "billing duration",
        "year value",
        "support url",
    ],
)
def test_literal_and_attribute_candidates_are_not_classes(term):
    assert classify_class(term, observed_property_terms=set()).role == "literal"


@pytest.mark.parametrize(
    "term", ["animal", "city", "format", "meter", "software", "tariff"]
)
def test_concept_candidates_are_retained(term):
    assert classify_class(term, observed_property_terms=set()).role == "class"
