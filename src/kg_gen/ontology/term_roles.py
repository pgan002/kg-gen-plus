"""Conservative role classification for normalized ontology-term candidates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

OntologyCandidateRole = Literal["class", "property", "literal", "ignore"]

_COPULAR_PROPERTIES = {
    "are",
    "be",
    "has type",
    "is",
    "is a",
    "is an",
    "is type of",
    "type of",
    "was",
    "were",
}
_LITERAL_TERMS = {
    "boolean",
    "date",
    "datetime",
    "decimal",
    "hour",
    "integer",
    "money",
    "number",
    "string",
    "timestamp",
    "url",
    "value",
}
_ATTRIBUTE_HEADS = {
    "amount",
    "count",
    "date",
    "datetime",
    "duration",
    "identifier",
    "number",
    "percentage",
    "price",
    "timestamp",
    "url",
    "value",
    "version",
}
_LITERAL_PATTERN = re.compile(
    r"^(?:"
    r"[-+]?\d+(?:\.\d+)?"
    r"|\d{4}-\d{2}-\d{2}(?:t\S+)?"
    r"|https?://\S+"
    r"|\S+@\S+\.\S+"
    r")$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RoleClassification:
    role: OntologyCandidateRole
    reason: str


def classify_property(term: str) -> RoleClassification:
    """Classify a normalized relation label as a property or support phrase."""
    if term in _COPULAR_PROPERTIES:
        return RoleClassification("ignore", "copular_or_type_support_relation")
    return RoleClassification("property", "relation_predicate")


def classify_class(
    term: str, *, observed_property_terms: set[str]
) -> RoleClassification:
    """Classify a normalized entity label conservatively for ontology export."""
    if term in observed_property_terms:
        return RoleClassification("property", "also_observed_as_relation_predicate")
    if term in _LITERAL_TERMS or _LITERAL_PATTERN.fullmatch(term):
        return RoleClassification("literal", "literal_or_scalar_value")
    words = term.split()
    if words and words[-1] in _ATTRIBUTE_HEADS:
        return RoleClassification("literal", "attribute_or_value_phrase")
    return RoleClassification("class", "concept_candidate")


def filter_ontology_term_roles(
    class_assignments: Mapping[str, set[str]],
    property_assignments: Mapping[str, set[str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, dict[str, str]]]:
    """Filter assignments and return per-term exclusion reasons.

    Exact class/property overlap is resolved in favor of the property role.
    Property evidence is collected domain-wide so a predicate is not retained as
    a class merely because it was extracted from a different CQ.
    """
    cq_ids = list(dict.fromkeys((*class_assignments, *property_assignments)))
    observed_properties = {
        term for terms in property_assignments.values() for term in terms
    }
    classes: dict[str, set[str]] = {}
    properties: dict[str, set[str]] = {}
    excluded: dict[str, dict[str, str]] = {}

    for cq_id in cq_ids:
        classes[cq_id] = set()
        properties[cq_id] = set()
        excluded[cq_id] = {}
        for term in class_assignments.get(cq_id, set()):
            classification = classify_class(
                term, observed_property_terms=observed_properties
            )
            if classification.role == "class":
                classes[cq_id].add(term)
            else:
                excluded[cq_id][f"class:{term}"] = classification.reason
        for term in property_assignments.get(cq_id, set()):
            classification = classify_property(term)
            if classification.role == "property":
                properties[cq_id].add(term)
            else:
                excluded[cq_id][f"property:{term}"] = classification.reason
    return classes, properties, excluded
