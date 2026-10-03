"""Ontology-specific class and property label normalization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

import inflect

from kg_gen.ontology.provenance_validation import (
    validate_canonical_assignment_provenance,
)
from kg_gen.utils.term_normalization import normalized_label_key

OntologyTermKind = Literal["class", "property"]

_INFLECT = inflect.engine()
_CLASS_ALIASES = {
    "carnivorous animal": "carnivore",
    "herbivorous animal": "herbivore",
    "omnivorous animal": "omnivore",
}
_NON_SINGULAR_CLASS_HEADS = {"data", "media", "software"}
_NON_SINGULAR_CLASS_SUFFIXES = ("is", "ness", "os", "ss", "us")
_NON_LEMMATIZED_PROPERTY_VERBS = {"has", "is", "was", "does"}


def singularize_class_head(value: str) -> str:
    """Singularize a class phrase's head noun without truncating mass nouns."""
    words = value.split()
    if (
        not words
        or words[-1] in _NON_SINGULAR_CLASS_HEADS
        or words[-1].endswith(_NON_SINGULAR_CLASS_SUFFIXES)
    ):
        return value
    singular = _INFLECT.singular_noun(cast(Any, words[-1]))
    if isinstance(singular, str):
        words[-1] = singular
    return " ".join(words)


def lemmatize_property_verb(value: str) -> str:
    """Conservatively lemmatize a present-tense predicate's first word."""
    words = value.split()
    if not words or words[0] in _NON_LEMMATIZED_PROPERTY_VERBS:
        return value
    verb = words[0]
    if verb.endswith("ies") and len(verb) > 3:
        verb = f"{verb[:-3]}y"
    elif verb.endswith(("ches", "shes", "sses", "xes", "zes")):
        verb = verb[:-2]
    elif verb.endswith("s") and not verb.endswith("ss"):
        verb = verb[:-1]
    words[0] = verb
    return " ".join(words)


def normalize_ontology_term(value: str, kind: OntologyTermKind) -> str:
    """Normalize a class or property label for ontology vocabulary matching."""
    normalized = normalized_label_key(value)
    if kind == "class":
        normalized = singularize_class_head(normalized)
        return _CLASS_ALIASES.get(normalized, normalized)
    return lemmatize_property_verb(normalized)


def normalize_term_assignments(
    assignments: Mapping[str, Iterable[str]], kind: OntologyTermKind
) -> dict[str, set[str]]:
    """Normalize labels while asserting exact preservation of valid provenance."""
    materialized = {source: set(terms) for source, terms in assignments.items()}
    canonical_map = {
        term: normalized
        for term in {term for terms in materialized.values() for term in terms}
        if (normalized := normalize_ontology_term(term, kind))
    }
    valid_original = {
        source: {term for term in terms if term in canonical_map}
        for source, terms in materialized.items()
    }
    normalized_assignments = {
        source: {canonical_map[term] for term in terms}
        for source, terms in valid_original.items()
    }
    validate_canonical_assignment_provenance(
        valid_original,
        normalized_assignments,
        canonical_map,
        stage=f"{kind} lexical/morphological normalization",
    )
    return normalized_assignments
