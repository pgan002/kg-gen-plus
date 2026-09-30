"""Ontology-specific class and property label normalization."""

from __future__ import annotations

from typing import Any, Literal, cast

import inflect

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
