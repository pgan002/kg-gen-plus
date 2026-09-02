"""Match model-produced labels back to the ontology terms they mean.

The model is shown ontology terms as *labels* and answers with labels, so the
label is the only join key between an extracted relation and the ontology. That
join used to be exact string equality, which is brittle in a way that loses
data silently: on a 16,131-document MuSiQue run the model returned
``birth_date``, ``birth_place``, ``death_date`` and ``death_place`` where the
ontology says ``birth date``, ``birth place``, ... Each relation stayed in the
graph with no URI, belonging to neither the ontology nor its extensions.

Normalization here is deliberately conservative -- case, word separators and
camelCase boundaries only. Nothing guesses: ``preceeded by`` still does not
match ``preceded by``. Approximate matching cannot be checked by the caller,
and on that same ontology ``country`` and ``county`` are 0.92 similar, so a
plausible cutoff would silently merge two real classes.

SKOS labels are honoured because they are the ontology author's own statement
of which words mean this term: ``skos:prefLabel``, ``skos:altLabel`` and
``skos:hiddenLabel`` (the last exists precisely for spelling variants).
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, TypeVar

from rdflib import Graph as RDFGraph, URIRef, RDFS
from rdflib.namespace import SKOS

# Split camelCase and PascalCase, but leave acronyms intact: "birthPlace" ->
# "birth Place", "HTTPServer" -> "HTTP Server", "CEO" -> "CEO".
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# The SKOS properties that assert "this string names this term".
SKOS_LABEL_PROPERTIES = (SKOS.prefLabel, SKOS.altLabel, SKOS.hiddenLabel)

T = TypeVar("T")


def normalize_label(label: Optional[str]) -> str:
    """Reduce a label to its comparison form.

    ``birth_place``, ``birthPlace``, ``Birth Place`` and ``birth place`` all
    become ``birth place``. Returns ``""`` for a missing label, which never
    matches anything in an index built by :func:`build_alias_index`.
    """
    if not label:
        return ""
    spaced = _CAMEL_BOUNDARY.sub(" ", str(label))
    return _NON_ALNUM.sub(" ", spaced.casefold()).strip()


def uri_local_name(uri: Optional[str]) -> str:
    """The last path/fragment segment of a URI, or ``""``."""
    if not uri:
        return ""
    return re.split(r"[#/]", str(uri))[-1]


def skos_labels(ontology: Optional[RDFGraph], uri: Optional[str]) -> list[str]:
    """Every ``skos:prefLabel``/``altLabel``/``hiddenLabel`` declared for ``uri``."""
    if ontology is None or not uri:
        return []
    subject = URIRef(uri)
    return [
        str(obj)
        for prop in SKOS_LABEL_PROPERTIES
        for obj in ontology.objects(subject, prop)
    ]


def build_alias_index(
    items: Iterable[T], ontology: Optional[RDFGraph] = None
) -> dict[str, T]:
    """Map every known alias of each item to that item, keyed by normalized form.

    Items need a ``label`` and may have a ``uri`` (``EntityType`` and
    ``OntologyPredicate`` both qualify). Aliases are added in priority order --
    primary labels for all items first, then URI local names, then SKOS labels
    -- and an alias never displaces an entry already present, so one term's
    synonym cannot shadow another term's primary label. Pass ``ontology`` (the
    parsed RDF graph) to pick up the SKOS labels; without it the index still
    covers the label and URI forms.
    """
    index: dict[str, T] = {}
    items = list(items)

    def add(alias: Optional[str], item: T) -> None:
        key = normalize_label(alias)
        if key and key not in index:
            index[key] = item

    for item in items:
        add(getattr(item, "label", None), item)
    for item in items:
        add(uri_local_name(getattr(item, "uri", None)), item)
    if ontology is not None:
        for item in items:
            for alias in skos_labels(ontology, getattr(item, "uri", None)):
                add(alias, item)
    return index


def match_label(index: dict[str, T], label: Optional[str]) -> Optional[T]:
    """Look ``label`` up in an alias index, or return ``None``."""
    return index.get(normalize_label(label))


def resolve_uri_by_label(
    ontology: Optional[RDFGraph], label: Optional[str]
) -> Optional[URIRef]:
    """The URI of the ontology term named ``label``, by any of its labels.

    Used where only a label is in hand and the class hierarchy has to be
    consulted -- the model answers with labels, while ``rdfs:subClassOf`` is
    keyed by URI.
    """
    target = normalize_label(label)
    if ontology is None or not target:
        return None
    for prop in (RDFS.label, *SKOS_LABEL_PROPERTIES):
        for subject, _, value in ontology.triples((None, prop, None)):
            if isinstance(subject, URIRef) and normalize_label(str(value)) == target:
                return subject
    return None


def find_normalized_label_collisions(ontology: RDFGraph) -> list[tuple[str, list[str]]]:
    """Ontology terms that share a normalized label, as ``(form, [uri, ...])``.

    Normalizing makes matching forgiving, but it also means two terms whose
    labels differ only in case or punctuation become indistinguishable and one
    of them can never be matched. That is an ontology defect worth reporting
    rather than resolving silently, so callers log it at parse time.
    """
    by_form: dict[str, list[str]] = {}
    for subject, _, label in ontology.triples((None, RDFS.label, None)):
        if not isinstance(subject, URIRef):
            continue
        form = normalize_label(str(label))
        if not form:
            continue
        uris = by_form.setdefault(form, [])
        if str(subject) not in uris:
            uris.append(str(subject))
    return [(form, uris) for form, uris in sorted(by_form.items()) if len(uris) > 1]
