"""Conservative semantic clustering for normalized ontology-term labels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

_SEMANTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "with",
}
_GENERIC_TERMS = {
    "attribute",
    "class",
    "concept",
    "entity",
    "event",
    "location",
    "object",
    "process",
    "property",
    "relation",
    "relationship",
    "role",
    "state",
    "thing",
    "type",
}
_NEGATION_TOKENS = {"no", "non", "not", "without"}


class TermEncoder(Protocol):
    def encode(
        self, sentences: list[str], *, normalize_embeddings: bool
    ) -> Any: ...


def content_tokens(term: str) -> set[str]:
    return set(term.split()) - _SEMANTIC_STOPWORDS


def semantic_pair_allowed(left: str, right: str) -> bool:
    """Reject risky semantic merges before considering embedding similarity."""
    if left == right or left in _GENERIC_TERMS or right in _GENERIC_TERMS:
        return False
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if bool(left_tokens & _NEGATION_TOKENS) != bool(right_tokens & _NEGATION_TOKENS):
        return False
    if not left_tokens & right_tokens:
        return False
    # A strict subset usually means broader/narrower rather than equivalent:
    # "plant" versus "plant part". Stopwords are excluded, so reordered forms
    # such as "plant part" and "part of plant" remain eligible.
    return not (left_tokens < right_tokens or right_tokens < left_tokens)


class OntologyTermClusterer:
    """Complete-link semantic clustering for already-normalized labels."""

    def __init__(
        self,
        model_name: str,
        threshold: float,
        *,
        encoder: TermEncoder | None = None,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.encoder = encoder or SentenceTransformer(model_name)
        self.threshold = threshold

    def canonical_map(self, occurrence_counts: Mapping[str, int]) -> dict[str, str]:
        terms = sorted(occurrence_counts)
        if len(terms) < 2:
            return {term: term for term in terms}
        embeddings = np.asarray(
            self.encoder.encode(terms, normalize_embeddings=True), dtype=float
        )
        similarities = embeddings @ embeddings.T
        clusters: list[list[int]] = []
        for candidate, term in enumerate(terms):
            matching_cluster = next(
                (
                    cluster
                    for cluster in clusters
                    if all(
                        similarities[candidate, member] >= self.threshold
                        and semantic_pair_allowed(term, terms[member])
                        for member in cluster
                    )
                ),
                None,
            )
            if matching_cluster is None:
                clusters.append([candidate])
            else:
                matching_cluster.append(candidate)

        mapping = {}
        for cluster in clusters:
            members = [terms[index] for index in cluster]
            canonical = min(
                members,
                key=lambda term: (
                    -occurrence_counts[term],
                    len(term.split()),
                    len(term),
                    term,
                ),
            )
            mapping.update({member: canonical for member in members})
        return mapping


def cluster_term_assignments(
    assignments: Mapping[str, Iterable[str]], clusterer: OntologyTermClusterer
) -> dict[str, set[str]]:
    """Canonicalize labels while retaining every source/provenance assignment."""
    occurrence_counts: dict[str, int] = defaultdict(int)
    materialized = {source: set(terms) for source, terms in assignments.items()}
    for terms in materialized.values():
        for term in terms:
            occurrence_counts[term] += 1
    mapping = clusterer.canonical_map(occurrence_counts)
    return {
        source: {mapping[term] for term in terms}
        for source, terms in materialized.items()
    }
