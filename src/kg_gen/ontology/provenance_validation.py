"""Runtime invariants for provenance-preserving term canonicalization."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping


class ProvenanceInvariantError(ValueError):
    """Raised when canonicalization loses or invents source assignments."""


def provenance_by_term(
    assignments: Mapping[str, Iterable[str]],
) -> dict[str, set[str]]:
    provenance: dict[str, set[str]] = defaultdict(set)
    for source, terms in assignments.items():
        for term in terms:
            provenance[term].add(source)
    return dict(provenance)


def validate_canonical_assignment_provenance(
    original_assignments: Mapping[str, Iterable[str]],
    canonical_assignments: Mapping[str, Iterable[str]],
    canonical_map: Mapping[str, str],
    *,
    stage: str,
) -> None:
    """Assert exact source preservation under an alias-to-canonical mapping.

    Every original term must occur in ``canonical_map``. For each source/CQ, the
    actual canonical assignments must exactly equal application of that map.
    The second global check produces a direct per-term provenance diagnostic.
    """
    original = {source: set(terms) for source, terms in original_assignments.items()}
    actual = {source: set(terms) for source, terms in canonical_assignments.items()}
    source_keys = set(original) | set(actual)
    unmapped = sorted(
        {term for terms in original.values() for term in terms if term not in canonical_map}
    )
    if unmapped:
        raise ProvenanceInvariantError(
            f"{stage}: canonical map is missing source terms: {unmapped}"
        )

    expected = {
        source: {canonical_map[term] for term in original.get(source, set())}
        for source in source_keys
    }
    actual_complete = {source: actual.get(source, set()) for source in source_keys}
    if expected != actual_complete:
        details = {
            source: {
                "missing": sorted(expected[source] - actual_complete[source]),
                "unexpected": sorted(actual_complete[source] - expected[source]),
            }
            for source in sorted(source_keys)
            if expected[source] != actual_complete[source]
        }
        raise ProvenanceInvariantError(
            f"{stage}: canonical assignments changed source provenance: {details}"
        )

    expected_provenance = provenance_by_term(expected)
    actual_provenance = provenance_by_term(actual_complete)
    if expected_provenance != actual_provenance:
        terms = set(expected_provenance) | set(actual_provenance)
        details = {
            term: {
                "missing_sources": sorted(
                    expected_provenance.get(term, set())
                    - actual_provenance.get(term, set())
                ),
                "unexpected_sources": sorted(
                    actual_provenance.get(term, set())
                    - expected_provenance.get(term, set())
                ),
            }
            for term in sorted(terms)
            if expected_provenance.get(term, set())
            != actual_provenance.get(term, set())
        }
        raise ProvenanceInvariantError(
            f"{stage}: canonical term provenance union is invalid: {details}"
        )
