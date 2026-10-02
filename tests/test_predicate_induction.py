from __future__ import annotations

import pytest

from kg_gen.ontology.predicate_induction import (
    CanonicalPredicate,
    PredicateInductionResult,
    PredicateOccurrence,
    RejectedPredicate,
    canonical_predicate_assignments,
    induce_canonical_predicates,
)


class FakeProvider:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def induce(self, occurrences, *, feedback=None):
        self.calls.append((occurrences, feedback))
        return self.result


def occurrences():
    return [
        PredicateOccurrence(
            label="can read",
            subject="software",
            object="data item",
            cq_id="CQ1",
            question="What software can read this data?",
        ),
        PredicateOccurrence(
            label="accept input",
            subject="software",
            object="data item",
            cq_id="CQ2",
            question="Which software accepts this input?",
        ),
        PredicateOccurrence(
            label="is a",
            subject="software",
            object="tool",
            cq_id="CQ3",
            question="Is this software a tool?",
        ),
    ]


def valid_result():
    return PredicateInductionResult(
        predicates=[
            CanonicalPredicate(
                canonical_label="has specified data inputs",
                aliases=["can read", "accept input"],
                reason="same directed input relation",
            )
        ]
    )


def test_induction_normalizes_canonical_label_and_propagates_alias_provenance():
    result = induce_canonical_predicates(occurrences(), FakeProvider(valid_result()))

    assert result.alias_map == {
        "accept input": "has specified data input",
        "can read": "has specified data input",
    }
    assert canonical_predicate_assignments(occurrences(), result) == {
        "CQ1": {"has specified data input"},
        "CQ2": {"has specified data input"},
    }


@pytest.mark.parametrize(
    "result,match",
    [
        (
            PredicateInductionResult(
                predicates=[
                    CanonicalPredicate(
                        canonical_label="has input", aliases=["can read"]
                    )
                ]
            ),
            "missing=\\['accept input'\\]",
        ),
        (
            PredicateInductionResult(
                predicates=[
                    CanonicalPredicate(
                        canonical_label="has input",
                        aliases=["can read", "accept input", "invented"],
                    )
                ],
                rejected=[],
            ),
            "unexpected=\\['invented'\\]",
        ),
        (
            PredicateInductionResult(
                predicates=[
                    CanonicalPredicate(
                        canonical_label="has input",
                        aliases=["can read", "accept input"],
                    )
                ],
                rejected=[
                    RejectedPredicate(label="can read", reason="too_generic"),
                ],
            ),
            "duplicates=\\['can read'\\]",
        ),
    ],
)
def test_induction_rejects_incomplete_open_or_overlapping_assignments(result, match):
    with pytest.raises(ValueError, match=match):
        induce_canonical_predicates(occurrences(), FakeProvider(result))


def test_empty_occurrences_do_not_call_provider():
    class FailingProvider:
        def induce(self, occurrences, *, feedback=None):
            raise AssertionError("provider should not be called")

    assert induce_canonical_predicates([], FailingProvider()).predicates == []


def test_synthesized_canonical_label_repeated_as_alias_is_ignored():
    values = [
        PredicateOccurrence(
            label="reside at",
            subject="person",
            object="place",
            cq_id="CQ1",
            question="Where does this person live?",
        )
    ]
    provider = FakeProvider(
        PredicateInductionResult(
            predicates=[
                CanonicalPredicate(
                    canonical_label="live",
                    aliases=["reside at", "live"],
                )
            ]
        )
    )

    result = induce_canonical_predicates(values, provider)

    assert result.alias_map == {"reside at": "live"}
    assert result.predicates[0].aliases == ["reside at"]


def test_synthesized_canonical_label_repeated_as_rejected_is_ignored():
    values = [
        PredicateOccurrence(
            label="reside at",
            subject="person",
            object="place",
            cq_id="CQ1",
            question="Where does this person live?",
        )
    ]
    provider = FakeProvider(
        PredicateInductionResult(
            predicates=[
                CanonicalPredicate(
                    canonical_label="live",
                    aliases=["reside at"],
                )
            ],
            rejected=[RejectedPredicate(label="live", reason="too_generic")],
        )
    )

    result = induce_canonical_predicates(values, provider)

    assert result.alias_map == {"reside at": "live"}
    assert result.rejected == []


def test_invented_rejected_explanatory_variant_is_ignored():
    values = [
        PredicateOccurrence(
            label="live in",
            subject="person",
            object="place",
            cq_id="CQ1",
            question="Where does this person live?",
        )
    ]
    provider = FakeProvider(
        PredicateInductionResult(
            predicates=[
                CanonicalPredicate(
                    canonical_label="lives in location",
                    aliases=["live in"],
                )
            ],
            rejected=[RejectedPredicate(label="live", reason="too_generic")],
        )
    )

    result = induce_canonical_predicates(values, provider)

    assert result.alias_map == {"live in": "live in location"}
    assert result.rejected == []


def test_synthesized_canonical_label_repeated_under_other_group_is_ignored():
    values = [
        PredicateOccurrence(
            label="reside at",
            subject="person",
            object="place",
            cq_id="CQ1",
            question="Where does this person live?",
        ),
        PredicateOccurrence(
            label="work for",
            subject="person",
            object="organization",
            cq_id="CQ2",
            question="Where does this person work?",
        ),
    ]
    provider = FakeProvider(
        PredicateInductionResult(
            predicates=[
                CanonicalPredicate(
                    canonical_label="live",
                    aliases=["reside at"],
                ),
                CanonicalPredicate(
                    canonical_label="work for",
                    aliases=["work for", "live"],
                ),
            ]
        )
    )

    result = induce_canonical_predicates(values, provider)

    assert result.alias_map == {"reside at": "live", "work for": "work for"}


def test_normalized_alias_spelling_is_mapped_to_supplied_vocabulary():
    values = [
        PredicateOccurrence(
            label="purchase game offering",
            subject="player",
            object="game offering",
            cq_id="CQ1",
            question="Which offering did the player purchase?",
        )
    ]
    provider = FakeProvider(
        PredicateInductionResult(
            predicates=[
                CanonicalPredicate(
                    canonical_label="purchase game offering",
                    aliases=["purchases game offerings"],
                )
            ]
        )
    )

    result = induce_canonical_predicates(values, provider)

    assert result.predicates[0].aliases == ["purchase game offering"]


def test_generic_has_and_class_membership_are_rejected_before_llm_call():
    provider = FakeProvider(
        PredicateInductionResult(
            predicates=[
                CanonicalPredicate(canonical_label="eat", aliases=["eat"])
            ]
        )
    )
    values = [
        PredicateOccurrence(
            label=label,
            subject="animal",
            object="plant",
            cq_id="CQ1",
            question="Which animal eats plants?",
        )
        for label in ("eat", "has", "is a")
    ]

    result = induce_canonical_predicates(values, provider)

    assert {item.label: item.reason for item in result.rejected} == {
        "has": "too_generic",
        "is a": "class_membership",
    }
    assert {item.label for item in provider.calls[0][0]} == {"eat"}


def test_invalid_duplicate_response_is_retried_with_feedback():
    invalid = PredicateInductionResult(
        predicates=[
            CanonicalPredicate(
                canonical_label="has input",
                aliases=["can read", "accept input"],
            )
        ],
        rejected=[RejectedPredicate(label="can read", reason="too_generic")],
    )
    valid = PredicateInductionResult(
        predicates=[
            CanonicalPredicate(
                canonical_label="has input",
                aliases=["can read", "accept input"],
            )
        ]
    )

    class FlakyProvider:
        def __init__(self):
            self.feedback = []

        def induce(self, occurrences, *, feedback=None):
            self.feedback.append(feedback)
            return invalid if len(self.feedback) == 1 else valid

    provider = FlakyProvider()
    result = induce_canonical_predicates(occurrences(), provider, max_attempts=2)

    assert result.alias_map["can read"] == "has input"
    assert provider.feedback[0] is None
    assert "duplicates=['can read']" in provider.feedback[1]
