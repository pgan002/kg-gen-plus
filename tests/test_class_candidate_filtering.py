from __future__ import annotations

from typing import Any

import pytest

from kg_gen.ontology.class_candidate_filtering import (
    ClassCandidateDecision,
    filter_class_candidates,
)


class FakeProvider:
    def __init__(self, decisions):
        self.decisions = decisions
        self.received = None

    def decide(self, questions, candidates, relations_by_cq):
        self.received = (questions, candidates, relations_by_cq)
        return self.decisions


def decision(term: str, keep: bool, reason: Any) -> ClassCandidateDecision:
    return ClassCandidateDecision(
        term=term,
        keep=keep,
        reason=reason,
        explanation="test",
    )


def test_filter_removes_incidental_surface_concepts_and_preserves_provenance():
    provider = FakeProvider(
        [
            decision("software", True, "answer_or_variable_type"),
            decision("collaboration", False, "incidental_surface_concept"),
            decision("developer", False, "named_individual_or_example"),
        ]
    )
    assignments = {
        "CQ1": {"software", "developer"},
        "CQ2": {"software", "collaboration"},
    }
    filtered, decisions = filter_class_candidates(
        [
            {"id": "CQ1", "value": "Who developed this software?"},
            {"id": "CQ2", "value": "Can this software support collaboration?"},
        ],
        assignments,
        {
            "CQ1": ["software --developed by--> developer"],
            "CQ2": ["software --support--> collaboration"],
        },
        provider,
    )

    assert filtered == {"CQ1": {"software"}, "CQ2": {"software"}}
    assert len(decisions) == 3
    assert provider.received is not None
    assert provider.received[1] == {
        "software": {"CQ1", "CQ2"},
        "developer": {"CQ1"},
        "collaboration": {"CQ2"},
    }


@pytest.mark.parametrize(
    "decisions,match",
    [
        (
            [decision("software", True, "central_domain_concept")],
            "missing=\\['developer'\\]",
        ),
        (
            [
                decision("software", True, "central_domain_concept"),
                decision("developer", True, "central_domain_concept"),
                decision("extra", False, "unsupported"),
            ],
            "unexpected=\\['extra'\\]",
        ),
        (
            [
                decision("software", True, "central_domain_concept"),
                decision("developer", True, "central_domain_concept"),
                decision("developer", False, "unsupported"),
            ],
            "duplicates=\\['developer'\\]",
        ),
    ],
)
def test_filter_rejects_incomplete_or_open_vocabulary_decisions(decisions, match):
    provider = FakeProvider(decisions)
    with pytest.raises(ValueError, match=match):
        filter_class_candidates(
            [{"id": "CQ1", "value": "Who developed this software?"}],
            {"CQ1": {"software", "developer"}},
            {},
            provider,
        )


def test_filter_rejects_unknown_cq_provenance():
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="unknown CQ"):
        filter_class_candidates(
            [{"id": "CQ1", "value": "Which software?"}],
            {"CQ2": {"software"}},
            {},
            provider,
        )
