from __future__ import annotations

import numpy as np
import pytest

from kg_gen.ontology.cq_term_assignment import (
    CanonicalCQTermAssigner,
    reassign_canonical_terms,
)


class FakeEncoder:
    def encode(self, sentences: list[str], *, normalize_embeddings: bool):
        assert normalize_embeddings is True
        vectors = []
        for sentence in sentences:
            if sentence.startswith("Competency question: Which software"):
                vectors.append([1.0, 0.0])
            elif sentence.startswith("Competency question: Which animal"):
                vectors.append([0.0, 1.0])
            elif sentence == "Ontology class: software":
                vectors.append([1.0, 0.0])
            elif sentence == "Ontology class: data item":
                vectors.append([0.8, 0.6])
            elif sentence == "Ontology class: animal":
                vectors.append([0.0, 1.0])
            elif sentence == "Ontology property: use software":
                vectors.append([1.0, 0.0])
            elif sentence == "Ontology property: eat":
                vectors.append([0.0, 1.0])
            else:
                raise AssertionError(f"Unexpected text: {sentence}")
        return np.asarray(vectors, dtype=float)


def test_assign_preserves_existing_and_adds_bounded_canonical_terms():
    assigner = CanonicalCQTermAssigner(
        "unused", threshold=0.7, max_additions=1, encoder=FakeEncoder()
    )
    result = assigner.assign(
        {
            "CQ1": "Which software can use this data item?",
            "CQ2": "Which animal eats another animal?",
        },
        {"CQ1": {"data item"}, "CQ2": {"animal", "software"}},
        "class",
    )

    assert result["CQ1"] == {"data item", "software"}
    assert result["CQ2"] == {"animal", "software"}


def test_assign_uses_only_terms_in_the_domain_vocabulary():
    assigner = CanonicalCQTermAssigner(
        "unused", threshold=0.5, max_additions=3, encoder=FakeEncoder()
    )
    result = assigner.assign(
        {"CQ1": "Which software can use this data item?"},
        {"CQ1": {"data item"}},
        "class",
    )

    assert result == {"CQ1": {"data item"}}


def test_assign_keeps_class_and_property_vocabularies_separate():
    assigner = CanonicalCQTermAssigner(
        "unused", threshold=0.5, max_additions=1, encoder=FakeEncoder()
    )
    result = assigner.assign(
        {
            "CQ1": "Which software can use this data item?",
            "CQ2": "Which animal eats another animal?",
        },
        {"CQ1": set(), "CQ2": {"eat", "use software"}},
        "property",
    )

    assert result["CQ1"] == {"use software"}
    assert result["CQ2"] == {"eat", "use software"}


def test_reassign_canonical_terms_rejects_invalid_question_records():
    assigner = CanonicalCQTermAssigner(
        "unused", threshold=0.5, max_additions=1, encoder=FakeEncoder()
    )

    with pytest.raises(ValueError, match="Invalid question"):
        reassign_canonical_terms(
            [{"id": "CQ1"}], {"CQ1": {"software"}}, "class", assigner
        )


@pytest.mark.parametrize(
    ("threshold", "max_additions", "message"),
    [(-0.1, 1, "threshold"), (1.1, 1, "threshold"), (0.5, 0, "max_additions")],
)
def test_assigner_validates_configuration(threshold, max_additions, message):
    with pytest.raises(ValueError, match=message):
        CanonicalCQTermAssigner(
            "unused",
            threshold=threshold,
            max_additions=max_additions,
            encoder=FakeEncoder(),
        )
