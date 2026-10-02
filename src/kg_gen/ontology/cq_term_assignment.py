"""Assign a domain's canonical ontology vocabulary back to its source CQs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from kg_gen.ontology.term_normalization import OntologyTermKind


class TermEncoder(Protocol):
    def encode(
        self, sentences: list[str], *, normalize_embeddings: bool
    ) -> Any: ...


class CanonicalCQTermAssigner:
    """Expand CQ-local terms from a bounded canonical domain vocabulary.

    Existing assignments are retained. New terms must exceed ``threshold`` and
    are capped per CQ, making this a conservative reassignment experiment rather
    than unconstrained extraction from the question text.
    """

    def __init__(
        self,
        model_name: str,
        threshold: float,
        max_additions: int,
        *,
        encoder: TermEncoder | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if max_additions < 1:
            raise ValueError("max_additions must be at least 1")
        self.encoder = encoder or SentenceTransformer(model_name)
        self.threshold = threshold
        self.max_additions = max_additions

    @staticmethod
    def _question_text(
        question: str, existing_terms: set[str], kind: OntologyTermKind
    ) -> str:
        role = "class" if kind == "class" else "property"
        existing = ", ".join(sorted(existing_terms)) or "none"
        return (
            f"Competency question: {question}\n"
            f"Existing ontology {role} terms: {existing}\n"
            f"Find other required ontology {role} terms."
        )

    @staticmethod
    def _term_text(term: str, kind: OntologyTermKind) -> str:
        role = "class" if kind == "class" else "property"
        return f"Ontology {role}: {term}"

    def assign(
        self,
        questions: Mapping[str, str],
        assignments: Mapping[str, set[str]],
        kind: OntologyTermKind,
    ) -> dict[str, set[str]]:
        """Return assignments expanded from terms occurring anywhere in the domain."""
        cq_ids = list(questions)
        materialized = {cq_id: set(assignments.get(cq_id, set())) for cq_id in cq_ids}
        vocabulary = sorted({term for terms in materialized.values() for term in terms})
        if not vocabulary or not cq_ids:
            return materialized

        query_texts = [
            self._question_text(questions[cq_id], materialized[cq_id], kind)
            for cq_id in cq_ids
        ]
        term_texts = [self._term_text(term, kind) for term in vocabulary]
        query_embeddings = np.asarray(
            self.encoder.encode(query_texts, normalize_embeddings=True), dtype=float
        )
        term_embeddings = np.asarray(
            self.encoder.encode(term_texts, normalize_embeddings=True), dtype=float
        )
        similarities = query_embeddings @ term_embeddings.T

        for row, cq_id in enumerate(cq_ids):
            existing = materialized[cq_id]
            candidates = sorted(
                (
                    (float(similarities[row, column]), term)
                    for column, term in enumerate(vocabulary)
                    if term not in existing
                    and similarities[row, column] >= self.threshold
                ),
                key=lambda item: (-item[0], item[1]),
            )
            existing.update(term for _, term in candidates[: self.max_additions])
        return materialized


def reassign_canonical_terms(
    questions: Sequence[dict[str, Any]],
    assignments: Mapping[str, set[str]],
    kind: OntologyTermKind,
    assigner: CanonicalCQTermAssigner,
) -> dict[str, set[str]]:
    """Validate CQ records and apply canonical-vocabulary reassignment."""
    question_map = {}
    for index, question in enumerate(questions):
        if not isinstance(question, dict) or not {"id", "value"} <= question.keys():
            raise ValueError(f"Invalid question at index {index}: {question!r}")
        question_map[str(question["id"])] = str(question["value"])
    return assigner.assign(question_map, assignments, kind)
