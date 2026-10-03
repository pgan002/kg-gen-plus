"""Domain-wide induction of canonical ontology predicates from CQ relations."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, Field

from kg_gen.ontology.provenance_validation import (
    validate_canonical_assignment_provenance,
)
from kg_gen.ontology.term_normalization import (
    normalize_ontology_term,
    singularize_class_head,
)


class PredicateOccurrence(BaseModel):
    label: str
    subject: str
    object: str
    cq_id: str
    question: str


class CanonicalPredicate(BaseModel):
    canonical_label: str
    aliases: list[str]
    reason: str = ""


RejectionReason = Literal[
    "copular_support",
    "class_membership",
    "too_generic",
    "not_an_ontology_relation",
    "unsupported",
]


class RejectedPredicate(BaseModel):
    label: str
    reason: RejectionReason
    explanation: str = ""


class PredicateInductionResult(BaseModel):
    predicates: list[CanonicalPredicate] = Field(default_factory=list)
    rejected: list[RejectedPredicate] = Field(default_factory=list)

    @property
    def alias_map(self) -> dict[str, str]:
        return {
            alias: predicate.canonical_label
            for predicate in self.predicates
            for alias in predicate.aliases
        }


class PredicateInductionProvider(Protocol):
    def induce(
        self,
        occurrences: Sequence[PredicateOccurrence],
        *,
        feedback: str | None = None,
    ) -> PredicateInductionResult: ...


_PRE_REJECTIONS: dict[str, tuple[RejectionReason, str]] = {
    "are": ("copular_support", "Grammatical copula, not an ontology property."),
    "be": ("copular_support", "Grammatical copula, not an ontology property."),
    "has": ("too_generic", "Bare 'has' does not identify a specific relation."),
    "has type": ("class_membership", "Type assertion, not an object property."),
    "is": ("copular_support", "Grammatical copula, not an ontology property."),
    "is a": ("class_membership", "Class membership, not an object property."),
    "is an": ("class_membership", "Class membership, not an object property."),
    "type of": ("class_membership", "Type assertion, not an object property."),
    "was": ("copular_support", "Grammatical copula, not an ontology property."),
    "were": ("copular_support", "Grammatical copula, not an ontology property."),
}


class OpenAIPredicateInductionProvider:
    """Induce canonical predicates through an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        *,
        api_base: str,
        api_key: str,
        max_tokens: int = 8192,
        enable_thinking: bool = False,
    ) -> None:
        self.model = model.removeprefix("openai/")
        self.client = OpenAI(api_key=api_key, base_url=api_base.rstrip("/"))
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

    @staticmethod
    def _prompt(
        occurrences: Sequence[PredicateOccurrence], feedback: str | None = None
    ) -> str:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for occurrence in occurrences:
            grouped[occurrence.label].append(
                {
                    "subject": occurrence.subject,
                    "object": occurrence.object,
                    "cq_id": occurrence.cq_id,
                    "question": occurrence.question,
                }
            )
        evidence = [
            {"label": label, "occurrences": values}
            for label, values in sorted(grouped.items())
        ]
        return (
            "Induce a canonical ontology property vocabulary from the supplied "
            "raw relation labels and all their competency-question evidence. "
            "Assign every supplied raw label exactly once: either to one canonical "
            "predicate group as an alias, or to rejected. Do not invent aliases, "
            "omit labels, or place one alias in multiple groups. Preserve relation "
            "direction and do not merge relations differing in direction, modality, "
            "temporality, causality, or specificity. Prefer a specific concise "
            "present-tense predicate label such as 'uses platform' or 'has specified "
            "data input'; avoid generic labels such as 'use', 'provide', or 'has' "
            "when a more precise alias is supported. The canonical_label may be a "
            "new normalized label justified by the aliases, but must not introduce "
            "new semantics. Reject grammatical copulas, class-membership support, "
            "generic non-relations, and unsupported phrases. Return JSON only with "
            "shape {\"predicates\":[{\"canonical_label\":str,\"aliases\":[str],"
            "\"reason\":str}],\"rejected\":[{\"label\":str,\"reason\":one of "
            "[\"copular_support\",\"class_membership\",\"too_generic\","
            "\"not_an_ontology_relation\",\"unsupported\"],"
            "\"explanation\":str}]}.\n\n"
            f"Predicate evidence:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
            + (
                "\n\nYour previous response failed validation. Correct it without "
                f"changing the input vocabulary. Validation error: {feedback}"
                if feedback
                else ""
            )
        )

    def induce(
        self,
        occurrences: Sequence[PredicateOccurrence],
        *,
        feedback: str | None = None,
    ) -> PredicateInductionResult:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conservative ontology engineer performing "
                        "closed-vocabulary predicate induction."
                    ),
                },
                {
                    "role": "user",
                    "content": self._prompt(occurrences, feedback),
                },
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            extra_body={
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            },
        )
        choice = response.choices[0]
        content = choice.message.content
        if not content:
            reasoning = getattr(choice.message, "reasoning_content", None)
            raise ValueError(
                "Predicate induction returned an empty response"
                + (f"; reasoning_content={reasoning!r}" if reasoning else "")
            )
        return PredicateInductionResult.model_validate_json(content)


def _normalized_predicate_label(label: str) -> str:
    return singularize_class_head(normalize_ontology_term(label, "property"))


def validate_predicate_induction(
    occurrences: Sequence[PredicateOccurrence], result: PredicateInductionResult
) -> PredicateInductionResult:
    """Validate complete assignment while tolerating harmless label formatting."""
    expected = {occurrence.label for occurrence in occurrences}
    expected_by_normalized = {
        _normalized_predicate_label(label): label for label in expected
    }
    response_canonical_labels = {
        _normalized_predicate_label(predicate.canonical_label)
        for predicate in result.predicates
    }
    sanitized_predicates = []
    for predicate in result.predicates:
        canonical = _normalized_predicate_label(predicate.canonical_label)
        aliases = []
        for alias in predicate.aliases:
            normalized_alias = _normalized_predicate_label(alias)
            supplied_alias = expected_by_normalized.get(normalized_alias)
            if supplied_alias is not None:
                aliases.append(supplied_alias)
            elif normalized_alias in response_canonical_labels:
                # Models often repeat synthesized canonical labels as aliases,
                # sometimes under another group. They carry no source evidence.
                continue
            else:
                aliases.append(alias)
        sanitized_predicates.append(
            CanonicalPredicate(
                canonical_label=canonical,
                aliases=aliases,
                reason=predicate.reason,
            )
        )
    sanitized_rejected = []
    for rejected in result.rejected:
        normalized = _normalized_predicate_label(rejected.label)
        if normalized not in expected_by_normalized:
            # Rejected entries do not contribute to the induced vocabulary or
            # provenance. Models sometimes add stems or explanatory variants
            # (for example, reject "live" while assigning source alias "live in").
            # Discard any rejection that was not actually supplied as input.
            continue
        sanitized_rejected.append(
            RejectedPredicate(
                label=expected_by_normalized.get(normalized, rejected.label),
                reason=rejected.reason,
                explanation=rejected.explanation,
            )
        )
    result = PredicateInductionResult(
        predicates=sanitized_predicates,
        rejected=sanitized_rejected,
    )
    returned = [
        alias for predicate in result.predicates for alias in predicate.aliases
    ] + [rejected.label for rejected in result.rejected]
    duplicates = sorted({label for label in returned if returned.count(label) > 1})
    missing = sorted(expected - set(returned))
    unexpected = sorted(set(returned) - expected)
    if duplicates or missing or unexpected:
        raise ValueError(
            "Invalid predicate induction: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )

    normalized_predicates = []
    canonical_labels = set()
    for predicate in result.predicates:
        canonical = predicate.canonical_label
        if not canonical:
            raise ValueError(
                f"Empty canonical predicate for aliases {predicate.aliases!r}"
            )
        if canonical in canonical_labels:
            raise ValueError(f"Duplicate canonical predicate label: {canonical!r}")
        canonical_labels.add(canonical)
        normalized_predicates.append(
            CanonicalPredicate(
                canonical_label=canonical,
                aliases=sorted(predicate.aliases),
                reason=predicate.reason,
            )
        )
    return PredicateInductionResult(
        predicates=normalized_predicates,
        rejected=sorted(result.rejected, key=lambda item: item.label),
    )


def induce_canonical_predicates(
    occurrences: Sequence[PredicateOccurrence],
    provider: PredicateInductionProvider,
    *,
    max_attempts: int = 3,
) -> PredicateInductionResult:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if not occurrences:
        return PredicateInductionResult()

    labels = {occurrence.label for occurrence in occurrences}
    pre_rejected = [
        RejectedPredicate(label=label, reason=reason, explanation=explanation)
        for label in sorted(labels & _PRE_REJECTIONS.keys())
        for reason, explanation in [_PRE_REJECTIONS[label]]
    ]
    filtered = [
        occurrence
        for occurrence in occurrences
        if occurrence.label not in _PRE_REJECTIONS
    ]
    if not filtered:
        print(
            "[predicate-induction] no labels remain after deterministic rejection",
            flush=True,
        )
        return PredicateInductionResult(rejected=pre_rejected)

    feedback = None
    last_error: ValueError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw_result = provider.induce(filtered, feedback=feedback)
            induced = validate_predicate_induction(filtered, raw_result)
            combined = PredicateInductionResult(
                predicates=induced.predicates,
                rejected=[*induced.rejected, *pre_rejected],
            )
            validated = validate_predicate_induction(occurrences, combined)
            return validated
        except ValueError as error:
            last_error = error
            feedback = str(error)
    raise ValueError(
        f"Predicate induction failed after {max_attempts} attempts: {last_error}"
    ) from last_error


def canonical_predicate_assignments(
    occurrences: Sequence[PredicateOccurrence], result: PredicateInductionResult
) -> dict[str, set[str]]:
    """Propagate canonical labels to every CQ contributing an accepted alias."""
    alias_map = result.alias_map
    original_assignments: dict[str, set[str]] = defaultdict(set)
    assignments: dict[str, set[str]] = defaultdict(set)
    for occurrence in occurrences:
        canonical = alias_map.get(occurrence.label)
        if canonical:
            original_assignments[occurrence.cq_id].add(occurrence.label)
            assignments[occurrence.cq_id].add(canonical)
    output = dict(assignments)
    validate_canonical_assignment_provenance(
        original_assignments,
        output,
        alias_map,
        stage="predicate induction",
    )
    return output
