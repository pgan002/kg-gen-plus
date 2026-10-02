"""Contextual filtering of plausible but unnecessary ontology class candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, Field


class ClassCandidateDecision(BaseModel):
    term: str
    keep: bool
    reason: Literal[
        "answer_or_variable_type",
        "restriction_participant",
        "central_domain_concept",
        "incidental_surface_concept",
        "attribute_not_class",
        "named_individual_or_example",
        "unsupported",
    ]
    explanation: str = ""


class ClassCandidateDecisions(BaseModel):
    decisions: list[ClassCandidateDecision] = Field(default_factory=list)


class ClassCandidateDecisionProvider(Protocol):
    def decide(
        self,
        questions: Mapping[str, str],
        candidates: Mapping[str, set[str]],
        relations_by_cq: Mapping[str, Sequence[str]],
    ) -> list[ClassCandidateDecision]: ...


class OpenAIClassCandidateDecisionProvider:
    """Obtain constrained candidate decisions from an OpenAI-compatible LLM."""

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
        questions: Mapping[str, str],
        candidates: Mapping[str, set[str]],
        relations_by_cq: Mapping[str, Sequence[str]],
    ) -> str:
        evidence = []
        for term in sorted(candidates):
            cq_ids = sorted(candidates[term])
            evidence.append(
                {
                    "term": term,
                    "supporting_cqs": [
                        {
                            "id": cq_id,
                            "question": questions[cq_id],
                            "extracted_relations": list(relations_by_cq.get(cq_id, [])),
                        }
                        for cq_id in cq_ids
                    ],
                }
            )
        return (
            "Decide whether each supplied candidate is required as an ontology "
            "CLASS to represent or answer at least one supporting competency "
            "question. Evaluate only the supplied terms; do not invent, rename, "
            "merge, or omit terms. Keep answer/variable types, restriction "
            "participants, and central domain concepts. Remove incidental wording, "
            "named examples/individuals, unsupported concepts, and attributes that "
            "should be represented as properties or values. A concept may be "
            "plausible in the domain but should still be removed if it is not "
            "required by its supporting CQs. Return JSON only with shape "
            "{\"decisions\":[{\"term\":str,\"keep\":bool,\"reason\":one of "
            "[\"answer_or_variable_type\",\"restriction_participant\","
            "\"central_domain_concept\",\"incidental_surface_concept\","
            "\"attribute_not_class\",\"named_individual_or_example\","
            "\"unsupported\"],\"explanation\":str}]}.\n\n"
            f"Candidate evidence:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
        )

    def decide(
        self,
        questions: Mapping[str, str],
        candidates: Mapping[str, set[str]],
        relations_by_cq: Mapping[str, Sequence[str]],
    ) -> list[ClassCandidateDecision]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conservative ontology engineer performing a "
                        "closed-vocabulary class-candidate review."
                    ),
                },
                {
                    "role": "user",
                    "content": self._prompt(questions, candidates, relations_by_cq),
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
            reasoning_detail = (
                f", reasoning_content_length={len(reasoning)}"
                if isinstance(reasoning, str)
                else ""
            )
            raise ValueError(
                "Class candidate filter returned an empty final response "
                f"(finish_reason={choice.finish_reason!r}{reasoning_detail}). "
                "Thinking is disabled by default; if it was explicitly enabled, "
                "disable it or increase --class-filter-max-tokens."
            )
        return ClassCandidateDecisions.model_validate_json(content).decisions


def filter_class_candidates(
    questions: Sequence[dict[str, Any]],
    assignments: Mapping[str, set[str]],
    relations_by_cq: Mapping[str, Sequence[str]],
    provider: ClassCandidateDecisionProvider,
) -> tuple[dict[str, set[str]], list[ClassCandidateDecision]]:
    """Filter canonical classes after validating closed-vocabulary decisions."""
    question_map = {}
    for index, question in enumerate(questions):
        if not isinstance(question, dict) or not {"id", "value"} <= question.keys():
            raise ValueError(f"Invalid question at index {index}: {question!r}")
        question_map[str(question["id"])] = str(question["value"])

    candidate_sources: dict[str, set[str]] = {}
    for cq_id, terms in assignments.items():
        if cq_id not in question_map:
            raise ValueError(f"Assignment references unknown CQ: {cq_id!r}")
        for term in terms:
            candidate_sources.setdefault(term, set()).add(cq_id)
    if not candidate_sources:
        return {cq_id: set(terms) for cq_id, terms in assignments.items()}, []

    decisions = provider.decide(question_map, candidate_sources, relations_by_cq)
    expected = set(candidate_sources)
    returned = [decision.term for decision in decisions]
    duplicates = sorted({term for term in returned if returned.count(term) > 1})
    missing = sorted(expected - set(returned))
    unexpected = sorted(set(returned) - expected)
    if duplicates or missing or unexpected:
        raise ValueError(
            "Invalid class candidate decisions: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )
    keep = {decision.term for decision in decisions if decision.keep}
    filtered = {
        cq_id: {term for term in terms if term in keep}
        for cq_id, terms in assignments.items()
    }
    return filtered, decisions
