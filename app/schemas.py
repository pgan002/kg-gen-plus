from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from kg_gen.models import EntityType, OntologyPredicate, TypedEntity


class HeartBeatResponse(BaseModel):
    is_alive: bool = True


class OntologyConversionInput(BaseModel):
    classes: list[EntityType]
    predicates: list[OntologyPredicate]


class GenerateSingleInput(BaseModel):
    id: str
    text: str
    terms: list[TypedEntity]


class GenerationMetadata(BaseModel):
    model: str = Field(
        "openai/gpt-5.4-mini",
        description="The model to use for generation, e.g. 'openai/gpt-4-turbo'",
    )
    api_base: Optional[str] = Field(
        None,
        description="The base URL for the LLM API. Use this for non-OpenAI models or custom endpoints.",
    )
    temperature: float | None = Field(
        None,
        description="The temperature to use for generation. Higher values mean more creative but less predictable results.",
    )
    max_tokens: int = Field(
        16000,
        gt=0,
        description="Maximum number of tokens the model may generate per call. "
        "Increase if responses are being truncated (see the LM truncation warning).",
    )
    entity_context: Optional[str] = Field(
        None,
        description="Use this field to provide additional instruction for processing entities, "
        "for example, describe which classes should be used if no ontology is provided.",
    )
    relation_context: Optional[str] = Field(
        None,
        description="Use this field to provide additional instruction for extracting relation, "
        "for example, describe which predicates should be used if no ontology is provided.",
    )
    enable_thinking: Optional[bool] = Field(
        False, description="Allows to control thinking in models like qwen"
    )
    enforce_type_conformance: bool = Field(
        False,
        description="Encourage entity type conformance. Note: best-effort, not strictly enforced.",
    )
    enforce_domain_conformance: bool = Field(
        True,
        description="Encourage predicate domain conformance. Note: best-effort, not strictly enforced.",
    )
    enforce_range_conformance: bool = Field(
        True,
        description="Encourage predicate range conformance. Note: best-effort, not strictly enforced.",
    )
    enforce_predicate_conformance: bool = Field(
        False,
        description="Encourage predicate conformance. Note: best-effort, not strictly enforced.",
    )
    deduplicate: bool = Field(
        True, description="Whether to deduplicate the generated graph."
    )
    retrieval_model: Optional[str] = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="The retrieval model to use for producing embedding.",
    )
    entity_threshold: float = Field(
        0.8,
        ge=0,
        le=1,
        description="Entity similarity threshold for deduplication.",
    )
    predicate_threshold: float = Field(
        0.9,
        ge=0,
        le=1,
        description="Predicate similarity threshold for deduplication.",
    )
    n_parallel: int = Field(
        10, description="The number of parallel calls to the LLM for graph generation."
    )


class DeduplicationMetadata(BaseModel):
    deduplicate: bool = Field(
        True, description="Whether to deduplicate the generated graph."
    )
    retrieval_model: Optional[str] = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="The retrieval model to use for producing embedding.",
    )
    entity_threshold: float = Field(
        0.9,
        ge=0,
        le=1,
        description="Entity similarity threshold for deduplication.",
    )
    predicate_threshold: float = Field(
        0.75,
        ge=0,
        le=1,
        description="Predicate similarity threshold for deduplication.",
    )
