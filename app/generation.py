"""Generation helpers shared by the HTTP handlers and the background worker.

These were originally private helpers inside ``app.apis.kg_construct``. They live
here so ``app.worker`` can run a job without importing the HTTP router module --
the worker serves no HTTP and should not depend on the API layer.
"""

from __future__ import annotations

import io
from typing import Optional

import dspy
from fastapi import HTTPException
from pydantic import ValidationError
from rdflib import Graph as RDFGraph
from rdflib.exceptions import ParserError

from app.apis.deps import get_kg_gen
from app.kggen_logger import kggen_logger
from app.schemas import GenerationMetadata
from app.utils import parse_ontology
from kg_gen.kg_gen import KGGen
from kg_gen.models import InputData, KnowledgeGraph, Ontology


def configure_dspy_cache() -> None:
    """Turn DSPy's LM caches off for any process that runs generation.

    Lives here, next to the code that drives the LM, because *every* such process
    needs it and they must not drift apart. The API used to set this at the bottom
    of ``app.server``; the worker does not import that module, so it would
    otherwise silently inherit DSPy's defaults -- a **30 GB** on-disk cache
    (``DSPY_CACHE_LIMIT``) of prompts and completions, i.e. document text at rest,
    plus a 1,000,000-entry in-memory cache. The worker is where all LM calls
    happen now, so that is exactly the wrong process to leave on defaults.
    """
    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
    )


def prepare_generation_inputs(
    corpus_bytes: bytes,
    ontology_bytes: Optional[bytes],
) -> tuple[
    list[InputData],
    Optional[Ontology],
    Optional[RDFGraph],
    Optional[list],
    Optional[list],
]:
    """Parse the ontology and corpus into the objects ``kg_gen.generate`` needs.

    Raises ``HTTPException`` on malformed input so callers can surface a 4xx to
    the client synchronously, before any (potentially long-running) generation.
    """
    onto: Optional[Ontology] = None
    rdflib_onto: Optional[RDFGraph] = None
    types = None
    predicates = None
    if ontology_bytes:
        try:
            onto, rdflib_onto = parse_ontology(io.BytesIO(ontology_bytes))
            types = list(onto.classes)
            predicates = list(onto.predicates)
        except (ParserError, SyntaxError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not parse the provided ontology: {exc}"
            )

    if onto:
        kggen_logger.info(
            f"With ontology: {len(onto.classes) = }, {len(onto.predicates) = }"
        )
    else:
        kggen_logger.info("No ontology provided")

    inputs: list[InputData] = []
    for line in corpus_bytes.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            inputs.append(InputData.model_validate_json(line))
        except ValidationError as e:
            raise HTTPException(
                status_code=422, detail=f"Invalid document format: {e.errors()}"
            )
    kggen_logger.info(f"Loaded {len(inputs)} document(s) from corpus")
    return inputs, onto, rdflib_onto, types, predicates


def build_stats_headers(total_gen_stats) -> dict[str, str]:
    """Build the ``X-KG-Gen-*`` response headers from generation stats."""
    overall = total_gen_stats.overall_usage
    headers = {
        "X-KG-Gen-Stats": total_gen_stats.model_dump_json(),
        "X-KG-Gen-Time": str(overall.execution_time),
        "X-KG-Gen-Input-Tokens": str(overall.lm_usage.prompt_tokens),
        "X-KG-Gen-Output-Tokens": str(overall.lm_usage.completion_tokens),
    }
    if total_gen_stats.deduplicate:
        headers["X-KG-Gen-Dedup-Stats"] = total_gen_stats.deduplicate.model_dump_json()
    return headers


async def execute_generation(
    kg_gen: KGGen,
    inputs: list[InputData],
    onto: Optional[Ontology],
    rdflib_onto: Optional[RDFGraph],
    types: Optional[list],
    predicates: Optional[list],
    generation_params: GenerationMetadata,
    progress_callback=None,
) -> tuple[KnowledgeGraph, dict[str, str]]:
    """Run generation and return the KnowledgeGraph plus stat headers."""
    final_graph, total_gen_stats = await kg_gen.generate(
        input_data=inputs,
        n_parallel=generation_params.n_parallel,
        ontology=rdflib_onto,
        types=types,
        predicate_domain_range=predicates,
        entity_context=generation_params.entity_context or "",
        relation_context=generation_params.relation_context or "",
        deduplicate=generation_params.deduplicate,
        temperature=generation_params.temperature,
        enforce_type_conformance=generation_params.enforce_type_conformance,
        enforce_domain_conformance=generation_params.enforce_domain_conformance,
        enforce_range_conformance=generation_params.enforce_range_conformance,
        enforce_predicate_conformance=generation_params.enforce_predicate_conformance,
        entity_similarity_threshold=generation_params.entity_threshold,
        edge_similarity_threshold=generation_params.predicate_threshold,
        progress_callback=progress_callback,
    )
    kg = final_graph.to_knowledge_graph(total_gen_stats, onto)
    return kg, build_stats_headers(total_gen_stats)


def kg_gen_from_params(
    generation_params: GenerationMetadata, x_api_key: Optional[str]
) -> KGGen:
    return get_kg_gen(
        api_key=x_api_key,
        api_base=generation_params.api_base,
        model=generation_params.model,
        max_tokens=generation_params.max_tokens,
        temperature=generation_params.temperature
        if generation_params.temperature is not None
        else 0.0,
        retrieval_model=generation_params.retrieval_model,
        enforce_type_conformance=generation_params.enforce_type_conformance,
        enforce_domain_conformance=generation_params.enforce_domain_conformance,
        enforce_range_conformance=generation_params.enforce_range_conformance,
        enforce_predicate_conformance=generation_params.enforce_predicate_conformance,
    )
