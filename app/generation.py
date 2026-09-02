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


def configure_dspy_concurrency(n_parallel: int) -> None:
    """Let ``n_parallel`` documents actually reach the LLM at once.

    Entity extraction runs through ``dspy.asyncify`` -- ``dspy.Refine`` has no
    ``aforward``, so there is no native async path -- and ``dspy.asyncify``
    acquires a process-global ``anyio.CapacityLimiter`` sized by
    ``dspy.settings.async_max_workers``, which defaults to **8**. That caps the
    first LLM step of every document at 8 concurrent no matter what
    ``n_parallel`` says, and the semaphore in ``kg_gen.generate`` cannot make up
    for it.

    Measured on the 16,131-document MuSiQue corpus: ``n_parallel=20`` and
    ``n_parallel=50`` both delivered ~1,150 docs/h, with the vLLM endpoint
    reporting 0 queued requests and single-digit KV-cache use -- starved, not
    saturated -- and the worker at 2% CPU. Raising ``n_parallel`` alone buys
    nothing.

    Raised, never lowered: DSPy resizes one global limiter, and another job may
    already be running in this process with a higher setting.
    """
    current = dspy.settings.get("async_max_workers") or 0
    if n_parallel > current:
        dspy.settings.configure(async_max_workers=n_parallel)
        kggen_logger.info(
            f"dspy async_max_workers raised {current} -> {n_parallel} so "
            f"n_parallel={n_parallel} is not throttled to {current} by asyncify"
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


# HTTP header values must be latin-1 encodable, and servers cap the header
# block (uvicorn allows 8 KiB by default). Stat headers are telemetry: a
# response must never fail because of them, hence the cap and the skip below.
MAX_STAT_HEADER_BYTES = 1024


def build_stats_headers(total_gen_stats) -> dict[str, str]:
    """Build the ``X-KG-Gen-*`` response headers from generation stats.

    Scalars only, by design. This used to also dump the whole ``KGGenStats``
    JSON into ``X-KG-Gen-Stats``, which grows without bound: it embeds one
    record per failed document, each carrying that document's full LM response.
    A 16,131-document run with 14 failures produced a 142 KB value containing an
    en dash, so setting the header raised ``UnicodeEncodeError`` and *every*
    fetch of that job's result returned 500 -- after the graph had already been
    generated. The detail is in the response body (``KnowledgeGraph.stats``), so
    keeping the headers small costs nothing.
    """
    overall = total_gen_stats.overall_usage
    headers = {
        "X-KG-Gen-Time": str(overall.execution_time),
        "X-KG-Gen-Input-Tokens": str(overall.lm_usage.prompt_tokens),
        "X-KG-Gen-Output-Tokens": str(overall.lm_usage.completion_tokens),
        "X-KG-Gen-Total-Tokens": str(overall.lm_usage.total_tokens),
        "X-KG-Gen-Failed-Documents": str(len(total_gen_stats.failed_documents)),
    }
    if total_gen_stats.deduplicate:
        headers["X-KG-Gen-Dedup-Stats"] = total_gen_stats.deduplicate.model_dump_json()
    return headers


def apply_stat_headers(response, headers: Optional[dict[str, str]]) -> None:
    """Copy stat headers onto a response, dropping any that cannot be sent.

    Assigning a non-latin-1 or oversized value to ``response.headers`` raises,
    and the exception escapes as a bare 500 that discards an otherwise valid
    response body. Telemetry is never worth that, so unsendable values are
    logged and skipped instead.
    """
    for key, value in (headers or {}).items():
        try:
            encoded = value.encode("latin-1")
        except UnicodeEncodeError:
            kggen_logger.warning(
                f"Skipping stat header {key}: value is not latin-1 encodable. "
                f"Full stats are in the response body."
            )
            continue
        if len(encoded) > MAX_STAT_HEADER_BYTES:
            kggen_logger.warning(
                f"Skipping stat header {key}: {len(encoded)} bytes exceeds the "
                f"{MAX_STAT_HEADER_BYTES}-byte cap. Full stats are in the "
                f"response body."
            )
            continue
        response.headers[key] = value


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
    configure_dspy_concurrency(generation_params.n_parallel)
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
        deduplicate_with_embeddings=generation_params.deduplicate_with_embeddings,
        progress_callback=progress_callback,
    )
    kg = final_graph.to_knowledge_graph(total_gen_stats, onto, rdflib_onto)
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
