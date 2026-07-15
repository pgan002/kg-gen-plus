from __future__ import annotations

import asyncio
import io
import time

from rdflib.exceptions import ParserError
from typing import Optional, Annotated

from fastapi import (
    UploadFile,
    File,
    Query,
    HTTPException,
    APIRouter,
    Response,
    Header,
    Request,
)
from pydantic import ValidationError
from rdflib import Graph as RDFGraph

from app.apis.deps import get_kg_gen
from app.jobs import JobStatus, job_store
from app.kggen_logger import kggen_logger
from app.schemas import (
    OntologyConversionInput,
    GenerationMetadata,
    DeduplicationMetadata,
)
from app.utils import parse_ontology, serialize_ontology_to_ttl
from kg_gen.kg_gen import KGGen
from kg_gen.models import (
    Ontology,
    Graph,
    InputData,
    KnowledgeGraph,
    OntologyExtensions,
)


kgc_router = APIRouter(prefix="/api")


@kgc_router.post("/convert_ontology", tags=["KG Construction"])
async def convert_ontology(input_data: OntologyConversionInput) -> Response:
    """
    Converts a structured ontology definition into a Turtle (.ttl) RDF file.

    This endpoint takes a list of entity types (classes) and predicates (properties),
    including their domains and ranges, and generates a standards-compliant RDF
    ontology that can be used for Knowledge Graph generation.
    """
    ttl_content = serialize_ontology_to_ttl(input_data.classes, input_data.predicates)

    return Response(content=ttl_content, media_type="application/x-turtle")


@kgc_router.post("/parse_ontology", tags=["KG Construction"])
async def parse_ontology_api(
    ontology_file: UploadFile = File(
        ...,
        description=parse_ontology.__doc__,
    ),
) -> Ontology:
    """
    Endpoint to parse an ontology file and return the parsed ontology as an object.

    Parses the provided ontology file using the `parse_ontology` function and
    returns the resulting `Ontology` object.

    Args:
        ontology_file: The ontology file to be parsed. Must be uploaded as an
            `UploadFile`. The description of this parameter is derived from
            the documentation of the `parse_ontology` function.

    Returns:
        Ontology: The parsed ontology object.
    """
    onto, g = parse_ontology(ontology_file.file)
    return onto


def _prepare_generation_inputs(
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


def _build_stats_headers(total_gen_stats) -> dict[str, str]:
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


async def _execute_generation(
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
    return kg, _build_stats_headers(total_gen_stats)


def _kg_gen_from_params(
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


@kgc_router.post("/generate", tags=["KG Construction"])
async def generate_graph(
    response: Response,
    generation_params: Annotated[GenerationMetadata, Query()],
    x_api_key: Optional[str] = Header(
        None, description="API key for the LLM provider."
    ),
    ontology_file: Optional[UploadFile] = File(
        None,
        description="Optional ontology file in Turtle (.ttl) format to guide extraction and conformance.",
    ),
    corpus_file: UploadFile = File(
        ...,
        description="A JSONL file where each line is a JSON object representing a document. "
        "Each object must have `id`, `text`, and optional `terms` fields. "
        "The `terms` should be instances of TypedEntity.",
    ),
) -> KnowledgeGraph:
    """
    Generates a Knowledge Graph from a multi-line text corpus.

    This endpoint processes each line of the input .jsonl file in parallel.
    It performs entity extraction, typing, and relation extraction based on the
    provided ontology.

    The request blocks until generation completes. For large corpora prefer
    `/generate_async`, which returns immediately with a job id and keeps
    processing even if the client disconnects.

    Note: Conformance to the ontology (types, domains, ranges) is best-effort and
    not strictly guaranteed.
    """
    if corpus_file.filename is not None and not corpus_file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Corpus must be a .jsonl file")

    corpus_bytes = await corpus_file.read()
    ontology_bytes = await ontology_file.read() if ontology_file else None

    inputs, onto, rdflib_onto, types, predicates = _prepare_generation_inputs(
        corpus_bytes, ontology_bytes
    )
    kggen_logger.info(
        f"{generation_params.entity_context = }\n{generation_params.relation_context = }"
    )

    if not inputs:
        return KnowledgeGraph(
            entities={},
            relations=[],
            ontology_extensions=OntologyExtensions(),
            stats={},
        )

    kg_gen = _kg_gen_from_params(generation_params, x_api_key)
    kggen_logger.info(f"Generating graph via KGGen: {generation_params.model = }")

    try:
        kg, headers = await _execute_generation(
            kg_gen, inputs, onto, rdflib_onto, types, predicates, generation_params
        )
    except ValidationError as exc:
        kggen_logger.exception("KGGen returned validation error")
        raise HTTPException(status_code=400, detail=f"Invalid graph result: {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        kggen_logger.exception("KGGen generation failed")
        raise HTTPException(status_code=500, detail=f"KGGen failed: {exc}")

    for key, value in headers.items():
        response.headers[key] = value
    return kg


@kgc_router.post("/generate_async", tags=["KG Construction"], status_code=202)
async def generate_graph_async(
    request: Request,
    generation_params: Annotated[GenerationMetadata, Query()],
    x_api_key: Optional[str] = Header(
        None, description="API key for the LLM provider."
    ),
    ontology_file: Optional[UploadFile] = File(
        None,
        description="Optional ontology file in Turtle (.ttl) format to guide extraction and conformance.",
    ),
    corpus_file: UploadFile = File(
        ...,
        description="A JSONL file where each line is a JSON object representing a document.",
    ),
) -> dict:
    """
    Start Knowledge Graph generation as a background job.

    The uploaded files are validated and read up front (so malformed input still
    returns a 4xx immediately), then generation runs on the server event loop and
    continues even if the client disconnects. Returns a `job_id`; poll
    `GET /api/jobs/{job_id}` for progress and fetch the graph from
    `GET /api/jobs/{job_id}/result` once the status is `completed`.
    """
    if corpus_file.filename is not None and not corpus_file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Corpus must be a .jsonl file")

    # Read the uploads now: the file handles are tied to this request and are
    # closed once it returns, so the background task cannot read them later.
    corpus_bytes = await corpus_file.read()
    ontology_bytes = await ontology_file.read() if ontology_file else None

    inputs, onto, rdflib_onto, types, predicates = _prepare_generation_inputs(
        corpus_bytes, ontology_bytes
    )
    if not inputs:
        raise HTTPException(status_code=400, detail="Corpus contains no documents")

    kg_gen = _kg_gen_from_params(generation_params, x_api_key)

    job = job_store.create()
    job.total_docs = len(inputs)

    async def _run() -> None:
        job.status = JobStatus.running
        job.started_at = time.time()
        kggen_logger.info(
            f"[job {job.id}] generating graph for {job.total_docs} docs "
            f"via KGGen: {generation_params.model = }"
        )
        try:
            kg, headers = await _execute_generation(
                kg_gen,
                inputs,
                onto,
                rdflib_onto,
                types,
                predicates,
                generation_params,
                progress_callback=job.update_progress,
            )
            job.result = kg
            job.headers = headers
            job.status = JobStatus.completed
            job.processed_docs = job.total_docs
            kggen_logger.info(f"[job {job.id}] completed")
        except Exception as exc:
            kggen_logger.exception(f"[job {job.id}] generation failed")
            job.error = str(exc)
            job.status_code = exc.status_code if isinstance(exc, HTTPException) else 500
            job.status = JobStatus.failed
        finally:
            job.finished_at = time.time()

    # Keep a reference so the task is not garbage-collected mid-flight.
    job.task = asyncio.create_task(_run())

    return {
        "job_id": job.id,
        "status": job.status.value,
        "total_docs": job.total_docs,
        "status_url": str(request.url_for("get_job_status", job_id=job.id)),
        "result_url": str(request.url_for("get_job_result", job_id=job.id)),
    }


@kgc_router.get("/jobs", tags=["KG Construction"], name="list_jobs")
async def list_jobs() -> list[dict]:
    """List all tracked background generation jobs, newest first.

    Only jobs still held in the in-memory store are returned (see the note on
    `/generate_async`): finished jobs may be evicted once the store is full, and
    all jobs are lost on a server restart.
    """
    return [job.to_status_dict() for job in job_store.list()]


@kgc_router.get("/jobs/{job_id}", tags=["KG Construction"], name="get_job_status")
async def get_job_status(job_id: str) -> dict:
    """Return the status and tqdm-like progress of a background generation job."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return job.to_status_dict()


@kgc_router.get(
    "/jobs/{job_id}/result", tags=["KG Construction"], name="get_job_result"
)
async def get_job_result(job_id: str, response: Response) -> KnowledgeGraph:
    """Fetch the generated Knowledge Graph for a completed job.

    Returns 404 for an unknown job, 409 while it is still pending/running, and
    re-raises the original error status if the job failed.
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    if job.status is JobStatus.failed:
        raise HTTPException(
            status_code=job.status_code, detail=f"Job failed: {job.error}"
        )
    if job.status is not JobStatus.completed:
        raise HTTPException(
            status_code=409,
            detail=f"Job not finished (status: {job.status.value})",
        )
    for key, value in (job.headers or {}).items():
        response.headers[key] = value
    return job.result


@kgc_router.post("/aggregate_and_deduplicate", tags=["DEPRECATED"])
async def aggregate_and_deduplicate_graphs(
    response: Response,
    graphs: list[Graph],
    meta: Annotated[DeduplicationMetadata, Query()],
) -> Graph:
    """
    Aggregates a list of graphs and performs deduplication.
    """
    if not graphs:
        raise HTTPException(
            status_code=400, detail="No graphs provided for aggregation."
        )

    kg_gen = get_kg_gen(retrieval_model=meta.retrieval_model)
    kggen_logger.info(f"Aggregating {len(graphs)} graphs.")
    aggregated_graph = kg_gen.aggregate(graphs)
    kggen_logger.info(
        "Aggregation complete: entities=%s relations=%s",
        len(aggregated_graph.entities),
        len(aggregated_graph.relations),
    )

    if meta.deduplicate:
        kggen_logger.info("Performing deduplication.")
        deduplicated_graph, dedup_stats = kg_gen.deduplicate(
            aggregated_graph,
            entity_similarity_threshold=meta.entity_threshold,
            edge_similarity_threshold=meta.predicate_threshold,
        )
        kggen_logger.info(
            "Deduplication complete: entities=%s relations=%s",
            len(deduplicated_graph.entities),
            len(deduplicated_graph.relations),
        )
        response.headers["X-KG-Gen-Dedup-Stats"] = dedup_stats.model_dump_json()
        return deduplicated_graph
    else:
        return aggregated_graph
