from __future__ import annotations

import asyncio
import time

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

from app import blobs, settings
from app.apis.deps import get_kg_gen
from app.generation import (
    apply_stat_headers,
    execute_generation,
    kg_gen_from_params,
    prepare_generation_inputs,
)
from app.job_queue import (
    JobFailed,
    JobNotFinished,
    JobNotFound,
    get_queue,
)
from app.jobs import JobStatus, job_store
from app.kggen_logger import kggen_logger
from app.schemas import (
    OntologyConversionInput,
    GenerationMetadata,
    DeduplicationMetadata,
)
from app.utils import parse_ontology, serialize_ontology_to_ttl
from kg_gen.models import (
    Ontology,
    Graph,
    KnowledgeGraph,
    OntologyExtensions,
)


kgc_router = APIRouter(prefix="/api")


@kgc_router.post("/blobs", tags=["KG Construction"], name="upload_blob")
async def upload_blob(
    request: Request,
    file: Optional[UploadFile] = File(
        None, description="The data to store, as a multipart upload."
    ),
) -> dict:
    """Store data and return a short handle for it.

    Exists so an agent can hand a large list to a tool *by reference*. Tool
    arguments are written by the calling model, so an inline 702-entity list
    costs it ~32,000 output tokens; uploading here costs an HTTP request the
    model never has to type, and the handle that comes back is ~20 tokens. The
    tools that accept `typed_entities`/`relations`/cluster lists, and
    `ontology_ttl`, all take a `blob:<id>` handle in place of the data.

    Unlike the file-path form those tools also accept, this works when the agent
    and the server share no filesystem, which is the normal case for a remote
    MCP server.

        curl -sF file=@entities.json http://HOST/api/blobs
        curl -s --data-binary @entities.json http://HOST/api/blobs

    Returns `{"blob": "blob:<id>", "bytes": <n>}`. The handle is content-
    addressed, so uploading the same bytes twice yields the same handle.
    """
    data = await file.read() if file is not None else await request.body()
    if not data:
        raise HTTPException(
            status_code=400,
            detail="No data to store. Send a multipart 'file' field or a raw body.",
        )
    if len(data) > settings.MAX_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Blob of {len(data)} bytes exceeds the "
            f"{settings.MAX_PAYLOAD_BYTES}-byte limit.",
        )
    handle = await blobs.store(data)
    return {"blob": handle, "bytes": len(data)}


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

    inputs, onto, rdflib_onto, types, predicates = prepare_generation_inputs(
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

    kg_gen = kg_gen_from_params(generation_params, x_api_key)
    kggen_logger.info(f"Generating graph via KGGen: {generation_params.model = }")

    try:
        kg, headers = await execute_generation(
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

    apply_stat_headers(response, headers)
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
    returns a 4xx immediately). Returns a `job_id`; poll
    `GET /api/jobs/{job_id}` for progress and fetch the graph from
    `GET /api/jobs/{job_id}/result` once the status is `completed`.

    Where the work runs depends on deployment:

    * `KGGEN_REDIS_URL` set -- the job is published to a durable Redis stream and
      executed by a separate worker process. It survives an API restart, and a
      worker crash mid-job causes the job to be redelivered and re-run.
    * unset -- the job runs on this process's event loop (the original
      behaviour). It continues if the client disconnects, but is lost on restart.
    """
    if corpus_file.filename is not None and not corpus_file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Corpus must be a .jsonl file")

    # Read the uploads now: the file handles are tied to this request and are
    # closed once it returns, so neither a background task nor a worker could
    # read them later.
    corpus_bytes = await corpus_file.read()
    ontology_bytes = await ontology_file.read() if ontology_file else None

    if len(corpus_bytes) > settings.MAX_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Corpus is {len(corpus_bytes)} bytes, over the "
                f"{settings.MAX_PAYLOAD_BYTES} byte limit "
                "(KGGEN_MAX_PAYLOAD_BYTES)"
            ),
        )

    # Validate here even on the queued path: the client gets its 4xx
    # synchronously, and the worker re-parses the raw bytes rather than having
    # rdflib graphs and Pydantic models serialised across the process boundary.
    inputs, onto, rdflib_onto, types, predicates = prepare_generation_inputs(
        corpus_bytes, ontology_bytes
    )
    if not inputs:
        raise HTTPException(status_code=400, detail="Corpus contains no documents")

    queue = get_queue()
    if queue is not None:
        job_id = await queue.enqueue(
            total_docs=len(inputs),
            corpus_bytes=corpus_bytes,
            ontology_bytes=ontology_bytes,
            generation_params=generation_params.model_dump(mode="json"),
            api_key=x_api_key,
        )
        return {
            "job_id": job_id,
            "status": JobStatus.pending.value,
            "total_docs": len(inputs),
            "status_url": str(request.url_for("get_job_status", job_id=job_id)),
            "result_url": str(request.url_for("get_job_result", job_id=job_id)),
        }

    kg_gen = kg_gen_from_params(generation_params, x_api_key)

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
            kg, headers = await execute_generation(
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

    With Redis configured, jobs are listed from the shared store and survive
    restarts (each expiring after `KGGEN_RESULT_TTL_SECONDS`). Without it, only
    jobs still held in this process's in-memory store are returned: finished jobs
    may be evicted once the store is full, and all jobs are lost on a restart.
    """
    queue = get_queue()
    if queue is not None:
        return await queue.list_statuses()
    return [job.to_status_dict() for job in job_store.list()]


@kgc_router.get("/jobs/{job_id}", tags=["KG Construction"], name="get_job_status")
async def get_job_status(job_id: str) -> dict:
    """Return the status and tqdm-like progress of a background generation job."""
    queue = get_queue()
    if queue is not None:
        try:
            return await queue.get_status(job_id)
        except JobNotFound:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")

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
    queue = get_queue()
    if queue is not None:
        try:
            graph, headers = await queue.get_result(job_id)
        except JobNotFound:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        except JobFailed as exc:
            raise HTTPException(
                status_code=exc.status_code, detail=f"Job failed: {exc.error}"
            )
        except JobNotFinished as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        apply_stat_headers(response, headers)
        return graph

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
    apply_stat_headers(response, job.headers)
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
            use_embeddings=meta.deduplicate_with_embeddings,
        )
        kggen_logger.info(
            "Deduplication complete: entities=%s relations=%s",
            len(deduplicated_graph.entities),
            len(deduplicated_graph.relations),
        )
        apply_stat_headers(
            response, {"X-KG-Gen-Dedup-Stats": dedup_stats.model_dump_json()}
        )
        return deduplicated_graph
    else:
        return aggregated_graph
