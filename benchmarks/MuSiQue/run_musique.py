"""Submit MuSiQue KG generation as a background job (step 1 of 2).

This script builds the corpus, submits it to the async kg-gen endpoint, and
records the returned job id to a stable file (``job_record_path``). Because the
work runs server-side as a background job, this process can exit immediately;
run ``fetch_musique.py`` later to poll for progress and download the result.
"""

import json
import logging
import time
from pathlib import Path

import requests

from benchmarks.MuSiQue.musique_config import (
    musique_chunks_path,
    musique_onto_path,
    KG_GEN_URL,
    KG_GENERATION_PARAMS,
    X_API_KEY,
    i_start,
    i_end,
    output_base_path,
    job_record_path,
)
from benchmarks.MuSiQue.musique_utils import iter_musique_chunks_jsonl

output_stem = "musique_kg"

# The async endpoints live alongside KG_GEN_URL under the same `/api` root, e.g.
# ".../api/generate" -> ".../api/generate_async" and ".../api/jobs/{id}".
_API_ROOT = KG_GEN_URL.rsplit("/", 1)[0]
GENERATE_ASYNC_URL = f"{_API_ROOT}/generate_async"
JOBS_URL = f"{_API_ROOT}/jobs"

# How often fetch_musique.py polls the job status while generation runs.
POLL_INTERVAL_SECONDS = 15


def submit_kg_generation_job(
    corpus_jsonl: str,
    ontology_content: bytes,
) -> str:
    """Submit the corpus to the async kg-gen endpoint and return the job id.

    kg-gen performs entity extraction, typing, relation extraction, aggregation
    and deduplication server-side (parallelised via ``n_parallel``). Because the
    work runs as a background job, it keeps going even if this client
    disconnects, and the result can be fetched later.
    """
    logging.info("Submitting KG generation job for the full corpus...")

    files = {
        "ontology_file": ("ontology.ttl", ontology_content, "text/turtle"),
        "corpus_file": ("corpus.jsonl", corpus_jsonl, "application/x-ndjson"),
    }
    headers = {"X-API-Key": X_API_KEY} if X_API_KEY else None

    response = requests.post(
        GENERATE_ASYNC_URL,
        params=KG_GENERATION_PARAMS,
        files=files,
        headers=headers,
        timeout=(30, 300),
    )
    response.raise_for_status()
    payload = response.json()
    job_id = payload["job_id"]
    logging.info(f"Job submitted: id={job_id} total_docs={payload.get('total_docs')}")
    return job_id


def wait_for_job(job_id: str) -> None:
    """Poll the job status until it completes, logging tqdm-like progress."""
    status_url = f"{JOBS_URL}/{job_id}"
    while True:
        response = requests.get(status_url, timeout=(30, 60))
        response.raise_for_status()
        status = response.json()

        state = status["status"]
        pct = status.get("percent_complete")
        eta = status.get("eta_seconds")
        logging.info(
            f"Job {job_id}: {state} "
            f"{status.get('processed_docs')}/{status.get('total_docs')} docs "
            f"({pct if pct is not None else '?'}%) "
            f"elapsed={status.get('elapsed_seconds')}s "
            f"ETA={eta if eta is not None else '?'}s"
        )

        if state == "completed":
            return
        if state == "failed":
            raise RuntimeError(f"KG generation job failed: {status.get('error')}")

        time.sleep(POLL_INTERVAL_SECONDS)


def fetch_job_result(job_id: str) -> dict:
    """Fetch the generated KnowledgeGraph for a completed job."""
    response = requests.get(f"{JOBS_URL}/{job_id}/result", timeout=(30, 300))
    response.raise_for_status()
    graph = response.json()
    logging.info("Successfully fetched the generated knowledge graph.")
    return graph


def build_corpus_jsonl(data_path: Path) -> str:
    """Build a JSONL corpus (one ``{"id", "text"}`` per line) from the chunks.

    Terms are intentionally omitted: kg-gen extracts and types entities itself.
    """
    lines = []
    for i, chunk in enumerate(iter_musique_chunks_jsonl(data_path), start=1):
        if i < i_start or i > i_end:
            continue
        lines.append(json.dumps({"id": chunk.chunk_id, "text": chunk.content}))
    return "\n".join(lines)


def record_job(job_id: str, output_path: Path, num_docs: int) -> None:
    """Persist the job id (and where its result should be saved) to disk.

    fetch_musique.py reads this file to know which job to poll and where to
    write the downloaded graph.
    """
    record = {
        "job_id": job_id,
        "num_docs": num_docs,
        "output_path": str(output_path),
        "jobs_url": JOBS_URL,
    }
    with open(job_record_path, "w") as f:
        json.dump(record, f, indent=2)
    logging.info(f"Recorded job {job_id} to {job_record_path}")


def main(data_path: Path, ontology_path: Path, output_stem: str):
    """Submit the MuSiQue corpus as a background job and record its id."""
    try:
        ontology_content = ontology_path.read_bytes()
    except FileNotFoundError:
        logging.error(f"Ontology file not found at {ontology_path}")
        return

    corpus_jsonl = build_corpus_jsonl(data_path)
    num_docs = corpus_jsonl.count("\n") + 1 if corpus_jsonl else 0
    if not num_docs:
        logging.warning("No chunks to process for the configured slice.")
        return
    logging.info(f"Collected {num_docs} chunks to process.")

    try:
        job_id = submit_kg_generation_job(corpus_jsonl, ontology_content)
    except requests.RequestException as exc:
        logging.error(f"KG generation request failed: {exc}")
        return

    output_path = (
        output_base_path / f"{output_stem}__{ontology_path.stem}_final_graph.json"
    )
    record_job(job_id, output_path, num_docs)
    logging.info(
        "Submission complete. Run fetch_musique.py to poll and download the result."
    )


if __name__ == "__main__":
    if musique_onto_path.exists():
        logging.info(f"Starting processing MuSiQue with {musique_onto_path.name}")
        main(
            data_path=musique_chunks_path,
            ontology_path=musique_onto_path,
            output_stem=output_stem,
        )
    else:
        logging.error(f"Ontology file not found: {musique_onto_path}")
