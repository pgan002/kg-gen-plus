import asyncio
import json
import logging
from pathlib import Path

import aiohttp

from benchmarks.MuSiQue.musique_config import (
    musique_chunks_path,
    musique_onto_path,
    KG_GEN_URL,
    KG_GENERATION_PARAMS,
    X_API_KEY,
    i_start,
    i_end,
    output_base_path,
)
from benchmarks.MuSiQue.musique_utils import iter_musique_chunks_jsonl

output_stem = "musique_kg"


async def call_kg_generation_service(
    session: aiohttp.ClientSession,
    corpus_jsonl: str,
    ontology_content: bytes,
) -> dict:
    """Send the whole corpus to the kg-gen single parallel endpoint.

    kg-gen performs entity extraction, typing, relation extraction, aggregation
    and deduplication server-side (parallelised via ``n_parallel``) and returns
    a single KnowledgeGraph. External term extraction is no longer used.
    """
    logging.info("Calling KG generation service for the full corpus...")

    form_data = aiohttp.FormData()
    form_data.add_field(
        "ontology_file",
        ontology_content,
        filename="ontology.ttl",
        content_type="text/turtle",
    )
    form_data.add_field(
        "corpus_file",
        corpus_jsonl,
        filename="corpus.jsonl",
        content_type="application/x-ndjson",
    )

    headers = {"X-API-Key": X_API_KEY} if X_API_KEY else None

    async with session.post(
        KG_GEN_URL, params=KG_GENERATION_PARAMS, data=form_data, headers=headers
    ) as response:
        response.raise_for_status()
        graph = await response.json()
        logging.info("Successfully generated the knowledge graph.")
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


async def main(data_path: Path, ontology_path: Path, output_stem: str):
    """Main async function to process the MuSiQue corpus."""
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

    # No client-side timeout: server-side generation of a full corpus is slow.
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            final_graph = await call_kg_generation_service(
                session, corpus_jsonl, ontology_content
            )
        except aiohttp.ClientError as exc:
            logging.error(f"KG generation request failed: {exc}")
            return

    if not final_graph:
        logging.warning("No graph was returned.")
        return

    # results_dir = musique_chunks_path.parent / "results"
    # results_dir.mkdir(exist_ok=True)
    output_path = (
        output_base_path / f"{output_stem}__{ontology_path.stem}_final_graph.json"
    )
    with open(output_path, "w") as f:
        json.dump(final_graph, f, indent=2)
    logging.info(f"Final knowledge graph saved to {output_path}")


if __name__ == "__main__":
    if musique_onto_path.exists():
        logging.info(f"Starting processing MuSiQue with {musique_onto_path.name}")
        asyncio.run(
            main(
                data_path=musique_chunks_path,
                ontology_path=musique_onto_path,
                output_stem=output_stem,
            )
        )
    else:
        logging.error(f"Ontology file not found: {musique_onto_path}")
