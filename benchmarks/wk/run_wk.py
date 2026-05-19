import logging
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import aiohttp

from benchmarks.wk.wk_config import (
    wk_base_data_path,
    configure_logging,
    TERM_EXTRACTION_URL,
    KG_GEN_URL,
    AGGREGATE_URL,
    NUM_WORKERS,
)


# --- Async Service Client Functions ---


async def chunk_large_file(file_path: Path) -> List[str]:
    """
    Placeholder for a document chunking service for a large file.
    For now, it reads the file and returns its content as a single chunk.
    """
    logging.info(f"Reading and chunking file {file_path} (placeholder)...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # In the future, this will make a request to the chunking service.
        # e.g., async with session.post(CHUNK_SERVICE_URL, json={"text": content}) as response:
        # chunks = await response.json()
        # return chunks["chunks"]
        return [content]  # Returning the whole content as one chunk for now
    except FileNotFoundError:
        logging.error(f"Data file not found at {file_path}")
        return []


async def call_term_extraction_service(
    session: aiohttp.ClientSession, text: str
) -> List[str]:
    """Calls the term extraction service asynchronously."""
    logging.info("Calling term extraction service...")
    try:
        async with session.post(TERM_EXTRACTION_URL, json={"text": text}) as response:
            response.raise_for_status()
            data = await response.json()
            terms = data.get("terms", [])
            logging.info(f"Extracted {len(terms)} terms.")
            return terms
    except aiohttp.ClientError as e:
        logging.error(f"Error calling term extraction service: {e}")
        return []


async def call_kg_generation_service(
    session: aiohttp.ClientSession,
    text: str,
    item_id: str,
    terms: list,
    ontology_content: bytes,
) -> dict:
    """Calls the KG generation service asynchronously."""
    logging.info(f"Calling KG generation service for item {item_id}...")
    payload = {
        "id": item_id,
        "text": text,
        "terms": terms,
    }
    form_data = aiohttp.FormData()
    form_data.add_field(
        "input_data", json.dumps(payload), content_type="application/json"
    )
    form_data.add_field(
        "ontology_file",
        ontology_content,
        filename="ontology.ttl",
        content_type="application/x-turtle",
    )

    try:
        async with session.post(KG_GEN_URL, data=form_data) as response:
            response.raise_for_status()
            logging.info(f"Successfully generated graph for item {item_id}.")
            return await response.json()
    except aiohttp.ClientError as e:
        logging.error(f"Error calling KG generation service for item {item_id}: {e}")
        return {}


async def call_aggregate_and_deduplicate_service(
    session: aiohttp.ClientSession, graphs: List[dict]
) -> dict:
    """Calls the aggregation and deduplication service asynchronously."""
    logging.info(f"Calling aggregation service for {len(graphs)} graphs...")
    try:
        async with session.post(AGGREGATE_URL, json={"graphs": graphs}) as response:
            response.raise_for_status()
            logging.info("Successfully aggregated and deduplicated graphs.")
            return await response.json()
    except aiohttp.ClientError as e:
        logging.error(f"Error calling aggregation service: {e}")
        return {}


async def process_chunk_with_sem(
    sem: asyncio.Semaphore, session: aiohttp.ClientSession, chunk_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Worker coroutine that takes a chunk, extracts terms, and generates a graph,
    while being constrained by a semaphore.
    """
    async with sem:
        chunk_id = chunk_data["chunk_id"]
        chunk_text = chunk_data["chunk_text"]
        ontology_content = chunk_data["ontology_content"]

        logging.info(f"Worker processing chunk {chunk_id}...")

        terms = await call_term_extraction_service(session, chunk_text)
        if not terms:
            logging.warning(f"No terms extracted for chunk {chunk_id}.")
            return {}

        graph = await call_kg_generation_service(
            session=session,
            text=chunk_text,
            item_id=chunk_id,
            terms=terms,
            ontology_content=ontology_content,
        )
        return graph


async def main(data_path: Path, ontology_path: Path):
    """
    Main async function to process a data file by orchestrating calls to chunking,
    term extraction, and KG generation services.
    """
    try:
        with open(ontology_path, "rb") as f:
            ontology_content = f.read()
    except FileNotFoundError:
        logging.error(f"Ontology file not found at {ontology_path}")
        return

    document_chunks = await chunk_large_file(data_path)
    if not document_chunks:
        logging.error("No chunks were created from the data file.")
        return

    chunks_to_process = [
        {
            "chunk_id": f"doc_{data_path.stem}_chunk_{i}",
            "chunk_text": chunk_text,
            "ontology_content": ontology_content,
        }
        for i, chunk_text in enumerate(document_chunks)
    ]

    logging.info(f"Collected {len(chunks_to_process)} chunks to process.")

    sem = asyncio.Semaphore(NUM_WORKERS)
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_chunk_with_sem(sem, session, chunk) for chunk in chunks_to_process
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    graphs = [res for res in results if isinstance(res, dict) and res]
    exceptions = [res for res in results if isinstance(res, Exception)]

    logging.info(f"Successfully generated {len(graphs)} graphs.")
    if exceptions:
        logging.error(f"Encountered {len(exceptions)} errors during processing.")
        for exc in exceptions:
            logging.error(f"- {exc}")

    if graphs:
        final_graph = await call_aggregate_and_deduplicate_service(session, graphs)
        if final_graph:
            results_dir = wk_base_data_path / "results"
            results_dir.mkdir(exist_ok=True)
            output_path = (
                results_dir / f"{data_path.stem}__{ontology_path.stem}_final_graph.json"
            )
            with open(output_path, "w") as f:
                json.dump(final_graph, f, indent=2)
            logging.info(f"Final aggregated graph saved to {output_path}")
    else:
        logging.warning("No graphs were generated to aggregate.")


if __name__ == "__main__":
    # Define the paths for the single large data file and the ontology
    data_file = wk_base_data_path / "data.json"  # Assuming a single large data file
    ontology_file = (
        wk_base_data_path / "ontologies" / "wk_ontology.ttl"
    )  # Example ontology name

    if data_file.exists() and ontology_file.exists():
        # Configure logging
        results_dir = wk_base_data_path / "results"
        results_dir.mkdir(exist_ok=True)
        log_file_path = results_dir / f"{data_file.stem}__{ontology_file.stem}.log"
        configure_logging(log_file_path=str(log_file_path))

        logging.info(
            f"Starting processing for {data_file.name} with {ontology_file.name}"
        )
        asyncio.run(main(data_path=data_file, ontology_path=ontology_file))
    else:
        if not data_file.exists():
            logging.error(f"Data file not found: {data_file}")
        if not ontology_file.exists():
            logging.error(f"Ontology file not found: {ontology_file}")
