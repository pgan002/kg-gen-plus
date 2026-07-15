import logging
import json
import asyncio
from pathlib import Path

import rdflib
from typing import Any
import tqdm
import aiohttp
from pydantic import ValidationError

from benchmarks.wk.schemas import ParseResult, Term
from benchmarks.wk.wk_config import (
    wk_base_data_path,
    KG_GEN_URL,
    AGGREGATE_URL,
    NUM_WORKERS,
    CHUNK_SERVICE_URL,
    TERM_EXTRACTION_URL,
    configure_logging,
    output_stem,
    ontology_file,
    CHUNKING_HEADERS,
    TERM_EXTRACTION_PARAMS,
    TERM_EXTRACTION_HEADERS,
    KG_GENERATION_PARAMS,
    chunked_path,
)
from benchmarks.wk.wk_utils import iter_wk_chunked_jsonl
from kg_gen.models import TypedEntity


# --- Helper Functions ---


# def extract_content_from_sections(sections: List[Section]) -> List[str]:
#     """
#     Recursively extracts content from a list of sections and their children.
#     """
#     content = []
#     for section in sections:
#         if section.section_content:
#             content.append(section.section_content)
#         if section.children:
#             content.extend(extract_content_from_sections(section.children))
#     return content


# --- Async Service Client Functions ---


# async def chunk_large_file(
#     session: aiohttp.ClientSession, file_path: Path
# ) -> str | None:
#     """
#     Submits a large file to the parsing service and returns a task ID.
#     """
#     logging.info(f"Submitting file {file_path} to parsing service...")
#
#     form_data = aiohttp.FormData()
#     form_data.add_field(
#         "configuration",
#         json.dumps(CHUNKING_CONFIGURATION),
#         content_type="application/json",
#     )
#
#     try:
#         content_type, _ = mimetypes.guess_type(file_path)
#         if content_type is None:
#             content_type = "application/octet-stream"
#
#         with open(file_path, "rb") as f:
#             form_data.add_field(
#                 "file", f, filename=file_path.name, content_type=content_type
#             )
#
#             async with session.post(
#                 CHUNK_SERVICE_URL, headers=CHUNKING_HEADERS, data=form_data
#             ) as response:
#                 response.raise_for_status()
#                 return (await response.json())["task_id"]
#
#     except FileNotFoundError:
#         logging.error(f"Data file not found at {file_path}")
#         return None
#     except aiohttp.ClientError as e:
#         logging.error(f"Error submitting file to parsing service: {e}")
#         return None


async def get_parse_result(
    session: aiohttp.ClientSession, task_id: str
) -> ParseResult | None:
    """
    Polls the parsing service for the result of a task.
    """
    output_url = f"{CHUNK_SERVICE_URL}/{task_id}/output/json"

    while True:
        try:
            async with session.get(output_url, headers=CHUNKING_HEADERS) as response:
                if response.status == 200:
                    data = await response.json()
                    try:
                        return ParseResult(**data)
                    except ValidationError as e:
                        logging.error(f"Error validating parsing service response: {e}")
                        return None
                elif response.status == 202:
                    logging.info(
                        f"Task {task_id} is still processing. Retrying in 5 seconds..."
                    )
                    await asyncio.sleep(5)
                else:
                    response.raise_for_status()
        except aiohttp.ClientError as e:
            logging.error(f"Error polling for parse result: {e}")
            return None


async def call_term_extraction_service(
    session: aiohttp.ClientSession, text: str, categories: list[str] | None = None
) -> list[Term]:
    """Calls the term extraction service asynchronously."""
    logging.info("Calling term extraction service...")

    if categories:
        te_params = {**TERM_EXTRACTION_PARAMS, "categories": categories}

    try:
        async with session.post(
            TERM_EXTRACTION_URL,
            params=te_params,
            headers=TERM_EXTRACTION_HEADERS,
            data=json.dumps(text),
        ) as response:
            response.raise_for_status()
            logging.info(f"Term extraction headers: {response.headers}")
            data = await response.json()

            try:
                terms = [Term(**item) for item in data]
                logging.info(f"Extracted {len(terms)} terms.")
                return terms
            except ValidationError as e:
                logging.error(f"Error validating term extraction response: {e}")
                return []

    except aiohttp.ClientError as e:
        logging.error(f"Error calling term extraction service: {e}")
        return []


async def call_kg_generation_service(
    session: aiohttp.ClientSession,
    text: str,
    item_id: str,
    terms: list[Term],
    ontology_content: bytes,
) -> dict:
    """Calls the KG generation service asynchronously."""
    logging.info(
        f"Calling KG generation service for item {item_id}, with {len(text) = }..."
    )

    kg_input_terms = []
    for term in terms:
        categories = term.categories or []
        alt_labels = term.alt_labels or []
        term_description = f"{term.definition}\nAlternative labels: {', '.join(alt_labels)}\nPredicted types: {', '.join(categories)}"
        kg_input_terms.append(
            TypedEntity(
                surface_form=term.pref_label,
                description=term_description,
            )
        )

    corpus_data = {
        "text": text,
        "id": item_id,
        "terms": [term.model_dump() for term in kg_input_terms],
    }

    form_data = aiohttp.FormData()
    form_data.add_field(
        "ontology_file",
        ontology_content,
        filename="ontology.ttl",
        content_type="text/turtle",
    )
    form_data.add_field(
        "corpus_file",
        json.dumps(corpus_data),
        filename="corpus.jsonl",
    )

    try:
        async with session.post(
            KG_GEN_URL, params=KG_GENERATION_PARAMS, data=form_data
        ) as response:
            if not response.ok:
                error_detail = await response.text()
                logging.error(
                    f"Error calling KG generation service for item {item_id}: {response.status} - {error_detail}"
                )
                response.raise_for_status()

            logging.info(f"KG generation headers: {response.headers}")
            graph = (await response.json())[0]
            logging.info(
                f"Successfully generated graph for item {item_id}. {len(graph['typed_entities']) = }, {len(graph['relations_wo_class_assertions']) = }"
            )
            return graph
    except aiohttp.ClientError as e:
        logging.error(f"Error calling KG generation service for item {item_id}: {e}")
        return {}


async def call_aggregate_and_deduplicate_service(
    session: aiohttp.ClientSession, graphs: list[dict]
) -> dict:
    """Calls the aggregation and deduplication service asynchronously."""
    logging.info(f"Calling aggregation service for {len(graphs)} graphs...")
    try:
        async with session.post(AGGREGATE_URL, json=graphs) as response:
            if not response.ok:
                error_detail = await response.text()
                logging.error(
                    f"Error calling aggregation service: {response.status} - {error_detail}"
                )
                response.raise_for_status()

            logging.info("Successfully aggregated and deduplicated graphs.")
            return await response.json()
    except aiohttp.ClientError as e:
        logging.error(f"Error calling aggregation service: {e}")
        return {}


async def process_chunk_with_sem(
    sem: asyncio.Semaphore, session: aiohttp.ClientSession, chunk_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Worker coroutine that takes a chunk, extracts terms, and generates a graph,
    while being constrained by a semaphore.
    """
    async with sem:
        chunk_id = chunk_data["chunk_id"]
        chunk_text = chunk_data["chunk_text"]
        ontology_content = chunk_data["ontology_content"]
        onto_class_names = chunk_data.get("onto_class_names")

        logging.info(f"Worker processing chunk {chunk_id}...")

        terms_cache_dir = wk_base_data_path / "terms_cache"
        terms_cache_dir.mkdir(exist_ok=True)
        cache_file = terms_cache_dir / f"terms_{chunk_id}.json"

        # terms = []
        # if cache_file.exists():
        #     logging.info(f"Loading terms from cache for chunk {chunk_id}")
        #     with open(cache_file, "r") as f:
        #         terms_data = json.load(f)
        #         try:
        #             terms = [Term(**item) for item in terms_data]
        #         except ValidationError as e:
        #             logging.error(
        #                 f"Error validating cached terms for chunk {chunk_id}: {e}"
        #             )
        #             # Invalidate cache and re-fetch
        #             cache_file.unlink()
        #
        # if not terms:
        logging.info(
            f"No valid cache found, calling term extraction for chunk {chunk_id}"
        )
        terms = await call_term_extraction_service(
            session, chunk_text, categories=onto_class_names
        )
        if terms:
            logging.info(f"Saving {len(terms)} terms to cache for chunk {chunk_id}")
            with open(cache_file, "w") as f:
                json.dump([term.model_dump() for term in terms], f, indent=2)

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


async def main(data_path: Path, ontology_path: Path, output_stem: str):
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

    onto = rdflib.Graph()
    onto.parse(ontology_path)

    classes = set()
    for s in onto.subjects(rdflib.RDF.type, rdflib.RDFS.Class):
        if isinstance(s, rdflib.URIRef):
            classes.add(s)
    for s in onto.subjects(rdflib.RDF.type, rdflib.OWL.Class):
        if isinstance(s, rdflib.URIRef):
            classes.add(s)

    onto_class_names = [onto.value(c, rdflib.RDFS.label).toPython() for c in classes]
    label_to_uri = {
        str(label): str(s)
        for s, p, o in onto.triples((None, rdflib.RDFS.label, None))
        for label in [o]
    }
    logging.info(f"Found {len(onto_class_names)} ontology classes.")

    async with aiohttp.ClientSession() as session:
        document_chunks = iter_wk_chunked_jsonl(data_path)

        chunks_to_process = [
            {
                "chunk_id": chunk.id,
                "chunk_text": chunk.text,
                "ontology_content": ontology_content,
                "onto_class_names": onto_class_names,
                "label_to_uri": label_to_uri,
            }
            for i, chunk in enumerate(document_chunks)
        ]

        logging.info(f"Collected {len(chunks_to_process)} chunks to process.")
        ###
        # chunks_to_process = chunks_to_process[:20]
        # logging.info(f"Reduced size {len(chunks_to_process) = }.")
        ###

        sem = asyncio.Semaphore(NUM_WORKERS)
        tasks = [
            process_chunk_with_sem(sem, session, chunk) for chunk in chunks_to_process
        ]

        graphs = []
        exceptions = []

        with tqdm.tqdm(total=len(tasks), desc="Processing chunks") as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    if isinstance(result, dict) and result:
                        graphs.append(result)
                except Exception as e:
                    exceptions.append(e)
                pbar.update(1)

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
                    results_dir
                    / f"{output_stem}__{ontology_path.stem}_final_graph.json"
                )
                with open(output_path, "w") as f:
                    json.dump(final_graph, f, indent=2)
                logging.info(f"Final aggregated graph saved to {output_path}")
        else:
            logging.warning("No graphs were generated to aggregate.")


if __name__ == "__main__":
    if ontology_file.exists():
        # Configure logging
        results_dir = wk_base_data_path / "results"
        results_dir.mkdir(exist_ok=True)
        log_file_path = results_dir / f"{output_stem}__{ontology_file.stem}.log"
        configure_logging(log_file_path=str(log_file_path))

        logging.info(f"Starting processing chunked WK Script with {ontology_file.name}")
        asyncio.run(
            main(
                data_path=chunked_path,
                ontology_path=ontology_file,
                output_stem=output_stem,
            )
        )
    else:
        logging.error(f"Ontology file not found: {ontology_file}")
