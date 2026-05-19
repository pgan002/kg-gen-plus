import asyncio
import logging
import time
from pathlib import Path
from io import StringIO

from benchmarks.MuSiQue.musique_config import (
    llm_model,
    retrieval_model,
    i_start,
    i_end,
    output_base_path,
    musique_chunks_path,
    musique_onto_path,
    num_workers,
)
from benchmarks.MuSiQue.musique_utils import (
    iter_musique_chunks_jsonl,
    parse_ontology,
    MusiqueChunk,
    extract_terms_for_text,
)
from kg_gen.kg_gen import KGGen
from kg_gen.models import Graph, InputData, Ontology, KGGenStats, TypedEntity


async def _process_single_chunk(
    item: MusiqueChunk, kg: KGGen, onto: Ontology
) -> tuple[Graph, KGGenStats, str]:
    """
    Processes a single MusiqueChunk item, capturing logs in a string buffer.
    Returns the generated graph, usage statistics, and captured log messages.
    """
    log_stream = StringIO()
    # Create a temporary logger for this chunk
    chunk_logger = logging.getLogger(f"chunk_{item.chunk_id}")
    chunk_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(log_stream)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    chunk_logger.addHandler(handler)
    chunk_logger.propagate = False  # Prevent duplicate logging to the root logger

    try:
        chunk_logger.info(f"Processing chunk {item.chunk_id}")
        chunk_logger.info(f"Extracting terms for item {item.chunk_id}...")
        start_time = time.time()
        item_entities = await extract_terms_for_text(item.content)
        out_terms = []
        for term in item_entities:
            term_as_typed_ent = TypedEntity(
                surface_form=term.term, uri=None, description=term.definition, type=None
            )
            out_terms.append(term_as_typed_ent)
        item.terms = out_terms

        chunk_logger.info(
            f"Extracted {len(item.terms)} terms for {item.chunk_id}. it took {time.time() - start_time:0.2f}s"
        )

        g, usage = await kg.generate(
            input_data=InputData(text=item.content, id=item.chunk_id),
            terms=item.terms,
            types=list(onto.classes),
            predicate_domain_range=list(onto.predicates),
        )
        chunk_logger.info(f"Graph generated for {item.chunk_id}. {g.stats()}")
        return g, usage, log_stream.getvalue()
    finally:
        # Clean up the handler and logger
        chunk_logger.removeHandler(handler)
        handler.close()


async def process_file(test_path: Path, kg: KGGen, onto: Ontology) -> None:
    gs: list[Graph] = []
    total_usage = KGGenStats()

    main_logger = logging.getLogger()
    main_logger.info(f"Starting parallel processing with {num_workers} workers.")

    # Use a Semaphore to limit the number of concurrent tasks
    sem = asyncio.Semaphore(num_workers)

    async def _process_with_sem(item):
        async with sem:
            return await _process_single_chunk(item, kg, onto)

    tasks = []
    for i, item in enumerate(iter_musique_chunks_jsonl(test_path), start=1):
        if i < i_start:
            continue
        if i > i_end:
            break

        tasks.append(asyncio.create_task(_process_with_sem(item)))

    for task in asyncio.as_completed(tasks):
        try:
            g, usage, chunk_logs = await task
            gs.append(g)
            total_usage += usage
            main_logger.info(
                f"\n--- Logs for chunk  ---\n{chunk_logs.strip()}\n--- End logs for chunk ---\n"
            )
            main_logger.info(
                f"Finished processing chunk {len(gs)}/{len(tasks)}. Chunk total usage: {usage.overall_usage}"
            )
        except Exception as exc:
            main_logger.error(f"A chunk generated an exception: {exc}")

    main_logger.info(
        f"All relevant chunks processed. Final total usage: {total_usage.overall_usage}"
    )

    if gs:
        agg_g = kg.aggregate(gs)
        kg.export_graph(graph=agg_g, output_path=f"{output_base_path}.json")
        kg.visualize(agg_g, f"{output_base_path}.html", False)
        main_logger.info(
            f"Knowledge graph exported to {output_base_path}.json and visualized at {output_base_path}.html"
        )
    return


if __name__ == "__main__":
    kg_gen = KGGen(
        model=llm_model,
        temperature=1.0,
        retrieval_model=retrieval_model,
    )

    onto = parse_ontology(musique_onto_path)
    if musique_chunks_path.exists():
        logging.info(f"Processing {musique_chunks_path.name} with {llm_model = }")
        asyncio.run(process_file(test_path=musique_chunks_path, kg=kg_gen, onto=onto))
    else:
        logging.error(f"Input file not found: {musique_chunks_path}")
