import asyncio
import json
import logging
from pathlib import Path

from benchmarks.MuSiQue.musique_config import (
    musique_chunks_path,
    num_workers,
    i_start,
    i_end,
)
from benchmarks.MuSiQue.musique_utils import (
    iter_musique_chunks_jsonl,
    MusiqueChunk,
    extract_terms_for_text,
)
from kg_gen.models import TypedEntity


async def _process_single_chunk(item: MusiqueChunk) -> MusiqueChunk:
    """
    Processes a single MusiqueChunk item, capturing logs in a string buffer.
    Returns the generated graph, usage statistics, and captured log messages.
    """
    item_entities = await extract_terms_for_text(item.content)
    out_terms = []
    for term in item_entities:
        term_as_typed_ent = TypedEntity(
            surface_form=term.term, uri=None, description=term.definition, type=None
        )
        out_terms.append(term_as_typed_ent)
    item.terms = out_terms
    return item


async def process_file(test_path: Path, output_path: Path) -> None:
    main_logger = logging.getLogger()
    main_logger.info(f"Starting parallel processing with {num_workers} workers.")

    if output_path.exists():
        output_path.write_text("")

    # Use a Semaphore to limit the number of concurrent tasks
    sem = asyncio.Semaphore(num_workers)

    async def _process_with_sem(item):
        async with sem:
            return await _process_single_chunk(item)

    tasks = []
    for i, item in enumerate(iter_musique_chunks_jsonl(test_path), start=1):
        if i < i_start:
            continue
        if i > i_end:
            break
        tasks.append(asyncio.create_task(_process_with_sem(item)))

    proc_items = []
    for task in asyncio.as_completed(tasks):
        try:
            input_item = await task
            proc_items.append(input_item)
            main_logger.info(
                f"Finished processing chunk {len(proc_items)}/{len(tasks)}"
            )
            output_path.open("a").write(json.dumps(input_item.model_dump()) + "\n")
        except Exception as exc:
            main_logger.error(f"A chunk generated an exception: {exc}")

    return


if __name__ == "__main__":
    if musique_chunks_path.exists():
        logging.info(f"Processing {musique_chunks_path.name}")
        asyncio.run(
            process_file(
                test_path=musique_chunks_path,
                output_path=musique_chunks_path.parent
                / f"terms_{i_start}_{i_end}.jsonl",
            )
        )
    else:
        logging.error(f"Input file not found: {musique_chunks_path}")
