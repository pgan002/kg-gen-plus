import logging
import json
import asyncio
from pathlib import Path
from typing import Iterator

from app.utils import parse_ontology
from benchmarks.wk.wk_config import (
    wk_base_data_path,
    configure_logging,
)
from kg_gen.kg_gen import KGGen
from kg_gen.models import InputData


def iter_jsonl(data_path: Path) -> Iterator[InputData]:
    with open(data_path) as f:
        for i, line in enumerate(f):
            data_line = json.loads(line)
            item = InputData(**data_line)
            yield item


async def main(data_path: Path, ontology_path: Path | None = None):
    """
    Main async function to process a data file by orchestrating calls to chunking,
    term extraction, and KG generation services.
    """
    if ontology_path is not None:
        try:
            with open(ontology_path, "rb") as f:
                onto = parse_ontology(f)
                logging.info(f"Parsed ontology: {onto}")
        except FileNotFoundError:
            logging.error(f"Ontology file not found at {ontology_path}")
            onto = None
    else:
        onto = None

    kg_gen = KGGen(
        # Use 'openai/' prefix to force standard HTTP client with Bearer token auth
        # model="openai/gpt-oss:120b",
        model="openai/qwen3.5",
        temperature=0.1,
        api_base="http://192.168.129.20:7777/v1",
        # api_key=keycloak_token
        retrieval_model=None,
    )
    gs = []
    total_usage = None
    for i, item in enumerate(iter_jsonl(data_path), start=1):
        # if i < i_start:
        #     continue
        # if i > i_end:
        #     break
        logging.info(
            f"\n\n{i = }, {item.id = }, {len(item.terms) if item.terms else None = }\n{item.text = }"
        )

        g, usage = await kg_gen.generate(
            input_data=item,
            types=onto.classes if onto else None,
            predicate_domain_range=onto.predicates if onto else None,
            deduplicate=False,
        )
        if total_usage is not None:
            total_usage += usage
        else:
            total_usage = usage
        logging.info(f"{usage = }\n")
        gs.append(g)
    logging.info(f"{total_usage = }")

    # graphs = [res for res in results if isinstance(res, dict) and res]
    # exceptions = [res for res in results if isinstance(res, Exception)]
    #
    # logging.info(f"Successfully generated {len(graphs)} graphs.")
    # if exceptions:
    #     logging.error(f"Encountered {len(exceptions)} errors during processing.")
    #     for exc in exceptions:
    #         logging.error(f"- {exc}")
    #
    # if graphs:
    #     final_graph = await call_aggregate_and_deduplicate_service(session, graphs)
    #     if final_graph:
    #         results_dir = wk_base_data_path / "results"
    #         results_dir.mkdir(exist_ok=True)
    #         output_path = (
    #             results_dir / f"{data_path.stem}__{ontology_path.stem}_final_graph.json"
    #         )
    #         with open(output_path, "w") as f:
    #             json.dump(final_graph, f, indent=2)
    #         logging.info(f"Final aggregated graph saved to {output_path}")
    # else:
    #     logging.warning("No graphs were generated to aggregate.")


if __name__ == "__main__":
    # Define the paths for the single large data file and the ontology
    data_file = (
        wk_base_data_path / "terms_4002.jsonl"
    )  # Assuming a single large data file

    if data_file.exists():
        # Configure logging
        results_dir = wk_base_data_path / "results"
        results_dir.mkdir(exist_ok=True)
        log_file_path = results_dir / f"{data_file.stem}.log"
        configure_logging(log_file_path=str(log_file_path))

        logging.info(f"Starting processing for {data_file.name}")
        asyncio.run(main(data_path=data_file))
    else:
        if not data_file.exists():
            logging.error(f"Data file not found: {data_file}")
