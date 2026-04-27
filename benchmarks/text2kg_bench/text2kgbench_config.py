import logging
from pathlib import Path

import dspy

text2kgbench_base_data_path = Path(__file__).parent / "data" / "wikidata_tekgen"
ground_truth_dir = text2kgbench_base_data_path / "ground_truth"
ontologies_dir = text2kgbench_base_data_path / "ontologies" / "owl"
llm_model = "openai/gpt-5.4-mini"
retrieval_model = "sentence-transformers/all-mpnet-base-v2"
# Remove triples that do not appear in the text from evaluation
clean_triples = True
# Disable dspy caching to produce new results every call -- useful for getting mean and std dev
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
# Slice items
i_start = 1
i_end = 1000
# Use gol entities or call the term extraction
use_gold_entities = False


def configure_logging(log_file_path: str):
    # Configure logging
    # log_file_path = f"{text2kgbench_base_results_path}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    # Create new handlers
    file_handler = logging.FileHandler(log_file_path)
    stream_handler = logging.StreamHandler()
    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logging.info(f"Logging to {log_file_path}")
