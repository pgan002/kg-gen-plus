import logging
from pathlib import Path

import dspy

musique_base_data_path = Path(__file__).parent / "data"
musique_chunks_path = musique_base_data_path / "musique_chunks.jsonl"
musique_onto_path = musique_base_data_path / "musique_ontology_refined.ttl"
output_base_path = musique_base_data_path / "results" / "musique_kg"
output_base_path.parent.mkdir(exist_ok=True)
llm_model = "openai/gpt-5.4-mini"
retrieval_model = None
# Disable dspy caching to produce new results every call -- useful for getting mean and std dev
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
# Slice items
i_start = 4001
i_end = 4010
num_workers = 10  # Number of threads for parallel processing


def configure_logging(log_file_path: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file_path)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logging.info(f"Logging to {log_file_path}")


configure_logging(log_file_path=f"{output_base_path}.log")
