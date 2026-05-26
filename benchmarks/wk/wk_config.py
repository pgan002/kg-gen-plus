import os

import logging
from pathlib import Path


# --- Configuration ---
NUM_WORKERS = 10  # Number of concurrent async tasks

# --- Service Configuration ---
CHUNK_SERVICE_URL = "http://dsx-gws-rai-docker-dmo-apl-n-01:8088/api/v1/parse"
TERM_EXTRACTION_URL = "http://dsx-gws-rai-docker-dmo-apl-n-01:8089/extract_from_text"
KG_GEN_URL = "http://dsx-gws-rai-docker-dmo-apl-n-01:8087/api/generate"
AGGREGATE_URL = (
    "http://dsx-gws-rai-docker-dmo-apl-n-01:8087/api/aggregate_and_deduplicate"
)

wk_base_data_path = Path(__file__).parent / "data"
chunking_auth_key = os.getenv("CHUNKING_AUTH_KEY")
chunking_task_id = "461f297c-3d98-4a19-82f5-b7d347604398"
output_stem = "skript_crashkurs_sose2018"
ontology_file = wk_base_data_path / "wkg_ontology_refined.ttl"

# --- Chunking Service Parameters ---
CHUNKING_HEADERS = {"accept": "application/json", "Authorization": chunking_auth_key}
CHUNKING_CONFIGURATION = {
    "parser": {"id": "docling"},
    "chunker": {"id": "default", "chunk_size": 64},
}

# --- Term Extraction Service Parameters ---
TERM_EXTRACTION_PARAMS = {
    "model": "gpt-mini-4o",
    "window_size": "24000",
    "window_overlap_size": "1000",
}
TERM_EXTRACTION_HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

# --- KG Generation Service Parameters ---
KG_GENERATION_PARAMS = {
    "model": "openai/qwen3.5",
    "api_base": "http://192.168.129.20:7777/v1",
    "enable_thinking": "false",
}


# Slice items
i_start = 1
i_end = 1000


def configure_logging(log_file_path: str):
    # Configure logging
    # log_file_path = f"{wk_base_results_path}.log"
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
