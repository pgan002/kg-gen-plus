from datetime import datetime

import logging
import os
from pathlib import Path

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
musique_base_data_path = Path(__file__).parent / "data"
musique_chunks_path = musique_base_data_path / "musique_chunks.jsonl"
musique_onto_path = musique_base_data_path / "musique_ontology_refined.ttl"
output_base_path = musique_base_data_path / "results" / f"musique_kg_{timestamp}"
output_base_path.parent.mkdir(exist_ok=True)

# Slice items (1-based, inclusive) to process.
i_start = 0
i_end = 10000

# --- Configuration ---
# Number of documents kg-gen processes concurrently server-side (n_parallel).
NUM_WORKERS = 10

# --- Service Configuration ---
# The kg-gen "single parallel endpoint": it takes the whole JSONL corpus plus an
# optional ontology, performs entity extraction, typing, relation extraction,
# aggregation and deduplication server-side, and returns one KnowledgeGraph.
# Term extraction is no longer done externally -- kg-gen handles it.
KG_GEN_URL = "http://dsx-gws-rai-docker-dmo-apl-n-01:8087/api/generate"

# Optional API key for the LLM provider, forwarded as the X-API-Key header.
# Not needed for the local (api_base) models below.
X_API_KEY = os.getenv("KG_GEN_API_KEY")

# --- KG Generation Parameters (sent as query parameters to KG_GEN_URL) ---
# These mirror app.schemas.GenerationMetadata. Values are strings because they
# are passed as URL query parameters.
KG_GENERATION_PARAMS = {
    "model": "openai/qwen3.5",
    "api_base": "http://192.168.129.20:7777/v1",
    "enable_thinking": "false",
    "enforce_type_conformance": "true",
    "enforce_domain_conformance": "true",
    "enforce_range_conformance": "true",
    "enforce_predicate_conformance": "true",
    "deduplicate": "true",
    "n_parallel": str(NUM_WORKERS),
    # A small non-zero temperature reduces repetition loops that cause local
    # models to run to the token limit at temperature 0.0.
    "temperature": "0.3",
    # Headroom above the 16000 default to avoid truncated extractions. Keep this
    # within the model's context window (prompt + output). Lower if your model
    # rejects it.
    "max_tokens": "32000",
}


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
