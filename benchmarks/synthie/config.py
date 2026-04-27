import logging
from pathlib import Path

import dspy

synthie_base_data_path = Path(__file__).parent / "data" / "synthie"
test_path = synthie_base_data_path / "test_small_ordered.jsonl"
llm_model = "openai/gpt-5.4-mini"
retrieval_model = "sentence-transformers/all-mpnet-base-v2"
logging.basicConfig(level=logging.INFO)
# Disable dspy caching to produce new results every call -- useful for getting mean and std dev
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
# Slice items
i_start = 1
i_end = 100
