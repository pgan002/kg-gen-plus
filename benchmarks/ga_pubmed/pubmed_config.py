from pathlib import Path

pubmed_base_data_path = Path(__file__).parent / "data"
corpus_path = pubmed_base_data_path / "dataset_medium_corpus.jsonl"
ontology_path = pubmed_base_data_path / "pubmed_medium_ontology_refined.ttl"
output_dir = pubmed_base_data_path / "results"

KG_GEN_URL = "http://localhost:5000/api/generate"
# KG_GEN_URL = "http://dsx-gws-rai-docker-dmo-apl-n-01:8087/api/generate"

KG_GENERATION_PARAMS = {
    # "model": "openai/gpt-5.4-mini",
    "n_parallel": 20,
    "enforce_type_conformance": "true",
    "enforce_domain_conformance": "true",
    "enforce_range_conformance": "true",
    "enforce_predicate_conformance": "true",
    "retrieval_model": "mixedbread-ai/mxbai-embed-large-v1",
    "entity_threshold": 0.9,
    "predicate_threshold": 0.9,
    "deduplicate": "true",
    "model": "openai/gemma4",
    "api_base": "http://192.168.129.20:7777/v1",
}

KG_GENERATION_HEADERS = {"Content-Type": "application/json", "X-Api-Key": None}

# Execution params
LIMIT = 500000
