# PubMed Data Processing and Benchmarking

This directory contains scripts to process PubMed data using the remote KGGen service and collect detailed statistics.

## Scripts

1. `prepare_pubmed_corpus.py`: Converts PubMed metadata and text files into a single JSONL corpus file expected by the KGGen API.
2. `run_pubmed_benchmark.py`: Sends the prepared corpus and ontology to the KGGen API and collects extensive statistics.

## Usage

### 1. Prepare the corpus

First, convert the raw PubMed data into the JSONL format:

```bash
python3 benchmarks/ga_pubmed/prepare_pubmed_corpus.py \
    --dataset benchmarks/ga_pubmed/data/dataset_small \
    --output benchmarks/ga_pubmed/data/dataset_small_corpus.jsonl
```

For the medium dataset:

```bash
python3 benchmarks/ga_pubmed/prepare_pubmed_corpus.py \
    --dataset benchmarks/ga_pubmed/data/dataset_medium \
    --output benchmarks/ga_pubmed/data/dataset_medium_corpus.jsonl
```

### 2. Run the benchmark

Run the benchmark using the prepared corpus. Settings are managed in `benchmarks/ga_pubmed/pubmed_config.py`.

```bash
PYTHONPATH=. python3 benchmarks/ga_pubmed/run_pubmed_benchmark.py
```

### Configuration (`pubmed_config.py`)

The benchmark behavior is controlled by variables in `benchmarks/ga_pubmed/pubmed_config.py`:

- `corpus_path`: Path to the prepared JSONL corpus file.
- `ontology_path`: Path to the ontology Turtle file.
- `output_dir`: Directory to save results.
- `KG_GEN_URL`: KGGen API URL.
- `KG_GENERATION_PARAMS`: Dictionary of parameters passed to the API, including:
    - `model`: LLM model to use.
    - `retrieval_model`: Model used for embeddings during deduplication.
    - `entity_threshold`: Similarity threshold for entity deduplication.
    - `predicate_threshold`: Similarity threshold for predicate deduplication.
    - `n_parallel`: Number of parallel requests.
- `PER_DOC`: Boolean, if `True`, processes documents individually.
- `LIMIT`: Integer, limits the number of documents to process.

## Collected Statistics

The scripts collect and calculate the following metrics:
- **Wall Time**: Total time taken from the client perspective.
- **API Time**: Generation time reported by the KGGen service.
- **Tokens**: Input (prompt) and Output (completion) tokens.
- **Characters**: Number of characters in the input text.
- **Graph Size**: Number of entities and relations extracted.
- **Throughput**: Tokens per second (both API and Wall time) and characters per second.
- **Step-by-step Stats**: Granular statistics for each generation step (extracted from `X-KG-Gen-Stats` header).
- **Deduplication Stats**: Statistics from the deduplication process (extracted from `X-KG-Gen-Dedup-Stats` header).

Results are saved as:
- `pubmed_{model}_{timestamp}_graph.json`: The generated knowledge graph.
- `pubmed_{model}_{timestamp}_results.json`: A detailed JSON summary of all collected metrics and statistics.
