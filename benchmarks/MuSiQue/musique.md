# MuSiQue Dataset KG Generation

This document explains how to generate a knowledge graph from the MuSiQue dataset
using the kg-gen service.

> **Note:** Term extraction is no longer performed by an external service. The
> kg-gen `/api/generate` endpoint extracts and types entities, extracts
> relations, and aggregates + deduplicates the whole corpus itself.

## 1. Prerequisites

- A running kg-gen service reachable at the `KG_GEN_URL` configured in
  `musique_config.py` (the `/api/generate` endpoint).
- If your LLM provider needs a key, export it as `KG_GEN_API_KEY`; it is
  forwarded to the service as the `X-API-Key` header. (Not needed for the local
  `api_base` model configured by default.)

## 2. Dataset

Place `musique_chunks.jsonl` in `benchmarks/MuSiQue/data/`. Each line is a chunk
with `chunk_id`, `source_doc`, and `content`. It is loaded via
`musique_utils.iter_musique_chunks_jsonl`. If you only have `musique_chunks.json`
(with a `chunks` array), convert it first with `json_to_jsonl.py`.

The ontology (`musique_ontology_refined.ttl` by default, set via
`musique_onto_path`) must also be in `benchmarks/MuSiQue/data/`.

## 3. Configuration

Configure your settings in `musique_config.py`:

- `KG_GEN_URL` — URL of the kg-gen `/api/generate` endpoint.
- `KG_GENERATION_PARAMS` — generation options sent as query parameters; these
  mirror `app.schemas.GenerationMetadata` (`model`, `api_base`,
  `enforce_*_conformance`, `enable_thinking`, `deduplicate`, `n_parallel`, …).
- `NUM_WORKERS` — how many documents kg-gen processes concurrently
  (used as `n_parallel`).
- `i_start` / `i_end` — 1-based, inclusive slice of the dataset to process.

## 4. Running the Script

Run from the project root:

```bash
python -m benchmarks.MuSiQue.run_musique
```

The whole (sliced) corpus is sent to kg-gen in a single request; generation runs
server-side and may take a while.

## 5. Results

Outputs are written to `benchmarks/MuSiQue/data/results/`:

- `musique_kg__<ontology_stem>_final_graph.json` — the generated `KnowledgeGraph`
  (entities, relations, ontology extensions, clusters, and stats).
- `musique_kg.log` — logging information.

To convert the resulting graph to RDF/Turtle (with `rdf:type` assertions, typed
literals, and RDF-star provenance linking triples back to their source chunks),
use `json_to_rdf.py`:

```bash
python -m benchmarks.MuSiQue.json_to_rdf \
    benchmarks/MuSiQue/data/results/musique_kg__musique_ontology_refined_final_graph.json \
    benchmarks/MuSiQue/data/musique_chunks.jsonl \
    output.ttl
```
