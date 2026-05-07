# MuSiQue Dataset KG Generation

This document provides instructions on how to generate a knowledge graph from the MuSiQue dataset.

## 1. Prerequisites

- Create a `.env` file by copying the `.env.example` file from the project root.
- Update the `.env` file with your LLM provider credentials.

## 2. Dataset

The `musique_chunks.jsonl` file should be placed in the `benchmarks/MuSiQue/data` directory. The data is loaded via `musique_utils.iter_musique_chunks_jsonl`.

## 3. Configuration

Before running the script, configure your settings in `musique_config.py`:

-   Set up your LLM model and retrieval model.
-   Optionally, you can control which items from the dataset are used by setting the `i_start` and `i_end` variables (1-based indexing).

## 4. Running the Script

To run the script, execute `run_musique.py` from the project root:

```bash
source .env && python benchmarks/MuSiQue/run_musique.py
```

## 5. Results

The script will generate a knowledge graph from the MuSiQue dataset. The output includes:

-   A `musique_kg.json` file containing the generated graph.
-   A `musique_kg.html` file for visualizing the graph.
-   A `musique_kg.log` file with logging information, including token usage.
