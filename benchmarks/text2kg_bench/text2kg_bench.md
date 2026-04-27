# Text2KGBench Dataset Evaluation

This document provides instructions on how to evaluate the Text2KGBench dataset.

## 1. Prerequisites

- Create a `.env` file by copying the `.env.example` file from the project root.
- Update the `.env` file with your LLM provider credentials.

## 2. Dataset

The dataset should be downloaded from [here](https://github.com/cenguix/Text2KGBench) and put into the `data` directory.
The dataset is loaded via `text2kgbench_utils.iter_text2kgbench_jsonl` with an optional parameter `clean_triples` to remove the triples that do not appear in the text explicitly -- checked if subject and object both appear in the text.

## 3. Configuration

Before running the evaluation, configure your settings in `text2kgbench_config.py`:

-   Set up your LLM.
-   Optionally, you can control which items from the dataset are used in the evaluation by setting the `i_start` and `i_end` variables (1-based indexing).
-   Specify if you want to use gold entities or extract with `use_gold_entities`.

## 4. Running the Evaluation

To run the evaluation, you first need to source the environment variables from the `.env` file. Execute the `run_text2kgbench.py` script from the project root:

```bash
source .env && python benchmarks/text2kg_bench/run_text2kgbench.py
```

## 5. Results

The evaluation script logs the results for each item and provides aggregated metrics at the end. The output includes:

-   Precision, Recall, and F1-score.
-   Usage data, such as token count and execution time.
