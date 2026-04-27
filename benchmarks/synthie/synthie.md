# SynthIE Dataset Evaluation

This document provides instructions on how to evaluate the SynthIE dataset.

## 1. Prerequisites

- Create a `.env` file by copying the `.env.example` file from the project root.
- Update the `.env` file with your LLM provider credentials.

## 2. Download the Dataset

First, download the evaluation file in `.jsonl` format from the [Hugging Face repository](https://huggingface.co/datasets/martinjosifoski/SynthIE/tree/main).

## 3. Configuration

Before running the evaluation, configure your settings in `config.py`:

-   Set up your LLM.
-   Specify the path to the downloaded dataset.
-   Optionally, you can control which items from the dataset are used in the evaluation by setting the `i_start` and `i_end` variables (1-based indexing).

## 4. Running the Evaluation

To run the evaluation, you first need to source the environment variables from the `.env` file.

### `run_synthie.py`

This script uses the ground-truth (gold) entities from the dataset itself.

To run it:
```bash
source .env && python benchmarks/synthie/run_synthie.py
```

### `run_synthie_w_termext.py`

This script uses a term extraction service to identify entities.

**Note:** The service URL is hardcoded in the script. If you need to use a different endpoint, please modify the URL inside `run_synthie_w_termext.py`. The default is `http://dsx-gws-rai-docker-dmo-apl-n-01:8089/extract`.

To run it:
```bash
source .env && python benchmarks/synthie/run_synthie_w_termext.py
```

## 5. Results

The evaluation script logs the results for each item and provides aggregated metrics at the end. The output includes:

-   Micro/Macro Precision, Recall, and F1-score.
-   Usage data, such as token count and execution time.
