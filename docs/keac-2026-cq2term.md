# Reproducing the Graphwise KEAC 2026 CQ2Term submission

This document describes how to reproduce the Graphwise submission
`kggen-cq-reassign-t035-k5` for the Knowledge Engineering Automation Challenge
2026 CQ2Term sub-task.

The submitted predictions were produced from run 03 of six repeated KGGen runs.
The method constructs one domain graph from all competency questions, performs
string graph deduplication, exports normalized and semantically clustered class
and property terms, and then reassigns the canonical domain vocabulary to each
competency question.

## Submitted configuration

- Base model: `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`
- vLLM served model name: `gemma4`
- Generation temperature: `0.7`
- Maximum completion tokens: `4096`
- Parallel documents: `10`
- Repeated runs: `6`
- KGGen graph deduplication: string deduplication enabled
- CQ2Term class source: entity surface forms
- Lexical normalization: enabled by the exporter
- Morphological normalization: enabled by the exporter
- Semantic clustering model: `sentence-transformers/all-MiniLM-L6-v2`
- Semantic clustering threshold: `0.9`
- CQ reassignment threshold: `0.35`
- Maximum reassignment additions: `5` per CQ and term kind
- Submitted repeated run: `03`

The submission does not use class-candidate filtering, ontology role filtering,
predicate induction, or embedding-based graph deduplication.

## 1. Clone the repositories

```bash
git clone https://github.com/pgan002/kg-gen-plus.git
cd kg-gen-plus
git checkout 0a879370776ce7859708f6aa01c8221dc82ccfd7
cd ..
git clone https://codeberg.org/ke-automation-challenge/challenge-catalog.git
```

Commit `0a879370776ce7859708f6aa01c8221dc82ccfd7` is the exact KGGen+ code
version used by this submission and is also recorded as the metadata entry
point. Checking it out avoids depending on the repository's changing default
branch.

For optional local evaluation, also clone CQ4OE:

```bash
git clone https://github.com/oeg-upm/cq4oe-benchmark.git
```

A convenient sibling layout is:

```text
workspace/
├── kg-gen-plus/
├── challenge-catalog/
└── cq4oe-benchmark/
```

## 2. Install KGGen+

The experiment was run with Python 3.12. From `kg-gen-plus`:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The export step downloads the configured SentenceTransformer model on first
use. Set `HF_TOKEN` if needed to avoid anonymous Hugging Face rate limits.

## 3. Serve Gemma with vLLM

Install vLLM in an environment appropriate for the available GPU hardware. One
representative command is:

```bash
vllm serve cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
  --served-model-name gemma4 \
  --port 7777
```

Confirm the served name:

```bash
curl http://localhost:7777/v1/models
```

The response must contain model ID `gemma4`. The server exposes an
OpenAI-compatible API; KGGen appends `/v1` when needed.

## 4. Generate six repeated domain graphs

From `kg-gen-plus`, run:

```bash
python misc/run_domain.py \
  --model gemma4 \
  --api-base http://localhost:7777 \
  --api-key dummy \
  --dataset-dir ../challenge-catalog/keac-2026/dataset/CQ2Onto \
  --output-root results \
  --label dedup \
  --runs 6 \
  --n-parallel 10 \
  --temperature 0.7 \
  --max-tokens 4096 \
  --deduplicate
```

This creates:

```text
results/gemma4-dedup/
├── awo/{manifest.json,runs/01.json,...,runs/06.json}
├── odrl/{manifest.json,runs/01.json,...,runs/06.json}
├── swo/{manifest.json,runs/01.json,...,runs/06.json}
├── vgo/{manifest.json,runs/01.json,...,runs/06.json}
├── water/{manifest.json,runs/01.json,...,runs/06.json}
└── wine/{manifest.json,runs/01.json,...,runs/06.json}
```

Generation is stochastic because the temperature is `0.7`. A rerun should
reproduce the method and configuration, but is not expected to reproduce the
submitted JSON byte for byte unless the complete model-serving environment and
random-number state are also reproduced.

## 5. Export CQ2Term predictions

From `kg-gen-plus`, with `cq4oe-benchmark` in the sibling location shown above:

```bash
python misc/export_cq4oe.py \
  --task cq2term \
  --results-dir results/gemma4-dedup \
  --benchmark-root ../cq4oe-benchmark \
  --class-source surface \
  --semantic-clustering \
  --semantic-model sentence-transformers/all-MiniLM-L6-v2 \
  --semantic-threshold 0.9 \
  --cq-reassignment \
  --cq-reassignment-threshold 0.35 \
  --cq-reassignment-max-additions 5 \
  --export-label surface-lex-morph-semantic-cq-reassign-t035-k5
```

The relevant staged folders are:

```text
../cq4oe-benchmark/CQ2Term/01_predictions/
└── kggen-gemma4-dedup-surface-lex-morph-semantic-cq-reassign-t035-k5-run-01/
    ...
└── kggen-gemma4-dedup-surface-lex-morph-semantic-cq-reassign-t035-k5-run-06/
```

The exporter is resumable. Repeating the same command skips complete output
whose manifest and required files match; it refuses to overwrite mismatched or
incomplete output.

## 6. Validate or evaluate the repeated runs (optional)

From `cq4oe-benchmark`:

```bash
python scripts/evaluate_and_aggregate_runs.py \
  --task cq2term \
  --run-prefix kggen-gemma4-dedup-surface-lex-morph-semantic-cq-reassign-t035-k5-run- \
  --group kggen/gemma4/string-dedup/surface-lex-morph-semantic-cq-reassign-t035-k5
```

Across the six public-data runs, run 03 was closest to the condition's mean
CQ-Mean coverage and was selected as the representative submission run. The
challenge organisers evaluate the submitted files independently, including on
hidden data where applicable.

## 7. Assemble the challenge submission

Use the six files under:

```text
../cq4oe-benchmark/CQ2Term/01_predictions/
kggen-gemma4-dedup-surface-lex-morph-semantic-cq-reassign-t035-k5-run-03/terms/
```

Place them in the challenge catalog as:

```text
keac-2026/CQ2Term/graphwise/kggen-cq-reassign-t035-k5/
├── metadata.yaml
├── awo/awo_cq2terms_terms.json
├── odrl/odrl_cq2terms_terms.json
├── swo/swo_cq2terms_terms.json
├── vgo/vgo_cq2terms_terms.json
├── water/water_cq2terms_terms.json
└── wine/wine_cq2terms_terms.json
```

Each domain directory must contain only its result file. Do not include local
CQ4OE evaluation results, aggregate summaries, or KGGen metadata traces in the
challenge submission.
