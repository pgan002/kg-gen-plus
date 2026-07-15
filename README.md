# KGGen+ — Ontology-Guided Knowledge Graph Generation

`KGGen+` extracts knowledge graphs from plain text using LLMs. It builds on
[kg-gen](https://github.com/stair-lab/kg-gen) and adds **ontology-guided
extraction** (typed entities and predicates with best-effort domain/range
conformance), a **FastAPI service** with a visual UI, entity/relation
**deduplication**, and a set of **benchmarks**.

Use it if you want to:
- Create a graph to assist with RAG (Retrieval-Augmented Generation)
- Generate graph-structured synthetic data for model training and testing
- Structure text into a graph that conforms to an ontology you provide
- Analyze the relationships between concepts in your source text

Models are pluggable via [LiteLLM](https://docs.litellm.ai/docs/providers)
(OpenAI, Ollama, Anthropic, Gemini, Deepseek, and others), and structured output
is produced with [DSPy](https://dspy.ai/).

## Powered by a model of your choice

Pass any LiteLLM-supported model id, e.g.:
- `openai/gpt-5.4-mini`
- `ollama/gpt-oss:120b`

## Quick start (Docker)

The application is containerized and run with Docker Compose.

**1. Configure the environment**

Create a `.env` file in the project root by copying `.env.example`, then fill in
your keys:

```dotenv
LLM_MODEL=openai/gpt-5.4-mini
LLM_TEMPERATURE=1.0
RETRIEVAL_MODEL=all-MiniLM-L6-v2

OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=
# GEMINI_API_KEY=

IMAGE_NAME="maven.ontotext.com/nlp/kg-gen-plus"
```

**2. Run the application**

```bash
docker-compose up --build
```

The server is then available at `http://localhost:5000` (override with `PORT`):
- **Interactive API docs** at `/docs`
- **Web APIs** at `/api` — generate graphs from text, parse/convert ontologies
- **Graphical UI** at `/ui` — visually investigate generated graphs

## Using the HTTP API

Generate a graph from a `.jsonl` corpus (one `{"id", "text"}` object per line),
optionally guided by a Turtle (`.ttl`) ontology. Two modes are available:

- **`POST /api/generate`** — synchronous; blocks until generation finishes and
  returns the `KnowledgeGraph`. Best for small corpora.
- **`POST /api/generate_async`** — starts a background job and returns `202` with
  a `job_id`. Generation continues even if the client disconnects. Poll
  `GET /api/jobs/{job_id}` for progress (with a live ETA) and fetch the graph
  from `GET /api/jobs/{job_id}/result` once the status is `completed`. Use
  `GET /api/jobs` to list all jobs. Recommended for large corpora.

> **Note:** background jobs are held **in memory** in the server process — they
> are lost on restart, the store is bounded (oldest finished jobs are evicted
> when full), and it does not work across multiple workers. See
> [`app/api_doku.md`](app/api_doku.md) for details and the full endpoint
> reference.

Example (background job):

```bash
# Start the job
curl -sX POST "http://localhost:5000/api/generate_async?model=openai/gpt-5.4-mini" \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -F "corpus_file=@corpus.jsonl" \
  -F "ontology_file=@ontology.ttl"
# -> {"job_id": "...", "status": "pending", "status_url": "...", "result_url": "..."}

# Poll progress, then fetch the result
curl -s "http://localhost:5000/api/jobs/<job_id>"
curl -s "http://localhost:5000/api/jobs/<job_id>/result"
```

## MCP server (agent-in-the-loop)

Besides the batch HTTP API, an [MCP](https://modelcontextprotocol.io) server is
mounted in-process at **`/mcp`** and starts together with the app. It powers an
interactive, agent-driven extraction flow: the agent does NER, entity typing, and
relation extraction, while the MCP tools handle the deterministic parts —
ontology parsing, predicate suggestion, conformance/schema validation, graph
serialization, and deduplication.

Point any MCP client at `http://localhost:5000/mcp`. The companion
[`kg-extract` skill](skills/kg-extract/SKILL.md) documents the workflow and the
expected ontology format. The server can also run standalone with
`fastmcp run mcp/server.py`. It requires the optional `mcp`
extra (`pip install 'kg-gen[mcp]'`); without it the `/mcp` endpoint is simply not
mounted. See [`app/api_doku.md`](app/api_doku.md) and
[`docs/skill-mcp-plan.md`](misc/skill-mcp-plan.md) for details.

## Using `kg-gen` as a library

Install it:

```bash
pip install .
```

`KGGen.generate` is asynchronous, so run it inside an event loop (e.g. with
`asyncio.run`):

```python
import asyncio

from kg_gen.kg_gen import KGGen
from kg_gen.models import InputData

kg = KGGen(
    model="openai/gpt-5.4-mini",  # any LiteLLM model id
    temperature=0.0,
    api_key="YOUR_API_KEY",       # optional if set in the environment / local model
)

text = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."

graph, stats = asyncio.run(
    kg.generate(
        input_data=InputData(id="text_1", text=text),
        entity_context="Family relationships",
    )
)
print(graph)   # typed entities and relations
print(stats)   # token usage, timings, class/predicate usage
```

`input_data` also accepts a `list[InputData]`, in which case documents are
processed in parallel (see `n_parallel`) and the resulting graphs are aggregated
and deduplicated into one. Progress is logged in a tqdm-like form.

### Visualizing graphs

```python
KGGen.visualize(graph, output_path="graph.html", open_in_browser=True)
```

## Benchmarks

The `benchmarks/` directory contains runners for several datasets (e.g. MuSiQue,
SynthIE, Text2KGBench). They call the service's `/api/generate` endpoint or the
library directly; see each subdirectory for configuration.

## Reference

This repository was forked from [kg-gen on GitHub](https://github.com/stair-lab/kg-gen).
See the original repo for additional background.
